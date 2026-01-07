from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import cvxpy as cp
import logging

from .battery import Battery
from .solution import DaySolution
from .day_input import DayInput

logger = logging.getLogger(__name__)


class DaySolver:
    """
    Day-ahead co-optimization for a battery with perfect foresight over one day.

    Changes vs your current version (more realistic constraints, still convex):
      1) Energy buffer (tau) made configurable and properly tied to SoC bounds
      2) FCR symmetry enforced with dedicated up/down headroom + SoC margin (typical for FCR)
      3) Activation "average" constraints made dimensionally consistent and optionally applied per block
      4) Prevent "free" reserve saturation by adding optional small reserve-holding penalty
         (helps when rho_up is always attractive and you want smoother allocations)
      5) Make sure any activation/energy links are consistent: a_up/a_down are MW and enter SoC via *dt

    Notes:
      - Battery.soc_min/max are ratios, soc is in MWh.
      - Prices rho_* are assumed already per time step (€/MW/interval). If your data are €/MW/h,
        you must rescale (see comment in code).
    """

    def __init__(self, battery: Battery, config: Dict[str, Any]) -> None:
        self.battery = battery
        self.config = config

        act = (config.get("activation", {}) or {})
        self.alpha_up = float(act.get("alpha_up", 0.25))
        self.alpha_down = float(act.get("alpha_down", 0.25))

        # Apply activation ratio constraint either:
        # - "daily": over the full day (as you do now)
        # - "block": per reserve price interval block (more realistic, still convex)
        self.activation_mode = str(act.get("mode", "daily"))  # "daily" or "block"
        self.reserve_price_interval_minutes = int(act.get("reserve_price_interval_minutes", 15))

        tp = (config.get("throughput_penalty", {}) or {})
        self.c_throughput_eur_per_mwh = float(tp.get("c_eur_per_mwh", 0.0))

        # NEW: optional small penalty on holding reserves (avoid always-max bids when prices are positive)
        # Keep it tiny (e.g. 1e-3 to 1e-2 €/MW/interval) – purely a regularizer.
        rp = (config.get("reserve_penalty", {}) or {})
        self.c_reserve_eur_per_mw = float(rp.get("c_eur_per_mw", 0.0))

        # Energy buffer / deliverability horizon tau (hours)
        eb = (config.get("energy_buffer", {}) or {})
        self.use_energy_buffer = bool(eb.get("enabled", True))
        self.tau_hours = float(eb.get("tau_hours", 0.25))  # typical 0.25h for 15min deliverability check

        # NEW: enforce FCR symmetry explicitly (up and down headroom)
        fcr_cfg = (config.get("fcr", {}) or {})
        self.enforce_fcr_symmetry = bool(fcr_cfg.get("enforce_symmetry", True))
        # Some operators derate FCR (e.g., only a fraction is assumed deliverable). Optional.
        self.fcr_derate = float(fcr_cfg.get("derate", 1.0))  # 1.0 = no derating

        eod = (config.get("end_of_day", {}) or {})
        self.use_end_of_day = bool(eod.get("enabled", False))
        self.end_of_day_min_soc_mwh = eod.get("min_soc_mwh", None)

        solver_cfg = (config.get("solver", {}) or {})
        self.solver_name = str(solver_cfg.get("name", "ECOS"))
        self.solver_opts = dict(solver_cfg.get("options", {}) or {})

        self._vars: Dict[str, cp.Expression] = {}
        self._params: Dict[str, cp.Parameter] = {}
        self._prob: Optional[cp.Problem] = None

    def solve_day(self, day_input: DayInput) -> DaySolution:
        T = int(day_input.T)
        dt_hours = float(day_input.dt)
        soc0 = float(day_input.soc0)

        self._compile(T=T, dt_hours=dt_hours)

        self._params["pi"].value = np.asarray(day_input.price_energy).reshape(T)
        self._params["rho_fcr"].value = np.asarray(day_input.price_fcr).reshape(T)
        self._params["rho_up"].value = np.asarray(day_input.price_afrr_up).reshape(T)
        self._params["rho_down"].value = np.asarray(day_input.price_afrr_down).reshape(T)
        self._params["soc0"].value = soc0

        try:
            self._prob.solve(solver=getattr(cp, self.solver_name), **self.solver_opts)
        except Exception as e:
            logger.error(f"Solver failed: {e}")
            return DaySolution(
                date=pd.Timestamp(day_input.index_ts[0].date(), tz="UTC"),
                schedule=pd.DataFrame(index=day_input.index_ts),
                status="failed",
                solver=self.solver_name,
                input=None,
            )

        logger.info(f"\tSolver status: {self._prob.status}")

        v = self._vars

        def clean_nonneg(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
            x = np.asarray(x).reshape(-1)
            x[np.abs(x) < eps] = 0.0
            return np.maximum(x, 0.0)

        res = pd.DataFrame(
            {
                "p_ch_mw": clean_nonneg(v["p_ch"].value),
                "p_dis_mw": clean_nonneg(v["p_dis"].value),
                "r_fcr_mw": clean_nonneg(v["r_fcr"].value),
                "r_afrr_up_mw": clean_nonneg(v["r_up"].value),
                "r_afrr_down_mw": clean_nonneg(v["r_down"].value),
                "a_act_up_mw": clean_nonneg(v["a_up"].value),
                "a_act_down_mw": clean_nonneg(v["a_down"].value),
                "soc_mwh": np.asarray(v["soc"].value).reshape(-1)[:-1],
            },
            index=day_input.index_ts,
        )

        return DaySolution(
            date=pd.Timestamp(day_input.index_ts[0].date(), tz="UTC"),
            schedule=res,
            status=str(self._prob.status),
            solver=self.solver_name,
            input=None,
        )

    def _compile(self, T: int, dt_hours: float) -> None:
        b = self.battery

        # ================
        # Variables
        # ================
        p_ch = cp.Variable(T, nonneg=True)
        p_dis = cp.Variable(T, nonneg=True)

        r_fcr = cp.Variable(T, nonneg=True)
        r_up = cp.Variable(T, nonneg=True)
        r_down = cp.Variable(T, nonneg=True)

        # Model activation as MW (average/expected) – enters energy balance via *dt_hours
        a_up = cp.Variable(T, nonneg=True)
        a_down = cp.Variable(T, nonneg=True)

        soc = cp.Variable(T + 1)

        # ================
        # Parameters
        # ================
        pi = cp.Parameter(T)
        rho_fcr = cp.Parameter(T)
        rho_up = cp.Parameter(T)
        rho_down = cp.Parameter(T)
        soc0 = cp.Parameter()

        # ================
        # Bounds helpers
        # ================
        soc_min_mwh = b.soc_min * b.e_max_mwh
        soc_max_mwh = b.soc_max * b.e_max_mwh
        tau = float(self.tau_hours)

        # ================
        # Constraints
        # ================
        constraints = []
        constraints += [soc[0] == soc0]
        constraints += [soc >= soc_min_mwh, soc <= soc_max_mwh]

        for t in range(T):
            # ---- Energy dynamics (MWh) ----
            # Activation a_down means "extra charging", a_up means "extra discharging"
            constraints += [
                soc[t + 1] == soc[t]
                + b.eta_ch * (p_ch[t] + a_down[t]) * dt_hours
                - (1.0 / b.eta_dis) * (p_dis[t] + a_up[t]) * dt_hours
            ]

            # ---- Headroom for planned dispatch + reserves ----
            # aFRR is directional and uses the corresponding converter direction.
            # FCR is symmetric -> needs headroom in BOTH directions (see below).
            constraints += [
                p_dis[t] + r_up[t] <= b.p_dis_max_mw,
                p_ch[t] + r_down[t] <= b.p_ch_max_mw,
            ]

            # ---- Activation must be within reserved capacity ----
            constraints += [
                a_up[t] <= r_up[t],
                a_down[t] <= r_down[t],
            ]

            # ---- FCR symmetry (more realistic) ----
            # If you offer r_fcr, you must be able to:
            #  - increase net injection by r_fcr (up direction)
            #  - decrease net injection by r_fcr (down direction)
            # This implies converter headroom BOTH ways.
            if self.enforce_fcr_symmetry:
                f = self.fcr_derate * r_fcr[t]
                constraints += [
                    # Upward response uses discharge capability (reduce charge or increase discharge)
                    p_dis[t] + r_up[t] + f <= b.p_dis_max_mw,
                    # Downward response uses charge capability (reduce discharge or increase charge)
                    p_ch[t] + r_down[t] + f <= b.p_ch_max_mw,
                ]
            else:
                # Fallback (less strict): share headroom with both directions (your previous form)
                constraints += [
                    p_dis[t] + r_fcr[t] + r_up[t] <= b.p_dis_max_mw,
                    p_ch[t] + r_fcr[t] + r_down[t] <= b.p_ch_max_mw,
                ]

            # ---- Energy deliverability buffer (critical realism) ----
            # To offer upward reserves (FCR+UP), you need energy above soc_min.
            # To offer downward reserves (FCR+DOWN), you need empty space below soc_max.
            if self.use_energy_buffer:
                f = self.fcr_derate * r_fcr[t]
                constraints += [
                    soc[t] >= soc_min_mwh + (f + r_up[t]) * tau,
                    soc[t] <= soc_max_mwh - (f + r_down[t]) * tau,
                ]

        # ---- Activation ratio constraints (dimensionally consistent) ----
        # a_* and r_* are MW. Summing them over T gives "MW-steps". Ratio is fine without dt.
        # (If you ever switch to defining a_* as MWh, you'd need dt. Here you do not.)
        if self.activation_mode == "daily":
            constraints += [
                cp.sum(a_up) == self.alpha_up * cp.sum(r_up),
                cp.sum(a_down) == self.alpha_down * cp.sum(r_down),
            ]
        elif self.activation_mode == "block":
            # Enforce the average activation ratio over blocks of reserve-price intervals.
            # This avoids pathological "all activation in one step" while staying convex.
            step_minutes = dt_hours * 60.0
            block_len = int(round(self.reserve_price_interval_minutes / step_minutes))
            block_len = max(block_len, 1)

            for k0 in range(0, T, block_len):
                k1 = min(T, k0 + block_len)
                constraints += [
                    cp.sum(a_up[k0:k1]) == self.alpha_up * cp.sum(r_up[k0:k1]),
                    cp.sum(a_down[k0:k1]) == self.alpha_down * cp.sum(r_down[k0:k1]),
                ]
        else:
            raise ValueError("activation.mode must be 'daily' or 'block'")

        # ---- End-of-day constraint ----
        if self.use_end_of_day:
            if self.end_of_day_min_soc_mwh is not None:
                constraints += [soc[T] >= float(self.end_of_day_min_soc_mwh)]
            else:
                constraints += [soc[T] >= soc0]
        else:
            constraints += [soc[T] >= soc0]

        # ================
        # Objective
        # ================
        # Energy revenue (€/MWh) * (MW) * dt (h)
        rev_energy = cp.sum(cp.multiply(pi, (p_dis - p_ch + a_up - a_down)) * dt_hours)

        # Reserve capacity revenue (assumed €/MW/interval already)
        # If instead your rho_* are €/MW/h, multiply by dt_hours here.
        rev_reserve = cp.sum(
            cp.multiply(rho_fcr, r_fcr)
            + cp.multiply(rho_up, r_up)
            + cp.multiply(rho_down, r_down)
        )

        # Throughput penalty discourages simultaneous charge/discharge
        cost_throughput = 0.0
        if self.c_throughput_eur_per_mwh > 0:
            cost_throughput = self.c_throughput_eur_per_mwh * cp.sum((p_ch + p_dis) * dt_hours)

        # Optional reserve-holding penalty (tiny regularizer)
        cost_reserve = 0.0
        if self.c_reserve_eur_per_mw > 0:
            cost_reserve = self.c_reserve_eur_per_mw * cp.sum(r_fcr + r_up + r_down)

        objective = cp.Maximize(rev_energy + rev_reserve - cost_throughput - cost_reserve)

        self._prob = cp.Problem(objective, constraints)

        self._params = {
            "pi": pi,
            "rho_fcr": rho_fcr,
            "rho_up": rho_up,
            "rho_down": rho_down,
            "soc0": soc0,
        }
        self._vars = {
            "p_ch": p_ch,
            "p_dis": p_dis,
            "r_fcr": r_fcr,
            "r_up": r_up,
            "r_down": r_down,
            "a_up": a_up,
            "a_down": a_down,
            "soc": soc,
        }
