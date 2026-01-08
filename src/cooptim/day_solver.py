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
    Day-ahead co-optimization for a battery (energy + reserves)

    Optimizes delta-t grid scheduling of a battery over one day (T time steps) with
    a given forecast of energy and reserve prices. The model selects planned
    charge/discharge powers, reserve capacities and expected average activations,
    while respecting energy dynamics, converter power limits, deliverability
    buffers and operational rules (activation ratios, optional FCR symmetry,
    end-of-day constraints).

    All decision variables are per time step t.
    Some constraints are optional and configurable in the global config
    as well as most parameters.

    -----------------------------------
    The optimization is formulated and solved using CVXPY.
    Find below a summary of the formulated problem with decision variables,
    constraints and objective function :

    Decision variables (per time step t):
        - p_ch[t] : scheduled charging power (MW).
        - p_dis[t]: scheduled discharging power (MW).
        - r_fcr[t]: FCR capacity offered (MW).
        - r_up[t] : aFRR up capacity offered (MW).
        - r_down[t]: aFRR down capacity offered (MW).
        - a_up[t] : activation in "up" direction (MW).
        - a_down[t]: activation in "down" direction (MW).
        - soc[t]  : state-of-charge (MWh), with soc[0] == soc0 and soc[T] constrained at day end.

    Key constraints:
        1. Energy balance (MWh):
           soc[t+1] = soc[t] + eta_ch * (p_ch[t] + a_down[t]) * dt_hours
                                - (1/eta_dis) * (p_dis[t] + a_up[t]) * dt_hours

           Note:
           - a_down increases stored energy (activation that charges),
           - a_up decreases stored energy (activation that discharges).

        2. Converter power limits (MW):
           p_dis[t] + r_up[t] <= p_dis_max
           p_ch[t] + r_down[t] <= p_ch_max

        3. Activation bounded by reserved capacity:
           a_up[t] <= r_up[t]
           a_down[t] <= r_down[t]

        4. FCR symmetry (optional):
           If `enforce_fcr_symmetry` is true, FCR capacity requires headroom in both
           directions via f = fcr_derate * r_fcr[t]:
             p_dis[t] + r_up[t] + f <= p_dis_max
             p_ch[t] + r_down[t] + f <= p_ch_max
           Otherwise FCR is added to the respective side in a less strict form.

        5. Energy deliverability buffer (optional):
           If enabled, the battery must reserve energy margin to be able to deliver
           reserves over an horizon `tau_hours`:
             soc[t] >= soc_min + (f + r_up[t]) * tau
             soc[t] <= soc_max - (f + r_down[t]) * tau
           with f = fcr_derate * r_fcr[t].

        6. Activation average (ratio) constraints:
           - Mode `daily` (default):
               sum(a_up) == alpha_up * sum(r_up)
               sum(a_down) == alpha_down * sum(r_down)
           - Mode `block`:
             The same ratio is enforced per block of length
             `reserve_price_interval_minutes` (converted to number of steps),
             preventing all activation in a single step while remaining convex.

        7. End-of-day:
           - Default ensures soc[T] >= soc0 (do not end lower than start).
           - If `end_of_day.enabled` is set, can force soc[T] >= configured minimum.

    Objective:
        Maximize net revenue:
          Max rev_energy + rev_reserve - cost_throughput - cost_reserve

        - rev_energy: sum_t pi[t] * (p_dis - p_ch + a_up - a_down) * dt_hours
          (net energy sold).
        - rev_reserve: sum_t (rho_fcr*r_fcr + rho_up*r_up + rho_down*r_down) * dt_hours
          (capacity reserve revenues).
        - cost_throughput: throughput penalty c_throughput_eur_per_mwh * sum((p_ch+p_dis)*dt_hours)
          to discourage simultaneous charge/discharge.
        - cost_reserve: small regularizer c_reserve_eur_per_mw * sum(r_fcr + r_up + r_down).
    """

    def __init__(self, battery: Battery, config: Dict[str, Any]) -> None:
        self.battery = battery
        self.config = config

        act = (config.get("activation", {}) or {})
        self.alpha_up = float(act.get("alpha_up", 0.25))
        self.alpha_down = float(act.get("alpha_down", 0.25))

        self.activation_mode = str(act.get("mode", "daily"))
        self.reserve_price_interval_minutes = int(act.get("reserve_price_interval_minutes", 15))

        tp = (config.get("throughput_penalty", {}) or {})
        self.c_throughput_eur_per_mwh = float(tp.get("c_eur_per_mwh", 0.0))

        rp = (config.get("reserve_penalty", {}) or {})
        self.c_reserve_eur_per_mw = float(rp.get("c_eur_per_mw", 0.0))

        eb = (config.get("energy_buffer", {}) or {})
        self.use_energy_buffer = bool(eb.get("enabled", True))
        self.tau_hours = float(eb.get("tau_hours", 0.25))

        fcr_cfg = (config.get("fcr", {}) or {})
        self.enforce_fcr_symmetry = bool(fcr_cfg.get("enforce_symmetry", True))
        self.fcr_derate = float(fcr_cfg.get("derate", 1.0))

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
        """
        Solve the day-ahead optimization for a given day's input data.

        Parameters
        ----------
        day_input : DayInput
            Input data for the day, including price forecasts and initial SoC.

        Returns
        -------
        DaySolution
            The optimization result, including the schedule and status.
        """
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
        """
        Formulate the optimization problem for CVXPY using CVXPY variables and parameters.

        Parameters
        ----------
        T : int
            Number of time steps in the day.
        dt_hours : float
            Duration of each time step in hours.
        """
        b = self.battery

        # ================
        # Decision Variables
        # ================
        p_ch = cp.Variable(T, nonneg=True)
        p_dis = cp.Variable(T, nonneg=True)

        r_fcr = cp.Variable(T, nonneg=True)
        r_up = cp.Variable(T, nonneg=True)
        r_down = cp.Variable(T, nonneg=True)

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
            constraints += [
                soc[t + 1] == soc[t]
                + b.eta_ch * (p_ch[t] + a_down[t]) * dt_hours
                - (1.0 / b.eta_dis) * (p_dis[t] + a_up[t]) * dt_hours
            ]

            # ---- Headroom for planned dispatch + reserves ----
            constraints += [
                p_dis[t] + r_up[t] <= b.p_dis_max_mw,
                p_ch[t] + r_down[t] <= b.p_ch_max_mw,
            ]

            # ---- Activation must be within reserved capacity ----
            constraints += [
                a_up[t] <= r_up[t],
                a_down[t] <= r_down[t],
            ]

            # ---- FCR symmetry---
            if self.enforce_fcr_symmetry:
                f = self.fcr_derate * r_fcr[t]
                constraints += [
                    p_dis[t] + r_up[t] + f <= b.p_dis_max_mw,
                    p_ch[t] + r_down[t] + f <= b.p_ch_max_mw,
                ]
            else:
                constraints += [
                    p_dis[t] + r_fcr[t] + r_up[t] <= b.p_dis_max_mw,
                    p_ch[t] + r_fcr[t] + r_down[t] <= b.p_ch_max_mw,
                ]

            # ---- Energy buffer----
            if self.use_energy_buffer:
                f = self.fcr_derate * r_fcr[t]
                constraints += [
                    soc[t] >= soc_min_mwh + (f + r_up[t]) * tau,
                    soc[t] <= soc_max_mwh - (f + r_down[t]) * tau,
                ]

        # ---- Activation ratio ----
        if self.activation_mode == "daily":
            constraints += [
                cp.sum(a_up) == self.alpha_up * cp.sum(r_up),
                cp.sum(a_down) == self.alpha_down * cp.sum(r_down),
            ]
        elif self.activation_mode == "block":
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

        # Reserve capacity revenue.
        rev_reserve = cp.sum((
            cp.multiply(rho_fcr, r_fcr)
            + cp.multiply(rho_up, r_up)
            + cp.multiply(rho_down, r_down)
        ) * dt_hours)

        # Throughput penalty discourages simultaneous charge/discharge
        cost_throughput = 0.0
        if self.c_throughput_eur_per_mwh > 0:
            cost_throughput = self.c_throughput_eur_per_mwh * cp.sum((p_ch + p_dis) * dt_hours)

        # Optional reserve-holding penalty
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
