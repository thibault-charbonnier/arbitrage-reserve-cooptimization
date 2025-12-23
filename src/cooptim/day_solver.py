from typing import Any, Dict, Optional, Tuple
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
    Class representing the day-ahead co-optimization problem.
    It uses cvxpy to formulate and solve the optimization (LP/QP).

    Because we assume perfect foresight over the next day, we only perform
    one solve per day as everything is known.

    This module takes as input a 'DayInput' object.

    The output is a `DaySolution` with the optimal charge/discharge
    powers, reserve allocations, and state of charge over the day.
    Columns: p_ch_mw, p_dis_mw, r_fcr_mw, r_afrr_up_mw, r_afrr_down_mw, soc_mwh.

    The input and output DataFrames are indexed by the same UTC timestamps.
    Typically for our project we use a 15-minute time grid.
    """

    def __init__(self, battery: Battery, config: Dict[str, Any]) -> None:

        self.battery = battery
        self.config = config

        self._compiled: bool = False
        self._T: Optional[int] = None
        self._dt_hours: Optional[float] = None
        self._reserve_price_interval_minutes: int = int(
            self.config.get("reserve_price_interval_minutes", 15)
        )

        # Column mapping (allow flexible names)
        cols = self.config.get("columns", {}) or {}
        self.col_energy = cols.get("energy", "price_energy")
        self.col_fcr = cols.get("fcr", "price_fcr")
        self.col_afrr_up = cols.get("afrr_up", "price_afrr_up")
        self.col_afrr_down = cols.get("afrr_down", "price_afrr_down")

        # Constraints toggles
        constraints = self.config.get("constraints", {}) or {}
        self.use_power_headroom = bool(constraints.get("power_headroom", True))

        # fraction of activation over the horizon (0.25 = 25%)
        act = constraints.get("activation", {}) or {}
        self.alpha_up = float(act.get("alpha_up", 0.25))
        self.alpha_down = float(act.get("alpha_down", 0.25))

        eb = constraints.get("energy_buffer", {}) or {}
        self.use_energy_buffer = bool(eb.get("enabled", True))
        self.tau_hours = float(eb.get("tau_hours", 0.25))

        eod = constraints.get("end_of_day", {}) or {}
        self.use_end_of_day = bool(eod.get("enabled", False))
        self.end_of_day_mode = str(eod.get("mode", "equal"))  # "equal" or "min"
        self.end_of_day_target_mwh = eod.get("target_mwh", None)

        tp = constraints.get("throughput_penalty", {}) or {}
        self.use_throughput_penalty = bool(tp.get("enabled", True))
        self.c_throughput_eur_per_mwh = float(tp.get("c_eur_per_mwh", 0.0))

        # Solver settings
        solver_cfg = self.config.get("solver", {}) or {}
        self.solver_name = str(solver_cfg.get("name", "ECOS"))
        self.solver_opts = dict(solver_cfg.get("options", {}) or {})

        # cvxpy internals
        self._vars = {}
        self._params = {}
        self._prob: Optional[cp.Problem] = None

    def solve_day(self, day_input: DayInput) -> DaySolution:
        """
        Solves the day-ahead co-optimization problem for a single day as follows:
            1. Compile the cvxpy problem (declare variables, parameters, constraints, objective)
            2. Fill in the parameters from the DayInput
            3. Solve the problem
            4. Extract results and format into DaySolution
        """
        T = day_input.T
        dt_hours = day_input.dt
        soc0 = day_input.soc0

        self._compile(T=T, dt_hours=dt_hours)

        self._params["pi"].value = day_input.price_energy
        self._params["rho_fcr"].value = day_input.price_fcr
        self._params["rho_up"].value = day_input.price_afrr_up
        self._params["rho_down"].value = day_input.price_afrr_down
        self._params["soc0"].value = soc0

        self._prob.solve(solver=getattr(cp, self.solver_name), **self.solver_opts)

        logger.info(f"\tSolver status: {self._prob.status}")

        status = str(self._prob.status)
        solver_used = self.solver_name

        v = self._vars
        res = pd.DataFrame(
            {
                "p_ch_mw": np.asarray(v["p_ch"].value).reshape(-1),
                "p_dis_mw": np.asarray(v["p_dis"].value).reshape(-1),
                "r_fcr_mw": np.asarray(v["r_fcr"].value).reshape(-1),
                "r_afrr_up_mw": np.asarray(v["r_up"].value).reshape(-1),
                "r_afrr_down_mw": np.asarray(v["r_down"].value).reshape(-1),
                "soc_mwh": np.asarray(v["soc"].value).reshape(-1)[:-1],
                "a_act_up_mw": np.asarray(v["a_up"].value).reshape(-1),
                "a_act_down_mw": np.asarray(v["a_down"].value).reshape(-1),
            },
            index=day_input.index_ts,
        )

        logger.info(f"\tExtracting results ...")

        return DaySolution(
            date=pd.Timestamp(day_input.index_ts[0].date(), tz="UTC"),
            schedule=res,
            status=status,
            solver=solver_used,
            input=None
        )

    def _compile(self, T: int, dt_hours: float) -> None:
        """
        Compiles the cvxpy problem by defining variables, parameters, constraints, and objective.
        This is required before solving the problem.
        """
        logger.info("\tCompiling optimization problem ...")

        b: Battery = self.battery

        # Variables
        p_ch = cp.Variable(T, nonneg=True)
        p_dis = cp.Variable(T, nonneg=True)
        r_fcr = cp.Variable(T, nonneg=True)
        r_up = cp.Variable(T, nonneg=True)
        r_down = cp.Variable(T, nonneg=True)
        soc = cp.Variable(T + 1)
        a_up = cp.Variable(T, nonneg=True)
        a_down = cp.Variable(T, nonneg=True)

        # Parameters
        pi = cp.Parameter(T)
        rho_fcr = cp.Parameter(T)
        rho_up = cp.Parameter(T)
        rho_down = cp.Parameter(T)
        soc0 = cp.Parameter()

        # Constraints
        constraints = []
        constraints += [soc[0] == soc0]
        constraints += [b.soc_min <= soc, soc <= b.soc_max]

        for t in range(T):
            # SoC dynamics are defined as constraints
            # Energy at next time step = current + charged - discharged
            # where charge is charging power planned at that step * efficiency * dt (hours)
            # and discharge is discharging power / efficiency * dt (hours)
            # we also account for the energy used to provide aFRR reserves (coefficients a_up, a_down)
            constraints += [
                soc[t + 1] == soc[t] + b.eta_ch * (p_ch[t] + a_down[t] * dt_hours) \
                              - (1.0 / b.eta_dis) * (p_dis[t] + a_up[t] * dt_hours)
            ]

            # Power headroom 
            # Charging power can not exceed max power minus reserves allocated
            # Discharging power can not exceed max power minus reserves allocated
            constraints += [
                p_dis[t] + r_fcr[t] + r_up[t] <= b.p_dis_max_mw,
                p_ch[t] + r_fcr[t] + r_down[t] <= b.p_ch_max_mw,
            ]

            # 
            constraints += [
                a_up[t] <= r_up[t],
                a_down[t] <= r_down[t],
            ]

            # Energy buffer
            # Ensure that enough energy is reserved to provide frequency reserves
            # during the time interval tau
            if self.use_energy_buffer:
                tau = self.tau_hours
                constraints += [
                    soc[t] >= b.soc_min + (r_fcr[t] + r_up[t]) * tau,
                    soc[t] <= b.soc_max - (r_fcr[t] + r_down[t]) * tau,
                ]

        # End-of-day policy for SoC
        # One wants to ensure that the battery level meets some value at the end of the day
        # to avoid depleting it entirely over time and starting the next day empty.
        if self.use_end_of_day:
            if self.end_of_day_mode == "equal":
                constraints += [soc[T] == soc0]
            elif self.end_of_day_mode == "min":
                if self.end_of_day_target_mwh is None:
                    raise ValueError("end_of_day.mode='min' requires end_of_day.target_mwh in config")
                constraints += [soc[T] >= float(self.end_of_day_target_mwh)]
            else:
                raise ValueError("end_of_day.mode must be 'equal' or 'min'")

        # aFRR activation constraints
        # Ensure that the total activated aFRR energy equals the allocated reserve energy
        constraints += [
            cp.sum(a_up) * dt_hours == self.alpha_up * cp.sum(r_up) * dt_hours,
            cp.sum(a_down) * dt_hours == self.alpha_down * cp.sum(r_down) * dt_hours,
        ]
                
        # Objective
        ## Revenues
        rev_energy = cp.sum(cp.multiply(pi, (p_dis - p_ch)) * dt_hours)
        dt_minutes = dt_hours * 60.0
        scale = dt_minutes / float(self._reserve_price_interval_minutes)
        rev_reserve = cp.sum(
            (cp.multiply(rho_fcr, r_fcr) + cp.multiply(rho_up, r_up) + cp.multiply(rho_down, r_down)) * scale
        )

        ## Costs
        cost_throughput = 0.0
        if self.use_throughput_penalty and self.c_throughput_eur_per_mwh > 0:
            cost_throughput = float(self.c_throughput_eur_per_mwh) * cp.sum((p_ch + p_dis) * dt_hours)

        ## aFRR activation revenue
        rev_activation = cp.sum(cp.multiply(pi, (a_up - a_down)) * dt_hours)

        # Formulate cvxpy maximization problem
        objective = cp.Maximize(rev_energy + rev_reserve + rev_activation - cost_throughput)
        prob = cp.Problem(objective, constraints)

        self._vars = {
            "p_ch": p_ch,
            "p_dis": p_dis,
            "r_fcr": r_fcr,
            "r_up": r_up,
            "r_down": r_down,
            "soc": soc,
            "a_up": a_up,
            "a_down": a_down,
        }
        self._params = {
            "pi": pi,
            "rho_fcr": rho_fcr,
            "rho_up": rho_up,
            "rho_down": rho_down,
            "soc0": soc0,
        }
        self._prob = prob
        self._compiled = True
        self._T = T
        self._dt_hours = dt_hours
