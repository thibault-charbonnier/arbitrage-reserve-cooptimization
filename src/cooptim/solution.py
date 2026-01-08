from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


@dataclass()
class DaySolution:
    """
    Represents the solution of the optimization for a single day.

    Attributes
    ----------
    date : pd.Timestamp
        The date of the optimization day.
    input : pd.DataFrame
        The input data used for the optimization (prices, etc.).
    schedule : pd.DataFrame
        The resulting schedule from the optimization (bids, SoC, etc.).
    status : str
        The status of the optimization (e.g., "optimal", "infeasible").
    solver : str
        The solver name used for the optimization.
    """
    date: pd.Timestamp
    input: pd.DataFrame
    schedule: pd.DataFrame
    status: str
    solver: str

    def plot_results(self, config: Optional[Dict[str, str]] = None):
        """
        Create a multi-panel plot showing:
            (1) Reserve bids (FCR, aFRR UP/DOWN) and SoC
            (2) Energy prices over time
            (3) Reserve capacity prices over time

        Parameters
        ----------
        config : Optional[Dict[str, str]]
            Configuration dictionary containing column names for prices.
        """
        inp = self.input.copy()
        sch = self.schedule.copy()

        sch = sch.reindex(inp.index)

        c_price_e = config["columns"]["energy"]
        c_price_fcr = config["columns"]["fcr"]
        c_price_up = config["columns"]["afrr_up"]
        c_price_down = config["columns"]["afrr_down"]

        c_r_fcr = "r_fcr_mw"
        c_r_up = "r_afrr_up_mw"
        c_r_down = "r_afrr_down_mw"
        c_soc = "soc_mwh"

        x = inp.index
        if len(x) >= 2:
            dt_seconds = float(np.median(np.diff(x.view("int64")) / 1e9))
        else:
            dt_seconds = 900.0
        width_days = (dt_seconds / 86400.0) * 0.85

        r_fcr = sch[c_r_fcr].to_numpy(dtype=float)
        r_up = sch[c_r_up].to_numpy(dtype=float)
        r_down = sch[c_r_down].to_numpy(dtype=float)
        soc = sch[c_soc].to_numpy(dtype=float)

        def nan_to_zero(a: np.ndarray) -> np.ndarray:
            return np.nan_to_num(a, nan=0.0)

        r_fcr = nan_to_zero(r_fcr)
        r_up = nan_to_zero(r_up)
        r_down = nan_to_zero(r_down)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

        ax1.set_title(f"Reserve Bids and SoC — {self.date.date()} ({self.status}, {self.solver})")

        x_num = mdates.date2num(x.to_pydatetime())
        w = width_days
        ax1.bar(x_num - w/3, r_fcr, width=w/3, label="FCR bid (MW)")
        ax1.bar(x_num, r_down, width=w/3, label="aFRR DOWN bid (MW)")
        ax1.bar(x_num + w/3,  r_up,   width=w/3, label="aFRR UP bid (MW)")
        ax1.set_ylabel("Bid size (MW)")
        ax1.grid(True, alpha=0.3)

        ax1b = ax1.twinx()
        ax1b.plot(x, soc, linewidth=2.0, label="SoC")
        ax1b.set_ylabel("SoC (MWh)")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1b.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)

        ax2.plot(x, inp[c_price_e].to_numpy(dtype=float), label="Spot / DA Price")
        ax2.set_ylabel("Price (€/MWh)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper left", frameon=True)
        ax2.set_title("Energy Prices Over Time")

        ax3.plot(x, inp[c_price_fcr].to_numpy(dtype=float), label="FCR Price")
        ax3.plot(x, inp[c_price_up].to_numpy(dtype=float), label="aFRR UP Price")
        ax3.plot(x, inp[c_price_down].to_numpy(dtype=float), label="aFRR DOWN Price")
        ax3.set_ylabel("Reserve price (€/MW/interval)")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper right", frameon=True)
        ax3.set_title("Reserve Capacity Prices Over Time")

        ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.autofmt_xdate(rotation=0)

        fig.tight_layout()
        fig.show()

def plot_global_results(solutions: List[DaySolution], config: Dict[str, Any]):
    """
    Concatenate all daily solutions and plot the full horizon results.

    New layout (more readable):
      (1) FCR bids + SoC
      (2) aFRR UP/DOWN bids + SoC
      (3) Spot/DA energy price
      (4) Reserve capacity prices (FCR, aFRR UP, aFRR DOWN)

    Parameters
    ----------
    solutions : List[DaySolution]
        List of daily solutions to concatenate and plot.
    config : Dict[str, Any]
        Configuration dictionary containing column names for prices.
    """

    full_input = pd.concat([s.input for s in solutions])
    full_schedule = pd.concat([s.schedule for s in solutions])

    full_input.sort_index(inplace=True)
    full_schedule.sort_index(inplace=True)

    c_price_e = config["columns"]["energy"]
    c_price_fcr = config["columns"]["fcr"]
    c_price_up = config["columns"]["afrr_up"]
    c_price_down = config["columns"]["afrr_down"]

    x = full_input.index

    dt_seconds = (x[1] - x[0]).total_seconds()
    width_days = (dt_seconds / 86400.0) * 0.85

    r_fcr = full_schedule["r_fcr_mw"].reindex(x).fillna(0.0).to_numpy()
    r_up = full_schedule["r_afrr_up_mw"].reindex(x).fillna(0.0).to_numpy()
    r_down = full_schedule["r_afrr_down_mw"].reindex(x).fillna(0.0).to_numpy()
    soc = full_schedule["soc_mwh"].reindex(x).to_numpy()

    x_num = mdates.date2num(x.to_pydatetime())
    w = width_days

    fig, (ax_fcr, ax_afrr, ax_energy, ax_prices) = plt.subplots(
        4, 1, figsize=(12, 12), sharex=True
    )

    start_str = x[0].date()
    end_str = x[-1].date()
    fig.suptitle(f"Global Optimization Results: {start_str} to {end_str}", y=0.995)

    ax_fcr.bar(x_num, r_fcr, width=w, label="FCR bid (MW)")
    ax_fcr.set_ylabel("FCR (MW)")
    ax_fcr.grid(True, alpha=0.3)
    ax_fcr.legend(loc="upper left")

    ax_fcr_soc = ax_fcr.twinx()
    ax_fcr_soc.plot(x, soc, "k", linewidth=1.2, label="SoC")
    ax_fcr_soc.set_ylabel("SoC (MWh)")
    ax_fcr_soc.legend(loc="upper right")

    ax_afrr.bar(x_num, r_down, color='tab:green', width=w, label="aFRR DOWN bid (MW)")
    ax_afrr.bar(x_num, r_up, color='tab:orange', width=w, bottom=r_down, label="aFRR UP bid (MW)")
    ax_afrr.set_ylabel("aFRR (MW)")
    ax_afrr.grid(True, alpha=0.3)
    ax_afrr.legend(loc="upper left")

    ax_afrr_soc = ax_afrr.twinx()
    ax_afrr_soc.plot(x, soc, "k", linewidth=1.2, label="SoC")
    ax_afrr_soc.set_ylabel("SoC (MWh)")

    ax_energy.plot(x, full_input[c_price_e].to_numpy(dtype=float), label="Spot / DA Price")
    ax_energy.set_ylabel("€/MWh")
    ax_energy.grid(True, alpha=0.3)
    ax_energy.legend(loc="upper left")

    ax_prices.plot(x, full_input[c_price_fcr].to_numpy(dtype=float), label="FCR Price")
    ax_prices.plot(x, full_input[c_price_up].to_numpy(dtype=float), label="aFRR UP Price")
    ax_prices.plot(x, full_input[c_price_down].to_numpy(dtype=float), label="aFRR DOWN Price")
    ax_prices.set_ylabel("€/MW/interval")
    ax_prices.grid(True, alpha=0.3)
    ax_prices.legend(loc="upper left")

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax_prices.xaxis.set_major_locator(locator)
    ax_prices.xaxis.set_major_formatter(formatter)

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    plt.show()
