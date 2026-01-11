from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class DayInput:
    """
    Data class representing the input data for a single day of optimization.
    For this project, we used perfect foresight, meaning that
    the prices are known in advance for the entire day.
    In practice, one could input forecasted prices, but it's a topic on its own.

    index_ts : Timestamps for each time step in the day.
    T : Number of time steps in the day.
    dt : Duration of each time step in hours.
    price_energy :  Energy prices for each time step.
    price_fcr : FCR prices for each time step.
    price_afrr_up : aFRR up prices for each time step.
    price_afrr_down : aFRR down prices for each time step.
    soc0 : Initial state of charge as a fraction of the battery capacity.
    """
    index_ts: pd.DatetimeIndex
    T: int
    dt: float

    price_energy: np.ndarray[float]
    price_fcr: np.ndarray[float]
    price_afrr_up: np.ndarray[float]
    price_afrr_down: np.ndarray[float]

    soc0: Optional[float] = 10.0

    @classmethod
    def from_df(cls,
                day_df: pd.DataFrame,
                config: dict,
                soc0: Optional[float] = 10.0) -> "DayInput":
        """
        Infers DayInput from a DataFrame for a single day.

        day_df : DataFrame containing the data for a single day. The index should be Timestamps.
        config : Configuration dictionary.
        soc0 : Initial state of charge as a fraction of the battery capacity

        """
        return cls(
            index_ts=day_df.index,
            T=len(day_df),
            dt=(day_df.index[1] - day_df.index[0]).total_seconds() / 3600.0,
            price_energy=day_df[config["columns"]["energy"]].to_numpy(),
            price_fcr=day_df[config["columns"]["fcr"]].to_numpy(),
            price_afrr_up=day_df[config["columns"]["afrr_up"]].to_numpy(),
            price_afrr_down=day_df[config["columns"]["afrr_down"]].to_numpy(),
            soc0=soc0,
        )