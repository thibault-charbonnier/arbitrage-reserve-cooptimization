from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class DayInput:

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

        Parameters
        ----------
        day_df : pd.DataFrame
            DataFrame containing the data for a single day. The index should be Timestamps.
        config : dict
            Configuration dictionary.
        soc0 : Optional[float], optional, default is 0.5
            Initial state of charge as a fraction of the battery capacity, by default 0.5.
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