import pandas as pd
import numpy as np
from typing import List
import logging

from src.cooptim.battery import Battery
from src.cooptim.day_input import DayInput
from src.cooptim.solution import DaySolution
from src.cooptim.day_solver import DaySolver

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestrator class to manage the co-optimization process over multiple days.

    Responsible for:
        - Loading data from parquet files.
        - Initializing the Battery instance from configuration.
        - Iterating over the specified date range to solve daily optimization problems.
    """
    def __init__(self, config: dict):
        self.config = config
        self.battery = self._load_battery()
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """
        Load energy and reserve price data from parquet files and create a unique dataframe.

        Returns
        -------
        pd.DataFrame
            DataFrame containing joined energy and reserve prices.
        """
        logger.info("Loading data ...")

        energy_prices = pd.read_parquet(self.config["data"]["energy_prices_parquet"])
        reserve_prices = pd.read_parquet(self.config["data"]["reserve_prices_parquet"])

        return energy_prices.join(reserve_prices, how="inner")
    
    def _load_battery(self) -> Battery:
        """
        Load battery specifications from the configuration dictionary and create a Battery instance.

        Returns
        -------
        Battery
            Instance of the Battery class with specifications from config.
        """
        logger.info("Loading battery ...")

        battery_config = self.config["battery"]

        return Battery(
            e_max_mwh=battery_config["e_max_mwh"],
            p_ch_max_mw=battery_config["p_ch_max_mw"],
            p_dis_max_mw=battery_config["p_dis_max_mw"],
            eta_ch=battery_config["eta_ch"],
            eta_dis=battery_config["eta_dis"],
            soc_min=battery_config["soc_min"],
            soc_max=battery_config["soc_max"],
        )
    
    def run(self) -> List[DaySolution]:
        """
        Main orchestration method to run the co-optimization process.

        Returns
        -------
        List[DaySolution]
            List of DaySolution instances for each day in the specified date range.
        """
        start_date = pd.to_datetime(self.config["run"]["start_date"])
        end_date = pd.to_datetime(self.config["run"]["end_date"])

        logger.info(f"Running co-optimization from {start_date.date()} to {end_date.date()} ...")
        solutions = []
        current_date = start_date

        previous_soc_end = None
        while current_date <= end_date:

            logger.info(f"\tSolving for date: {current_date.date()}")

            day_data = self.data[self.data.index.normalize().date == current_date.date()]
            
            if day_data.empty:
                logger.warning(f"\tNo data available for date: {current_date.date()}, skipping.")
                current_date += pd.Timedelta(days=1)
                continue

            # By default, start at 50% SoC if no previous day to ensure multi-day continuity
            soc_init = previous_soc_end if previous_soc_end is not None else 5.0

            day_input = DayInput.from_df(
                day_df=day_data,
                config=self.config,
                soc0=soc_init
            )

            solver = DaySolver(battery=self.battery, config=self.config)
            day_solution: DaySolution = solver.solve_day(day_input=day_input)

            day_solution.input = day_data
            solutions.append(day_solution)

            if not day_solution.schedule.empty:
                previous_soc_end = day_solution.schedule["soc_mwh"].iloc[-1]

            current_date += pd.Timedelta(days=1)

        return solutions