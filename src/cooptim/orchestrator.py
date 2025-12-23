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

    def __init__(self, config: dict):
        self.config = config
        self.battery = self._load_battery()
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """
        Load energy and reserve price data from parquet files and create a unique dataframe.
        """
        logger.info("Loading data ...")

        energy_prices = pd.read_parquet(self.config["data"]["energy_prices_parquet"])
        reserve_prices = pd.read_parquet(self.config["data"]["reserve_prices_parquet"])

        data = energy_prices.join(reserve_prices, how="inner")

        return data
    
    def _load_battery(self) -> Battery:
        """
        Load battery specifications from the configuration dictionary and create a Battery instance.
        """
        logger.info("Loading battery ...")

        battery_config = self.config["battery"]
        battery = Battery(
            e_max_mwh=battery_config["e_max_mwh"],
            p_ch_max_mw=battery_config["p_ch_max_mw"],
            p_dis_max_mw=battery_config["p_dis_max_mw"],
            eta_ch=battery_config["eta_ch"],
            eta_dis=battery_config["eta_dis"],
            soc_min=battery_config["soc_min_mwh"],
            soc_max=battery_config["soc_max_mwh"],
        )
        return battery
    
    def run(self) -> List[DaySolution]:
        """
        Main orchestration method to run the co-optimization process.
        """
        start_date = pd.to_datetime(self.config["run"]["start_date"])
        end_date = pd.to_datetime(self.config["run"]["end_date"])

        logger.info(f"Running co-optimization from {start_date.date()} to {end_date.date()} ...")
        solutions = []
        current_date = start_date
        while current_date <= end_date:

            logger.info(f"\tSolving for date: {current_date.date()}")

            day_data = self.data[self.data.index.normalize().date == current_date.date()]
            if day_data.empty:
                logger.warning(f"\tNo data available for date: {current_date.date()}, skipping.")
                current_date += pd.Timedelta(days=1)
                continue

            day_input = DayInput.from_df(
                day_df=day_data,
                config=self.config
            )

            solver = DaySolver(battery=self.battery, config=self.config)
            day_solution: DaySolution = solver.solve_day(day_input=day_input)
            day_solution.input = day_data
            solutions.append(day_solution)

            day_solution.plot_results(config=self.config)
            current_date += pd.Timedelta(days=1)

        return solutions