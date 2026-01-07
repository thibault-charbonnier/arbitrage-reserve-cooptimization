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

        # Lecture des fichiers parquet définis dans la config
        energy_prices = pd.read_parquet(self.config["data"]["energy_prices_parquet"])
        reserve_prices = pd.read_parquet(self.config["data"]["reserve_prices_parquet"])

        # Fusion des données (Inner Join pour ne garder que les dates communes)
        data = energy_prices.join(reserve_prices, how="inner")

        return data
    
    def _load_battery(self) -> Battery:
        """
        Load battery specifications from the configuration dictionary and create a Battery instance.
        """
        logger.info("Loading battery ...")

        battery_config = self.config["battery"]
        
        # CORRECTION ICI : On utilise les clés "soc_min" et "soc_max" (ratios)
        # pour correspondre au nouveau config.json
        battery = Battery(
            e_max_mwh=battery_config["e_max_mwh"],
            p_ch_max_mw=battery_config["p_ch_max_mw"],
            p_dis_max_mw=battery_config["p_dis_max_mw"],
            eta_ch=battery_config["eta_ch"],
            eta_dis=battery_config["eta_dis"],
            soc_min=battery_config["soc_min"],  # <--- Corrigé (était soc_min_mwh)
            soc_max=battery_config["soc_max"],  # <--- Corrigé (était soc_max_mwh)
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
        
        # On peut avoir besoin du SoC final du jour précédent pour l'initialisation
        # Pour le premier jour, c'est None (le DayInput utilisera sa valeur par défaut ou fixée)
        previous_soc_end = None

        while current_date <= end_date:

            logger.info(f"\tSolving for date: {current_date.date()}")

            day_data = self.data[self.data.index.normalize().date == current_date.date()]
            
            if day_data.empty:
                logger.warning(f"\tNo data available for date: {current_date.date()}, skipping.")
                current_date += pd.Timedelta(days=1)
                continue
            
            # Gestion du SoC initial (enchaînement des jours)
            # Si on a un historique, on prend le soc final de la veille.
            # Sinon, on laisse le DayInput utiliser sa valeur par défaut (souvent 50% ou 0)
            soc_init = previous_soc_end if previous_soc_end is not None else 5.0 # Valeur par défaut explicite (5 MWh)

            day_input = DayInput.from_df(
                day_df=day_data,
                config=self.config,
                soc0=soc_init
            )

            solver = DaySolver(battery=self.battery, config=self.config)
            day_solution: DaySolution = solver.solve_day(day_input=day_input)
            
            # Stockage de l'input pour les graphiques plus tard
            day_solution.input = day_data
            solutions.append(day_solution)
            
            # Mise à jour du SoC initial pour le lendemain
            if not day_solution.schedule.empty:
                # On récupère le dernier SoC calculé
                previous_soc_end = day_solution.schedule["soc_mwh"].iloc[-1]

            current_date += pd.Timedelta(days=1)

        return solutions