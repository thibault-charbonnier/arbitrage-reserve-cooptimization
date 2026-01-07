import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
from src.cooptim import Orchestrator
from src.cooptim.solution import plot_global_results

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def run_scenario(name, data_modifier=None, start_date=None, end_date=None):
    logger.info(f"--- Démarrage du Scénario : {name} ---")

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # OVERRIDE AVANT Orchestrator
    if start_date is not None:
        config["run"]["start_date"] = start_date
    if end_date is not None:
        config["run"]["end_date"] = end_date

    orchestrator = Orchestrator(config=config)

    if data_modifier:
        logger.info(f"Application du modificateur de données pour : {name}")
        orchestrator.data = data_modifier(orchestrator.data)

    solutions = orchestrator.run()
    if solutions:
        logger.info(f"--> Scénario {name} terminé. {len(solutions)} jours simulés.")

    return solutions


# --- Modificateurs de données ---

def modifier_arbitrage_seul(df):
    """Force les prix de réserve à 0 pour simuler un arbitrage pur."""
    df_mod = df.copy()
    # On met à zéro les colonnes de prix de réserve
    # Assurez-vous que ces noms correspondent à ceux de vos fichiers Parquet
    cols_to_zero = ["price_fcr", "price_afrr_up", "price_afrr_down"]
    for col in cols_to_zero:
        if col in df_mod.columns:
            df_mod[col] = 0.0
    return df_mod

# --- Bloc Principal ---

if __name__ == "__main__":
    
    # 1. Scénario de Référence : Arbitrage Pur (Energy Only)
    # On veut voir combien la batterie gagne sans faire de réserves
    sols_arb = run_scenario("Arbitrage Pur", modifier_arbitrage_seul,
                        start_date="2025-05-10", end_date="2025-05-11")

    # 2. Scénario Cible : Co-optimisation (Energy + Reserves)
    # On utilise votre nouveau DaySolver complet
    sols_coopt = run_scenario("Co-optimisation", None,
                          start_date="2025-05-10", end_date="2025-05-11")


    # 3. Comparaison Graphique (pour un jour type)
    if sols_arb and sols_coopt:
        # On prend le premier jour qui a réussi
        idx = 0
        s_arb = sols_arb[idx]
        s_coopt = sols_coopt[idx]
        

        day_date = s_arb.date.date()

        plt.figure(figsize=(12, 6))

        plt.plot(
            s_arb.schedule.index, s_arb.schedule["soc_mwh"],
            label="SoC (Arbitrage Seul)", linestyle="--", color="gray"
        )

        plt.plot(
            s_coopt.schedule.index, s_coopt.schedule["soc_mwh"],
            label="SoC (Co-optimisé)", linewidth=2
        )

        plt.title(f"Impact des scénarios sur le cycle batterie ({day_date})")
        plt.ylabel("Energie Stockée (MWh)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # 4. Affichage des résultats complets de la co-optimisation
        # Utilise la fonction plot_global_results que nous avons créée plus tôt
        logger.info("Affichage des résultats globaux...")
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            plot_global_results(sols_coopt, config)
        except Exception as e:
            logger.warning(f"Impossible d'afficher les graphiques globaux : {e}")

    logger.info("Fin du programme.")