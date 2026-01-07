import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.cooptim import Orchestrator
from src.cooptim.solution import plot_global_results

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def run_scenario(name, data_modifier=None, start_date=None, end_date=None):
    """
    Exécute un scénario donné avec une configuration potentiellement modifiée par data_modifier.
    """
    logger.info(f"--- Démarrage du Scénario : {name} ---")

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # OVERRIDE des dates si fournies
    if start_date is not None:
        config["run"]["start_date"] = start_date
    if end_date is not None:
        config["run"]["end_date"] = end_date

    orchestrator = Orchestrator(config=config)

    # Application du modificateur de données 
    if data_modifier:
        logger.info(f"Application du modificateur de données pour : {name}")
        orchestrator.data = data_modifier(orchestrator.data, config)

    solutions = orchestrator.run()
    if solutions:
        logger.info(f"--> Scénario {name} terminé. {len(solutions)} jours simulés.")

    return solutions

# --- Modificateurs de données ---

def modifier_arbitrage_seul(df, config):
    """
    Force les prix de réserve à 0 pour simuler un arbitrage pur (Energy Only).
    """
    df_mod = df.copy()
    
    col_fcr = config["columns"]["fcr"]
    col_afrr_up = config["columns"]["afrr_up"]
    col_afrr_down = config["columns"]["afrr_down"]
    
    cols_to_zero = [col_fcr, col_afrr_up, col_afrr_down]
    
    for col in cols_to_zero:
        if col in df_mod.columns:
            df_mod[col] = 0.0
        else:
            logger.warning(f"Attention: La colonne '{col}' n'existe pas dans le DataFrame.")
            
    return df_mod

def modifier_reserve_seul(df, config):
    """
    Force le prix de l'énergie (Spot) à 0 pour simuler une stratégie focalisée uniquement
    sur les réserves (le stockage ne fait de l'énergie que pour gérer son SoC).
    """
    df_mod = df.copy()
    
    col_energy = config["columns"]["energy"]
    
    if col_energy in df_mod.columns:
        df_mod[col_energy] = 0.0
    else:
        logger.warning(f"Attention: La colonne '{col_energy}' n'existe pas dans le DataFrame.")
            
    return df_mod

# --- Reporting et PnL ---

def calculate_financials(solutions, config, scenario_name="Scénario"):
    """
    Calcule et affiche un rapport de performance financière sur la période.
    """
    if not solutions:
        print(f"Pas de solutions pour {scenario_name}")
        return

    total_rev_energy = 0.0
    total_rev_reserve = 0.0
    total_throughput_mwh = 0.0
    
    # Récupération des noms de colonnes prix
    c_price_e = config["columns"]["energy"]
    c_price_fcr = config["columns"]["fcr"]
    c_price_up = config["columns"]["afrr_up"]
    c_price_down = config["columns"]["afrr_down"]

    for s in solutions:
        inp = s.input
        sch = s.schedule
        
        # Calcul du pas de temps en heures
        dt_seconds = (inp.index[1] - inp.index[0]).total_seconds()
        dt_hours = dt_seconds / 3600.0

        # 1. Revenu Energie
        net_flow_mw = sch["p_dis_mw"] - sch["p_ch_mw"]
        if "a_act_up_mw" in sch.columns:
             net_flow_mw += (sch["a_act_up_mw"] - sch["a_act_down_mw"])
             
        rev_energy_day = (net_flow_mw * inp[c_price_e] * dt_hours).sum()
        total_rev_energy += rev_energy_day

        # 2. Revenu Réserve 
        rev_fcr = (sch["r_fcr_mw"] * inp[c_price_fcr]).sum() * dt_hours
        rev_up = (sch["r_afrr_up_mw"] * inp[c_price_up]).sum() * dt_hours
        rev_down = (sch["r_afrr_down_mw"] * inp[c_price_down]).sum() * dt_hours
        
        total_rev_reserve += (rev_fcr + rev_up + rev_down)

        # 3. Throughput
        total_throughput_mwh += (sch["p_dis_mw"] * dt_hours).sum()

    total_revenue = total_rev_energy + total_rev_reserve
    
    print(f"\n=== RÉSULTATS FINANCIERS : {scenario_name} ===")
    print(f"Revenu Total       : {total_revenue:,.2f} €")
    print(f"  > Dont Energie   : {total_rev_energy:,.2f} €")
    print(f"  > Dont Réserve   : {total_rev_reserve:,.2f} €")
    print(f"Volume Déchargé    : {total_throughput_mwh:,.2f} MWh")
    
    e_max = config["battery"]["e_max_mwh"]
    cycles = total_throughput_mwh / e_max
    print(f"Cycles Équivalents : {cycles:.2f}")
    print("============================================\n")

    return total_revenue

# --- Bloc Principal ---

if __name__ == "__main__":
    
    with open("config.json", "r", encoding="utf-8") as f:
        global_config = json.load(f)
    
    # Période de simulation
    START = "2025-01-15"
    END = "2025-01-15"

    # 1. Scénario : Arbitrage Pur (Energy Only)
    sols_arb = run_scenario("Arbitrage Pur", modifier_arbitrage_seul,
                            start_date=START, end_date=END)

    # 2. Scénario : Réserve Seule (Reserves Only) - AJOUTÉ
    sols_res = run_scenario("Réserve Seule", modifier_reserve_seul,
                            start_date=START, end_date=END)

    # 3. Scénario : Co-optimisation (Energy + Reserves)
    sols_coopt = run_scenario("Co-optimisation", None,
                              start_date=START, end_date=END)

    # 4. Calcul et Affichage du PnL
    calculate_financials(sols_arb, global_config, "Arbitrage Pur")
    calculate_financials(sols_res, global_config, "Réserve Seule") # AJOUTÉ
    calculate_financials(sols_coopt, global_config, "Co-optimisation")

# 5. Comparaison Graphique (SoC)
    if sols_arb and sols_coopt and sols_res:
        idx = 0 
        if idx < len(sols_arb) and idx < len(sols_coopt):
            s_arb = sols_arb[idx]
            s_res = sols_res[idx]
            s_coopt = sols_coopt[idx]
            day_date = s_arb.date.date()

            plt.figure(figsize=(12, 6))

            # SoC Arbitrage
            plt.plot(
                s_arb.schedule.index, s_arb.schedule["soc_mwh"],
                label="SoC (Arbitrage Seul)", linestyle="--", color="gray", linewidth=1.5
            )

            # SoC Réserve Seule
            plt.plot(
                s_res.schedule.index, s_res.schedule["soc_mwh"],
                label="SoC (Réserve Seule)", linestyle="-.", color="green", linewidth=1.5
            )

            # SoC Co-optimisé
            plt.plot(
                s_coopt.schedule.index, s_coopt.schedule["soc_mwh"],
                label="SoC (Co-optimisé)", color="tab:blue", linewidth=2.5
            )
            
            # --- SUPPRESSION DU BLOC AX2 / FCR ICI ---
            
            plt.title(f"Comparaison Stratégies : Arbitrage vs Réserve vs Co-opti ({day_date})")
            plt.xlabel("Heure (UTC)")
            plt.ylabel("Energie Stockée (MWh)")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.show()

        # Affichage détaillé de la co-optimisation
        logger.info("Affichage des graphiques détaillés pour la Co-optimisation...")
        try:
            plot_global_results(sols_coopt, global_config)
        except Exception as e:
            logger.warning(f"Impossible d'afficher les graphiques globaux : {e}")

    logger.info("Fin du programme.")