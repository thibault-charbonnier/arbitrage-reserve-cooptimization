import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src.cooptim import Orchestrator
from src.cooptim.solution import plot_global_results

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def run_scenario(name, data_modifier=None, start_date=None, end_date=None):
    """ Exécute un scénario donné. """
    logger.info(f"--- Démarrage du Scénario : {name} ---")

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    if start_date is not None:
        config["run"]["start_date"] = start_date
    if end_date is not None:
        config["run"]["end_date"] = end_date

    orchestrator = Orchestrator(config=config)

    if data_modifier:
        logger.info(f"Application du modificateur de données pour : {name}")
        orchestrator.data = data_modifier(orchestrator.data, config)

    solutions = orchestrator.run()
    if solutions:
        logger.info(f"--> Scénario {name} terminé. {len(solutions)} jours simulés.")

    return solutions

# --- Modificateurs de données ---

def modifier_arbitrage_seul(df, config):
    df_mod = df.copy()
    col_fcr = config["columns"]["fcr"]
    col_afrr_up = config["columns"]["afrr_up"]
    col_afrr_down = config["columns"]["afrr_down"]
    for col in [col_fcr, col_afrr_up, col_afrr_down]:
        if col in df_mod.columns:
            df_mod[col] = 0.0
    return df_mod

def modifier_reserve_seul(df, config):
    df_mod = df.copy()
    col_energy = config["columns"]["energy"]
    if col_energy in df_mod.columns:
        df_mod[col_energy] = 0.0
    return df_mod

# --- Reporting et PnL ---

def calculate_financials(solutions, config, scenario_name="Scénario"):
    """ Retourne un dictionnaire de résultats financiers. """
    if not solutions:
        logger.warning(f"Pas de solutions pour {scenario_name}")
        return None

    total_rev_energy = 0.0
    total_rev_reserve = 0.0
    total_throughput_mwh = 0.0
    
    c_price_e = config["columns"]["energy"]
    c_price_fcr = config["columns"]["fcr"]
    c_price_up = config["columns"]["afrr_up"]
    c_price_down = config["columns"]["afrr_down"]

    for s in solutions:
        inp = s.input
        sch = s.schedule
        
        dt_seconds = (inp.index[1] - inp.index[0]).total_seconds()
        dt_hours = dt_seconds / 3600.0

        # Revenu Energie
        net_flow_mw = sch["p_dis_mw"] - sch["p_ch_mw"]
        if "a_act_up_mw" in sch.columns:
             net_flow_mw += (sch["a_act_up_mw"] - sch["a_act_down_mw"])
        rev_energy_day = (net_flow_mw * inp[c_price_e] * dt_hours).sum()
        total_rev_energy += rev_energy_day

        # Revenu Réserve
        rev_fcr = (sch["r_fcr_mw"] * inp[c_price_fcr]).sum() * dt_hours
        rev_up = (sch["r_afrr_up_mw"] * inp[c_price_up]).sum() * dt_hours
        rev_down = (sch["r_afrr_down_mw"] * inp[c_price_down]).sum() * dt_hours
        total_rev_reserve += (rev_fcr + rev_up + rev_down)

        # Throughput
        act_up = sch["a_act_up_mw"] if "a_act_up_mw" in sch.columns else 0.0
        total_throughput_mwh += ((sch["p_dis_mw"] + act_up) * dt_hours).sum()

    total_revenue = total_rev_energy + total_rev_reserve
    e_max = config["battery"]["e_max_mwh"]
    cycles = (total_throughput_mwh / e_max) if e_max > 0 else 0.0

    return {
        "Scénario": scenario_name,
        "Revenu Total (€)": total_revenue,
        "Dont Energie (€)": total_rev_energy,
        "Dont Réserve (€)": total_rev_reserve,
        "Volume Déchargé (MWh)": total_throughput_mwh,
        "Cycles Équivalents": cycles
    }

# --- Génération PDF ---

def export_results_to_pdf(results_list, filename="rapport_financier.pdf"):
    if not results_list:
        return

    df = pd.DataFrame(results_list).set_index("Scénario").T
    df_display = df.copy()
    for col in df_display.columns:
        df_display[col] = df_display[col].apply(lambda x: f"{x:,.2f}")

    with PdfPages(filename) as pdf:
        fig, ax = plt.subplots(figsize=(8,3))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(
            cellText=df_display.values,
            rowLabels=df_display.index,
            colLabels=df_display.columns,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 1.8)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#40466e')
            elif col == -1:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f2f2f2')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        logger.info(f"Rapport PDF généré : {filename}")

        
def get_daily_pnl_series(solutions, config):
    """
    Extrait la série des revenus Journalier pour chaque strat

    """
    if not solutions:
        return pd.Series(dtype=float)

    dates = []
    revenues = []
    
    # Récupération des colonnes de prix
    c_e = config["columns"]["energy"]
    c_fcr = config["columns"]["fcr"]
    c_up = config["columns"]["afrr_up"]
    c_down = config["columns"]["afrr_down"]

    for s in solutions:
        inp = s.input
        sch = s.schedule
        
        # Durée du pas de temps en heures
        dt_hours = (inp.index[1] - inp.index[0]).total_seconds() / 3600.0
        
        # Revenu Energie (Flux net * Prix Spot)
        net_flow_mw = sch["p_dis_mw"] - sch["p_ch_mw"]
        if "a_act_up_mw" in sch.columns:
             net_flow_mw += (sch["a_act_up_mw"] - sch["a_act_down_mw"])
             
        rev_energy = (net_flow_mw * inp[c_e]).sum() * dt_hours

        # Revenu Réserve (Capacité * Prix)
        rev_reserve = (
            sch["r_fcr_mw"] * inp[c_fcr] +
            sch["r_afrr_up_mw"] * inp[c_up] +
            sch["r_afrr_down_mw"] * inp[c_down]
        ).sum() * dt_hours
        
        # On stocke la date (sans l'heure) et le revenu total du jour
        dates.append(s.date.date()) 
        revenues.append(rev_energy + rev_reserve)
        
    return pd.Series(data=revenues, index=pd.to_datetime(dates)).sort_index()

# --- Bloc Principal ---

if __name__ == "__main__":
    
    with open("config.json", "r", encoding="utf-8") as f:
        global_config = json.load(f)
    
    START = "2025-01-15"
    END = "2025-01-15"
    financial_results = []

 
    #  Exécution des simulations (Calculs Physiques)
    
    sols_arb = run_scenario("Arbitrage Pur", modifier_arbitrage_seul, start_date=START, end_date=END)
    sols_res = run_scenario("Réserve Seule", modifier_reserve_seul, start_date=START, end_date=END)
    sols_coopt = run_scenario("Co-optimisation", None, start_date=START, end_date=END)

    # On injecte les vrais prix (ceux de Co-opt) dans les résultats de Réserve
    # pour que le calcul financier soit réaliste (coût de recharge non nul).
    if sols_res and sols_coopt:
        logger.info("Correction des prix Spot pour le calcul PnL 'Réserve Seule'...")
        col_energy = global_config["columns"]["energy"]
        
        for i, s_res in enumerate(sols_res):
            # on vérifie qu'on compare les mêmes jours
            if i < len(sols_coopt) and s_res.date == sols_coopt[i].date:
                s_res.input[col_energy] = sols_coopt[i].input[col_energy]


    #  Calculs Financiers 

    # Arbitrage
    res_arb = calculate_financials(sols_arb, global_config, "Arbitrage Pur")
    if res_arb: financial_results.append(res_arb)

    # Réserve 
    res_res = calculate_financials(sols_res, global_config, "Réserve Seule")
    if res_res: financial_results.append(res_res)

    # Co-optimisation
    res_coopt = calculate_financials(sols_coopt, global_config, "Co-optimisation")
    if res_coopt: financial_results.append(res_coopt)

 
    # Génération du PDF et Graphiques 

    export_results_to_pdf(financial_results, filename="Resultats_Simulation.pdf")

    # Comparaison Graphique (SoC) 
    if sols_arb and sols_coopt and sols_res:
        idx = 0 
        if idx < len(sols_arb) and idx < len(sols_coopt):
            s_arb = sols_arb[idx]
            s_res = sols_res[idx]
            s_coopt = sols_coopt[idx]
            
            plt.figure(figsize=(12, 6))
            plt.plot(s_arb.schedule.index, s_arb.schedule["soc_mwh"], label="SoC (Arbitrage)", linestyle="--", color="gray")
            plt.plot(s_res.schedule.index, s_res.schedule["soc_mwh"], label="SoC (Réserve)", linestyle="-.", color="green")
            plt.plot(s_coopt.schedule.index, s_coopt.schedule["soc_mwh"], label="SoC (Co-opti)", color="tab:blue", linewidth=2.5)
            
            plt.title("Comparaison des SoC")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    # Affichage détaillé Co-optimisation 
    logger.info("Affichage des graphiques détaillés pour la Co-optimisation...")
    try:
        plot_global_results(sols_coopt, global_config)
    except Exception as e:
        logger.warning(f"Impossible d'afficher les graphiques globaux : {e}")

    logger.info("Fin du programme.")

   
    
    # Graphique d'évolution du PnL sur la période
    if sols_arb and sols_coopt and sols_res:
        
        # Extraction des séries temporelles
        ts_arb = get_daily_pnl_series(sols_arb, global_config)
        ts_res = get_daily_pnl_series(sols_res, global_config)
        ts_coopt = get_daily_pnl_series(sols_coopt, global_config)
        
        # Création du graphique
        plt.figure(figsize=(12, 6))
        
        # Tracé des courbes avec marqueurs
        plt.plot(ts_arb.index, ts_arb.values, label='Arbitrage Pur', marker='o', linestyle='--', color='gray', alpha=0.7)
        plt.plot(ts_res.index, ts_res.values, label='Réserve Seule', marker='s', linestyle='-.', color='green', alpha=0.7)
        plt.plot(ts_coopt.index, ts_coopt.values, label='Co-optimisation', marker='^', linestyle='-', color='tab:blue', linewidth=2)
        
        # Mise en forme
        plt.title(f"Comparaison de la Rentabilité Journalière ({START} au {END})")
        plt.ylabel("Revenu Net Journalier (€)")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Formatage des dates sur l'axe X
        plt.gcf().autofmt_xdate()
        
        # Sauvegarde / Affichage
        plt.savefig("pnl_evolution.png", dpi=150)
        logger.info("Graphique d'évolution PnL sauvegardé : 'pnl_evolution.png'")
        
        plt.show() 