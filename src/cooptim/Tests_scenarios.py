# src/cooptim/scenarios.py
from typing import Callable, Dict
import pandas as pd

DataModifier = Callable[[pd.DataFrame], pd.DataFrame]

def modifier_arbitrage_seul(df: pd.DataFrame) -> pd.DataFrame:
    """Force reserve prices to 0 -> energy-only arbitrage."""
    df_mod = df.copy()
    for col in ["price_fcr", "price_afrr_up", "price_afrr_down"]:
        if col in df_mod.columns:
            df_mod[col] = 0.0
    return df_mod

def modifier_reserve_low(df: pd.DataFrame, scale: float = 0.2) -> pd.DataFrame:
    """Sensitivity test: downscale reserve prices."""
    df_mod = df.copy()
    for col in ["price_fcr", "price_afrr_up", "price_afrr_down"]:
        if col in df_mod.columns:
            df_mod[col] = df_mod[col] * scale
    return df_mod

# Registry = "nom de scénario" -> fonction modif
SCENARIOS: Dict[str, DataModifier] = {
    "energy_only": modifier_arbitrage_seul,
    # si vous voulez figer à 0.2 sans paramètre :
    "reserve_low": lambda df: modifier_reserve_low(df, scale=0.2),
}
