from dataclasses import dataclass

@dataclass(frozen=True)
class Battery:
    """
    Battery specs.
    soc_min and soc_max are RATIOS (0.0 to 1.0), not MWh.
    """
    e_max_mwh: float
    p_ch_max_mw: float
    p_dis_max_mw: float
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    soc_min: float = 0.0  # 0%
    soc_max: float = 1.0  # 100%