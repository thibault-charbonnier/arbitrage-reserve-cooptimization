from dataclasses import dataclass

@dataclass(frozen=True)
class Battery:
    """
    Structure representing a battery energy storage system and stores the battery specs.
    """
    e_max_mwh: float
    p_ch_max_mw: float
    p_dis_max_mw: float
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    soc_min: float = 0.0
    soc_max: float = 1.0
