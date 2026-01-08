from dataclasses import dataclass

@dataclass(frozen=True)
class Battery:
    """
    Data class representing a battery energy storage system.

    Attributes
    ----------
    e_max_mwh : float
        Maximum energy capacity of the battery in MWh.
    p_ch_max_mw : float
        Maximum charging power of the battery in MW.
    p_dis_max_mw : float
        Maximum discharging power of the battery in MW.
    eta_ch : float
        Charging efficiency of the battery (default is 0.95).
    eta_dis : float
        Discharging efficiency of the battery (default is 0.95).
    soc_min : float
        Minimum state of charge as a fraction of the battery capacity (default is 0.0
        representing 0%).
    soc_max : float
        Maximum state of charge as a fraction of the battery capacity (default is 1.0
        representing 100%).
    """
    e_max_mwh: float
    p_ch_max_mw: float
    p_dis_max_mw: float
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    soc_min: float = 0.0  # 0%
    soc_max: float = 1.0  # 100%