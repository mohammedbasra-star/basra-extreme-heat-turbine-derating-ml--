"""
Psychrometric Evaporative Fogging Cooling Potential for Basra Climate.
Calculates wet-bulb temperature (Stull, 2011) and power recovery potential via inlet air chilling.
"""

import numpy as np


def compute_wet_bulb_temperature(T_celsius: float, RH_percent: float) -> float:
    """
    Computes wet-bulb temperature Twb (°C) using the empirical Stull (2011) formula.
    Valid for T between -20°C and 55°C and RH between 5% and 99%.
    """
    T = T_celsius
    RH = RH_percent

    Twb = (
        T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
        + np.arctan(T + RH)
        - np.arctan(RH - 1.676331)
        + 0.00391838 * (RH ** 1.5) * np.arctan(0.023101 * RH)
        - 4.686035
    )
    return float(Twb)


def evaluate_fogging_potential(
    T_ambient_c: float,
    RH_percent: float,
    eta_evaporator: float = 0.85,
) -> dict:
    """
    Evaluates evaporative cooling potential:
      - Wet-bulb depression: Delta_T_wb = T_amb - T_wb
      - Achievable inlet temperature drop: Delta_T_fog = eta_evap * Delta_T_wb
      - Cooled compressor inlet temperature: T_inlet = T_amb - Delta_T_fog
    """
    Twb = compute_wet_bulb_temperature(T_ambient_c, RH_percent)
    wb_depression = max(0.0, T_ambient_c - Twb)
    temp_drop = eta_evaporator * wb_depression
    T_cooled = T_ambient_c - temp_drop

    # Estimated power gain: ~0.7% power recovery per 1°C inlet temperature reduction
    estimated_power_recovery_pct = temp_drop * 0.72

    return {
        "wet_bulb_c": float(Twb),
        "wet_bulb_depression_c": float(wb_depression),
        "achievable_cooling_c": float(temp_drop),
        "cooled_inlet_temp_c": float(T_cooled),
        "power_recovery_percent": float(estimated_power_recovery_pct),
    }
