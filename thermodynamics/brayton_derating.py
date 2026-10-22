"""
Thermodynamic Brayton Cycle Model for Gas Turbine Derating in Extreme Climates.
Calibrated to heavy-duty industrial gas turbines (e.g., GE 9E / SGT-800, nominal 125 MW at ISO 15°C).
"""

import numpy as np

# Specific gas constant for dry air (J / kg K)
R_AIR = 287.058
# Specific heat ratio and cp
GAMMA = 1.4
CP_AIR = 1005.0  # J / kg K


def compute_air_density(T_celsius: float, P_mbar: float = 1013.25, RH_percent: float = 30.0) -> float:
    """
    Computes moist air density (kg/m^3) using ideal gas law with water vapor partial pressure.
    """
    T_kelvin = T_celsius + 273.15
    P_pa = P_mbar * 100.0

    # Saturation vapor pressure (Tetens formula in Pa)
    p_sat = 610.78 * np.exp(17.27 * T_celsius / (T_celsius + 237.3))
    p_v = (RH_percent / 100.0) * p_sat
    p_d = P_pa - p_v

    # Moist air density: rho = (p_d / (R_d * T)) + (p_v / (R_v * T))
    R_v = 461.5
    rho = (p_d / (R_AIR * T_kelvin)) + (p_v / (R_v * T_kelvin))
    return float(rho)


def simulate_turbine_power(
    T_celsius: float,
    P_mbar: float = 1013.25,
    RH_percent: float = 30.0,
    P_nominal_mw: float = 125.0,
    TIT_kelvin: float = 1400.0,
    pressure_ratio: float = 12.5,
    eta_compressor: float = 0.86,
    eta_turbine: float = 0.89,
) -> dict:
    """
    Simulates net power output (MW) and thermal efficiency across ambient temperatures (10°C to 55°C).
    """
    T_amb_k = T_celsius + 273.15
    rho_iso = compute_air_density(15.0, 1013.25, 60.0)
    rho_actual = compute_air_density(T_celsius, P_mbar, RH_percent)

    # Mass flow derating relative to ISO (nominal ~400 kg/s)
    mass_flow_iso = 410.0  # kg/s
    mass_flow = mass_flow_iso * (rho_actual / rho_iso)

    # Compressor work per kg: wc = cp * T_amb * (rp^((gamma-1)/gamma) - 1) / eta_c
    rp_factor = pressure_ratio ** ((GAMMA - 1.0) / GAMMA)
    w_comp = CP_AIR * T_amb_k * (rp_factor - 1.0) / eta_compressor  # J/kg

    # Turbine work per kg: wt = cp * TIT * (1 - rp^-((gamma-1)/gamma)) * eta_t
    w_turb = CP_AIR * TIT_kelvin * (1.0 - 1.0 / rp_factor) * eta_turbine  # J/kg

    # Net specific work and total power
    w_net = w_turb - w_comp  # J/kg
    power_mw = (mass_flow * w_net) / 1e6

    # Heat added in combustor
    T_comp_out = T_amb_k + (T_amb_k * (rp_factor - 1.0) / eta_compressor)
    q_in = CP_AIR * (TIT_kelvin - T_comp_out)
    thermal_efficiency = w_net / max(q_in, 1e-4)

    # Derating percentage relative to ISO 15°C
    # Reference at 15°C
    power_iso = P_nominal_mw
    derate_percent = max(0.0, (power_iso - power_mw) / power_iso * 100.0)

    return {
        "power_mw": float(power_mw),
        "thermal_efficiency": float(thermal_efficiency),
        "mass_flow_kg_s": float(mass_flow),
        "air_density_kg_m3": float(rho_actual),
        "derate_percent": float(derate_percent),
    }
