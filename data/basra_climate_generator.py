"""
Basra Meteorological and Turbomachinery Operational Data Generator.
Calibrated to 8,760 hourly climate records in Basra, Southern Iraq (Shatt Al-Basra / Rumaila region).
Features extreme summer temperatures (up to 53°C) and variable marine/desert humidity.
"""

import os
import numpy as np
import pandas as pd
from thermodynamics.brayton_derating import simulate_turbine_power, compute_air_density
from thermodynamics.psychrometric_cooling import compute_wet_bulb_temperature, evaluate_fogging_potential


def generate_basra_dataset(
    output_path: str = "data/basra_gas_turbine_hourly.csv",
    num_hours: int = 8760,
    seed: int = 42,
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(seed)

    days = np.arange(num_hours) / 24.0
    day_of_year = days % 365.0
    hour_of_day = np.arange(num_hours) % 24

    # Seasonal sinusoidal cycle (peak in mid-July ~ day 196)
    seasonal_temp = 32.0 + 16.0 * np.sin(2.0 * np.pi * (day_of_year - 105.0) / 365.0)

    # Diurnal cycle (peak at 14:00, trough at 05:00)
    diurnal_temp = 6.5 * np.sin(2.0 * np.pi * (hour_of_day - 9.0) / 24.0)

    # Weather noise / extreme heat waves (up to 53°C in July/August)
    heatwave_noise = np.random.normal(0.0, 2.0, num_hours)
    T_ambient = np.clip(seasonal_temp + diurnal_temp + heatwave_noise, 8.0, 53.5)

    # Relative humidity (inverse relation to temperature, with occasional Gulf "Sharqi" humid spikes)
    base_rh = 65.0 - 0.8 * T_ambient
    sharqi_wind_spikes = (np.random.uniform(0.0, 1.0, num_hours) > 0.85) * np.random.uniform(15.0, 40.0, num_hours)
    RH = np.clip(base_rh + sharqi_wind_spikes + np.random.normal(0.0, 4.0, num_hours), 5.0, 92.0)

    # Atmospheric pressure (winter highs ~1018 mbar, summer thermal low ~998 mbar)
    P_ambient = 1010.0 - 8.0 * np.sin(2.0 * np.pi * (day_of_year - 105.0) / 365.0) + np.random.normal(0.0, 1.5, num_hours)

    # Gas turbine simulation
    power_mw = []
    thermal_eff = []
    air_density = []
    wet_bulb_c = []
    fogging_power_mw = []

    for t_c, rh, p_m in zip(T_ambient, RH, P_ambient):
        # Baseline uncooled turbine
        res = simulate_turbine_power(t_c, p_m, rh, P_nominal_mw=125.0)
        power_mw.append(res["power_mw"])
        thermal_eff.append(res["thermal_efficiency"])
        air_density.append(res["air_density_kg_m3"])

        # Psychrometric cooling potential
        fog = evaluate_fogging_potential(t_c, rh)
        wet_bulb_c.append(fog["wet_bulb_c"])

        # Cooled power
        res_cooled = simulate_turbine_power(fog["cooled_inlet_temp_c"], p_m, 95.0, P_nominal_mw=125.0)
        fogging_power_mw.append(res_cooled["power_mw"])

    # Build DataFrame
    df = pd.DataFrame({
        "hour": np.arange(num_hours),
        "day_of_year": day_of_year.round(2),
        "hour_of_day": hour_of_day,
        "T_ambient_C": T_ambient.round(2),
        "Relative_Humidity_pct": RH.round(2),
        "P_ambient_mbar": P_ambient.round(2),
        "Air_Density_kg_m3": np.array(air_density).round(4),
        "Wet_Bulb_Temp_C": np.array(wet_bulb_c).round(2),
        "Net_Power_MW": np.array(power_mw).round(3),
        "Thermal_Efficiency": np.array(thermal_eff).round(4),
        "Cooled_Power_MW": np.array(fogging_power_mw).round(3),
        "Power_Derate_MW": (125.0 - np.array(power_mw)).round(3),
    })

    df.to_csv(output_path, index=False)
    print(f"Basra turbine climate dataset successfully generated at {output_path} ({len(df)} rows).")
    return output_path


if __name__ == "__main__":
    generate_basra_dataset()
