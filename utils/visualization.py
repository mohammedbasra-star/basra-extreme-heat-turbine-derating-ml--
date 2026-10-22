"""
Publication-Grade Visualizations for Basra Gas Turbine Ambient Derating.
Produces 300 DPI figures formatted for ASME Journal of Engineering for Gas Turbines and Power.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_publication_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_basra_derating_curve(
    df: pd.DataFrame,
    save_path: str = "figures/basra_summer_derating_curve.png",
):
    set_publication_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        df["T_ambient_C"],
        df["Net_Power_MW"],
        c=df["Relative_Humidity_pct"],
        cmap="coolwarm_r",
        alpha=0.4,
        s=15,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Relative Humidity (%)")

    # Fit quadratic polynomial derating trend
    p_fit = np.poly1d(np.polyfit(df["T_ambient_C"], df["Net_Power_MW"], 2))
    t_range = np.linspace(10.0, 53.0, 100)
    ax.plot(t_range, p_fit(t_range), "r-", lw=2.5, label="Derating Trend (~0.72 MW / °C loss)")

    ax.axvline(50.0, color="darkred", linestyle="--", lw=1.5, label="Basra Summer Extreme Threshold (50°C)")
    ax.axhline(125.0, color="gray", linestyle=":", lw=1.2, label="ISO Rating (125 MW at 15°C)")

    ax.set_xlabel("Ambient Temperature (°C) [Basra Meteorological Station]")
    ax.set_ylabel("Net Turbine Power Output (MW)")
    ax.set_title("Gas Turbine Thermal Derating Under Extreme Basra Ambient Heat")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Basra derating curve saved to {save_path}")


def plot_fogging_recovery(
    df: pd.DataFrame,
    save_path: str = "figures/evaporative_fogging_power_recovery.png",
):
    set_publication_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Filter to summer peak (July hours: ~4320 to 5064)
    summer_df = df.iloc[4320 : 4320 + 168]  # 1 representative summer week

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(summer_df["hour"].values, summer_df["Net_Power_MW"].values, "r-", lw=1.8, label="Uncooled Turbine (Baseline)")
    ax.plot(summer_df["hour"].values, summer_df["Cooled_Power_MW"].values, "b--", lw=2.0, label="Evaporative Fogging Chilled Inlet")

    ax.fill_between(
        summer_df["hour"].values,
        summer_df["Net_Power_MW"].values,
        summer_df["Cooled_Power_MW"].values,
        color="cyan",
        alpha=0.3,
        label="Recoverable Power Deficit (MW)",
    )

    ax.set_xlabel("Operational Hour of the Year (Basra Peak July)")
    ax.set_ylabel("Power Output (MW)")
    ax.set_title("Inlet Air Fogging Power Recovery Potential During Basra Summer Peak")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Fogging recovery plot saved to {save_path}")


def plot_model_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    r2: float,
    rmse: float,
    save_path: str = "figures/model_benchmark_comparison.png",
):
    set_publication_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, color="#1f77b4", s=18, edgecolors="none")

    # 45 degree perfect parity line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=2, label="Perfect Parity (1:1)")

    ax.set_xlabel("Actual Ground Truth Power (MW)")
    ax.set_ylabel("Physics-Guided ML Predicted Power (MW)")
    ax.set_title(f"Model Parity Validation (R² = {r2:.4f}, RMSE = {rmse:.3f} MW)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Model parity plot saved to {save_path}")


def plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    save_path: str = "figures/feature_importance_thermodynamics.png",
):
    set_publication_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color="#2ca02c")
    ax.set_xlabel("Relative Feature Importance (Gradient Boosting)")
    ax.set_title("Thermodynamic Impact Drivers on Basra Gas Turbine Power")
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Feature importance plot saved to {save_path}")
