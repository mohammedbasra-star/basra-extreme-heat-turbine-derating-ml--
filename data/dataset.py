"""
Data Loading and Preprocessing Pipeline for Basra Gas Turbine Climate Dataset.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data.basra_climate_generator import generate_basra_dataset


FEATURE_COLS = [
    "T_ambient_C",
    "Relative_Humidity_pct",
    "P_ambient_mbar",
    "Air_Density_kg_m3",
    "Wet_Bulb_Temp_C",
    "hour_of_day",
]
TARGET_COL = "Net_Power_MW"


class BasraTurbineDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_basra_data_splits(
    csv_path: str = "data/basra_gas_turbine_hourly.csv",
    batch_size: int = 64,
    test_size: float = 0.2,
    seed: int = 42,
):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Generating...")
        generate_basra_dataset(output_path=csv_path)

    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Train/Val/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=seed)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    train_ds = BasraTurbineDataset(X_train_scaled, y_train)
    val_ds = BasraTurbineDataset(X_val_scaled, y_val)
    test_ds = BasraTurbineDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler_X": scaler_X,
        "df_full": df,
    }
