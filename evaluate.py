"""
Independent Evaluation Script for Basra Gas Turbine Derating Models.
"""

import argparse
import os
import json
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data.dataset import get_basra_data_splits, FEATURE_COLS
from models.physics_guided_regressor import PhysicsGuidedMLP


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Basra PGNN model on: {device}")

    model = PhysicsGuidedMLP(in_features=len(FEATURE_COLS), hidden_dim=args.hidden_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    splits = get_basra_data_splits(csv_path=args.data_path)
    test_loader = splits["test_loader"]

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            p = model(X_b).cpu().numpy().ravel()
            all_preds.extend(p)
            all_targets.extend(y_b.numpy().ravel())

    y_test = np.array(all_targets)
    y_pred = np.array(all_preds)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100.0

    print("\n" + "=" * 55)
    print("      BASRA GAS TURBINE AMBIENT DERATING EVALUATION     ")
    print("=" * 55)
    print(f"  {'R2 Score':<35}: {r2:.4f}")
    print(f"  {'Root Mean Squared Error (RMSE)':<35}: {rmse:.3f} MW")
    print(f"  {'Mean Absolute Error (MAE)':<35}: {mae:.3f} MW")
    print(f"  {'Mean Absolute Percentage Error (MAPE)':<35}: {mape:.2f} %")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_pgnn_model.pt")
    parser.add_argument("--data-path", type=str, default="data/basra_gas_turbine_hourly.csv")
    parser.add_argument("--hidden-dim", type=int, default=64)

    args = parser.parse_args()
    evaluate(args)
