"""
Training Pipeline for Basra Gas Turbine Ambient Derating Models.
Trains Physics-Guided Neural Network (PGNN) and Scikit-Learn baselines on 8,760 hourly Basra records.
"""

import argparse
import os
import json
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

from data.dataset import get_basra_data_splits, FEATURE_COLS
from models.physics_guided_regressor import PhysicsGuidedMLP, physics_loss_fn
from models.baseline_models import train_baseline_suite
from utils.visualization import (
    plot_basra_derating_curve,
    plot_fogging_recovery,
    plot_model_parity,
    plot_feature_importance,
)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Basra Derating Models on device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # 1. Dataset splits
    splits = get_basra_data_splits(csv_path=args.data_path, batch_size=args.batch_size)
    train_loader = splits["train_loader"]
    val_loader = splits["val_loader"]
    test_loader = splits["test_loader"]

    # 2. Train Classical Baselines
    print("\n--- Training Classical ML Baselines (Ridge, RF, GBDT) ---")
    baseline_results = train_baseline_suite(
        splits["X_train"], splits["y_train"], splits["X_test"], splits["y_test"]
    )
    for name, res in baseline_results.items():
        print(f"  {name:<18} | R2: {res['r2']:.4f} | RMSE: {res['rmse']:.3f} MW | MAE: {res['mae']:.3f} MW")

    # 3. Train Physics-Guided MLP
    print("\n--- Training Physics-Guided Neural Network (PGNN) ---")
    model = PhysicsGuidedMLP(in_features=len(FEATURE_COLS), hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_pgnn_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            total_loss, l_data, l_mono = physics_loss_fn(
                model, X_b, y_b, t_ambient_col_idx=0, lambda_mono=args.lambda_mono
            )
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * len(X_b)

        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b)
                v_loss = torch.mean((pred - y_b) ** 2)
                running_val += v_loss.item() * len(X_b)

        val_loss = running_val / len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            torch.save(model.state_dict(), "results/best_pgnn_model.pt")

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Train Loss: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Best: {best_val_loss:.4f}")

    # Evaluate PGNN on test set
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            p = model(X_b).cpu().numpy().ravel()
            all_preds.extend(p)
            all_targets.extend(y_b.numpy().ravel())

    y_test_arr = np.array(all_targets)
    y_pred_arr = np.array(all_preds)
    pgnn_r2 = float(r2_score(y_test_arr, y_pred_arr))
    pgnn_rmse = float(np.sqrt(mean_squared_error(y_test_arr, y_pred_arr)))

    print(f"\nFinal PGNN Test Benchmark: R2 = {pgnn_r2:.4f}, RMSE = {pgnn_rmse:.3f} MW")

    # Generate Publication Figures
    print("\nGenerating publication figures...")
    plot_basra_derating_curve(splits["df_full"], save_path="figures/basra_summer_derating_curve.png")
    plot_fogging_recovery(splits["df_full"], save_path="figures/evaporative_fogging_power_recovery.png")
    plot_model_parity(y_test_arr, y_pred_arr, pgnn_r2, pgnn_rmse, save_path="figures/model_benchmark_comparison.png")

    gb_importances = baseline_results["GradientBoosting"]["model"].feature_importances_
    plot_feature_importance(FEATURE_COLS, gb_importances, save_path="figures/feature_importance_thermodynamics.png")

    # Save benchmark metrics
    summary = {
        "pgnn_r2": pgnn_r2,
        "pgnn_rmse_mw": pgnn_rmse,
        "gradient_boosting_r2": baseline_results["GradientBoosting"]["r2"],
        "gradient_boosting_rmse_mw": baseline_results["GradientBoosting"]["rmse"],
        "random_forest_r2": baseline_results["RandomForest"]["r2"],
        "random_forest_rmse_mw": baseline_results["RandomForest"]["rmse"],
        "ridge_r2": baseline_results["Ridge"]["r2"],
        "ridge_rmse_mw": baseline_results["Ridge"]["rmse"],
    }
    with open("results/test_evaluation.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("Benchmark saved to results/test_evaluation.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Basra Gas Turbine Derating Models")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lambda-mono", type=float, default=0.5)
    parser.add_argument("--data-path", type=str, default="data/basra_gas_turbine_hourly.csv")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    args = parser.parse_args()
    train(args)
