"""
run_phase4c.py

Phase 4c: Cross-Validated Out-of-Fold (OOF) Predictions & Sequence-Length (L=20 vs L=30) Experiment.
- 5-Fold Engine-Level Cross-Validation on 100 training engines (seed=42).
- Inner validation set (8 engines) per fold for early stopping (held-out fold strictly untouched).
- All-cycle OOF predictions with left-padding for both L=20 and L=30.
- Saves outputs/oof_predictions_L20.csv and outputs/oof_predictions_L30.csv.
- Leave-fold-out interval calibration for L=30 (calibrated on other 4 folds' OOF residuals).
- Generates 14_oof_predictions_scatter.png and 15_engine_trajectory_intervals.png.
- Measures accurate inference latencies for documentation.
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_cmapss
from src.preprocessing import (
    drop_uninformative_columns,
    compute_rul,
    scale_features,
    generate_sequences,
    get_test_sequences,
    RETAINED_SENSOR_COLUMNS,
)
from src.model import LSTMRegressor
from src.train import (
    train_model,
    compute_mae,
    compute_rmse,
    compute_nasa_score,
    DEFAULT_TARGET_SCALE,
    unscale_target,
    scale_target,
)
from src.uncertainty import (
    DEFAULT_BIN_EDGES,
    DEFAULT_RESIDUAL_CALIB_PATH,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_all_cycle_sequences_for_engines(
    df: pd.DataFrame,
    engine_ids: np.ndarray,
    sequence_length: int,
    feature_cols: list = RETAINED_SENSOR_COLUMNS,
):
    """
    Generate rolling sequence windows for EVERY cycle of each specified engine.
    For cycles < sequence_length, left-pads by repeating the first cycle.
    
    Returns
    -------
    X : np.ndarray of shape (N_total_cycles, sequence_length, num_features)
    meta_df : pd.DataFrame with columns [unit_number, cycle, true_rul_raw, true_rul_capped]
    """
    X_list = []
    meta_rows = []
    
    subset_df = df[df["unit_number"].isin(engine_ids)].copy()
    
    for unit_id, group in subset_df.groupby("unit_number"):
        group = group.sort_values("time_in_cycles")
        feats = group[feature_cols].values
        cycles = group["time_in_cycles"].values
        rul_raw = group["RUL"].values
        rul_capped = group["RUL_capped"].values
        num_cycles = len(group)
        
        for idx in range(num_cycles):
            current_cycle = cycles[idx]
            raw_rul = rul_raw[idx]
            cap_rul = rul_capped[idx]
            
            # Sub-sequence up to current cycle (0-indexed 0 to idx+1)
            observed_len = idx + 1
            if observed_len < sequence_length:
                pad_len = sequence_length - observed_len
                pad_matrix = np.repeat(feats[0:1, :], pad_len, axis=0)
                seq = np.vstack([pad_matrix, feats[:observed_len, :]])
            else:
                seq = feats[observed_len - sequence_length:observed_len, :]
                
            X_list.append(seq)
            meta_rows.append({
                "unit_number": int(unit_id),
                "cycle": int(current_cycle),
                "true_rul_raw": float(raw_rul),
                "true_rul_capped": float(cap_rul),
            })
            
    X = np.array(X_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)
    return X, meta_df


def run_5fold_cv_experiment(sequence_length: int, train_scaled_df: pd.DataFrame, test_scaled_df: pd.DataFrame, device: torch.device):
    print(f"\n" + "=" * 70)
    print(f"RUNNING 5-FOLD CV FOR SEQUENCE LENGTH L = {sequence_length}")
    print("=" * 70)
    
    unique_units = np.sort(train_scaled_df["unit_number"].unique()) # 100 engines
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_dfs = []
    fold_metrics = []
    fold_last_metrics = []
    fold_models = []
    
    # Pre-extract test sequences for this sequence length
    X_test, test_unit_order = get_test_sequences(test_scaled_df, sequence_length=sequence_length)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(unique_units), start=1):
        held_out_engines = unique_units[val_idx] # 20 engines
        train_engines_80 = unique_units[train_idx] # 80 engines
        
        # Carve inner validation set (8 engines) for early stopping from train_engines_80
        rng = np.random.default_rng(42 + fold_idx)
        shuffled_80 = rng.permutation(train_engines_80)
        inner_val_engines = np.sort(shuffled_80[:8])
        inner_train_engines = np.sort(shuffled_80[8:])
        
        # Generate training sequences
        train_df_inner = train_scaled_df[train_scaled_df["unit_number"].isin(inner_train_engines)]
        val_df_inner = train_scaled_df[train_scaled_df["unit_number"].isin(inner_val_engines)]
        
        X_inner_train, y_inner_train = generate_sequences(train_df_inner, sequence_length=sequence_length)
        X_inner_val, y_inner_val = generate_sequences(val_df_inner, sequence_length=sequence_length)
        
        # Scale targets
        y_inner_train_norm = scale_target(y_inner_train)
        y_inner_val_norm = scale_target(y_inner_val)
        
        # DataLoaders
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_inner_train, dtype=torch.float32), torch.tensor(y_inner_train_norm, dtype=torch.float32)),
            batch_size=64,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_inner_val, dtype=torch.float32), torch.tensor(y_inner_val_norm, dtype=torch.float32)),
            batch_size=64,
            shuffle=False,
        )
        
        # Model initialization (winning config: h=64, drop=0.2, 2 layers, lr=0.001)
        torch.manual_seed(42 + fold_idx)
        model = LSTMRegressor(input_size=14, hidden_size=64, num_layers=2, dropout_rate=0.2)
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=60,
            lr=0.001,
            patience=8,
            target_scale=125.0,
            device=device,
            verbose=False,
        )
        fold_models.append(model)
        
        # OOF Predictions on the 20 held-out engines (EVERY cycle)
        X_oof_fold, meta_oof_fold = generate_all_cycle_sequences_for_engines(
            train_scaled_df,
            held_out_engines,
            sequence_length=sequence_length,
        )
        
        model.eval()
        with torch.no_grad():
            preds_norm = model(torch.tensor(X_oof_fold, dtype=torch.float32).to(device)).cpu().numpy()
            preds_cycles = unscale_target(preds_norm, target_scale=125.0)
            
        meta_oof_fold["pred_rul"] = preds_cycles
        meta_oof_fold["fold"] = fold_idx
        oof_dfs.append(meta_oof_fold)
        
        # Fold-level metrics (all windows)
        mae_all = compute_mae(meta_oof_fold["true_rul_capped"].values, preds_cycles)
        rmse_all = compute_rmse(meta_oof_fold["true_rul_capped"].values, preds_cycles)
        fold_metrics.append({"fold": fold_idx, "mae_all": mae_all, "rmse_all": rmse_all})
        
        # Fold-level metrics (last window per engine)
        last_window_df = meta_oof_fold.groupby("unit_number").last().reset_index()
        mae_last = compute_mae(last_window_df["true_rul_capped"].values, last_window_df["pred_rul"].values)
        rmse_last = compute_rmse(last_window_df["true_rul_capped"].values, last_window_df["pred_rul"].values)
        fold_last_metrics.append({"fold": fold_idx, "mae_last": mae_last, "rmse_last": rmse_last})
        
        print(f"  Fold {fold_idx}/5 | Best Ep: {history['best_epoch']:02d} | Val MAE: {history['best_val_mae']:.2f}c | OOF MAE (all): {mae_all:.2f}c | OOF MAE (last): {mae_last:.2f}c")

    # Combine OOF DataFrame
    oof_full_df = pd.concat(oof_dfs, ignore_index=True)
    oof_full_df = oof_full_df.sort_values(["unit_number", "cycle"]).reset_index(drop=True)
    
    # Overall OOF metrics
    oof_mae_all = compute_mae(oof_full_df["true_rul_capped"].values, oof_full_df["pred_rul"].values)
    oof_rmse_all = compute_rmse(oof_full_df["true_rul_capped"].values, oof_full_df["pred_rul"].values)
    
    oof_last_df = oof_full_df.groupby("unit_number").last().reset_index()
    oof_mae_last = compute_mae(oof_last_df["true_rul_capped"].values, oof_last_df["pred_rul"].values)
    oof_rmse_last = compute_rmse(oof_last_df["true_rul_capped"].values, oof_last_df["pred_rul"].values)
    
    # Fold stats
    all_maes = [m["mae_all"] for m in fold_metrics]
    all_rmses = [m["rmse_all"] for m in fold_metrics]
    last_maes = [m["mae_last"] for m in fold_last_metrics]
    last_rmses = [m["rmse_last"] for m in fold_last_metrics]
    
    mean_mae_all, std_mae_all = np.mean(all_maes), np.std(all_maes)
    mean_rmse_all, std_rmse_all = np.mean(all_rmses), np.std(all_rmses)
    mean_mae_last, std_mae_last = np.mean(last_maes), np.std(last_maes)
    mean_rmse_last, std_rmse_last = np.mean(last_rmses), np.std(last_rmses)
    
    # Test Evaluation (5-model ensemble average)
    test_preds_list = []
    for model in fold_models:
        model.eval()
        with torch.no_grad():
            p_norm = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
            test_preds_list.append(unscale_target(p_norm, target_scale=125.0))
    test_pred_ensemble = np.mean(test_preds_list, axis=0) # shape (100,)
    
    # Load test ground truth
    rul_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "RUL_FD001.txt", sep=r"\s+", header=None)
    y_test_raw = rul_df.values.flatten().astype(np.float32)
    y_test_capped = np.minimum(y_test_raw, 125.0)
    
    test_mae_capped = compute_mae(y_test_capped, test_pred_ensemble)
    test_rmse_capped = compute_rmse(y_test_capped, test_pred_ensemble)
    test_nasa_capped = compute_nasa_score(y_test_capped, test_pred_ensemble)
    
    test_mae_raw = compute_mae(y_test_raw, test_pred_ensemble)
    test_rmse_raw = compute_rmse(y_test_raw, test_pred_ensemble)
    test_nasa_raw = compute_nasa_score(y_test_raw, test_pred_ensemble)
    
    results = {
        "sequence_length": sequence_length,
        "oof_df": oof_full_df,
        "oof_mae_all": oof_mae_all,
        "oof_rmse_all": oof_rmse_all,
        "oof_mae_last": oof_mae_last,
        "oof_rmse_last": oof_rmse_last,
        "mean_mae_all": mean_mae_all,
        "std_mae_all": std_mae_all,
        "mean_rmse_all": mean_rmse_all,
        "std_rmse_all": std_rmse_all,
        "mean_mae_last": mean_mae_last,
        "std_mae_last": std_mae_last,
        "mean_rmse_last": mean_rmse_last,
        "std_rmse_last": std_rmse_last,
        "test_mae_capped": test_mae_capped,
        "test_rmse_capped": test_rmse_capped,
        "test_nasa_capped": test_nasa_capped,
        "test_mae_raw": test_mae_raw,
        "test_rmse_raw": test_rmse_raw,
        "test_nasa_raw": test_nasa_raw,
        "test_pred_ensemble": test_pred_ensemble,
        "fold_models": fold_models,
        "X_test": X_test,
    }
    
    return results


def calibrate_leave_fold_out_intervals(oof_df: pd.DataFrame, bin_edges=DEFAULT_BIN_EDGES, alpha=0.10):
    """
    For engines in fold f, calculate binned residual quantiles (q05, q95)
    using OOF residuals from the OTHER 4 folds only.
    Then compute lower and upper interval bounds for fold f.
    """
    oof_df = oof_df.copy()
    oof_df["lower"] = 0.0
    oof_df["upper"] = 0.0
    
    q_low_p = (alpha / 2.0) * 100.0   # 5th percentile
    q_high_p = (1.0 - alpha / 2.0) * 100.0 # 95th percentile
    
    inner_edges = bin_edges[1:-1]
    num_bins = len(bin_edges) - 1
    
    unique_folds = sorted(oof_df["fold"].unique())
    fold_bin_stats = {}
    
    for f in unique_folds:
        # Calibration data: other 4 folds
        other_folds_df = oof_df[oof_df["fold"] != f].copy()
        other_res = other_folds_df["true_rul_capped"].values - other_folds_df["pred_rul"].values
        other_preds = other_folds_df["pred_rul"].values
        other_bins = np.digitize(other_preds, inner_edges)
        
        # Calculate quantiles per bin for other 4 folds
        bin_quantiles = {}
        for b in range(num_bins):
            mask_b = (other_bins == b)
            res_b = other_res[mask_b]
            if len(res_b) > 0:
                q05 = float(np.percentile(res_b, q_low_p))
                q95 = float(np.percentile(res_b, q_high_p))
            else:
                q05 = float(np.percentile(other_res, q_low_p))
                q95 = float(np.percentile(other_res, q_high_p))
            bin_quantiles[b] = {"q05": q05, "q95": q95, "count": int(np.sum(mask_b))}
            
        fold_bin_stats[f] = bin_quantiles
        
        # Apply to fold f
        fold_mask = (oof_df["fold"] == f)
        fold_preds = oof_df.loc[fold_mask, "pred_rul"].values
        fold_bins = np.digitize(fold_preds, inner_edges)
        fold_bins = np.clip(fold_bins, 0, num_bins - 1)
        
        f_lowers = np.zeros_like(fold_preds)
        f_uppers = np.zeros_like(fold_preds)
        
        for i, p in enumerate(fold_preds):
            b = fold_bins[i]
            q05 = bin_quantiles[b]["q05"]
            q95 = bin_quantiles[b]["q95"]
            f_lowers[i] = max(0.0, min(125.0, float(p + q05)))
            f_uppers[i] = min(125.0, max(0.0, float(p + q95)))
            
        oof_df.loc[fold_mask, "lower"] = f_lowers
        oof_df.loc[fold_mask, "upper"] = f_uppers

    return oof_df, fold_bin_stats


def compute_oof_interval_metrics(oof_df: pd.DataFrame, bin_edges=DEFAULT_BIN_EDGES):
    """Compute overall and per-bin empirical 90% coverage and interval widths."""
    y_true = oof_df["true_rul_capped"].values
    y_pred = oof_df["pred_rul"].values
    lower = oof_df["lower"].values
    upper = oof_df["upper"].values
    
    covered = (y_true >= lower) & (y_true <= upper)
    widths = upper - lower
    
    overall_cov = float(np.mean(covered))
    overall_width = float(np.mean(widths))
    
    # Safety critical (pred < 30)
    crit_mask = (y_pred < 30.0)
    crit_cov = float(np.mean(covered[crit_mask])) if np.sum(crit_mask) > 0 else 0.0
    crit_width = float(np.mean(widths[crit_mask])) if np.sum(crit_mask) > 0 else 0.0
    crit_count = int(np.sum(crit_mask))
    
    # Per-bin metrics
    inner_edges = bin_edges[1:-1]
    num_bins = len(bin_edges) - 1
    bin_indices = np.digitize(y_pred, inner_edges)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    bin_metrics = []
    for b in range(num_bins):
        low_e = bin_edges[b]
        high_e = bin_edges[b+1]
        m = (bin_indices == b)
        cnt = int(np.sum(m))
        if cnt > 0:
            cov_b = float(np.mean(covered[m]))
            w_b = float(np.mean(widths[m]))
            mae_b = float(np.mean(np.abs(y_true[m] - y_pred[m])))
        else:
            cov_b, w_b, mae_b = 0.0, 0.0, 0.0
            
        bin_metrics.append({
            "bin_idx": b,
            "range": f"[{low_e:.0f}, {high_e:.0f})",
            "count": cnt,
            "coverage": cov_b,
            "mean_width": w_b,
            "mae": mae_b,
        })
        
    return {
        "overall_coverage": overall_cov,
        "overall_mean_width": overall_width,
        "critical_coverage": crit_cov,
        "critical_mean_width": crit_width,
        "critical_count": crit_count,
        "bins": bin_metrics,
    }


def main():
    print("=" * 80)
    print("PHASE 4c: CROSS-VALIDATED PREDICTIONS & SEQUENCE-LENGTH EXPERIMENT (L=20 vs L=30)")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load and preprocess dataset
    train_df, test_df, rul_df = load_cmapss("FD001", data_dir=PROJECT_ROOT / "data" / "raw")
    train_clean = drop_uninformative_columns(train_df)
    test_clean = drop_uninformative_columns(test_df)
    train_rul = compute_rul(train_clean, max_rul=125, is_train=True)
    train_scaled, test_scaled, scaler = scale_features(train_rul, test_clean)
    
    # Run 5-fold CV for L=20 and L=30
    res_L20 = run_5fold_cv_experiment(sequence_length=20, train_scaled_df=train_scaled, test_scaled_df=test_scaled, device=device)
    res_L30 = run_5fold_cv_experiment(sequence_length=30, train_scaled_df=train_scaled, test_scaled_df=test_scaled, device=device)
    
    # Save initial OOF CSVs
    oof_path_L20 = PROJECT_ROOT / "outputs" / "oof_predictions_L20.csv"
    oof_path_L30 = PROJECT_ROOT / "outputs" / "oof_predictions_L30.csv"
    
    res_L20["oof_df"].to_csv(oof_path_L20, index=False)
    print(f"\nSaved OOF predictions (L=20): {oof_path_L20} ({len(res_L20['oof_df']):,} rows)")
    
    # Leave-fold-out interval calibration for L=30
    print("\n[Step 3] Calibrating Leave-Fold-Out Empirical Prediction Intervals for L=30...")
    oof_df_L30_calib, fold_bin_stats = calibrate_leave_fold_out_intervals(res_L30["oof_df"])
    oof_df_L30_calib.to_csv(oof_path_L30, index=False)
    print(f"Saved calibrated OOF predictions (L=30): {oof_path_L30} ({len(oof_df_L30_calib):,} rows)")
    
    # Compute OOF interval metrics
    interval_metrics_L30 = compute_oof_interval_metrics(oof_df_L30_calib)
    
    # -------------------------------------------------------------------------
    # Latency Benchmark (Accurate measurement)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Benchmarking exact inference latency (Deterministic vs 100-Pass MC-Dropout)...")
    sample_window = torch.tensor(res_L30["X_test"][:1], dtype=torch.float32).to(device)
    test_model = res_L30["fold_models"][0]
    test_model.eval()
    
    # Warmup
    for _ in range(20):
        _ = test_model(sample_window)
        
    # Benchmark deterministic single pass
    t0 = time.perf_counter()
    n_det = 500
    with torch.no_grad():
        for _ in range(n_det):
            _ = test_model(sample_window)
    t_det_ms = ((time.perf_counter() - t0) / n_det) * 1000.0
    
    # Benchmark MC-Dropout 100 passes
    test_model.enable_mc_dropout()
    t0 = time.perf_counter()
    n_mc_runs = 50
    with torch.no_grad():
        for _ in range(n_mc_runs):
            for _ in range(100):
                _ = test_model(sample_window)
    t_mc_ms = ((time.perf_counter() - t0) / n_mc_runs) * 1000.0
    test_model.disable_mc_dropout()
    
    print(f"  -> Single Deterministic Forward Pass Latency: {t_det_ms:.3f} ms / window (measured over {n_det} runs on {device})")
    print(f"  -> 100-Pass MC-Dropout Latency: {t_mc_ms:.3f} ms / window (measured over {n_mc_runs} runs on {device})")
    
    # -------------------------------------------------------------------------
    # Plots Generation
    # -------------------------------------------------------------------------
    print("\n[Step 5] Generating Publication-Grade Visualizations...")
    
    # Plot 14: OOF Predicted vs True RUL Scatter (L=20 vs L=30)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    for ax, res, title in zip(
        axes,
        [res_L20, res_L30],
        [f"Sequence Length L = 20\n(OOF MAE: {res_L20['oof_mae_all']:.2f}c, RMSE: {res_L20['oof_rmse_all']:.2f}c)",
         f"Sequence Length L = 30\n(OOF MAE: {res_L30['oof_mae_all']:.2f}c, RMSE: {res_L30['oof_rmse_all']:.2f}c)"]
    ):
        df_p = res["oof_df"]
        # Subsample for clear density visualization or use hexbin/scatter with alpha
        ax.scatter(df_p["true_rul_capped"], df_p["pred_rul"], alpha=0.15, s=12, color="#2563EB", edgecolors="none", label="OOF Windows (N=20,631)")
        ax.plot([0, 125], [0, 125], color="#DC2626", linestyle="--", linewidth=2.0, label="Ideal Parity ($y = \hat{y}$)")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("True Capped RUL (Cycles)", fontsize=11, fontweight="semibold")
        ax.set_ylabel("Predicted RUL (Cycles)", fontsize=11, fontweight="semibold")
        ax.set_xlim(-5, 135)
        ax.set_ylim(-5, 135)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", frameon=True, framealpha=0.9)
        
    plt.suptitle("NASA C-MAPSS FD001: 5-Fold Cross-Validated Out-of-Fold (OOF) Predictions Across All 20,631 Cycles", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot14_path = OUTPUT_DIR / "14_oof_predictions_scatter.png"
    plt.savefig(plot14_path, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved Plot 14: {plot14_path}")
    
    # Plot 15: 4 Sample Engines' Predicted RUL with Intervals Over Full Lifetime vs True RUL
    sample_units = [3, 24, 34, 100] # Representative diverse lifetime engines
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    axes = axes.flatten()
    
    for i, unit_id in enumerate(sample_units):
        ax = axes[i]
        eng_df = oof_df_L30_calib[oof_df_L30_calib["unit_number"] == unit_id].sort_values("cycle")
        
        cycles = eng_df["cycle"].values
        true_raw = eng_df["true_rul_raw"].values
        true_cap = eng_df["true_rul_capped"].values
        pred_rul = eng_df["pred_rul"].values
        lower_rul = eng_df["lower"].values
        upper_rul = eng_df["upper"].values
        
        # Plot True RUL trajectory
        ax.plot(cycles, true_cap, color="#111827", linestyle="--", linewidth=2.2, label="True Capped RUL")
        
        # Plot Prediction Interval Band
        ax.fill_between(cycles, lower_rul, upper_rul, color="#3B82F6", alpha=0.25, label="90% Prediction Interval")
        
        # Plot Model Prediction Line
        ax.plot(cycles, pred_rul, color="#1D4ED8", linewidth=2.0, label="Predicted RUL ($\hat{y}$)")
        
        # Mark threshold 30 cycles
        ax.axhline(30, color="#EF4444", linestyle=":", linewidth=1.5, alpha=0.8, label="Critical Alarm Threshold (30c)")
        
        total_cyc = int(cycles[-1])
        fold_id = int(eng_df["fold"].iloc[0])
        ax.set_title(f"Engine Unit #{unit_id} (Fold {fold_id}, Total Lifespan: {total_cyc} Cycles)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Operational Time in Cycles", fontsize=10, fontweight="semibold")
        ax.set_ylabel("Remaining Useful Life (Cycles)", fontsize=10, fontweight="semibold")
        ax.set_ylim(-5, 135)
        ax.set_xlim(0, total_cyc + 5)
        ax.grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
            
    plt.suptitle("Full-Lifetime Degradation Trajectories & 90% Uncertainty Bounds (Preview for Fleet Simulator)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plot15_path = OUTPUT_DIR / "15_engine_trajectory_intervals.png"
    plt.savefig(plot15_path, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved Plot 15: {plot15_path}")
    
    # -------------------------------------------------------------------------
    # Print Full Comparative Tables
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 SUMMARY: SEQUENCE LENGTH 20 vs 30")
    print("=" * 80)
    print(f"{'Metric':<40} | {'Sequence Length L=20':<22} | {'Sequence Length L=30':<22}")
    print("-" * 90)
    print(f"{'OOF MAE (All 20,631 Windows)':<40} | {res_L20['oof_mae_all']:>19.2f}c | {res_L30['oof_mae_all']:>19.2f}c")
    print(f"{'OOF RMSE (All 20,631 Windows)':<40} | {res_L20['oof_rmse_all']:>19.2f}c | {res_L30['oof_rmse_all']:>19.2f}c")
    print(f"{'OOF MAE (Last Window / Engine, N=100)':<40} | {res_L20['oof_mae_last']:>19.2f}c | {res_L30['oof_mae_last']:>19.2f}c")
    print(f"{'OOF RMSE (Last Window / Engine, N=100)':<40} | {res_L20['oof_rmse_last']:>19.2f}c | {res_L30['oof_rmse_last']:>19.2f}c")
    print(f"{'5-Fold Mean +/- Std MAE (All Windows)':<40} | {res_L20['mean_mae_all']:.2f} +/- {res_L20['std_mae_all']:.2f}c      | {res_L30['mean_mae_all']:.2f} +/- {res_L30['std_mae_all']:.2f}c")
    print(f"{'5-Fold Mean +/- Std RMSE (All Windows)':<40} | {res_L20['mean_rmse_all']:.2f} +/- {res_L20['std_rmse_all']:.2f}c      | {res_L30['mean_rmse_all']:.2f} +/- {res_L30['std_rmse_all']:.2f}c")
    print(f"{'5-Fold Mean +/- Std MAE (Last Window)':<40} | {res_L20['mean_mae_last']:.2f} +/- {res_L20['std_mae_last']:.2f}c      | {res_L30['mean_mae_last']:.2f} +/- {res_L30['std_mae_last']:.2f}c")
    print(f"{'5-Fold Mean +/- Std RMSE (Last Window)':<40} | {res_L20['mean_rmse_last']:.2f} +/- {res_L20['std_rmse_last']:.2f}c      | {res_L30['mean_rmse_last']:.2f} +/- {res_L30['std_rmse_last']:.2f}c")
    print("-" * 90)
    print(f"{'Test MAE (5-Model Ensemble, Capped)':<40} | {res_L20['test_mae_capped']:>19.2f}c | {res_L30['test_mae_capped']:>19.2f}c")
    print(f"{'Test RMSE (5-Model Ensemble, Capped)':<40} | {res_L20['test_rmse_capped']:>19.2f}c | {res_L30['test_rmse_capped']:>19.2f}c")
    print(f"{'Test NASA Score (Ensemble, Capped)':<40} | {res_L20['test_nasa_capped']:>19.2f}  | {res_L30['test_nasa_capped']:>19.2f} ")
    print(f"{'Test MAE (5-Model Ensemble, Raw RUL)':<40} | {res_L20['test_mae_raw']:>19.2f}c | {res_L30['test_mae_raw']:>19.2f}c")
    print(f"{'Test RMSE (5-Model Ensemble, Raw RUL)':<40} | {res_L20['test_rmse_raw']:>19.2f}c | {res_L30['test_rmse_raw']:>19.2f}c")
    print(f"{'Test NASA Score (Ensemble, Raw RUL)':<40} | {res_L20['test_nasa_raw']:>19.2f}  | {res_L30['test_nasa_raw']:>19.2f} ")
    
    print("\n" + "=" * 80)
    print("LEAVE-FOLD-OUT PREDICTION INTERVAL EVALUATION (L=30, N=20,631 OOF Windows)")
    print("=" * 80)
    print(f"Overall 90% Empirical Coverage: {interval_metrics_L30['overall_coverage']*100:.1f}% ({int(interval_metrics_L30['overall_coverage']*len(oof_df_L30_calib)):,}/{len(oof_df_L30_calib):,} windows)")
    print(f"Overall Mean Interval Width:    {interval_metrics_L30['overall_mean_width']:.2f} cycles")
    print(f"Safety-Critical (pred < 30c) Width:    {interval_metrics_L30['critical_mean_width']:.2f} cycles (Coverage: {interval_metrics_L30['critical_coverage']*100:.1f}%, N={interval_metrics_L30['critical_count']:,})")
    print("\nPer-Bin Breakdown:")
    print(f"{'Predicted RUL Bin':<20} | {'Count':<8} | {'90% Coverage':<14} | {'Mean Width':<12} | {'MAE':<10}")
    print("-" * 72)
    for b_info in interval_metrics_L30["bins"]:
        print(f"{b_info['range']:<20} | {b_info['count']:<8} | {b_info['coverage']*100:>11.1f}% | {b_info['mean_width']:>9.2f}c | {b_info['mae']:>7.2f}c")
        
    return res_L20, res_L30, interval_metrics_L30, t_det_ms, t_mc_ms


if __name__ == "__main__":
    main()
