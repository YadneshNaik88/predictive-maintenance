"""
run_training.py

Executes Phase 3: LSTM Model Training, Compact Tuning Grid, and Test Evaluation.
Performs engine-level train/validation split (80/20 engines), sweeps the 8-configuration
tuning grid with target normalization (y / 125.0) and early stopping on Validation MAE,
persists tuning results to outputs/tuning_results.csv, checkpoints the winning model to
outputs/best_lstm_model.pt (including target_scale=125.0), generates publication-quality
evaluation plots in outputs/eda_plots/, and builds notebooks/03_model_training.ipynb.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nbformat as nbf
import torch
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
    get_engine_train_val_split,
    RETAINED_SENSOR_COLUMNS,
)
from src.model import LSTMRegressor
from src.train import (
    train_model,
    evaluate_model,
    compute_mae,
    compute_rmse,
    compute_nasa_score,
    scale_target,
    unscale_target,
    DEFAULT_TARGET_SCALE,
)

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TUNING_CSV_PATH = PROJECT_ROOT / "outputs" / "tuning_results.csv"
BEST_MODEL_PATH = PROJECT_ROOT / "outputs" / "best_lstm_model.pt"

# Visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]


def main():
    print("=" * 80)
    print("PHASE 3: LSTM MODEL TRAINING, TUNING & EVALUATION (NASA C-MAPSS FD001)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------------------------------
    # Step 1: Ingestion, Preprocessing & Engine-Level Train/Val Split
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading and preprocessing dataset...")
    train_df, test_df, rul_df = load_cmapss("FD001", data_dir=PROJECT_ROOT / "data" / "raw")
    train_clean = drop_uninformative_columns(train_df)
    test_clean = drop_uninformative_columns(test_df)
    train_rul = compute_rul(train_clean, max_rul=125, is_train=True)
    train_scaled, test_scaled, scaler = scale_features(train_rul, test_clean)

    print("\n[Step 2] Performing ENGINE-LEVEL Train/Validation Split (80/20)...")
    X_train_split, y_train_split, X_val_split, y_val_split, train_units, val_units = get_engine_train_val_split(
        train_scaled,
        val_ratio=0.2,
        sequence_length=30,
        feature_cols=RETAINED_SENSOR_COLUMNS,
        label_col="RUL_capped",
        seed=42,
    )

    print(f"  -> Train Engines ({len(train_units)}): {train_units[:8]} ... {train_units[-4:]}")
    print(f"  -> Val Engines   ({len(val_units)}): {val_units[:8]} ... {val_units[-4:]}")
    print(f"  -> X_train shape: {X_train_split.shape} ({len(X_train_split):,} sequences)")
    print(f"  -> y_train shape: {y_train_split.shape}")
    print(f"  -> X_val shape:   {X_val_split.shape} ({len(X_val_split):,} sequences)")
    print(f"  -> y_val shape:   {y_val_split.shape}")

    # Generate Test Sequences
    X_test, test_unit_order = get_test_sequences(test_scaled, sequence_length=30)
    y_test_raw = rul_df["RUL"].values.astype(np.float32)
    y_test_capped = np.minimum(y_test_raw, 125.0).astype(np.float32)

    print(f"  -> X_test shape:  {X_test.shape} (100 test engines)")
    print(f"  -> y_test shape:  {y_test_raw.shape}")

    # Target normalization: train on y / 125.0
    y_train_norm = scale_target(y_train_split, DEFAULT_TARGET_SCALE)
    y_val_norm = scale_target(y_val_split, DEFAULT_TARGET_SCALE)

    # PyTorch DataLoaders
    batch_size = 64
    train_dataset = TensorDataset(torch.tensor(X_train_split, dtype=torch.float32), torch.tensor(y_train_norm, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_split, dtype=torch.float32), torch.tensor(y_val_norm, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test_capped, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # -------------------------------------------------------------------------
    # Step 3: Compact Hyperparameter Tuning Grid (8 Configurations)
    # -------------------------------------------------------------------------
    print("\n[Step 3] Running Compact Tuning Grid (8 Configurations)...")
    grid = [
        {"hidden_size": 32, "dropout": 0.2, "learning_rate": 0.0010},
        {"hidden_size": 32, "dropout": 0.2, "learning_rate": 0.0005},
        {"hidden_size": 32, "dropout": 0.3, "learning_rate": 0.0010},
        {"hidden_size": 32, "dropout": 0.3, "learning_rate": 0.0005},
        {"hidden_size": 64, "dropout": 0.2, "learning_rate": 0.0010},
        {"hidden_size": 64, "dropout": 0.2, "learning_rate": 0.0005},
        {"hidden_size": 64, "dropout": 0.3, "learning_rate": 0.0010},
        {"hidden_size": 64, "dropout": 0.3, "learning_rate": 0.0005},
    ]

    tuning_records = []
    models_history = {}
    models_dict = {}

    max_epochs = 60
    patience = 8

    for i, cfg in enumerate(grid, 1):
        print(f"\n--- [Config #{i}/8] hidden_size={cfg['hidden_size']}, dropout={cfg['dropout']}, lr={cfg['learning_rate']} ---")
        torch.manual_seed(42)
        model = LSTMRegressor(
            input_size=14,
            hidden_size=cfg["hidden_size"],
            num_layers=2,
            dropout_rate=cfg["dropout"],
        )

        trained_model, history = train_model(
            model,
            train_loader,
            val_loader,
            epochs=max_epochs,
            lr=cfg["learning_rate"],
            patience=patience,
            target_scale=DEFAULT_TARGET_SCALE,
            device=device,
            verbose=False,
        )

        # Evaluate on validation split (unscaled cycles)
        val_preds, val_targets, val_metrics = evaluate_model(
            trained_model, val_loader, target_scale=DEFAULT_TARGET_SCALE, device=device
        )

        best_ep = history["best_epoch"]
        tot_ep = history["epochs_trained"]
        near_max = best_ep >= (max_epochs - 3)

        record = {
            "config_id": i,
            "hidden_size": cfg["hidden_size"],
            "dropout": cfg["dropout"],
            "learning_rate": cfg["learning_rate"],
            "val_MAE": round(val_metrics["MAE"], 4),
            "val_RMSE": round(val_metrics["RMSE"], 4),
            "val_NASA_Score": round(val_metrics["NASA_Score"], 2),
            "epochs_trained": tot_ep,
            "best_epoch": best_ep,
            "near_max_flag": near_max,
        }
        tuning_records.append(record)
        models_history[i] = history
        models_dict[i] = trained_model

        print(f"  -> Val MAE: {val_metrics['MAE']:.2f} cycles | Val RMSE: {val_metrics['RMSE']:.2f} cycles | Best Epoch: {best_ep}/{tot_ep} {'[NEAR MAX]' if near_max else ''}")

    tuning_df = pd.DataFrame(tuning_records).sort_values(by="val_MAE").reset_index(drop=True)
    tuning_df.to_csv(TUNING_CSV_PATH, index=False)
    print("\n[Tuning Grid Summary Table (Ranked by Validation MAE)]:")
    print(tuning_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # Step 4: Best Model Selection & Checkpointing
    # -------------------------------------------------------------------------
    best_row = tuning_df.iloc[0]
    best_config_id = int(best_row["config_id"])
    best_model = models_dict[best_config_id]
    best_history = models_history[best_config_id]

    print("\n[Step 4] Best Configuration Selected Strictly by Validation MAE:")
    print(f"  • Config ID:       #{best_config_id}")
    print(f"  • Hidden Size:     {int(best_row['hidden_size'])}")
    print(f"  • Dropout:         {best_row['dropout']}")
    print(f"  • Learning Rate:   {best_row['learning_rate']}")
    print(f"  • Validation MAE:  {best_row['val_MAE']:.4f} cycles")
    print(f"  • Validation RMSE: {best_row['val_RMSE']:.4f} cycles")
    print(f"  • Best Epoch:      {int(best_row['best_epoch'])} / {int(best_row['epochs_trained'])}")

    # Save model checkpoint with target_scale=125.0
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "config": {
                "input_size": 14,
                "hidden_size": int(best_row["hidden_size"]),
                "num_layers": 2,
                "dropout_rate": float(best_row["dropout"]),
                "sequence_length": 30,
                "target_scale": DEFAULT_TARGET_SCALE,
            },
        },
        BEST_MODEL_PATH,
    )
    print(f"  -> Checkpointed winning model to: {BEST_MODEL_PATH}")

    # -------------------------------------------------------------------------
    # Step 5: Final Evaluation on Held-Out Test Set (100 Engines)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Evaluating Winning Model ONCE on Held-Out Test Set (100 Engines)...")
    test_preds, _, _ = evaluate_model(
        best_model, test_loader, target_scale=DEFAULT_TARGET_SCALE, device=device
    )

    test_mae_capped = compute_mae(y_test_capped, test_preds)
    test_rmse_capped = compute_rmse(y_test_capped, test_preds)
    test_nasa_capped = compute_nasa_score(y_test_capped, test_preds)

    test_mae_raw = compute_mae(y_test_raw, test_preds)
    test_rmse_raw = compute_rmse(y_test_raw, test_preds)
    test_nasa_raw = compute_nasa_score(y_test_raw, test_preds)

    print("\n--- Final Test Set Results ---")
    print(f"Target Capped (RUL <= 125): MAE = {test_mae_capped:6.3f} cycles | RMSE = {test_rmse_capped:6.3f} cycles | NASA Score = {test_nasa_capped:8.2f}")
    print(f"Target Raw True RUL:        MAE = {test_mae_raw:6.3f} cycles | RMSE = {test_rmse_raw:6.3f} cycles | NASA Score = {test_nasa_raw:8.2f}")
    print(f"Test Predictions Min / Max: {test_preds.min():.2f} / {test_preds.max():.2f} cycles")

    # -------------------------------------------------------------------------
    # Step 6: Evaluation Plots
    # -------------------------------------------------------------------------
    print("\n[Step 6] Generating Evaluation Visualizations...")

    # Plot 1: Predicted vs Actual RUL
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
    errors = test_preds - y_test_capped

    scatter = ax.scatter(
        y_test_capped,
        test_preds,
        c=np.abs(errors),
        cmap="coolwarm",
        s=70,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
        zorder=4,
    )
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute Error |Predicted - Actual| (Cycles)", fontsize=10, fontweight="bold")

    lims = [0, 140]
    ax.plot(lims, lims, color="#0f172a", linestyle="--", linewidth=1.8, label="Ideal Prediction (y = x)", zorder=3)
    ax.fill_between(lims, [l - 15 for l in lims], [l + 15 for l in lims], color="#3b82f6", alpha=0.10, label="±15 Cycles Error Band")
    ax.fill_between(lims, [l - 30 for l in lims], [l + 30 for l in lims], color="#64748b", alpha=0.06, label="±30 Cycles Error Band")

    metric_text = (
        f"Test Set Performance (100 Engines):\n"
        f"  • Capped MAE:  {test_mae_capped:.2f} cycles\n"
        f"  • Capped RMSE: {test_rmse_capped:.2f} cycles\n"
        f"  • NASA Score:  {test_nasa_capped:.1f}\n"
        f"  • Raw True MAE: {test_mae_raw:.2f} cycles\n"
        f"Model: LSTM (h={int(best_row['hidden_size'])}, drop={best_row['dropout']}, lr={best_row['learning_rate']})"
    )
    ax.text(
        0.04,
        0.96,
        metric_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#cbd5e1", alpha=0.95),
        zorder=5,
    )

    ax.set_xlim(0, 140)
    ax.set_ylim(0, 140)
    ax.set_xlabel("True Ground-Truth RUL (Cycles)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted RUL by LSTM (Cycles)", fontsize=11, fontweight="bold")
    ax.set_title("Predicted vs. Actual Remaining Useful Life (100 Test Engines)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    plot1_path = OUTPUT_DIR / "08_test_predictions_vs_actual.png"
    plt.tight_layout()
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved scatter plot to: {plot1_path}")

    # Plot 2: Training & Validation Loss Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    train_loss = best_history["train_loss"]
    val_loss = best_history["val_loss"]
    val_mae = best_history["val_mae"]
    epochs_range = range(1, len(train_loss) + 1)
    best_ep = best_history["best_epoch"]

    # Subplot 1: MSE Loss in Normalized Space
    ax1.plot(epochs_range, train_loss, label="Training Loss (Normalized MSE)", color="#2563eb", linewidth=2.0)
    ax1.plot(epochs_range, val_loss, label="Validation Loss (Normalized MSE)", color="#ea580c", linewidth=2.0)
    ax1.scatter([best_ep], [val_loss[best_ep - 1]], color="#16a34a", s=80, marker="o", zorder=5, label=f"Best Model (Epoch {best_ep})")
    ax1.axvline(best_ep, color="#16a34a", linestyle=":", linewidth=1.2, alpha=0.8)
    ax1.set_title("Normalized MSE Loss Curves", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Training Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Mean Squared Error (y / 125)", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Validation MAE in Cycle Units
    ax2.plot(epochs_range, val_mae, label="Validation MAE (Cycles)", color="#059669", linewidth=2.0)
    best_vl_mae = val_mae[best_ep - 1]
    ax2.scatter([best_ep], [best_vl_mae], color="#dc2626", s=80, marker="o", zorder=5, label=f"Min Val MAE = {best_vl_mae:.2f} cycles")
    ax2.axvline(best_ep, color="#dc2626", linestyle=":", linewidth=1.2, alpha=0.8)
    ax2.axhline(36.51, color="#64748b", linestyle="--", linewidth=1.2, label="Naive Mean Baseline (36.51 cycles)")
    ax2.set_title("Validation MAE Tracking (Unscaled Cycles)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Training Epoch", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Validation MAE (Cycles)", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"LSTM Training Dynamics (Config #{best_config_id}: h={int(best_row['hidden_size'])}, drop={best_row['dropout']}, lr={best_row['learning_rate']})", fontsize=13, fontweight="bold", y=1.02)
    plot2_path = OUTPUT_DIR / "09_loss_curves.png"
    plt.tight_layout()
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved loss curves to: {plot2_path}")

    # Copy to outputs directory for display
    artifact_dir = Path("outputs")
    if artifact_dir.exists():
        import shutil
        shutil.copy(plot1_path, artifact_dir / "08_test_predictions_vs_actual.png")
        shutil.copy(plot2_path, artifact_dir / "09_loss_curves.png")

    # -------------------------------------------------------------------------
    # Step 7: Build Interactive Notebook (03_model_training.ipynb)
    # -------------------------------------------------------------------------
    print("\n[Step 7] Generating interactive notebooks/03_model_training.ipynb...")
    build_notebook()
    print("\nPhase 3 execution completed successfully!")


def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Overview
    cells.append(nbf.v4.new_markdown_cell("""# Phase 3: LSTM Model Training, Hyperparameter Tuning & Evaluation

**Project:** Agentic Predictive Maintenance  
**Dataset:** NASA C-MAPSS FD001  
**Objective:** Train an LSTM sequence regression model to predict Remaining Useful Life (RUL), conduct an engine-level cross-validation tuning grid with target normalization ($y / 125.0$) and early stopping on Validation MAE, evaluate on held-out test engines, and report MAE, RMSE, and NASA asymmetric score.

---
### Phase 3 Key Workflow:
1. **Engine-Level Train/Validation Split (80/20):** Prevent sequence overlap data leakage by splitting across engine units ($80$ train engines, $20$ validation engines).
2. **Target Normalization:** Normalize targets ($y / 125.0$) to balance gradient flow and prevent bias collapse.
3. **LSTM Neural Architecture (`src/model.py`):** Stacked LSTM with post-recurrent Dropout head for Phase 4 MC-Dropout compatibility.
4. **Hyperparameter Tuning Study (8 Configurations):** Sweep `hidden_size` $\\in \\{32, 64\\}$, `dropout` $\\in \\{0.2, 0.3\\}$, and `learning_rate` $\\in \\{0.001, 0.0005\\}$.
5. **Early Stopping:** Monitored on **Validation MAE** (patience $= 8$, max epochs $= 60$).
6. **Final Test Evaluation (100 Engines):** Evaluate against true test ground truth (`RUL_FD001.txt`).
7. **NASA Asymmetric Scoring Function:** Compute penalty metric emphasizing safety against late failure overestimation.
"""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import TensorDataset, DataLoader

# Add project root to path
PROJECT_ROOT = Path("..").resolve() if Path("..").resolve().name == "predictive-maintenance" else Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_cmapss
from src.preprocessing import (
    drop_uninformative_columns,
    compute_rul,
    scale_features,
    generate_sequences,
    get_test_sequences,
    get_engine_train_val_split,
    RETAINED_SENSOR_COLUMNS,
)
from src.model import LSTMRegressor
from src.train import (
    train_model,
    evaluate_model,
    compute_mae,
    compute_rmse,
    compute_nasa_score,
    scale_target,
    unscale_target,
    DEFAULT_TARGET_SCALE,
)

# Visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.dpi"] = 120

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda_plots"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | PyTorch Version: {torch.__version__}")
"""))

    # Data Ingestion & Split
    cells.append(nbf.v4.new_markdown_cell("""## 1. Engine-Level Train/Validation Split
We partition the $100$ training engines into $80$ training units and $20$ validation units. Sequences are extracted separately to strictly prevent temporal sequence leakage."""))

    cells.append(nbf.v4.new_code_cell("""train_df, test_df, rul_df = load_cmapss("FD001", data_dir=PROJECT_ROOT / "data" / "raw")
train_clean = drop_uninformative_columns(train_df)
test_clean = drop_uninformative_columns(test_df)
train_rul = compute_rul(train_clean, max_rul=125, is_train=True)
train_scaled, test_scaled, scaler = scale_features(train_rul, test_clean)

X_train_split, y_train_split, X_val_split, y_val_split, train_units, val_units = get_engine_train_val_split(
    train_scaled,
    val_ratio=0.2,
    sequence_length=30,
    feature_cols=RETAINED_SENSOR_COLUMNS,
    label_col="RUL_capped",
    seed=42,
)

X_test, test_unit_order = get_test_sequences(test_scaled, sequence_length=30)
y_test_raw = rul_df["RUL"].values.astype(np.float32)
y_test_capped = np.minimum(y_test_raw, 125.0).astype(np.float32)

# Target normalization
y_train_norm = scale_target(y_train_split, DEFAULT_TARGET_SCALE)
y_val_norm = scale_target(y_val_split, DEFAULT_TARGET_SCALE)

print(f"Train Split: {len(train_units)} engines -> X_train: {X_train_split.shape}, y_train: {y_train_split.shape}")
print(f"Val Split:   {len(val_units)} engines -> X_val:   {X_val_split.shape}, y_val:   {y_val_split.shape}")
print(f"Test Set:    100 engines -> X_test:  {X_test.shape}, y_test:  {y_test_raw.shape}")
"""))

    # Dataloaders
    cells.append(nbf.v4.new_markdown_cell("""## 2. PyTorch DataLoaders"""))
    cells.append(nbf.v4.new_code_cell("""batch_size = 64
train_dataset = TensorDataset(torch.tensor(X_train_split, dtype=torch.float32), torch.tensor(y_train_norm, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val_split, dtype=torch.float32), torch.tensor(y_val_norm, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test_capped, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
print("DataLoaders ready.")
"""))

    # Tuning Grid
    cells.append(nbf.v4.new_markdown_cell("""## 3. Compact Hyperparameter Tuning Grid (8 Runs)
We sweep combinations of:
* `hidden_size`: $32, 64$
* `dropout`: $0.2, 0.3$
* `learning_rate`: $0.001, 0.0005$
* Early stopping: monitored on **Validation MAE** with patience $= 8$."""))

    cells.append(nbf.v4.new_code_cell("""grid = [
    {"hidden_size": 32, "dropout": 0.2, "learning_rate": 0.0010},
    {"hidden_size": 32, "dropout": 0.2, "learning_rate": 0.0005},
    {"hidden_size": 32, "dropout": 0.3, "learning_rate": 0.0010},
    {"hidden_size": 32, "dropout": 0.3, "learning_rate": 0.0005},
    {"hidden_size": 64, "dropout": 0.2, "learning_rate": 0.0010},
    {"hidden_size": 64, "dropout": 0.2, "learning_rate": 0.0005},
    {"hidden_size": 64, "dropout": 0.3, "learning_rate": 0.0010},
    {"hidden_size": 64, "dropout": 0.3, "learning_rate": 0.0005},
]

tuning_records = []
models_dict = {}

max_epochs = 60
patience = 8

for i, cfg in enumerate(grid, 1):
    torch.manual_seed(42)
    model = LSTMRegressor(
        input_size=14,
        hidden_size=cfg["hidden_size"],
        num_layers=2,
        dropout_rate=cfg["dropout"],
    )

    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=max_epochs,
        lr=cfg["learning_rate"],
        patience=patience,
        target_scale=DEFAULT_TARGET_SCALE,
        device=device,
        verbose=False,
    )

    val_preds, val_targets, val_metrics = evaluate_model(
        trained_model, val_loader, target_scale=DEFAULT_TARGET_SCALE, device=device
    )

    best_ep = history["best_epoch"]
    tot_ep = history["epochs_trained"]
    near_max = best_ep >= (max_epochs - 3)

    record = {
        "config_id": i,
        "hidden_size": cfg["hidden_size"],
        "dropout": cfg["dropout"],
        "learning_rate": cfg["learning_rate"],
        "val_MAE": round(val_metrics["MAE"], 4),
        "val_RMSE": round(val_metrics["RMSE"], 4),
        "val_NASA_Score": round(val_metrics["NASA_Score"], 2),
        "epochs_trained": tot_ep,
        "best_epoch": best_ep,
        "near_max_flag": near_max,
    }
    tuning_records.append(record)
    models_dict[i] = (trained_model, history, val_metrics)

tuning_df = pd.DataFrame(tuning_records).sort_values(by="val_MAE").reset_index(drop=True)
tuning_df.to_csv(PROJECT_ROOT / "outputs" / "tuning_results.csv", index=False)
tuning_df
"""))

    # Best Model Selection
    cells.append(nbf.v4.new_markdown_cell("""## 4. Best Model Selection & Checkpointing
We select the model configuration achieving the minimum **Validation MAE** (no test data used for selection)."""))

    cells.append(nbf.v4.new_code_cell("""best_row = tuning_df.iloc[0]
best_config_id = int(best_row["config_id"])
best_model, best_history, best_val_metrics = models_dict[best_config_id]

print(f"Selected Best Configuration (#{best_config_id}):")
print(f"  • Hidden Units:   {int(best_row['hidden_size'])}")
print(f"  • Dropout:        {best_row['dropout']}")
print(f"  • Learning Rate:  {best_row['learning_rate']}")
print(f"  • Validation MAE: {best_row['val_MAE']:.4f} cycles")
print(f"  • Validation RMSE:{best_row['val_RMSE']:.4f} cycles")
print(f"  • Best Epoch:     {int(best_row['best_epoch'])} / {int(best_row['epochs_trained'])}")

# Save model checkpoint with target_scale=125.0
torch.save(
    {
        "model_state_dict": best_model.state_dict(),
        "config": {
            "input_size": 14,
            "hidden_size": int(best_row["hidden_size"]),
            "num_layers": 2,
            "dropout_rate": float(best_row["dropout"]),
            "sequence_length": 30,
            "target_scale": DEFAULT_TARGET_SCALE,
        },
    },
    PROJECT_ROOT / "outputs" / "best_lstm_model.pt",
)
print("Saved winning model to outputs/best_lstm_model.pt")
"""))

    # Test Set Evaluation
    cells.append(nbf.v4.new_markdown_cell("""## 5. Final Evaluation on Held-Out Test Set (100 Engines)
Evaluated ONCE on the held-out test set against both capped target ($RUL \\le 125$) and raw ground-truth ($RUL$)."""))

    cells.append(nbf.v4.new_code_cell("""test_preds, _, _ = evaluate_model(
    best_model, test_loader, target_scale=DEFAULT_TARGET_SCALE, device=device
)

test_mae_capped = compute_mae(y_test_capped, test_preds)
test_rmse_capped = compute_rmse(y_test_capped, test_preds)
test_nasa_capped = compute_nasa_score(y_test_capped, test_preds)

test_mae_raw = compute_mae(y_test_raw, test_preds)
test_rmse_raw = compute_rmse(y_test_raw, test_preds)
test_nasa_raw = compute_nasa_score(y_test_raw, test_preds)

eval_summary = pd.DataFrame([
    {"Target": "Capped RUL (max=125)", "MAE": test_mae_capped, "RMSE": test_rmse_capped, "NASA Score": test_nasa_capped},
    {"Target": "Raw True RUL", "MAE": test_mae_raw, "RMSE": test_rmse_raw, "NASA Score": test_nasa_raw},
])
eval_summary
"""))

    # Visualizations
    cells.append(nbf.v4.new_markdown_cell("""## 6. Evaluation Visualizations"""))
    cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=150)
errors = test_preds - y_test_capped

scatter = ax.scatter(
    y_test_capped,
    test_preds,
    c=np.abs(errors),
    cmap="coolwarm",
    s=70,
    edgecolor="black",
    linewidth=0.6,
    alpha=0.9,
    zorder=4,
)
cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Absolute Error |Predicted - Actual| (Cycles)", fontsize=10, fontweight="bold")

lims = [0, 140]
ax.plot(lims, lims, color="#0f172a", linestyle="--", linewidth=1.8, label="Ideal Prediction (y = x)", zorder=3)
ax.fill_between(lims, [l - 15 for l in lims], [l + 15 for l in lims], color="#3b82f6", alpha=0.10, label="±15 Cycles Error Band")
ax.fill_between(lims, [l - 30 for l in lims], [l + 30 for l in lims], color="#64748b", alpha=0.06, label="±30 Cycles Error Band")

metric_text = (
    f"Test Performance (100 Engines):\\n"
    f"  • Capped MAE:  {test_mae_capped:.2f} cycles\\n"
    f"  • Capped RMSE: {test_rmse_capped:.2f} cycles\\n"
    f"  • NASA Score:  {test_nasa_capped:.1f}\\n"
    f"Model: LSTM (h={int(best_row['hidden_size'])}, drop={best_row['dropout']}, lr={best_row['learning_rate']})"
)
ax.text(
    0.04, 0.96, metric_text, transform=ax.transAxes, fontsize=9.5, verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#cbd5e1", alpha=0.95), zorder=5
)

ax.set_xlim(0, 140)
ax.set_ylim(0, 140)
ax.set_xlabel("True Ground-Truth RUL (Cycles)", fontsize=11, fontweight="bold")
ax.set_ylabel("Predicted RUL by LSTM (Cycles)", fontsize=11, fontweight="bold")
ax.set_title("Predicted vs. Actual Remaining Useful Life (100 Test Engines)", fontsize=12, fontweight="bold", pad=12)
ax.legend(loc="lower right", frameon=True, fontsize=9.5)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
"""))

    nb.cells = cells
    notebook_path = PROJECT_ROOT / "notebooks" / "03_model_training.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    main()
