"""
run_preprocessing.py

Executes the Phase 2 Preprocessing Pipeline for NASA C-MAPSS FD001.
Validates each transformation step, creates the RUL capping comparison plot,
persists processed array artifacts in data/processed/, and builds notebooks/02_preprocessing_check.ipynb.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nbformat as nbf
import joblib

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
    count_padded_engines,
    UNINFORMATIVE_COLUMNS,
    RETAINED_SENSOR_COLUMNS,
)

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SCALER_PATH = PROJECT_ROOT / "outputs" / "scaler.pkl"

# Visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]


def main():
    print("=" * 75)
    print("PHASE 2: PREPROCESSING PIPELINE EXECUTION & VERIFICATION")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Step 1: Ingestion
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading raw C-MAPSS FD001 dataset...")
    train_raw, test_raw, rul_df = load_cmapss(subset="FD001", data_dir=PROJECT_ROOT / "data" / "raw")
    print(f"  -> Raw train_df shape: {train_raw.shape} (100 engines, {train_raw.shape[1]} columns)")
    print(f"  -> Raw test_df shape:  {test_raw.shape} (100 engines, {test_raw.shape[1]} columns)")
    print(f"  -> Ground truth RUL:   {rul_df.shape} (100 test engines)")

    # -------------------------------------------------------------------------
    # Step 2: Drop Uninformative Columns
    # -------------------------------------------------------------------------
    print("\n[Step 2] Dropping 10 uninformative/zero-variance columns...")
    print(f"  Columns to drop: {UNINFORMATIVE_COLUMNS}")
    train_dropped = drop_uninformative_columns(train_raw)
    test_dropped = drop_uninformative_columns(test_raw)
    print(f"  -> Train shape after drop: {train_dropped.shape} ({len(RETAINED_SENSOR_COLUMNS)} sensor features + 2 index)")
    print(f"  -> Test shape after drop:  {test_dropped.shape}")

    # -------------------------------------------------------------------------
    # Step 3: Compute RUL and Piecewise Capping (max_rul = 125)
    # -------------------------------------------------------------------------
    print("\n[Step 3] Computing true RUL and piecewise-linear capped RUL (cap = 125)...")
    train_rul = compute_rul(train_dropped, max_rul=125, is_train=True)
    print(f"  -> Raw RUL range:        [{train_rul['RUL'].min()}, {train_rul['RUL'].max()}] cycles")
    print(f"  -> Capped RUL range:     [{train_rul['RUL_capped'].min()}, {train_rul['RUL_capped'].max()}] cycles")
    print(f"  -> Total capped rows:    {(train_rul['RUL'] > 125).sum():,} / {len(train_rul):,} ({((train_rul['RUL'] > 125).mean()*100):.1f}%)")

    # -------------------------------------------------------------------------
    # Step 4: Scale Features (MinMaxScaler [0, 1] fit ONLY on train)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Fitting MinMaxScaler on training sensors and transforming train & test...")
    train_scaled, test_scaled, scaler = scale_features(
        train_rul,
        test_dropped,
        feature_cols=RETAINED_SENSOR_COLUMNS,
        scaler_path=SCALER_PATH,
    )
    print(f"  -> Fitted scaler saved to: {SCALER_PATH}")
    print(f"  -> Scaled train features min: {train_scaled[RETAINED_SENSOR_COLUMNS].min().min():.4f}, max: {train_scaled[RETAINED_SENSOR_COLUMNS].max().max():.4f}")
    print(f"  -> Scaled test features min:  {test_scaled[RETAINED_SENSOR_COLUMNS].min().min():.4f}, max: {test_scaled[RETAINED_SENSOR_COLUMNS].max().max():.4f}")

    # -------------------------------------------------------------------------
    # Step 5: Sliding-Window Sequence Generation (Sequence Length = 30)
    # -------------------------------------------------------------------------
    SEQUENCE_LENGTH = 30
    print(f"\n[Step 5] Generating temporal sequences (sequence_length = {SEQUENCE_LENGTH})...")
    
    # Check padding requirements
    train_padded_count = count_padded_engines(train_scaled, sequence_length=SEQUENCE_LENGTH)
    test_padded_count = count_padded_engines(test_scaled, sequence_length=SEQUENCE_LENGTH)
    print(f"  -> Training engines requiring left-padding: {train_padded_count} / 100")
    print(f"  -> Test engines requiring left-padding:     {test_padded_count} / 100")

    X_train, y_train = generate_sequences(
        train_scaled,
        sequence_length=SEQUENCE_LENGTH,
        feature_cols=RETAINED_SENSOR_COLUMNS,
        label_col="RUL_capped",
    )
    print(f"  -> Final X_train shape: {X_train.shape} (dtype: {X_train.dtype})")
    print(f"  -> Final y_train shape: {y_train.shape} (dtype: {y_train.dtype})")

    X_test, test_unit_order = get_test_sequences(
        test_scaled,
        sequence_length=SEQUENCE_LENGTH,
        feature_cols=RETAINED_SENSOR_COLUMNS,
    )
    y_test = rul_df["RUL"].values.astype(np.float32)
    y_test_capped = np.minimum(y_test, 125.0).astype(np.float32)

    print(f"  -> Final X_test shape:  {X_test.shape} (dtype: {X_test.dtype})")
    print(f"  -> Test unit order:     {test_unit_order.shape} (IDs: {test_unit_order[0]}..{test_unit_order[-1]})")
    print(f"  -> Ground truth y_test: {y_test.shape}")

    # -------------------------------------------------------------------------
    # Step 6: Plot RUL Capping Comparison
    # -------------------------------------------------------------------------
    print("\n[Step 6] Plotting 'Before vs After Capping' comparison for 3 sample engines...")
    sample_units = [39, 1, 69]  # Shortest life (39: 128 cycles), Typical life (1: 192 cycles), Longest life (69: 362 cycles)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    colors = ["#d97706", "#2563eb", "#7c3aed"]

    for idx, unit_id in enumerate(sample_units):
        ax = axes[idx]
        unit_df = train_rul[train_rul["unit_number"] == unit_id]
        
        ax.plot(
            unit_df["time_in_cycles"],
            unit_df["RUL"],
            label="Raw Linear RUL (Uncapped)",
            color="#94a3b8",
            linestyle="--",
            linewidth=2.0,
        )
        ax.plot(
            unit_df["time_in_cycles"],
            unit_df["RUL_capped"],
            label="Piecewise Capped RUL (Cap = 125)",
            color=colors[idx],
            linewidth=2.5,
        )
        ax.axhline(125, color="#dc2626", linestyle=":", linewidth=1.2, alpha=0.8, label="Cap Threshold (125)")
        ax.axvline(unit_df["time_in_cycles"].max(), color="#ef4444", linestyle="-.", linewidth=1.2, label=f"Failure (Cycle {unit_df['time_in_cycles'].max()})")
        
        ax.set_title(f"Engine Unit {unit_id} (Lifespan: {len(unit_df)} cycles)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time in Cycles", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Remaining Useful Life (RUL)", fontsize=11)
        ax.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="white")
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Piecewise-Linear RUL Capping (Cap = 125) vs Raw Linear Degradation", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    capping_plot_path = OUTPUT_DIR / "07_rul_capping_comparison.png"
    plt.savefig(capping_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved capping comparison plot to: {capping_plot_path}")

    # -------------------------------------------------------------------------
    # Step 7: Persist Processed Numpy Arrays in data/processed/
    # -------------------------------------------------------------------------
    print("\n[Step 7] Persisting processed sequence arrays in data/processed/...")
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    np.save(PROCESSED_DIR / "y_test_capped.npy", y_test_capped)
    np.save(PROCESSED_DIR / "test_unit_order.npy", test_unit_order)
    print(f"  -> Saved X_train.npy:       {X_train.shape} ({(X_train.nbytes / (1024*1024)):.2f} MB)")
    print(f"  -> Saved y_train.npy:       {y_train.shape} ({(y_train.nbytes / 1024):.2f} KB)")
    print(f"  -> Saved X_test.npy:        {X_test.shape} ({(X_test.nbytes / 1024):.2f} KB)")
    print(f"  -> Saved y_test.npy:        {y_test.shape}")
    print(f"  -> Saved y_test_capped.npy: {y_test_capped.shape}")
    print(f"  -> Saved test_unit_order.npy: {test_unit_order.shape}")

    # -------------------------------------------------------------------------
    # Step 8: Build Jupyter Notebook notebooks/02_preprocessing_check.ipynb
    # -------------------------------------------------------------------------
    print("\n[Step 8] Building Jupyter Notebook notebooks/02_preprocessing_check.ipynb...")
    create_preprocessing_notebook()
    print("  -> Notebook successfully generated!")

    print("\n" + "=" * 75)
    print("PHASE 2 PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 75)


def create_preprocessing_notebook():
    """Create the Phase 2 Preprocessing verification Jupyter Notebook."""
    nb = nbf.v4.new_notebook()
    cells = []

    # Title Markdown
    cells.append(nbf.v4.new_markdown_cell("""# Phase 2: Preprocessing Pipeline & Verification

**Project:** Agentic Predictive Maintenance  
**Dataset:** NASA C-MAPSS FD001 (Single Operating Condition, Single Fault Mode)  
**Objective:** Transform raw sensor telemetry into structured sliding-window temporal sequences for deep learning sequence models (e.g., LSTM, GRU, Transformers) with zero data leakage.

---
### Pipeline Stages:
1. **Feature Screening:** Drop 10 confirmed uninformative/zero-variance columns.
2. **Piecewise-Linear RUL Capping:** Compute linear RUL and apply upper cap at $RUL = 125$ cycles.
3. **MinMax Feature Scaling:** Normalize 14 retained sensors to range $[0, 1]$ fit solely on training data.
4. **Sliding-Window Sequence Generation:** Generate $(X, y)$ sequence pairs (window length $= 30$).
5. **Test Set Final-Window Extraction:** Extract final $30$-cycle evaluation sequence for each test engine.
6. **Artifact Persistence:** Persist numpy arrays to `data/processed/` and scaler to `outputs/scaler.pkl`.
"""))

    # Cell 1: Setup & Imports
    cells.append(nbf.v4.new_code_cell("""import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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
    count_padded_engines,
    UNINFORMATIVE_COLUMNS,
    RETAINED_SENSOR_COLUMNS,
)

# Visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.dpi"] = 120

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda_plots"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Environment and preprocessing modules loaded.")
"""))

    # Cell 2: Raw Ingestion
    cells.append(nbf.v4.new_markdown_cell("""## 1. Raw Data Ingestion
Loading raw C-MAPSS FD001 dataset via `load_cmapss()`."""))

    cells.append(nbf.v4.new_code_cell("""train_raw, test_raw, rul_df = load_cmapss(subset="FD001", data_dir=PROJECT_ROOT / "data" / "raw")

print(f"Raw Train Shape: {train_raw.shape} (Engines: {train_raw['unit_number'].nunique()})")
print(f"Raw Test Shape:  {test_raw.shape} (Engines: {test_raw['unit_number'].nunique()})")
print(f"Test Ground Truth RUL: {rul_df.shape}")
"""))

    # Cell 3: Feature Dropping
    cells.append(nbf.v4.new_markdown_cell("""## 2. Feature Screening: Dropping Uninformative Columns
Dropping 10 uninformative/constant features identified during Phase 1 EDA:
* 3 Operational Settings: `operational_setting_1`, `operational_setting_2`, `operational_setting_3`
* 7 Constant/Near-Constant Sensors: `sensor_measurement_1`, `5`, `6`, `10`, `16`, `18`, `19`

Retaining **14 informative sensor measurements**: `2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21`."""))

    cells.append(nbf.v4.new_code_cell("""train_dropped = drop_uninformative_columns(train_raw)
test_dropped = drop_uninformative_columns(test_raw)

print(f"Train Shape after dropping: {train_dropped.shape}")
print(f"Test Shape after dropping:  {test_dropped.shape}")
print(f"Retained Sensor Columns ({len(RETAINED_SENSOR_COLUMNS)}):")
print(RETAINED_SENSOR_COLUMNS)
"""))

    # Cell 4: RUL Capping
    cells.append(nbf.v4.new_markdown_cell("""## 3. Ground-Truth RUL & Piecewise-Linear Capping
For each run-to-failure engine, linear RUL is computed as:
$$\\text{RUL}(t) = t_{\\text{max}} - t$$
In standard prognostic literature for C-MAPSS FD001, early operational cycles exhibit negligible physical degradation. An upper threshold of **$125$ cycles** is applied:
$$\\text{RUL}_{\\text{capped}}(t) = \\min(\\text{RUL}(t), 125)$$"""))

    cells.append(nbf.v4.new_code_cell("""train_rul = compute_rul(train_dropped, max_rul=125, is_train=True)

print(f"Raw RUL Range:    [{train_rul['RUL'].min()}, {train_rul['RUL'].max()}] cycles")
print(f"Capped RUL Range: [{train_rul['RUL_capped'].min()}, {train_rul['RUL_capped'].max()}] cycles")
print(f"Rows with Capped Value (=125): {(train_rul['RUL_capped'] == 125).sum():,} / {len(train_rul):,} ({(train_rul['RUL_capped'] == 125).mean()*100:.1f}%)")

train_rul[["unit_number", "time_in_cycles", "RUL", "RUL_capped"]].head(10)
"""))

    # Cell 5: RUL Capping Visualization
    cells.append(nbf.v4.new_markdown_cell("""### Visual Verification of Piecewise Capping
Overlaying raw linear RUL vs capped RUL for 3 sample engines spanning minimum, typical, and maximum lifetimes."""))

    cells.append(nbf.v4.new_code_cell("""sample_units = [39, 1, 69]  # Min life (39: 128), Typical (1: 192), Max life (69: 362)
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
colors = ["#d97706", "#2563eb", "#7c3aed"]

for idx, unit_id in enumerate(sample_units):
    ax = axes[idx]
    unit_df = train_rul[train_rul["unit_number"] == unit_id]
    
    ax.plot(
        unit_df["time_in_cycles"],
        unit_df["RUL"],
        label="Raw Linear RUL",
        color="#94a3b8",
        linestyle="--",
        linewidth=2.0,
    )
    ax.plot(
        unit_df["time_in_cycles"],
        unit_df["RUL_capped"],
        label="Capped RUL (Cap = 125)",
        color=colors[idx],
        linewidth=2.5,
    )
    ax.axhline(125, color="#dc2626", linestyle=":", linewidth=1.2, label="Cap Threshold (125)")
    ax.axvline(unit_df["time_in_cycles"].max(), color="#ef4444", linestyle="-.", linewidth=1.2, label=f"Failure ({unit_df['time_in_cycles'].max()} cycles)")
    
    ax.set_title(f"Engine Unit {unit_id} (Lifespan: {len(unit_df)} cycles)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time in Cycles", fontsize=11)
    if idx == 0:
        ax.set_ylabel("Remaining Useful Life (RUL)", fontsize=11)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="white")
    ax.grid(True, linestyle="--", alpha=0.6)

plt.suptitle("Piecewise-Linear RUL Capping (Cap = 125) vs Raw Linear Degradation", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_rul_capping_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
"""))

    # Cell 6: Scaling
    cells.append(nbf.v4.new_markdown_cell("""## 4. MinMax Feature Scaling (Zero Leakage)
A `MinMaxScaler(range=(0, 1))` is fit strictly on training features and subsequently used to transform both train and test sets. The fitted scaler is persisted to `outputs/scaler.pkl`."""))

    cells.append(nbf.v4.new_code_cell("""train_scaled, test_scaled, scaler = scale_features(
    train_rul,
    test_dropped,
    feature_cols=RETAINED_SENSOR_COLUMNS,
    scaler_path=PROJECT_ROOT / "outputs" / "scaler.pkl",
)

print(f"Scaler saved to: {PROJECT_ROOT / 'outputs' / 'scaler.pkl'}")
print(f"Train Scaled Min: {train_scaled[RETAINED_SENSOR_COLUMNS].min().min():.4f}, Max: {train_scaled[RETAINED_SENSOR_COLUMNS].max().max():.4f}")
print(f"Test Scaled Min:  {test_scaled[RETAINED_SENSOR_COLUMNS].min().min():.4f}, Max: {test_scaled[RETAINED_SENSOR_COLUMNS].max().max():.4f}")
"""))

    # Cell 7: Sequence Generation
    cells.append(nbf.v4.new_markdown_cell("""## 5. Temporal Sequence Generation (Sliding Window = 30)
* **Training Sequences:** A sliding window of $30$ cycles slides over each engine's trajectory to create $(X, y)$ pairs, where $X \\in \\mathbb{R}^{30 \\times 14}$ and $y = \\text{RUL}_{\\text{capped}}$ at the window's final cycle.
* **Test Sequences:** Exactly one $30$-cycle sequence is generated per test engine (the final $30$ observed cycles) for direct evaluation against `RUL_FD001.txt`.
* **Padding Check:** Any engine with total cycles $< 30$ is left-padded with its initial cycle repeated."""))

    cells.append(nbf.v4.new_code_cell("""SEQUENCE_LENGTH = 30

train_padded = count_padded_engines(train_scaled, sequence_length=SEQUENCE_LENGTH)
test_padded = count_padded_engines(test_scaled, sequence_length=SEQUENCE_LENGTH)

print(f"Training Engines requiring Left-Padding: {train_padded} / 100")
print(f"Test Engines requiring Left-Padding:     {test_padded} / 100")

# Generate Training Sequences
X_train, y_train = generate_sequences(
    train_scaled,
    sequence_length=SEQUENCE_LENGTH,
    feature_cols=RETAINED_SENSOR_COLUMNS,
    label_col="RUL_capped",
)

# Generate Test Sequences
X_test, test_unit_order = get_test_sequences(
    test_scaled,
    sequence_length=SEQUENCE_LENGTH,
    feature_cols=RETAINED_SENSOR_COLUMNS,
)

y_test = rul_df["RUL"].values.astype(np.float32)
y_test_capped = np.minimum(y_test, 125.0).astype(np.float32)

print(f"\\nFinal Processed Array Shapes:")
print(f"  X_train: {X_train.shape} (dtype: {X_train.dtype})")
print(f"  y_train: {y_train.shape} (dtype: {y_train.dtype})")
print(f"  X_test:  {X_test.shape} (dtype: {X_test.dtype})")
print(f"  y_test:  {y_test.shape} (dtype: {y_test.dtype})")
print(f"  Test unit IDs: {test_unit_order[:5]} ... {test_unit_order[-5:]}")
"""))

    # Cell 8: Saving Arrays
    cells.append(nbf.v4.new_markdown_cell("""## 6. Persisting Preprocessed Datasets
Saving arrays to `data/processed/` for direct consumption by deep learning models in Phase 3."""))

    cells.append(nbf.v4.new_code_cell("""np.save(PROCESSED_DIR / "X_train.npy", X_train)
np.save(PROCESSED_DIR / "y_train.npy", y_train)
np.save(PROCESSED_DIR / "X_test.npy", X_test)
np.save(PROCESSED_DIR / "y_test.npy", y_test)
np.save(PROCESSED_DIR / "y_test_capped.npy", y_test_capped)
np.save(PROCESSED_DIR / "test_unit_order.npy", test_unit_order)

print("Saved all preprocessed numpy arrays to data/processed/:")
for p in sorted(PROCESSED_DIR.glob("*.npy")):
    size_kb = p.stat().st_size / 1024
    print(f"  - {p.name:<22} ({size_kb:>8.2f} KB)")
"""))

    nb.cells = cells
    notebook_path = PROJECT_ROOT / "notebooks" / "02_preprocessing_check.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to: {notebook_path}")


if __name__ == "__main__":
    main()
