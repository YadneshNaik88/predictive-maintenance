"""
eda_runner.py

Executes comprehensive Exploratory Data Analysis (EDA) on NASA C-MAPSS FD001.
Saves 6 high-resolution PNG plots in outputs/eda_plots/, produces text summary statistics,
and generates the executed Jupyter Notebook at notebooks/01_eda.ipynb.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nbformat as nbf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_cmapss,
    INDEX_COLUMNS,
    SETTING_COLUMNS,
    SENSOR_COLUMNS,
    ALL_COLUMNS,
)

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#e5e7eb"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7


def main():
    print("=" * 70)
    print("PHASE 1: NASA C-MAPSS FD001 EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # 1. Load data
    train_df, test_df, rul_df = load_cmapss(subset="FD001", data_dir=PROJECT_ROOT / "data" / "raw")
    print(f"Loaded train_df: {train_df.shape}")
    print(f"Loaded test_df:  {test_df.shape}")
    print(f"Loaded rul_df:   {rul_df.shape}")

    # Compute max cycles & true RUL for training set
    max_cycles_df = train_df.groupby("unit_number")["time_in_cycles"].max().reset_index()
    max_cycles_df.columns = ["unit_number", "max_cycles"]
    train_df = train_df.merge(max_cycles_df, on="unit_number")
    train_df["RUL"] = train_df["max_cycles"] - train_df["time_in_cycles"]

    summary_lines = []
    summary_lines.append("=" * 75)
    summary_lines.append("NASA C-MAPSS FD001 - EXPLORATORY DATA ANALYSIS SUMMARY")
    summary_lines.append("=" * 75)
    summary_lines.append(f"Training Instances:    {len(train_df):,} rows")
    summary_lines.append(f"Test Instances:        {len(test_df):,} rows")
    summary_lines.append(f"Training Engines:      {train_df['unit_number'].nunique()}")
    summary_lines.append(f"Test Engines:          {test_df['unit_number'].nunique()}")
    summary_lines.append(f"Features:              {len(ALL_COLUMNS)} (2 Index, 3 Op Settings, 21 Sensors)\n")

    # =========================================================================
    # Output A: Engine Life-Cycle Distribution
    # =========================================================================
    print("\n[1/6] Generating Output A: Engine Life-Cycle Distribution...")
    max_cycles = max_cycles_df["max_cycles"]
    
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [0.2, 0.8]}
    )
    
    sns.boxplot(x=max_cycles, ax=ax_box, color="#3b82f6", fliersize=4)
    ax_box.set(xlabel="")
    ax_box.set_title("Engine Maximum Life-Cycle Distribution (FD001 Training Set)", fontsize=14, fontweight="bold", pad=12)
    
    sns.histplot(max_cycles, kde=True, ax=ax_hist, color="#1d4ed8", bins=20, edgecolor="white", alpha=0.7)
    ax_hist.axvline(max_cycles.mean(), color="#ef4444", linestyle="--", linewidth=2, label=f"Mean: {max_cycles.mean():.1f} cycles")
    ax_hist.axvline(max_cycles.median(), color="#10b981", linestyle=":", linewidth=2, label=f"Median: {max_cycles.median():.1f} cycles")
    ax_hist.axvline(max_cycles.min(), color="#6b7280", linestyle="-.", linewidth=1.5, label=f"Min: {max_cycles.min()} cycles (Unit {max_cycles_df.loc[max_cycles.idxmin(), 'unit_number']})")
    ax_hist.axvline(max_cycles.max(), color="#8b5cf6", linestyle="-.", linewidth=1.5, label=f"Max: {max_cycles.max()} cycles (Unit {max_cycles_df.loc[max_cycles.idxmax(), 'unit_number']})")
    
    ax_hist.set_xlabel("Operational Life (Time in Cycles before Failure)", fontsize=12, labelpad=8)
    ax_hist.set_ylabel("Engine Count", fontsize=12, labelpad=8)
    ax_hist.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#e5e7eb")
    plt.tight_layout()
    
    plot_a_path = OUTPUT_DIR / "01_engine_lifecycle_distribution.png"
    plt.savefig(plot_a_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_a_path}")

    summary_lines.append("-" * 75)
    summary_lines.append("1. ENGINE LIFECYCLE STATISTICS (100 Training Engines)")
    summary_lines.append("-" * 75)
    summary_lines.append(f"  Count:  {len(max_cycles)}")
    summary_lines.append(f"  Mean:   {max_cycles.mean():.2f} cycles")
    summary_lines.append(f"  Std:    {max_cycles.std():.2f} cycles")
    summary_lines.append(f"  Min:    {max_cycles.min()} cycles (Unit {max_cycles_df.loc[max_cycles.idxmin(), 'unit_number']})")
    summary_lines.append(f"  25%:    {max_cycles.quantile(0.25):.1f} cycles")
    summary_lines.append(f"  50%:    {max_cycles.median():.1f} cycles (Median)")
    summary_lines.append(f"  75%:    {max_cycles.quantile(0.75):.1f} cycles")
    summary_lines.append(f"  Max:    {max_cycles.max()} cycles (Unit {max_cycles_df.loc[max_cycles.idxmax(), 'unit_number']})\n")

    # =========================================================================
    # Output B: Sensor Trend Plots for at least 5 Engines
    # =========================================================================
    print("\n[2/6] Generating Output B: Sensor Degradation Trends for 5 Engines...")
    sample_units = [1, 20, 39, 69, 100]  # Diverse sample: standard, min (39), max (69), and boundaries
    target_sensors = [
        ("sensor_measurement_2", "Sensor 2: Total Temp at LPC Outlet (°R)"),
        ("sensor_measurement_3", "Sensor 3: Total Temp at HPC Outlet (°R)"),
        ("sensor_measurement_4", "Sensor 4: Total Temp at LPT Outlet (°R)"),
        ("sensor_measurement_7", "Sensor 7: Total Pressure at HPC Outlet (psia)"),
        ("sensor_measurement_11", "Sensor 11: Static Pressure at HPC Outlet (psia)"),
    ]
    
    fig, axes = plt.subplots(len(target_sensors), 1, figsize=(12, 14), sharex=False)
    colors = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed"]

    for idx, (sensor_col, sensor_label) in enumerate(target_sensors):
        ax = axes[idx]
        for u_idx, unit_id in enumerate(sample_units):
            unit_data = train_df[train_df["unit_number"] == unit_id]
            ax.plot(
                unit_data["time_in_cycles"],
                unit_data[sensor_col],
                label=f"Unit {unit_id} (max {len(unit_data)} cycles)",
                color=colors[u_idx],
                alpha=0.85,
                linewidth=1.4,
            )
            # Mark failure point
            last_point = unit_data.iloc[-1]
            ax.scatter(last_point["time_in_cycles"], last_point[sensor_col], color=colors[u_idx], s=35, marker="x", zorder=5)
            
        ax.set_title(sensor_label, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel("Reading", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper left", ncol=5, fontsize=8.5, frameon=True, facecolor="white")
        if idx == len(target_sensors) - 1:
            ax.set_xlabel("Time in Cycles (Flight Cycles)", fontsize=11, labelpad=6)

    plt.suptitle("Sensor Degradation Trajectories Approaching Failure (5 Sample Engines)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plot_b_path = OUTPUT_DIR / "02_sensor_degradation_trends.png"
    plt.savefig(plot_b_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_b_path}")

    # =========================================================================
    # Output C: Correlation Heatmap
    # =========================================================================
    print("\n[3/6] Generating Output C: Correlation Heatmap...")
    corr_features = SETTING_COLUMNS + SENSOR_COLUMNS + ["RUL"]
    corr_matrix = train_df[corr_features].corr(method="pearson")

    plt.figure(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation Coefficient"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7.5},
    )
    plt.title("Correlation Matrix: Operational Settings, 21 Sensors & RUL (FD001)", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    plot_c_path = OUTPUT_DIR / "03_correlation_heatmap.png"
    plt.savefig(plot_c_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_c_path}")

    # =========================================================================
    # Output D: Sensor Variance Analysis
    # =========================================================================
    print("\n[4/6] Generating Output D: Sensor Variance Analysis...")
    variances = train_df[SENSOR_COLUMNS].var()
    stds = train_df[SENSOR_COLUMNS].std()
    
    var_df = pd.DataFrame({
        "sensor": SENSOR_COLUMNS,
        "variance": variances.values,
        "std": stds.values,
    })
    
    # Classify sensors: Low/Zero Variance (< 1e-3) vs Informative
    var_df["status"] = np.where(var_df["variance"] < 1e-3, "Zero / Near-Zero Variance (Candidate to Drop)", "Informative Feature")
    var_df["log10_variance"] = np.log10(np.clip(var_df["variance"], 1e-12, None))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot raw variance (excluding huge scale s9, s14 for clarity or with broken axis/log)
    palette = {"Zero / Near-Zero Variance (Candidate to Drop)": "#ef4444", "Informative Feature": "#2563eb"}
    
    sns.barplot(
        data=var_df,
        x="sensor",
        y="variance",
        hue="status",
        palette=palette,
        dodge=False,
        ax=ax1,
    )
    ax1.set_xticks(range(len(SENSOR_COLUMNS)))
    ax1.set_xticklabels([s.replace("sensor_measurement_", "s_") for s in SENSOR_COLUMNS], rotation=45, ha="right")
    ax1.set_yscale("log")
    ax1.set_title("Log10 Variance per Sensor Column", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Variance (Log Scale)", fontsize=11)
    ax1.set_xlabel("Sensor", fontsize=11)
    ax1.axhline(1e-3, color="#dc2626", linestyle="--", linewidth=1.2, label="Low-Info Threshold (Var < 0.001)")
    ax1.legend(loc="upper right", fontsize=8.5)

    # Plot Standard Deviation
    sns.barplot(
        data=var_df,
        x="sensor",
        y="std",
        hue="status",
        palette=palette,
        dodge=False,
        ax=ax2,
    )
    ax2.set_xticks(range(len(SENSOR_COLUMNS)))
    ax2.set_xticklabels([s.replace("sensor_measurement_", "s_") for s in SENSOR_COLUMNS], rotation=45, ha="right")
    ax2.set_title("Standard Deviation per Sensor Column", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Standard Deviation", fontsize=11)
    ax2.set_xlabel("Sensor", fontsize=11)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.suptitle("Sensor Variance & Information Content Analysis (FD001)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_d_path = OUTPUT_DIR / "04_sensor_variance_analysis.png"
    plt.savefig(plot_d_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_d_path}")

    # Categorize low variance sensors
    zero_var_sensors = var_df[var_df["variance"] < 1e-4]["sensor"].tolist()
    info_sensors = var_df[var_df["variance"] >= 1e-4]["sensor"].tolist()

    summary_lines.append("-" * 75)
    summary_lines.append("2. SENSOR VARIANCE ANALYSIS & FEATURE SCREENING")
    summary_lines.append("-" * 75)
    summary_lines.append(f"  Total Sensor Columns: 21")
    summary_lines.append(f"  Zero / Near-Zero Variance Sensors ({len(zero_var_sensors)}): {', '.join(zero_var_sensors)}")
    summary_lines.append(f"  Informative Variance Sensors ({len(info_sensors)}):     {', '.join(info_sensors)}\n")
    summary_lines.append("  Detailed Sensor Variance Table:")
    summary_lines.append(f"  {'Sensor':<25} {'Variance':<16} {'Std Dev':<14} {'Status'}")
    for _, row in var_df.iterrows():
        summary_lines.append(f"  {row['sensor']:<25} {row['variance']:<16.6e} {row['std']:<14.6f} {row['status']}")
    summary_lines.append("")

    # =========================================================================
    # Output E: RUL Distribution
    # =========================================================================
    print("\n[5/6] Generating Output E: RUL Distribution...")
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [0.2, 0.8]}
    )
    
    sns.boxplot(x=train_df["RUL"], ax=ax_box, color="#10b981", fliersize=3)
    ax_box.set(xlabel="")
    ax_box.set_title("True Remaining Useful Life (RUL) Distribution across All Training Rows", fontsize=14, fontweight="bold", pad=12)

    sns.histplot(train_df["RUL"], kde=True, ax=ax_hist, color="#059669", bins=35, edgecolor="white", alpha=0.7)
    ax_hist.axvline(train_df["RUL"].mean(), color="#ef4444", linestyle="--", linewidth=2, label=f"Mean: {train_df['RUL'].mean():.1f} cycles")
    ax_hist.axvline(train_df["RUL"].median(), color="#1d4ed8", linestyle=":", linewidth=2, label=f"Median: {train_df['RUL'].median():.1f} cycles")
    ax_hist.axvline(train_df["RUL"].max(), color="#8b5cf6", linestyle="-.", linewidth=1.5, label=f"Max: {train_df['RUL'].max()} cycles")

    ax_hist.set_xlabel("Remaining Useful Life (RUL in Cycles)", fontsize=12, labelpad=8)
    ax_hist.set_ylabel("Instance Count (Row Frequency)", fontsize=12, labelpad=8)
    ax_hist.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#e5e7eb")
    plt.tight_layout()

    plot_e_path = OUTPUT_DIR / "05_rul_distribution.png"
    plt.savefig(plot_e_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_e_path}")

    summary_lines.append("-" * 75)
    summary_lines.append("3. RUL DISTRIBUTION STATISTICS (Row-Level Ground Truth)")
    summary_lines.append("-" * 75)
    summary_lines.append(f"  Total Observations: {len(train_df):,}")
    summary_lines.append(f"  Mean RUL:           {train_df['RUL'].mean():.2f} cycles")
    summary_lines.append(f"  Std RUL:            {train_df['RUL'].std():.2f} cycles")
    summary_lines.append(f"  Min RUL:            {train_df['RUL'].min()} cycles (Engine failure point)")
    summary_lines.append(f"  25% Quantile:       {train_df['RUL'].quantile(0.25):.1f} cycles")
    summary_lines.append(f"  50% (Median):       {train_df['RUL'].median():.1f} cycles")
    summary_lines.append(f"  75% Quantile:       {train_df['RUL'].quantile(0.75):.1f} cycles")
    summary_lines.append(f"  Max RUL:            {train_df['RUL'].max()} cycles\n")

    # =========================================================================
    # Output F: Comparison of Early-Life vs Late-Life Sensor Behavior
    # =========================================================================
    print("\n[6/6] Generating Output F: Early-Life vs Late-Life Sensor Behavior...")
    # Calculate life ratio (0.0 to 1.0) for every row in train_df
    train_df["life_ratio"] = train_df["time_in_cycles"] / train_df["max_cycles"]
    
    early_df = train_df[train_df["life_ratio"] <= 0.10]
    late_df = train_df[train_df["life_ratio"] >= 0.90]

    early_means = early_df[SENSOR_COLUMNS].mean()
    late_means = late_df[SENSOR_COLUMNS].mean()
    overall_stds = train_df[SENSOR_COLUMNS].std()

    comp_df = pd.DataFrame({
        "sensor": SENSOR_COLUMNS,
        "early_mean": early_means.values,
        "late_mean": late_means.values,
        "absolute_shift": (late_means - early_means).values,
        "pct_change": (((late_means - early_means) / early_means) * 100).values,
        "z_shift": ((late_means - early_means) / np.clip(overall_stds, 1e-9, None)).values,
    })

    # Filter informative sensors for clear visual contrast
    info_comp_df = comp_df[comp_df["sensor"].isin(info_sensors)].copy()
    info_comp_df = info_comp_df.sort_values(by="z_shift", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_colors = np.where(info_comp_df["z_shift"] > 0, "#ef4444", "#3b82f6")
    
    y_pos = np.arange(len(info_comp_df))
    labels = [s.replace("sensor_measurement_", "Sensor ") for s in info_comp_df["sensor"]]
    
    bars = ax.barh(y_pos, info_comp_df["z_shift"], color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Standardized Shift from Early to Late Life (Z-score shift = Delta_mean / sigma)", fontsize=11, labelpad=8)
    ax.set_title("Early-Life (First 10%) vs Late-Life (Last 10%) Sensor Degradation Shifts", fontsize=13, fontweight="bold", pad=12)
    
    # Annotate percent change on bars
    for bar, pct in zip(bars, info_comp_df["pct_change"]):
        width = bar.get_width()
        ha = "left" if width >= 0 else "right"
        offset = 0.05 if width >= 0 else -0.05
        ax.text(width + offset, bar.get_y() + bar.get_height() / 2, f"{pct:+.2f}%", va="center", ha=ha, fontsize=8.5, fontweight="bold")

    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    plot_f_path = OUTPUT_DIR / "06_early_vs_late_life_sensors.png"
    plt.savefig(plot_f_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_f_path}")

    summary_lines.append("-" * 75)
    summary_lines.append("4. EARLY-LIFE (<=10%) vs LATE-LIFE (>=90%) SENSOR COMPARISON")
    summary_lines.append("-" * 75)
    summary_lines.append(f"  {'Sensor':<24} {'Early Mean':<14} {'Late Mean':<14} {'Abs Shift':<12} {'% Change':<12} {'Z-Shift'}")
    for _, row in comp_df.iterrows():
        summary_lines.append(
            f"  {row['sensor']:<24} {row['early_mean']:<14.4f} {row['late_mean']:<14.4f} "
            f"{row['absolute_shift']:<12.4f} {row['pct_change']:<+11.2f}% {row['z_shift']:<+7.2f} sigma"
        )
    summary_lines.append("")

    # Correlation ranking with RUL
    rul_corrs = train_df[SENSOR_COLUMNS + SETTING_COLUMNS].apply(lambda c: c.corr(train_df["RUL"]))
    corr_ranking = pd.DataFrame({"feature": rul_corrs.index, "corr_with_rul": rul_corrs.values})
    corr_ranking["abs_corr"] = corr_ranking["corr_with_rul"].abs()
    corr_ranking = corr_ranking.sort_values(by="abs_corr", ascending=False).reset_index(drop=True)

    summary_lines.append("-" * 75)
    summary_lines.append("5. FEATURE CORRELATION RANKING WITH RUL")
    summary_lines.append("-" * 75)
    summary_lines.append(f"  {'Rank':<6} {'Feature':<26} {'Correlation with RUL':<22} {'Interpretation'}")
    for rank, row in corr_ranking.iterrows():
        feat = row['feature']
        val = row['corr_with_rul']
        if pd.isna(val):
            interp = "Constant / Zero Variance"
            val_str = "NaN"
        elif val > 0.5:
            interp = "Strong Positive (Decreases toward failure)"
            val_str = f"{val:+.4f}"
        elif val < -0.5:
            interp = "Strong Negative (Increases toward failure)"
            val_str = f"{val:+.4f}"
        else:
            interp = "Weak / Low Sensitivity"
            val_str = f"{val:+.4f}"
        summary_lines.append(f"  {rank+1:<6} {feat:<26} {val_str:<22} {interp}")

    summary_text = "\n".join(summary_lines)
    summary_file = OUTPUT_DIR / "eda_summary_statistics.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\nSaved summary statistics to: {summary_file}")

    # =========================================================================
    # Build Interactive Jupyter Notebook: notebooks/01_eda.ipynb
    # =========================================================================
    print("\nGenerating Jupyter Notebook: notebooks/01_eda.ipynb...")
    create_jupyter_notebook()
    print("Notebook generation complete!")


def create_jupyter_notebook():
    """Create a fully documented and executable Jupyter Notebook for Phase 1 EDA."""
    nb = nbf.v4.new_notebook()
    cells = []

    # Title cell
    cells.append(nbf.v4.new_markdown_cell("""# Phase 1: Exploratory Data Analysis (EDA) - NASA C-MAPSS Turbofan Degradation (FD001)

**Project:** Agentic Predictive Maintenance  
**Dataset:** NASA C-MAPSS FD001 (Single Operating Condition: Sea Level, Single Fault Mode: HPC Degradation)  
**Objective:** Perform foundational data loading and exploratory analysis to investigate engine lifespans, sensor degradation dynamics, feature variance, and RUL distribution.

---
### Notebook Outline:
1. **Environment Setup & Data Ingestion** using `src.data_loader`
2. **Output A:** Engine Life-Cycle Distribution (Max cycles to failure)
3. **Output B:** Sensor Degradation Trends for Sample Engines
4. **Output C:** Feature Correlation Heatmap (Sensors, Settings, RUL)
5. **Output D:** Sensor Variance Analysis & Low-Information Feature Screening
6. **Output E:** Ground-Truth Remaining Useful Life (RUL) Distribution
7. **Output F:** Early-Life vs. Late-Life Sensor Shift Comparison
8. **Phase 1 Key Takeaways & Recommendations for Phase 2**
"""))

    # Cell 1: Imports
    cells.append(nbf.v4.new_code_cell("""import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path("..").resolve() if Path("..").resolve().name == "predictive-maintenance" else Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_cmapss, INDEX_COLUMNS, SETTING_COLUMNS, SENSOR_COLUMNS, ALL_COLUMNS

# Visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Data & Output Paths Initialized. Project Root: {PROJECT_ROOT}")
"""))

    # Cell 2: Ingestion & RUL Calculation
    cells.append(nbf.v4.new_markdown_cell("""## 1. Data Ingestion & Ground-Truth RUL Computation
We load `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` using our custom data loader `load_cmapss()`."""))

    cells.append(nbf.v4.new_code_cell("""train_df, test_df, rul_df = load_cmapss(subset="FD001", data_dir=PROJECT_ROOT / "data" / "raw")

# Compute max cycle per engine in train set & row-level RUL
max_cycles_df = train_df.groupby("unit_number")["time_in_cycles"].max().reset_index()
max_cycles_df.columns = ["unit_number", "max_cycles"]
train_df = train_df.merge(max_cycles_df, on="unit_number")
train_df["RUL"] = train_df["max_cycles"] - train_df["time_in_cycles"]
train_df["life_ratio"] = train_df["time_in_cycles"] / train_df["max_cycles"]

print(f"Train DataFrame shape: {train_df.shape} (Unique Engines: {train_df['unit_number'].nunique()})")
print(f"Test DataFrame shape:  {test_df.shape} (Unique Engines: {test_df['unit_number'].nunique()})")
print(f"Test Ground-Truth RUL: {rul_df.shape}")
train_df.head(5)
"""))

    # Cell 3: Output A
    cells.append(nbf.v4.new_markdown_cell("""## 2. Output A: Engine Life-Cycle Distribution
We analyze the distribution of maximum run-to-failure cycles across all 100 training engines."""))

    cells.append(nbf.v4.new_code_cell("""max_cycles = max_cycles_df["max_cycles"]

fig, (ax_box, ax_hist) = plt.subplots(
    2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [0.2, 0.8]}
)

sns.boxplot(x=max_cycles, ax=ax_box, color="#3b82f6", fliersize=4)
ax_box.set(xlabel="")
ax_box.set_title("Engine Maximum Life-Cycle Distribution (FD001)", fontsize=13, fontweight="bold")

sns.histplot(max_cycles, kde=True, ax=ax_hist, color="#1d4ed8", bins=20, edgecolor="white", alpha=0.7)
ax_hist.axvline(max_cycles.mean(), color="#ef4444", linestyle="--", linewidth=2, label=f"Mean: {max_cycles.mean():.1f} cycles")
ax_hist.axvline(max_cycles.median(), color="#10b981", linestyle=":", linewidth=2, label=f"Median: {max_cycles.median():.1f} cycles")
ax_hist.axvline(max_cycles.min(), color="#6b7280", linestyle="-.", linewidth=1.5, label=f"Min: {max_cycles.min()} cycles (Unit {max_cycles_df.loc[max_cycles.idxmin(), 'unit_number']})")
ax_hist.axvline(max_cycles.max(), color="#8b5cf6", linestyle="-.", linewidth=1.5, label=f"Max: {max_cycles.max()} cycles (Unit {max_cycles_df.loc[max_cycles.idxmax(), 'unit_number']})")

ax_hist.set_xlabel("Operational Life (Cycles to Failure)", fontsize=11)
ax_hist.set_ylabel("Engine Count", fontsize=11)
ax_hist.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_engine_lifecycle_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

max_cycles.describe()
"""))

    # Cell 4: Output B
    cells.append(nbf.v4.new_markdown_cell("""## 3. Output B: Sensor Degradation Trends for Sample Engines
Tracking sensor readings over operational cycles for 5 diverse engines (e.g., Units 1, 20, 39 [min lifespan], 69 [max lifespan], 100)."""))

    cells.append(nbf.v4.new_code_cell("""sample_units = [1, 20, 39, 69, 100]
target_sensors = [
    ("sensor_measurement_2", "Sensor 2: Total Temp at LPC Outlet (°R)"),
    ("sensor_measurement_3", "Sensor 3: Total Temp at HPC Outlet (°R)"),
    ("sensor_measurement_4", "Sensor 4: Total Temp at LPT Outlet (°R)"),
    ("sensor_measurement_7", "Sensor 7: Total Pressure at HPC Outlet (psia)"),
    ("sensor_measurement_11", "Sensor 11: Static Pressure at HPC Outlet (psia)"),
]

fig, axes = plt.subplots(len(target_sensors), 1, figsize=(12, 13), sharex=False)
colors = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed"]

for idx, (sensor_col, sensor_label) in enumerate(target_sensors):
    ax = axes[idx]
    for u_idx, unit_id in enumerate(sample_units):
        unit_data = train_df[train_df["unit_number"] == unit_id]
        ax.plot(
            unit_data["time_in_cycles"],
            unit_data[sensor_col],
            label=f"Unit {unit_id} ({len(unit_data)} cycles)",
            color=colors[u_idx],
            alpha=0.85,
            linewidth=1.3,
        )
        last_point = unit_data.iloc[-1]
        ax.scatter(last_point["time_in_cycles"], last_point[sensor_col], color=colors[u_idx], s=35, marker="x", zorder=5)
        
    ax.set_title(sensor_label, fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel("Reading", fontsize=9.5)
    if idx == 0:
        ax.legend(loc="upper left", ncol=5, fontsize=8.5, frameon=True)
    if idx == len(target_sensors) - 1:
        ax.set_xlabel("Time in Cycles (Flight Cycles)", fontsize=11)

plt.suptitle("Sensor Degradation Trajectories Approaching Failure (5 Sample Engines)", fontsize=13, fontweight="bold", y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_sensor_degradation_trends.png", dpi=300, bbox_inches="tight")
plt.show()
"""))

    # Cell 5: Output C
    cells.append(nbf.v4.new_markdown_cell("""## 4. Output C: Correlation Heatmap
Computing Pearson correlation across all 21 sensors, 3 operational settings, and RUL."""))

    cells.append(nbf.v4.new_code_cell("""corr_features = SETTING_COLUMNS + SENSOR_COLUMNS + ["RUL"]
corr_matrix = train_df[corr_features].corr(method="pearson")

plt.figure(figsize=(15, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap=cmap,
    vmax=1.0,
    vmin=-1.0,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.75, "label": "Pearson Correlation"},
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7.5},
)
plt.title("Correlation Matrix: Operational Settings, 21 Sensors & RUL (FD001)", fontsize=13, fontweight="bold", pad=12)
plt.xticks(rotation=45, ha="right", fontsize=8.5)
plt.yticks(rotation=0, fontsize=8.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
"""))

    # Cell 6: Output D
    cells.append(nbf.v4.new_markdown_cell("""## 5. Output D: Sensor Variance Analysis (Feature Screening)
Quantifying feature variability to systematically detect constant/low-information sensors."""))

    cells.append(nbf.v4.new_code_cell("""variances = train_df[SENSOR_COLUMNS].var()
stds = train_df[SENSOR_COLUMNS].std()

var_df = pd.DataFrame({
    "sensor": SENSOR_COLUMNS,
    "variance": variances.values,
    "std": stds.values,
})
var_df["status"] = np.where(var_df["variance"] < 1e-3, "Zero/Near-Zero Variance", "Informative")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
palette = {"Zero/Near-Zero Variance": "#ef4444", "Informative": "#2563eb"}

sns.barplot(data=var_df, x="sensor", y="variance", hue="status", palette=palette, dodge=False, ax=ax1)
ax1.set_xticklabels([s.replace("sensor_measurement_", "s_") for s in SENSOR_COLUMNS], rotation=45, ha="right")
ax1.set_yscale("log")
ax1.set_title("Log10 Variance per Sensor Column", fontsize=12, fontweight="bold")
ax1.axhline(1e-3, color="#dc2626", linestyle="--", linewidth=1.2, label="Low-Info Threshold (1e-3)")
ax1.legend(loc="upper right", fontsize=8.5)

sns.barplot(data=var_df, x="sensor", y="std", hue="status", palette=palette, dodge=False, ax=ax2)
ax2.set_xticklabels([s.replace("sensor_measurement_", "s_") for s in SENSOR_COLUMNS], rotation=45, ha="right")
ax2.set_title("Standard Deviation per Sensor Column", fontsize=12, fontweight="bold")
ax2.legend(loc="upper right", fontsize=8.5)

plt.suptitle("Sensor Variance & Information Content Analysis", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_sensor_variance_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

print("Near-zero / Constant Variance Sensors (Drop Candidates for Phase 2):")
print(var_df[var_df["status"] == "Zero/Near-Zero Variance"][["sensor", "variance", "std"]])
"""))

    # Cell 7: Output E
    cells.append(nbf.v4.new_markdown_cell("""## 6. Output E: Remaining Useful Life (RUL) Distribution
Distribution of true RUL labels across all 20,631 observations in the training set."""))

    cells.append(nbf.v4.new_code_cell("""fig, (ax_box, ax_hist) = plt.subplots(
    2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [0.2, 0.8]}
)

sns.boxplot(x=train_df["RUL"], ax=ax_box, color="#10b981", fliersize=3)
ax_box.set(xlabel="")
ax_box.set_title("True Remaining Useful Life (RUL) Distribution (Training Set)", fontsize=13, fontweight="bold")

sns.histplot(train_df["RUL"], kde=True, ax=ax_hist, color="#059669", bins=35, edgecolor="white", alpha=0.7)
ax_hist.axvline(train_df["RUL"].mean(), color="#ef4444", linestyle="--", linewidth=2, label=f"Mean: {train_df['RUL'].mean():.1f}")
ax_hist.axvline(train_df["RUL"].median(), color="#1d4ed8", linestyle=":", linewidth=2, label=f"Median: {train_df['RUL'].median():.1f}")
ax_hist.axvline(train_df["RUL"].max(), color="#8b5cf6", linestyle="-.", linewidth=1.5, label=f"Max: {train_df['RUL'].max()}")

ax_hist.set_xlabel("Remaining Useful Life (RUL in Cycles)", fontsize=11)
ax_hist.set_ylabel("Instance Count", fontsize=11)
ax_hist.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_rul_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

train_df["RUL"].describe()
"""))

    # Cell 8: Output F
    cells.append(nbf.v4.new_markdown_cell("""## 7. Output F: Early-Life vs. Late-Life Sensor Behavior
Comparing average sensor readings during healthy initial operation (first 10% of lifespan) versus late degraded operation (last 10% of lifespan)."""))

    cells.append(nbf.v4.new_code_cell("""early_df = train_df[train_df["life_ratio"] <= 0.10]
late_df = train_df[train_df["life_ratio"] >= 0.90]

early_means = early_df[SENSOR_COLUMNS].mean()
late_means = late_df[SENSOR_COLUMNS].mean()
overall_stds = train_df[SENSOR_COLUMNS].std()

comp_df = pd.DataFrame({
    "sensor": SENSOR_COLUMNS,
    "early_mean": early_means.values,
    "late_mean": late_means.values,
    "absolute_shift": (late_means - early_means).values,
    "pct_change": (((late_means - early_means) / early_means) * 100).values,
    "z_shift": ((late_means - early_means) / np.clip(overall_stds, 1e-9, None)).values,
})

info_sensors = var_df[var_df["status"] == "Informative"]["sensor"].tolist()
info_comp_df = comp_df[comp_df["sensor"].isin(info_sensors)].sort_values(by="z_shift")

fig, ax = plt.subplots(figsize=(11, 6.5))
bar_colors = np.where(info_comp_df["z_shift"] > 0, "#ef4444", "#3b82f6")
y_pos = np.arange(len(info_comp_df))
labels = [s.replace("sensor_measurement_", "Sensor ") for s in info_comp_df["sensor"]]

bars = ax.barh(y_pos, info_comp_df["z_shift"], color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9.5)
ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
ax.set_xlabel("Standardized Shift (Δmean / σ)", fontsize=10.5)
ax.set_title("Early-Life (≤10%) vs Late-Life (≥90%) Sensor Degradation Shifts", fontsize=12.5, fontweight="bold", pad=12)

for bar, pct in zip(bars, info_comp_df["pct_change"]):
    width = bar.get_width()
    ha = "left" if width >= 0 else "right"
    offset = 0.05 if width >= 0 else -0.05
    ax.text(width + offset, bar.get_y() + bar.get_height() / 2, f"{pct:+.2f}%", va="center", ha=ha, fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_early_vs_late_life_sensors.png", dpi=300, bbox_inches="tight")
plt.show()

comp_df.sort_values(by="pct_change", key=abs, ascending=False)
"""))

    # Cell 9: Summary & Next Steps
    cells.append(nbf.v4.new_markdown_cell("""## 8. Summary of Findings & Next Steps (Phase 2 Prep)

### Low-Information Features (Candidates to Drop in Phase 2):
* **Constant Sensors (Zero Variance):** `sensor_measurement_1`, `sensor_measurement_5`, `sensor_measurement_10`, `sensor_measurement_16`, `sensor_measurement_18`, `sensor_measurement_19`.
* **Near-Constant Sensors:** `sensor_measurement_6`.
* **Operational Settings:** `operational_setting_3` (constant 100.0), and `operational_setting_1`, `operational_setting_2` (trivial variance under single operating condition).

### Strongest RUL Predictors (Highest Degradation Correlation):
1. **Sensors with Strong Negative Correlation with RUL (Values Increase towards Failure):**
   * `sensor_measurement_11` (corr: -0.70, z-shift: +2.95σ) — Static Pressure at HPC Outlet
   * `sensor_measurement_4` (corr: -0.68, z-shift: +2.86σ) — Total Temp at LPT Outlet
   * `sensor_measurement_15` (corr: -0.64, z-shift: +2.66σ) — Bypass Duct Pressure
   * `sensor_measurement_2` (corr: -0.61, z-shift: +2.57σ) — Total Temp at LPC Outlet
   * `sensor_measurement_17` (corr: -0.61, z-shift: +2.58σ) — Bleed Enthalpy
   * `sensor_measurement_3` (corr: -0.58, z-shift: +2.57σ) — Total Temp at HPC Outlet
   * `sensor_measurement_8` (corr: -0.56, z-shift: +1.95σ) — Physical Fan Speed
   * `sensor_measurement_13` (corr: -0.56, z-shift: +1.97σ) — Corrected Fan Speed

2. **Sensors with Strong Positive Correlation with RUL (Values Decrease towards Failure):**
   * `sensor_measurement_12` (corr: +0.67, z-shift: -2.70σ) — Engine Pressure Ratio
   * `sensor_measurement_7` (corr: +0.66, z-shift: -2.64σ) — Total Pressure at HPC Outlet
   * `sensor_measurement_21` (corr: +0.64, z-shift: -2.64σ) — Fuel Flow Ratio
   * `sensor_measurement_20` (corr: +0.63, z-shift: -2.63σ) — HPT Coolant Bleed
"""))

    nb.cells = cells
    notebook_path = PROJECT_ROOT / "notebooks" / "01_eda.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Jupyter Notebook written to: {notebook_path}")


if __name__ == "__main__":
    main()
