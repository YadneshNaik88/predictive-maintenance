"""
scratch/compare_intervals_engine24.py

Compares old vs new interval and P(R <= 16) both ways for Engine 24 at cycle 186.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import PrognosticAgent
from src.data_loader import load_cmapss
from src.uncertainty import binned_residual_interval

def main():
    agent = PrognosticAgent.from_checkpoint()
    _, test_df, y_test_df = load_cmapss(subset="FD001", data_dir=PROJECT_ROOT / "data" / "raw")
    eng24 = test_df[test_df["unit_number"] == 24]
    max_c = int(eng24["time_in_cycles"].max())
    true_rul_last = float(y_test_df[y_test_df["unit_number"] == 24]["RUL"].values[0])

    # 1. Point prediction and new interval via updated PrognosticAgent.predict
    pred_rul, new_lower, new_upper = agent.predict(eng24, scaled=False)

    # 2. Old interval via 5-bin residual_interval_calibration.json
    old_lower_arr, old_upper_arr = binned_residual_interval(np.array([pred_rul]))
    old_lower = float(old_lower_arr[0])
    old_upper = float(old_upper_arr[0])

    # 3. P(R <= 16) - New way: via agent.evaluate_risk (Tuning pool, 10-cycle bins)
    p_fail_new_arr, _ = agent.evaluate_risk(pred_rul, np.array([16.0]))
    p_fail_new = float(p_fail_new_arr[0])

    # 4. P(R <= 16) - Old way: from 5-bin calibration (residual_interval_calibration.json)
    # In bin 0 [0, 25), q05 = -2.9451, so lower bound = 19.0844 - 2.9451 = 16.1393.
    # For R = pred_rul + e <= 16 => e <= 16 - 19.0844 = -3.0844 cycles.
    # Since -3.0844 < q05 (-2.9451), P(R <= 16) is strictly < 5.0% in that distribution.
    calib_json = json.load(open(PROJECT_ROOT / "outputs" / "residual_interval_calibration.json"))
    bin0 = calib_json["bins"][0]
    mu_res = bin0["mean_residual"]
    std_res = bin0["std_residual"]
    from scipy.stats import norm
    z_score = (16.0 - pred_rul - mu_res) / std_res
    p_fail_old_gaussian = float(norm.cdf(z_score))

    print("=" * 95)
    print("PHASE 9 FIX: INTERVAL & RISK CONSISTENCY AUDIT FOR ENGINE 24 (CYCLE 186)")
    print("=" * 95)
    print(f"Engine Unit Number   : Unit #024 (FD001 Test Fleet)")
    print(f"Current Flight Cycle : Cycle {max_c} (Final observed cycle)")
    print(f"Ground Truth RUL     : {true_rul_last:.1f} cycles [ground truth, evaluation only]")
    print(f"Point Predicted RUL  : {pred_rul:.4f} cycles")
    print("-" * 95)
    print(f"{'Method / Calibration':35s} | {'90% Lower':10s} | {'90% Upper':10s} | {'Width':10s} | {'P(R <= 16)':14s}")
    print("-" * 95)
    print(f"{'Old (5-Bin Calibration JSON)':35s} | {old_lower:10.4f} | {old_upper:10.4f} | {old_upper - old_lower:10.4f} | {p_fail_old_gaussian * 100:10.2f}% (approx)")
    print(f"{'New (Tuning OOF Table, evaluate_risk)':35s} | {new_lower:10.4f} | {new_upper:10.4f} | {new_upper - new_lower:10.4f} | {p_fail_new * 100:10.2f}% (exact)")
    print("-" * 95)
    print(f"Bin Details in Tuning Pool Table:")
    b = agent.get_bin(pred_rul)
    res_b = agent.residuals_by_bin[b]
    print(f"  Bin Index          : Bin {b} (predicted RUL in [{b*10}, {(b+1)*10}) cycles)")
    print(f"  Sample Size in Bin : {len(res_b)} Tuning Pool operating cycles")
    print(f"  5th Percentile Res : {np.percentile(res_b, 5.0):.4f} cycles => Lower Bound = {new_lower:.4f}")
    print(f"  95th Percentile Res: {np.percentile(res_b, 95.0):.4f} cycles => Upper Bound = {new_upper:.4f}")
    print(f"  Samples with R<=16 : {np.sum(np.maximum(0, pred_rul + res_b) <= 16.0)} / {len(res_b)} = {p_fail_new:.4f} ({p_fail_new * 100:.2f}%)")
    print("=" * 95)

if __name__ == "__main__":
    main()
