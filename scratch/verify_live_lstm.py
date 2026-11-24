"""
scratch/verify_live_lstm.py

Step 2 Verification: Recompute test MAE, RMSE, and NASA score (capped and raw)
across the 100 test engines of FD001 using the live PrognosticAgent path,
and compare verbatim against stored values.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_cmapss
from src.agents import PrognosticAgent
from src.train import compute_mae, compute_rmse, compute_nasa_score

STORED_METRICS = {
    "Capped (RUL <= 125)": {
        "MAE": 10.2004,
        "RMSE": 13.8984,
        "NASA": 307.87,
    },
    "Raw True RUL": {
        "MAE": 11.2704,
        "RMSE": 15.3818,
        "NASA": 364.81,
    },
}

def main():
    agent = PrognosticAgent.from_checkpoint()

    _, test_df, y_test_df = load_cmapss(subset="FD001", data_dir=PROJECT_ROOT / "data" / "raw")
    y_test_raw = y_test_df["RUL"].values.astype(np.float32)
    y_test_capped = np.minimum(y_test_raw, 125.0)

    # Run live inference for each of the 100 test engines
    pred_ruls = []
    lower_ruls = []
    upper_ruls = []

    unique_units = np.sort(test_df["unit_number"].unique())
    for unit_id in unique_units:
        engine_df = test_df[test_df["unit_number"] == unit_id]
        p, low, up = agent.predict(engine_df, method="binned_residual")
        pred_ruls.append(p)
        lower_ruls.append(low)
        upper_ruls.append(up)

    preds = np.array(pred_ruls, dtype=np.float32)

    # Compute metrics
    live_mae_capped = compute_mae(y_test_capped, preds)
    live_rmse_capped = compute_rmse(y_test_capped, preds)
    live_nasa_capped = compute_nasa_score(y_test_capped, preds)

    live_mae_raw = compute_mae(y_test_raw, preds)
    live_rmse_raw = compute_rmse(y_test_raw, preds)
    live_nasa_raw = compute_nasa_score(y_test_raw, preds)

    print("=" * 85)
    print("PHASE 9: LIVE LSTM PROGNOSTIC VERIFICATION ON FD001 TEST ENGINES (100 UNITS)")
    print("=" * 85)
    print(f"{'Target':22s} | {'Metric':10s} | {'Live Path':12s} | {'Stored Value':14s} | {'Delta':10s} | {'Status':8s}")
    print("-" * 85)

    comparisons = [
        ("Capped (RUL <= 125)", "MAE", live_mae_capped, STORED_METRICS["Capped (RUL <= 125)"]["MAE"]),
        ("Capped (RUL <= 125)", "RMSE", live_rmse_capped, STORED_METRICS["Capped (RUL <= 125)"]["RMSE"]),
        ("Capped (RUL <= 125)", "NASA Score", live_nasa_capped, STORED_METRICS["Capped (RUL <= 125)"]["NASA"]),
        ("Raw True RUL", "MAE", live_mae_raw, STORED_METRICS["Raw True RUL"]["MAE"]),
        ("Raw True RUL", "RMSE", live_rmse_raw, STORED_METRICS["Raw True RUL"]["RMSE"]),
        ("Raw True RUL", "NASA Score", live_nasa_raw, STORED_METRICS["Raw True RUL"]["NASA"]),
    ]

    all_matched = True
    for target, metric, live_val, stored_val in comparisons:
        delta = abs(live_val - stored_val)
        matched = delta < 0.01
        if not matched:
            all_matched = False
        status_str = "MATCH" if matched else "MISMATCH"
        print(f"{target:22s} | {metric:10s} | {live_val:12.4f} | {stored_val:14.4f} | {delta:10.4f} | {status_str:8s}")

    print("-" * 85)
    if all_matched:
        print("[SUCCESS] All live LSTM prognostic metrics match stored values exactly!")
    else:
        print("[ERROR] Discrepancies detected between live path and stored values.")

if __name__ == "__main__":
    main()
