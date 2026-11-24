"""
test_dashboard.py - Unit test for Phase 8 Streamlit Operations Dashboard.
Verifies data loading, schema integrity matching outputs/results_final.json,
OOF predictions, decision logs, and information rules.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from app.app import (
    load_results_final,
    load_oof_predictions,
    load_decision_logs,
    load_phase7_results,
    load_sensor_telemetry,
    run_live_decision,
    PROJECT_ROOT,
    PAGES,
    RESULTS_FINAL_PATH,
    OOF_PREDICTIONS_PATH,
    DECISION_LOG_PATH,
    PHASE7_RESULTS_PATH,
    COMPONENT_COLORS,
    POLICY_COLORS,
    E017_FIXTURE,
)


def test_dashboard_integrity():
    """Comprehensive test validating dashboard data sources, schemas, and invariants."""
    # 1. Verify required files exist
    assert RESULTS_FINAL_PATH.exists(), f"Missing {RESULTS_FINAL_PATH}"
    assert OOF_PREDICTIONS_PATH.exists(), f"Missing {OOF_PREDICTIONS_PATH}"
    assert DECISION_LOG_PATH.exists(), f"Missing {DECISION_LOG_PATH}"
    assert PHASE7_RESULTS_PATH.exists(), f"Missing {PHASE7_RESULTS_PATH}"

    # 2. Verify results_final.json schema & exact cost conservation
    data = load_results_final()
    assert isinstance(data, dict), "results_final must be a dictionary"
    assert "metadata" in data and "scenarios" in data

    scenarios = data["scenarios"]
    for s_key in ["scenario_a", "scenario_b", "scenario_c"]:
        assert s_key in scenarios, f"Missing {s_key} in scenarios"
        scen_data = scenarios[s_key]
        assert "policies" in scen_data and "paired_vs_naive" in scen_data

        policies = scen_data["policies"]
        for p in ["Run-to-Failure", "Agent-Point", "Agent-Interval", "Agent-Perfect"]:
            assert p in policies, f"Missing policy {p} in {s_key}"

        for pol_name, pol_data in policies.items():
            cd = pol_data["cost_decomposition"]
            decomp_sum = (
                cd["preventive_overhaul"]
                + cd["spare_parts"]
                + cd["downtime_loss"]
                + cd["failure_penalty"]
            )
            assert np.isclose(decomp_sum, pol_data["mean_cost"], atol=0.01), (
                f"Cost decomposition sum ${decomp_sum} != mean_cost ${pol_data['mean_cost']} "
                f"for policy {pol_name} in {s_key}"
            )

    # 3. Verify OOF predictions dataframe schema
    oof_df = load_oof_predictions()
    assert isinstance(oof_df, pd.DataFrame) and not oof_df.empty
    required_cols = ["unit_number", "cycle", "pred_rul", "lower", "upper", "true_rul_raw"]
    for col in required_cols:
        assert col in oof_df.columns, f"Missing column {col} in oof_predictions"
    assert len(oof_df["unit_number"].unique()) == 100
    assert (oof_df["upper"] >= oof_df["lower"]).all()

    # 4. Verify decision log records and E-017 fixture
    logs = load_decision_logs(max_records=100)
    assert isinstance(logs, list) and len(logs) > 0
    for record in logs[:10]:
        for k in ["engine", "cycle", "predicted_rul", "interval", "chosen_cycle", "reason"]:
            assert k in record
    assert E017_FIXTURE["engine"] == "E-017 (Test Fixture)"
    assert E017_FIXTURE["chosen_cycle"] == 16
    assert len(E017_FIXTURE["candidates"]) == 8

    # 5. Verify Phase 7 sensitivity analysis data
    p7 = load_phase7_results()
    assert isinstance(p7, dict)
    for cond in ["L4_cf10000", "L8_cf10000", "L12_cf10000", "L16_cf10000", "L8_cf20000", "L8_cf40000"]:
        assert cond in p7, f"Missing condition {cond} in phase7_results"

    # 6. Verify sensor telemetry loader
    tel_df = load_sensor_telemetry(unit_number=1)
    assert isinstance(tel_df, pd.DataFrame)
    if not tel_df.empty:
        assert (tel_df["unit_number"] == 1).all()
        assert "time_in_cycles" in tel_df.columns

    # 7. Verify color consistency definitions
    for c in ["Preventive Overhaul", "Spare Parts", "Downtime Loss", "Failure Penalty"]:
        assert c in COMPONENT_COLORS
    for p in ["Run-to-Failure", "Fixed Interval", "Naive Threshold", "Agent-Point", "Agent-Interval", "Diagnostic Oracle", "Agent-Perfect"]:
        assert p in POLICY_COLORS


def test_live_lstm_metrics_match_stored():
    """Verify live PrognosticAgent LSTM inference matches stored benchmark values within 0.05."""
    from src.data_loader import load_cmapss
    from src.agents import PrognosticAgent
    from src.train import compute_mae, compute_rmse, compute_nasa_score

    prognostic = PrognosticAgent.from_checkpoint()
    _, test_df, y_test_df = load_cmapss(subset="FD001", data_dir=PROJECT_ROOT / "data" / "raw")
    y_test_raw = y_test_df["RUL"].values.astype(np.float32)
    y_test_capped = np.minimum(y_test_raw, 125.0)

    unique_units = np.sort(test_df["unit_number"].unique())
    preds = []
    for unit_id in unique_units:
        engine_df = test_df[test_df["unit_number"] == unit_id]
        p, _, _ = prognostic.predict(engine_df, method="binned_residual")
        preds.append(p)
    preds = np.array(preds, dtype=np.float32)

    live_mae_capped = compute_mae(y_test_capped, preds)
    live_rmse_capped = compute_rmse(y_test_capped, preds)
    live_nasa_capped = compute_nasa_score(y_test_capped, preds)

    live_mae_raw = compute_mae(y_test_raw, preds)
    live_rmse_raw = compute_rmse(y_test_raw, preds)
    live_nasa_raw = compute_nasa_score(y_test_raw, preds)

    # Assert within 0.05 of stored benchmark metrics
    assert abs(live_mae_capped - 10.2004) < 0.05, f"MAE capped mismatch: {live_mae_capped} vs 10.20"
    assert abs(live_rmse_capped - 13.8984) < 0.05, f"RMSE capped mismatch: {live_rmse_capped} vs 13.90"
    assert abs(live_nasa_capped - 307.87) < 0.05, f"NASA capped mismatch: {live_nasa_capped} vs 307.87"
    assert abs(live_mae_raw - 11.2704) < 0.05, f"MAE raw mismatch: {live_mae_raw} vs 11.27"
    assert abs(live_rmse_raw - 15.3818) < 0.05, f"RMSE raw mismatch: {live_rmse_raw} vs 15.38"
    assert abs(live_nasa_raw - 364.81) < 0.05, f"NASA raw mismatch: {live_nasa_raw} vs 364.81"


def test_e017_fixture_live_page_equivalence():
    """Verify E-017 benchmark fixture through live-page code path equals the existing fixture."""
    res = run_live_decision(is_fixture=True)
    rec = res["rec"]

    assert "E-017" in rec["engine"]
    assert rec["chosen_cycle"] == E017_FIXTURE["chosen_cycle"]
    assert rec["chosen_cycle"] == 16
    assert rec["escalated"] == E017_FIXTURE["escalated"]
    assert rec["escalated"] is False
    assert res["pred_rul"] == E017_FIXTURE["predicted_rul"]
    assert res["lower_rul"] == E017_FIXTURE["interval"][0]
    assert res["upper_rul"] == E017_FIXTURE["interval"][1]

    # Verify candidate cost rates and feasibility
    cand_map = {c["start_cycle"]: c for c in rec["candidates"]}
    assert cand_map[16]["feasible"] is True
    assert cand_map[22]["feasible"] is True
    assert np.isclose(cand_map[16]["cost_rate"], 18.38, atol=0.05)
    assert np.isclose(cand_map[22]["cost_rate"], 21.89, atol=0.05)
    assert cand_map[17]["feasible"] is False


def test_streamlit_apptest_all_pages():
    """Verify Streamlit AppTest opens every dashboard page with zero exceptions."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py")).run(timeout=30)
    assert len(at.exception) == 0, f"Exception on initial page load: {at.exception}"

    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run(timeout=30)
        assert len(at.exception) == 0, f"Exception on page '{page}': {at.exception}"


def test_page7_early_cycles_engine24_outside_planning_horizon():
    """Verify Page 7 at flight cycles 60 and 122 for engine 24 renders without exception and returns outside-horizon."""
    from streamlit.testing.v1 import AppTest

    # 1. Direct function check
    res_60 = run_live_decision(unit_id=24, current_cycle=60, lead_time=8)
    assert res_60["outside_planning_horizon"] is True
    assert res_60["planning_horizon"] == 68
    assert res_60["pred_rul"] > 68
    assert res_60["rec"] is None
    assert len(res_60["candidates"]) == 0

    res_122 = run_live_decision(unit_id=24, current_cycle=122, lead_time=8)
    assert res_122["outside_planning_horizon"] is True
    assert res_122["planning_horizon"] == 68
    assert res_122["pred_rul"] > 68
    assert res_122["rec"] is None
    assert len(res_122["candidates"]) == 0

    # 2. AppTest check
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py")).run(timeout=30)
    at.sidebar.radio[0].set_value("7. Live Decision Demo").run(timeout=30)
    assert len(at.exception) == 0, f"Exception on Page 7 load: {at.exception}"

    # Engine 24 is default or select explicitly
    at.selectbox[1].set_value(24).run(timeout=30)
    at.slider[0].set_value(60).run(timeout=30)
    assert len(at.exception) == 0, f"Exception on cycle 60: {at.exception}"
    assert any("No maintenance needed yet" in info.value for info in at.info)

    at.slider[0].set_value(122).run(timeout=30)
    assert len(at.exception) == 0, f"Exception on cycle 122: {at.exception}"
    assert any("No maintenance needed yet" in info.value for info in at.info)

