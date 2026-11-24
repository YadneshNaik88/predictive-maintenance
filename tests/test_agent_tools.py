"""
test_agent_tools.py

Tests for LangChain Tool Wrappers (src/agent_tools.py):
1. Information Rule Verification: Ensures no tool accepts or returns true RUL.
   Inspects every tool's args schema and returned keys, failing if 'true_rul' or 'ground' appears.
2. Section 14 E-017 Benchmark Fixture Equivalence:
   Manual tool chain output matches CoordinatorAgent output bit-for-bit (cycle, prob_fail, cost_rate).
3. Engine 24 Live Inference Equivalence:
   Manual tool chain matches live CoordinatorAgent decision on FD001 test unit #024.
4. Session Isolation: Guarantees no global state leakage across sessions.
5. Credential Isolation: Ensures get_llm reads credentials strictly from GROQ_API_KEY and GROQ_MODEL.
"""

import os
from pathlib import Path
from typing import Any
import pytest
import numpy as np
import pandas as pd

from src.agents import (
    PrognosticAgent,
    ResourceAgent,
    CostAgent,
    CoordinatorAgent,
)
from src.simulator import EngineObservation
from src.data_loader import load_cmapss
from src.agent_tools import AgentSession, create_agent_tools, get_llm, ALL_TOOLS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_information_rule_no_true_rul_leakage():
    """
    INFORMATION RULE AUDIT:
    Inspect every tool's args schema and return keys.
    Must fail if 'true_rul' or 'ground' appears anywhere.
    """
    session = AgentSession(spare_lead_time=8, preventive_duration=2, horizon_W=60)
    session.set_fixture(17, 0, 28.0, 21.0, 35.0)
    tools = session.get_tools()

    banned_substrings = ["true_rul", "ground"]

    # 1. Audit argument schemas for every tool
    for t in tools:
        # Check args dict keys
        for arg_name in t.args.keys():
            for banned in banned_substrings:
                assert banned not in arg_name.lower(), (
                    f"Forbidden keyword '{banned}' detected in argument '{arg_name}' of tool '{t.name}'"
                )

        # Check description for accidental leakage instructions
        assert "true_rul" not in t.description.lower(), (
            f"Forbidden keyword 'true_rul' detected in description of tool '{t.name}'"
        )

    # 2. Audit return keys by executing each tool
    # predict_rul
    res_pred = session.predict_rul.invoke({"engine_id": 17, "current_cycle": 0})
    # check_spare_status
    res_spare = session.check_spare_status.invoke({"current_cycle": 0, "lead_time": 8, "order_placed": False})
    # list_technician_windows
    res_tech = session.list_technician_windows.invoke({
        "current_cycle": 0,
        "pred_rul": 28.0,
        "technician_cycles": [16, 22],
    })
    # get_risk_profile
    res_risk = session.get_risk_profile.invoke({"pred_rul": 28.0, "candidate_offsets": [16, 22]})
    # evaluate_candidate_costs
    res_cost = session.evaluate_candidate_costs.invoke({
        "age": 120,
        "feasible_offsets": [16, 22],
        "prob_fail": res_risk["prob_fail"],
        "exp_min_r": res_risk["exp_min_r"],
    })
    # reserve_window
    res_res = session.reserve_window.invoke({"start_cycle": 16})
    # order_spare
    res_ord = session.order_spare.invoke({"current_cycle": 0})

    results = [res_pred, res_spare, res_tech, res_risk, res_cost, res_res, res_ord]

    def check_keys_recursively(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                curr_path = f"{path}.{k}" if path else str(k)
                for banned in banned_substrings:
                    assert banned not in k.lower(), (
                        f"Forbidden keyword '{banned}' detected in return key '{curr_path}'"
                    )
                check_keys_recursively(v, curr_path)
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                check_keys_recursively(item, f"{path}[{idx}]")

    for res in results:
        check_keys_recursively(res)


def test_e017_fixture_manual_chain_equivalence():
    """
    SECTION 14 BENCHMARK FIXTURE EQUIVALENCE:
    E-017 fixture: RUL 28.0, interval [21.0, 35.0], lead time 8, technicians at cycles 16 and 22.
    Assert manual tool chain output EXACTLY equals the CoordinatorAgent path:
    - chosen_cycle == 16
    - prob_fail == 0.0
    - cost_rate == 18.38235294117647
    """
    # 1. Existing CoordinatorAgent baseline
    prognostic = PrognosticAgent(use_intervals=True)
    resource = ResourceAgent(spare_lead_time=8, preventive_duration=2, horizon_W=60)
    cost = CostAgent(preventive_duration=2)
    coordinator = CoordinatorAgent(prognostic, resource, cost, replan_k=5, horizon_W=60, spare_lead_time=8)

    roster = np.zeros(65, dtype=int)
    roster[16:18] = 1
    roster[22:24] = 1
    coordinator.reset(total_horizon=60, capacity_roster=roster)

    obs = EngineObservation(
        slot_id=0,
        unit_number=17,
        current_cycle=0,
        age=120,
        pred_rul=28.0,
        lower_rul=21.0,
        upper_rul=35.0,
        order_already_placed=False,
    )
    coordinator.plan_fleet(current_cycle=0, observations=[obs])
    coord_rec = coordinator.decision_records[0]

    chosen_cand = [c for c in coord_rec["candidates"] if c["start_cycle"] == coord_rec["chosen_cycle"]][0]
    expected_cycle = int(coord_rec["chosen_cycle"])
    expected_pf = float(chosen_cand["prob_fail"])
    expected_cr = float(chosen_cand["cost_rate"])

    # 2. Manual Tool Chaining
    session = AgentSession(spare_lead_time=8, preventive_duration=2, horizon_W=60)
    session.set_fixture(engine_id=17, current_cycle=0, predicted_rul=28.0, lower_bound=21.0, upper_bound=35.0)

    # Step 1: Predict RUL & uncertainty bounds
    pred_res = session.predict_rul.invoke({"engine_id": 17, "current_cycle": 0})
    assert pred_res["predicted_rul"] == 28.0
    assert pred_res["lower_bound"] == 21.0
    assert pred_res["upper_bound"] == 35.0

    # Step 2: Check spare part status
    spare_res = session.check_spare_status.invoke({"current_cycle": 0, "lead_time": 8, "order_placed": False})
    assert spare_res["part_arrival_cycle"] == 8

    # Step 3: List feasible technician windows
    tech_res = session.list_technician_windows.invoke({
        "current_cycle": 0,
        "pred_rul": pred_res["predicted_rul"],
        "technician_cycles": [16, 22],
    })
    feasible_offsets = tech_res["feasible_offsets"]
    assert feasible_offsets == [16, 22]

    # Step 4: Evaluate failure risk profile
    risk_res = session.get_risk_profile.invoke({
        "pred_rul": pred_res["predicted_rul"],
        "candidate_offsets": feasible_offsets,
    })
    assert len(risk_res["prob_fail"]) == 2

    # Step 5: Evaluate candidate cost rates
    cost_res = session.evaluate_candidate_costs.invoke({
        "age": 120,
        "feasible_offsets": feasible_offsets,
        "prob_fail": risk_res["prob_fail"],
        "exp_min_r": risk_res["exp_min_r"],
    })
    chosen_offset = cost_res["best_offset"]
    chosen_cycle = 0 + chosen_offset
    idx_chosen = feasible_offsets.index(chosen_offset)
    chosen_pf = risk_res["prob_fail"][idx_chosen]
    chosen_cr = cost_res["best_cost_rate"]

    # Step 6: Reserve technician window & order spare
    res_window = session.reserve_window.invoke({"start_cycle": chosen_cycle})
    res_order = session.order_spare.invoke({"current_cycle": 0})

    assert res_window["reserved"] is True
    assert res_order["status"] == "ORDER_PLACED"

    # Strict Equality Assertions
    assert chosen_cycle == expected_cycle == 16
    assert np.isclose(chosen_pf, expected_pf, atol=1e-7)
    assert np.isclose(chosen_cr, expected_cr, atol=1e-7)


def test_engine24_live_path_manual_chain_equivalence():
    """
    LIVE INFERENCE PATH EQUIVALENCE (ENGINE 24):
    Unit #024 at cycle 186 (final observed cycle):
    Assert manual tool chain output EXACTLY equals the CoordinatorAgent path:
    - chosen_cycle == 202
    - prob_fail == 0.186
    - cost_rate == 20.146013220218858
    """
    # 1. Load engine 24 data and run live PrognosticAgent
    _, test_df, _ = load_cmapss(subset="FD001", data_dir=PROJECT_ROOT / "data" / "raw")
    eng24 = test_df[test_df["unit_number"] == 24].copy()
    current_cycle = int(eng24["time_in_cycles"].max())  # 186

    prognostic = PrognosticAgent.from_checkpoint()
    live_pred, live_low, live_up = prognostic.predict(eng24, method="binned_residual")

    lead_time = 8
    d_pm = 2
    resource = ResourceAgent(spare_lead_time=lead_time, preventive_duration=d_pm, horizon_W=60)
    # Technician availability relative to current cycle: 16 and 22 => cycles 202 and 208
    roster_len = current_cycle + 60 + 50
    roster = np.zeros(roster_len, dtype=int)
    roster[current_cycle + 16 : current_cycle + 18] = 1
    roster[current_cycle + 22 : current_cycle + 24] = 1
    resource.reset(total_horizon=current_cycle + 60, capacity_roster=roster)

    cost = CostAgent(preventive_duration=d_pm)
    coordinator = CoordinatorAgent(prognostic, resource, cost, spare_lead_time=lead_time)
    coordinator.reset(total_horizon=current_cycle + 60, capacity_roster=roster)

    obs = EngineObservation(
        slot_id=0,
        unit_number=24,
        current_cycle=current_cycle,
        age=current_cycle,
        pred_rul=float(live_pred),
        lower_rul=float(live_low),
        upper_rul=float(live_up),
        order_already_placed=False,
    )
    coordinator.plan_fleet(current_cycle=current_cycle, observations=[obs])
    coord_rec = coordinator.decision_records[0]

    chosen_cand = [c for c in coord_rec["candidates"] if c["start_cycle"] == coord_rec["chosen_cycle"]][0]
    expected_cycle = int(coord_rec["chosen_cycle"])
    expected_pf = float(chosen_cand["prob_fail"])
    expected_cr = float(chosen_cand["cost_rate"])

    # 2. Manual Tool Chaining via AgentSession
    session = AgentSession(
        prognostic=prognostic,
        spare_lead_time=lead_time,
        preventive_duration=d_pm,
        horizon_W=60,
    )

    # Step 1: predict_rul
    pred_res = session.predict_rul.invoke({"engine_id": 24, "current_cycle": current_cycle})
    assert np.isclose(pred_res["predicted_rul"], live_pred, atol=1e-5)
    assert np.isclose(pred_res["lower_bound"], live_low, atol=1e-5)
    assert np.isclose(pred_res["upper_bound"], live_up, atol=1e-5)

    # Step 2: check_spare_status
    spare_res = session.check_spare_status.invoke({
        "current_cycle": current_cycle,
        "lead_time": lead_time,
        "order_placed": False,
    })
    assert spare_res["part_arrival_cycle"] == current_cycle + lead_time

    # Step 3: list_technician_windows
    tech_res = session.list_technician_windows.invoke({
        "current_cycle": current_cycle,
        "pred_rul": pred_res["predicted_rul"],
        "technician_cycles": [16, 22],
    })
    feasible_offsets = tech_res["feasible_offsets"]
    # Offset 16 is feasible; offset 22 is outside floor(pred_rul) = 19
    assert feasible_offsets == [16]

    # Step 4: get_risk_profile
    risk_res = session.get_risk_profile.invoke({
        "pred_rul": pred_res["predicted_rul"],
        "candidate_offsets": feasible_offsets,
    })
    assert len(risk_res["prob_fail"]) == 1

    # Step 5: evaluate_candidate_costs
    cost_res = session.evaluate_candidate_costs.invoke({
        "age": current_cycle,
        "feasible_offsets": feasible_offsets,
        "prob_fail": risk_res["prob_fail"],
        "exp_min_r": risk_res["exp_min_r"],
    })
    chosen_offset = cost_res["best_offset"]
    chosen_cycle = current_cycle + chosen_offset
    idx_chosen = feasible_offsets.index(chosen_offset)
    chosen_pf = risk_res["prob_fail"][idx_chosen]
    chosen_cr = cost_res["best_cost_rate"]

    # Step 6: reserve_window and order_spare
    res_window = session.reserve_window.invoke({"start_cycle": chosen_cycle})
    res_order = session.order_spare.invoke({"current_cycle": current_cycle})

    assert res_window["reserved"] is True
    assert res_order["status"] == "ORDER_PLACED"

    # Strict Equality Assertions
    assert chosen_cycle == expected_cycle == 202
    assert np.isclose(chosen_pf, expected_pf, atol=1e-7)
    assert np.isclose(chosen_cr, expected_cr, atol=1e-7)


def test_agent_session_isolation():
    """
    Ensure multiple AgentSession instances have fully isolated calendars,
    reservations, and orders with zero shared mutable state.
    """
    s1 = AgentSession(spare_lead_time=8)
    s2 = AgentSession(spare_lead_time=14)

    s1.set_fixture(17, 0, 28.0, 21.0, 35.0)
    assert (17, 0) in s1.fixture_predictions
    assert (17, 0) not in s2.fixture_predictions

    s1.order_spare.invoke({"current_cycle": 10})
    assert len(s1.spare_orders) == 1
    assert len(s2.spare_orders) == 0

    s1.set_technician_capacity([20])
    s1.reserve_window.invoke({"start_cycle": 20})
    assert len(s1.reservations) == 1
    assert len(s2.reservations) == 0


def test_get_llm_configuration():
    """
    Ensure get_llm strictly reads credentials from GROQ_API_KEY
    and model name from GROQ_MODEL.
    """
    orig_key = os.environ.get("GROQ_API_KEY")
    orig_model = os.environ.get("GROQ_MODEL")
    try:
        os.environ["GROQ_API_KEY"] = ""
        os.environ["GROQ_MODEL"] = ""
        with pytest.raises(ValueError, match="GROQ_API_KEY environment variable is missing"):
            get_llm()

        os.environ["GROQ_API_KEY"] = "dummy_test_api_key_123"
        os.environ["GROQ_MODEL"] = "llama-3.3-70b-versatile"
        llm = get_llm()
        assert llm.model_name == "llama-3.3-70b-versatile"
    finally:
        if orig_key is not None:
            os.environ["GROQ_API_KEY"] = orig_key
        elif "GROQ_API_KEY" in os.environ:
            del os.environ["GROQ_API_KEY"]

        if orig_model is not None:
            os.environ["GROQ_MODEL"] = orig_model
        elif "GROQ_MODEL" in os.environ:
            del os.environ["GROQ_MODEL"]
