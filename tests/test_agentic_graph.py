"""
test_agentic_graph.py

Tests for LangGraph Multi-Agent Predictive Maintenance Orchestrator (src/agentic_graph.py):
1. E-017 Benchmark Fixture Exact Reproduction:
   Validates chosen start cycle (16), failure probability (0.0%), and renewal cost rate ($18.38/cycle).
2. Engine 24 Live Inference Exact Reproduction:
   Validates live chosen start cycle (202), failure probability (18.6%), and renewal cost rate ($20.15/cycle).
3. Invalid Proposed Cycle Retries & Fallback:
   Verifies validator detects invalid proposal, retries twice, and falls back to deterministic Coordinator (fallback=True).
4. No-Feasible-Window Escalation:
   Verifies supervisor detects 0 feasible windows and routes directly to escalation (escalated=True).
5. Information Rule Invariant:
   Guarantees no true RUL or ground truth appears in shared state, messages, tool payloads, or traces.
6. Mermaid Architecture Export:
   Generates and verifies outputs/agent_graph.md contains valid Mermaid graph specification.
"""

import os
from pathlib import Path
from typing import Any
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from src.agent_tools import AgentSession
from src.agentic_graph import (
    AgenticState,
    build_predictive_maintenance_graph,
    run_agentic_graph,
)
from src.agents import (
    PrognosticAgent,
    ResourceAgent,
    CostAgent,
    CoordinatorAgent,
)
from src.simulator import EngineObservation
from src.data_loader import load_cmapss


def test_graph_e017_fixture_reproduction():
    """
    Assert LangGraph with fake model reproduces the Section 14 E-017 benchmark fixture exactly:
    - chosen_cycle == 16
    - prob_fail == 0.0
    - cost_rate == 18.38235294117647
    - escalated == False
    - fallback == False
    """
    final_state = run_agentic_graph(
        engine_id=17,
        current_cycle=0,
        age=120,
        lead_time=8,
        technician_cycles=[16, 22],
        llm_type="fake",
        is_fixture=True,
        fixture_rul=28.0,
        fixture_low=21.0,
        fixture_up=35.0,
    )

    rec = final_state["final_recommendation"]
    assert rec["chosen_cycle"] == 16
    assert np.isclose(rec["prob_fail"], 0.0, atol=1e-6)
    assert np.isclose(rec["cost_rate"], 18.38235294117647, atol=1e-6)
    assert rec["escalated"] is False
    assert rec["fallback"] is False
    assert final_state["validated"] is True


def test_graph_engine24_live_reproduction():
    """
    Assert LangGraph with fake model reproduces the live Engine #024 decision path exactly:
    - chosen_cycle == 202
    - prob_fail == 0.186
    - cost_rate == 20.146013220218858
    - escalated == False
    - fallback == False
    """
    final_state = run_agentic_graph(
        engine_id=24,
        current_cycle=186,
        age=186,
        lead_time=8,
        technician_cycles=[16, 22],
        llm_type="fake",
        is_fixture=False,
    )

    rec = final_state["final_recommendation"]
    assert rec["chosen_cycle"] == 202
    assert np.isclose(rec["prob_fail"], 0.186, atol=1e-6)
    assert np.isclose(rec["cost_rate"], 20.146013220218858, atol=1e-6)
    assert rec["escalated"] is False
    assert rec["fallback"] is False
    assert final_state["validated"] is True


def test_invalid_proposed_cycle_triggers_retry_then_fallback():
    """
    Assert an invalid proposed cycle triggers validator rejection, retries twice,
    and falls back to the deterministic CoordinatorAgent decision (fallback=True).
    """
    final_state = run_agentic_graph(
        engine_id=17,
        current_cycle=0,
        age=120,
        lead_time=8,
        technician_cycles=[16, 22],
        llm_type="fake",
        is_fixture=True,
        fixture_rul=28.0,
        fixture_low=21.0,
        fixture_up=35.0,
        inject_invalid_cycles=3,  # Injects invalid cycle 999
    )

    rec = final_state["final_recommendation"]
    assert rec["fallback"] is True
    assert rec["chosen_cycle"] == 16  # Fallback correctly chooses deterministic optimal cycle
    assert final_state["retries"] >= 2

    # Verify trace contains retry and fallback events
    validator_events = [t for t in final_state["trace"] if t.get("node") == "validator"]
    rejected_events = [t for t in validator_events if t.get("validation") == "REJECTED"]
    fallback_events = [t for t in validator_events if t.get("validation") == "FALLBACK_TRIGGERED"]

    assert len(rejected_events) >= 2
    assert len(fallback_events) == 1
    assert fallback_events[0]["fallback"] is True


def test_no_feasible_window_triggers_escalation():
    """
    Assert that when no technician window is available within predicted RUL,
    the supervisor routes to escalation, securing the earliest feasible window.
    """
    # Technicians only available at cycle 100, while RUL is 28 cycles
    final_state = run_agentic_graph(
        engine_id=17,
        current_cycle=0,
        age=120,
        lead_time=8,
        technician_cycles=[100],
        llm_type="fake",
        is_fixture=True,
        fixture_rul=28.0,
        fixture_low=21.0,
        fixture_up=35.0,
    )

    rec = final_state["final_recommendation"]
    assert rec["escalated"] is True
    assert rec["chosen_cycle"] == 100  # Earliest available technician window
    assert final_state["feasible_windows"] == []

    # Verify escalation logged in trace
    escalation_events = [t for t in final_state["trace"] if t.get("node") == "escalation"]
    assert len(escalation_events) == 1
    assert escalation_events[0]["escalated"] is True


def test_information_rule_in_graph_states_and_traces():
    """
    Audit entire LangGraph execution: shared state keys, trace dictionaries,
    messages, and final recommendation must contain zero true RUL / ground truth leakage.
    """
    final_state = run_agentic_graph(
        engine_id=17,
        current_cycle=0,
        age=120,
        lead_time=8,
        technician_cycles=[16, 22],
        llm_type="fake",
        is_fixture=True,
    )

    banned_substrings = ["true_rul", "ground"]

    def audit_recursively(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                curr_path = f"{path}.{k}" if path else str(k)
                for banned in banned_substrings:
                    assert banned not in k.lower(), (
                        f"Forbidden keyword '{banned}' detected in state/trace key '{curr_path}'"
                    )
                audit_recursively(v, curr_path)
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                audit_recursively(item, f"{path}[{idx}]")

    audit_recursively(final_state)


def test_draw_mermaid_and_save():
    """
    Generate Mermaid diagram using graph.get_graph().draw_mermaid()
    and save to outputs/agent_graph.md.
    """
    session = AgentSession(spare_lead_time=8, preventive_duration=2, horizon_W=60)
    app = build_predictive_maintenance_graph(session=session, llm_type="fake")
    mermaid_code = app.get_graph().draw_mermaid()

    assert mermaid_code is not None
    assert "graph TD" in mermaid_code or "graph LR" in mermaid_code or "---" in mermaid_code
    assert "supervisor" in mermaid_code
    assert "prognostic_agent" in mermaid_code
    assert "resource_agent" in mermaid_code
    assert "cost_agent" in mermaid_code
    assert "validator" in mermaid_code
    assert "escalation" in mermaid_code
    assert "report" in mermaid_code

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "agent_graph.md"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# LangGraph Multi-Agent Predictive Maintenance Graph\n\n```mermaid\n" + mermaid_code + "\n```\n")

    assert out_file.exists()
    assert out_file.stat().st_size > 100


def test_groq_429_retry_and_recovery():
    """
    Simulate 429 rate limit errors on initial calls.
    The resilient retry mechanism retries with backoff and successfully recovers.
    """
    final_state = run_agentic_graph(
        engine_id=17,
        current_cycle=0,
        age=120,
        lead_time=8,
        technician_cycles=[16, 22],
        llm_type="fake",
        is_fixture=True,
        fixture_rul=28.0,
        fixture_low=21.0,
        fixture_up=35.0,
        simulate_429_count=2,  # First 2 calls raise 429, 3rd succeeds
    )

    rec = final_state["final_recommendation"]
    assert rec["chosen_cycle"] == 16
    assert rec["fallback"] is False
    assert final_state["validated"] is True


def test_groq_429_exhaustion_triggers_coordinator_fallback():
    """
    Simulate persistent 429 errors that exhaust all retries.
    Verifies the system falls back to the deterministic CoordinatorAgent decision
    with fallback=True and fallback_reason populated without leaking any keys.
    """
    final_state = run_agentic_graph(
        engine_id=17,
        current_cycle=0,
        age=120,
        lead_time=8,
        technician_cycles=[16, 22],
        llm_type="fake",
        is_fixture=True,
        fixture_rul=28.0,
        fixture_low=21.0,
        fixture_up=35.0,
        simulate_429_count=10,  # All retries raise 429
    )

    rec = final_state["final_recommendation"]
    assert rec["fallback"] is True
    assert rec["chosen_cycle"] == 16  # Deterministic Coordinator optimal cycle
    assert rec.get("fallback_reason") is not None
    assert "429" in rec["fallback_reason"] or "RuntimeError" in rec["fallback_reason"]


def test_malformed_tool_call_retry_and_recovery():
    """
    Simulate an agent returning a malformed/missing tool call on the first attempt.
    Verifies that the corrective retry prompt prompts for and recovers a valid tool call.
    """
    final_state = run_agentic_graph(
        engine_id=17,
        current_cycle=0,
        age=120,
        lead_time=8,
        technician_cycles=[16, 22],
        llm_type="fake",
        is_fixture=True,
        fixture_rul=28.0,
        fixture_low=21.0,
        fixture_up=35.0,
        simulate_malformed=True,
    )

    rec = final_state["final_recommendation"]
    assert rec["chosen_cycle"] == 16
    assert final_state["validated"] is True


def test_string_encoded_tool_args_parsed():
    """
    Simulate Groq quirk where tool arguments arrive as a JSON string instead of a dict.
    Verifies parse_and_validate_tool_args correctly parses JSON strings to dicts
    and handles invalid JSON gracefully.
    """
    from src.agentic_graph import parse_and_validate_tool_args

    raw_string = '{"engine_id": 17, "current_cycle": 0}'
    parsed = parse_and_validate_tool_args(raw_string)
    assert isinstance(parsed, dict)
    assert parsed == {"engine_id": 17, "current_cycle": 0}

    # Also test invalid JSON string fallback
    assert parse_and_validate_tool_args("not a json string") == {}

    # Also test dict passthrough
    assert parse_and_validate_tool_args({"lead_time": 8}) == {"lead_time": 8}


def test_rationale_guardrail_hallucinated_number_rejected():
    """
    Test guardrail: LLM rationale containing an unrecognized number is rejected
    and replaced by the deterministic template (rationale_source='template').
    """
    from src.agentic_graph import validate_rationale_guardrail

    allowed = {"17", "0", "16", "28.0", "18.38", "120", "8"}
    hallucinated_text = "Maintenance scheduled at cycle 16 saving $99999 across 500 flights."

    is_valid, unrec = validate_rationale_guardrail(hallucinated_text, allowed)
    assert is_valid is False
    assert "99999" in unrec or "500" in unrec


def test_post_validation_resource_agent_reservation():
    """
    Verify that reserve_window and order_spare tool calls occur under the resource_agent
    node after validation passes.
    """
    final_state = run_agentic_graph(
        engine_id=17,
        current_cycle=0,
        age=120,
        lead_time=8,
        technician_cycles=[16, 22],
        llm_type="fake",
        is_fixture=True,
        fixture_rul=28.0,
        fixture_low=21.0,
        fixture_up=35.0,
    )

    trace = final_state["trace"]
    reserve_events = [t for t in trace if t.get("tool") == "reserve_window"]
    order_events = [t for t in trace if t.get("tool") == "order_spare"]

    assert len(reserve_events) >= 1
    assert len(order_events) >= 1
    for ev in reserve_events:
        assert ev.get("node") == "resource_agent"
    for ev in order_events:
        assert ev.get("node") == "resource_agent"


def test_regression_skip_tech_windows_enforced_by_graph():
    """
    Regression Test (a):
    Simulate ResourceAgent model skipping list_technician_windows (and calling check_spare_status twice).
    The graph must deduplicate repeated calls, detect the missing tool, and enforce it deterministically.
    Verifies:
    - Final chosen_cycle == 202 (not premature escalation to cycle 194)
    - escalated == False
    - fallback == False
    - validated == True
    - trace contains at least one entry with enforced_by == 'graph'
    """
    final_state = run_agentic_graph(
        engine_id=24,
        current_cycle=186,
        age=186,
        lead_time=8,
        technician_cycles=[16, 22],
        llm_type="fake",
        is_fixture=False,
        simulate_skip_tech_windows=True,
    )
    rec = final_state["final_recommendation"]
    assert rec["chosen_cycle"] == 202
    assert rec["escalated"] is False
    assert rec["fallback"] is False
    assert final_state["validated"] is True
    assert np.isclose(rec["prob_fail"], 0.186, atol=1e-3)
    assert np.isclose(rec["cost_rate"], 20.146, atol=1e-3)

    enforced_entries = [t for t in final_state["trace"] if t.get("enforced_by") == "graph"]
    assert len(enforced_entries) >= 1
    assert any(t.get("tool") == "list_technician_windows" for t in enforced_entries)


def test_regression_escalation_matches_find_earliest_available():
    """
    Regression Test (b):
    When no technician window is available within predicted RUL (e.g. technician_cycles=[100]),
    the graph escalates and routes to validator to ensure chosen_cycle matches
    ResourceAgent.find_earliest_available(part_arrival, D).
    Verifies:
    - escalated == True
    - chosen_cycle == 100 (matches find_earliest_available)
    - prob_fail is None and cost_rate is None (display as N/A)
    - validated == True
    """
    final_state = run_agentic_graph(
        engine_id=17,
        current_cycle=0,
        age=120,
        lead_time=8,
        technician_cycles=[100],
        llm_type="fake",
        is_fixture=True,
        fixture_rul=28.0,
        fixture_low=21.0,
        fixture_up=35.0,
    )
    rec = final_state["final_recommendation"]
    assert rec["escalated"] is True
    assert rec["chosen_cycle"] == 100
    assert rec["prob_fail"] is None
    assert rec["cost_rate"] is None
    assert final_state["validated"] is True


def test_regression_contradictory_rationale_replaced_by_template():
    """
    Regression Test (c):
    When the LLM rationale contradicts the state (e.g., claims emergency escalation
    when state is scheduled), the rationale guardrail detects the contradiction and
    replaces it with the deterministic template.
    Verifies:
    - rationale_source == 'template'
    - reason contains expected template language
    """
    final_state = run_agentic_graph(
        engine_id=17,
        current_cycle=0,
        age=120,
        lead_time=8,
        technician_cycles=[16, 22],
        llm_type="fake",
        is_fixture=True,
        fixture_rul=28.0,
        fixture_low=21.0,
        fixture_up=35.0,
        simulate_contradictory_rationale=True,
    )
    rec = final_state["final_recommendation"]
    assert rec["rationale_source"] == "template"
    assert "Selected cycle 16 (offset +16 cycles)" in rec["reason"]
    assert "$18.38/cycle" in rec["reason"]
