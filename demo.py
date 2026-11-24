"""
demo.py - End-to-End Live Predictive Maintenance Demonstration Script

Phase 9: Live multi-agent prognostic & scheduling pipeline.
Usage:
    python demo.py --engine 24
    python demo.py --engine 17 --lead-time 8 --technicians 16,22
    python demo.py --brief-example
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv(override=False)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_cmapss
from src.agents import (
    PrognosticAgent,
    ResourceAgent,
    CostAgent,
    CoordinatorAgent,
)
from src.simulator import EngineObservation
from src.agentic_graph import run_agentic_graph

CONFIG_PATH = PROJECT_ROOT / "config" / "sim_params.yaml"


def parse_technician_cycles(tech_str: str, current_cycle: int = 0, horizon_W: int = 60) -> np.ndarray:
    """
    Parse comma-separated technician availability cycles into a capacity roster.
    If cycle numbers are relative offsets (c < current_cycle), maps them to (current_cycle + c).
    """
    total_len = current_cycle + horizon_W + 50
    roster = np.zeros(total_len, dtype=int)
    if not tech_str.strip():
        return roster
    cycles = [int(c.strip()) for c in tech_str.split(",") if c.strip()]
    for c in cycles:
        abs_c = (current_cycle + c) if (c < current_cycle and current_cycle > 0) else c
        # Each job takes D_pm = 2 consecutive cycles
        if abs_c < total_len - 1:
            roster[abs_c : abs_c + 2] = np.maximum(roster[abs_c : abs_c + 2], 1)
        elif abs_c < total_len:
            roster[abs_c] = max(roster[abs_c], 1)
    return roster


def run_agentic_demo_section(
    engine_id: int,
    current_cycle: int,
    age: int,
    lead_time: int,
    technician_cycles: List[int],
    is_fixture: bool,
    rule_rec: Optional[dict],
    llm_mode: str = "fake",
):
    """Execute and display the LangGraph multi-agent orchestrator comparison."""
    import json
    if rule_rec is None or not rule_rec:
        print("\n" + "=" * 90)
        print(f"LANGGRAPH MULTI-AGENT ORCHESTRATION PIPELINE (LLM BACKEND: {llm_mode.upper()})")
        print("=" * 90)
        print("No maintenance needed yet. Engine is operating safely outside the planning horizon.")
        print("=" * 90)
        return
    print("\n" + "=" * 90)
    print(f"LANGGRAPH MULTI-AGENT ORCHESTRATION PIPELINE (LLM BACKEND: {llm_mode.upper()})")
    print("=" * 90)

    # 1. Execute graph
    state = run_agentic_graph(
        engine_id=engine_id,
        current_cycle=current_cycle,
        age=age,
        lead_time=lead_time,
        technician_cycles=technician_cycles,
        llm_type=llm_mode,
        is_fixture=is_fixture,
    )
    rec = state["final_recommendation"]

    # 2. Print Tool-Call Trace
    print("\n[1. Structured Tool-Call Execution Trace]:")
    print(f"{'Step':4s} | {'Agent / Node':18s} | {'Tool Invoked':24s} | {'Key Arguments & Results'}")
    print("-" * 90)
    step = 1
    for t in state.get("trace", []):
        if "tool" in t:
            args_str = json.dumps(t.get("args", {}))
            res_summary = ""
            res = t.get("result", {})
            if isinstance(res, dict):
                if "predicted_rul" in res and "lower_bound" in res:
                    res_summary = f"pred_rul={res['predicted_rul']:.1f}, int=[{res['lower_bound']:.1f}, {res['upper_bound']:.1f}]"
                elif "feasible_offsets" in res:
                    res_summary = f"feasible_offsets={res['feasible_offsets']}, part_arrival={res.get('part_arrival_cycle')}"
                elif "prob_fail" in res:
                    res_summary = f"P(fail)={res['prob_fail']}"
                elif "best_offset" in res:
                    res_summary = f"best_offset=+{res['best_offset']}, best_cost_rate=${res['best_cost_rate']:.2f}"
                elif "reserved" in res:
                    res_summary = f"reserved={res['reserved']}"
                elif "status" in res:
                    res_summary = f"status={res['status']}"
                else:
                    res_summary = str(res)[:50]
            print(f"{step:4d} | {t.get('node', 'N/A'):18s} | {t.get('tool', 'N/A'):24s} | {args_str[:35]} -> {res_summary}")
            step += 1

    # 3. Print Per-Agent Messages
    print("\n[2. Per-Agent Inter-Agent Communications & Prompts]:")
    print("-" * 90)
    for i, msg in enumerate(state.get("messages", [])):
        role = getattr(msg, "type", "message")
        if role == "ai":
            if msg.tool_calls:
                tc_names = [tc["name"] for tc in msg.tool_calls]
                print(f"  [AI Message #{i+1}] Requested Tool Calls: {tc_names}")
            else:
                print(f"  [AI Message #{i+1}] Content: \"{msg.content.strip()}\"")
        elif role == "tool":
            content_preview = str(msg.content)[:100] + ("..." if len(str(msg.content)) > 100 else "")
            print(f"  [Tool Response #{i+1}] {content_preview}")

    # 4. Print Validator Audit & Fallback
    print("\n[3. Deterministic Non-LLM Guardrail Audit]:")
    print("-" * 90)
    validator_traces = [t for t in state.get("trace", []) if t.get("node") == "validator"]
    last_val = validator_traces[-1] if validator_traces else {}
    print(f"  Validation Status   : {last_val.get('validation', 'PASSED')}")
    print(f"  Retry Attempts      : {state.get('retries', 0)} of max 2 allowed")
    print(f"  Fallback Triggered  : {rec.get('fallback', False)}")
    if rec.get("fallback_reason"):
        print(f"  Fallback Reason     : {rec.get('fallback_reason')}")
    print(f"  Rationale Source    : {rec.get('rationale_source', 'N/A')}")
    if last_val.get("reason"):
        print(f"  Validation Notes    : {last_val.get('reason')}")

    # 5. Side-by-Side Comparison
    rule_chosen = rule_rec.get("chosen_cycle")
    graph_chosen = rec.get("chosen_cycle")
    agree = (rule_chosen == graph_chosen)

    rule_cand = next((c for c in rule_rec.get("candidates", []) if c.get("start_cycle") == rule_chosen), {})
    rule_pf = rule_cand.get("prob_fail")
    rule_cr = rule_cand.get("cost_rate")

    print("\n" + "=" * 90)
    print("SIDE-BY-SIDE DECISION COMPARISON: RULE-BASED COORDINATOR vs. LANGGRAPH AGENTS")
    print("=" * 90)
    print(f"{'Decision Metric':26s} | {'Rule-Based Coordinator':28s} | {'LangGraph Multi-Agent':28s}")
    print("-" * 90)
    print(f"{'Engine Identifier':26s} | E-{engine_id:03d}{'':23s} | E-{engine_id:03d}{'':23s}")
    print(f"{'Current Flight Cycle':26s} | Cycle {current_cycle:<22d} | Cycle {current_cycle:<22d}")
    print(f"{'Recommended Start Cycle':26s} | Cycle {rule_chosen:<22d} | Cycle {graph_chosen:<22d}")
    print(f"{'Start Offset s':26s} | +{rule_chosen - current_cycle:<21d} cycles | +{rec['chosen_offset']:<21d} cycles")
    pf_rule_str = f"{rule_pf * 100:.2f}%" if rule_pf is not None else "N/A"
    pf_graph_str = f"{rec['prob_fail'] * 100:.2f}%" if rec.get('prob_fail') is not None else "N/A"
    print(f"{'Failure Risk P(R <= s)':26s} | {pf_rule_str:<28s} | {pf_graph_str:<28s}")
    cr_rule_str = f"${rule_cr:.2f}/cycle" if rule_cr is not None else "N/A"
    cr_graph_str = f"${rec['cost_rate']:.2f}/cycle" if rec.get('cost_rate') is not None else "N/A"
    print(f"{'Expected Cost Rate C(s)/E':26s} | {cr_rule_str:<28s} | {cr_graph_str:<28s}")
    print(f"{'Escalation Flag':26s} | {str(rule_rec.get('escalated')):<28s} | {str(rec.get('escalated')):<28s}")
    print(f"{'Fallback Triggered':26s} | {'N/A (Deterministic Core)':<28s} | {str(rec.get('fallback')):<28s}")
    match_status = "MATCH: EXACT AGREEMENT" if agree else "DISCREPANCY"
    print(f"{'Decision Agreement':26s} | [{match_status}]{'':12s} | [{match_status}]{'':12s}")
    print("-" * 90)
    print("Rule-Based Operational Rationale :")
    print(f"  \"{rule_rec.get('reason', '')}\"")
    print("LangGraph Agent Operational Rationale :")
    print(f"  \"{rec.get('reason', '')}\"")
    print("=" * 90)


def run_e017_brief_fixture(agentic: bool = False, llm_mode: str = "fake"):
    """Execute the brief's Section 14 E-017 test fixture."""
    print("\n" + "=" * 90)
    print("SECTION 14 TEST FIXTURE: ENGINE E-017 BENCHMARK DEMONSTRATION")
    print("=" * 90)
    print("Fixture Inputs: Predicted RUL = 28.0 cycles | 90% Interval = [21.0, 35.0] cycles")
    print("                Engine Age = 120 cycles     | Spare Lead Time = 8 cycles")
    print("                Technician Capacity Available at Cycles: 16 and 22")
    print("-" * 90)

    # 1. Prognostic Agent (empirical residual risk table)
    prognostic = PrognosticAgent(use_intervals=True)

    # 2. Resource Agent
    resource = ResourceAgent(spare_lead_time=8, preventive_duration=2, horizon_W=60)

    # 3. Cost Agent
    cost = CostAgent()

    # 4. Coordinator Agent
    coordinator = CoordinatorAgent(
        prognostic=prognostic,
        resource=resource,
        cost=cost,
        replan_k=5,
        horizon_W=60,
        spare_lead_time=8,
    )

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
    rec = coordinator.decision_records[0]

    # Print Candidate Evaluation Table
    print("\n[Resource & Cost Agents] Candidate Start Windows Evaluation:")
    print(f"{'Cycle':7s} | {'Offset s':10s} | {'Part Ready':12s} | {'Tech Ready':12s} | {'Feasible':10s} | {'P(Fail)':10s} | {'Cost Rate':14s}")
    print("-" * 90)

    display_cycles = [8, 12, 16, 17, 20, 22, 23, 26, 28]
    for c in rec["candidates"]:
        cyc = c["start_cycle"]
        if c["feasible"] or cyc in display_cycles:
            pr_str = "YES" if c["part_ready"] else "NO"
            tr_str = "YES" if c["tech_ready"] else "NO"
            feas_str = "FEASIBLE" if c["feasible"] else "BLOCKED"
            pf_str = f"{c['prob_fail'] * 100:5.2f}%" if c["prob_fail"] is not None else "N/A"
            cr_str = f"${c['cost_rate']:6.2f}/cyc" if c["cost_rate"] is not None else "N/A"
            flag = " <-- CHOSEN OPTIMAL" if cyc == rec["chosen_cycle"] else ""
            print(f"{cyc:7d} | {c['offset_s']:10d} | {pr_str:12s} | {tr_str:12s} | {feas_str:10s} | {pf_str:10s} | {cr_str:14s}{flag}")

    print("-" * 90)
    print(f"\n[Coordinator Decision Outcome]:")
    print(f"  Recommended Maintenance Start Cycle: Cycle {rec['chosen_cycle']} (Offset +{rec['chosen_cycle']} cycles)")
    print(f"  Spare Part Order Dispatched         : Cycle {rec['spare_order_cycle']} (Arrival: Cycle {rec['spare_order_cycle'] + 8})")
    print(f"  Emergency Escalation Required       : {rec['escalated']}")
    print(f"  Plain-English Operational Rationale :")
    print(f"    \"{rec['reason']}\"")

    if agentic:
        run_agentic_demo_section(
            engine_id=17,
            current_cycle=0,
            age=120,
            lead_time=8,
            technician_cycles=[16, 22],
            is_fixture=True,
            rule_rec=rec,
            llm_mode=llm_mode,
        )


def run_live_engine_demo(
    unit_id: int,
    lead_time: int,
    tech_str: str,
    cycle: Optional[int] = None,
    agentic: bool = False,
    llm_mode: str = "fake",
):
    """Execute the live multi-agent prediction and decision pipeline on a test engine."""
    print("=" * 90)
    print(f"PHASE 9: LIVE MULTI-AGENT INFERENCE & DECISION DEMONSTRATION (ENGINE #{unit_id})")
    print("=" * 90)

    # Ingest raw telemetry
    _, test_df, y_test_df = load_cmapss(subset="FD001", data_dir=PROJECT_ROOT / "data" / "raw")
    engine_data = test_df[test_df["unit_number"] == unit_id].copy()

    if len(engine_data) == 0:
        raise ValueError(f"Engine unit {unit_id} not found in FD001 test set (valid range: 1..100).")

    max_obs_cycle = int(engine_data["time_in_cycles"].max())
    if cycle is not None:
        if cycle < 1 or cycle > max_obs_cycle:
            raise ValueError(f"Selected cycle {cycle} out of range [1, {max_obs_cycle}].")
        engine_subset = engine_data[engine_data["time_in_cycles"] <= cycle].copy()
        current_cycle = cycle
    else:
        engine_subset = engine_data
        current_cycle = max_obs_cycle

    # Retrieve ground truth RUL for evaluation line
    true_rul_last = float(y_test_df[y_test_df["unit_number"] == unit_id]["RUL"].values[0])
    # Ground truth remaining life at current cycle
    cycles_remaining_to_last = max_obs_cycle - current_cycle
    true_rul_at_current = true_rul_last + cycles_remaining_to_last

    print(f"Engine Identifier    : Unit #{unit_id:03d} (FD001 Test Fleet)")
    print(f"Current Flight Cycle : Cycle {current_cycle} of {max_obs_cycle} observed")
    print(f"Ground Truth RUL     : {true_rul_at_current:.1f} cycles [ground truth, evaluation only]")
    print(f"Configured Lead Time : L = {lead_time} cycles")
    print(f"Technician Schedule  : Slots at cycles {tech_str}")

    # 1. LIVE PROGNOSTIC AGENT
    print("\n--- 1. PROGNOSTIC AGENT (Live PyTorch LSTM Regressor) ---")
    prognostic = PrognosticAgent.from_checkpoint()
    pred_rul, lower_rul, upper_rul = prognostic.predict(engine_subset, method="binned_residual")
    print(f"  Model Checkpoint   : outputs/best_lstm_model.pt (target_scale = {prognostic.target_scale:.1f})")
    print(f"  Scaler Artifact    : outputs/scaler.pkl (14 retained sensors normalized)")
    print(f"  Live Predicted RUL : {pred_rul:6.2f} cycles")
    print(f"  90% Pred Interval  : [{lower_rul:6.2f}, {upper_rul:6.2f}] cycles (width: {upper_rul - lower_rul:5.2f} cycles)")

    # 2. RESOURCE AGENT
    print("\n--- 2. RESOURCE AGENT (Logistics & Roster Feasibility) ---")
    d_pm = 2
    resource = ResourceAgent(spare_lead_time=lead_time, preventive_duration=d_pm, horizon_W=60)
    roster = parse_technician_cycles(tech_str, current_cycle=current_cycle, horizon_W=60)
    resource.reset(total_horizon=current_cycle + 60, capacity_roster=roster)

    # 3. COST AGENT
    print("--- 3. COST AGENT (Renewal Reward Rate C(s)/E(s)) ---")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)
    costs = cfg_yaml.get("costs", {})
    cost_agent = CostAgent(
        cost_preventive=float(costs.get("preventive_maintenance", 2000.0)),
        cost_corrective=float(costs.get("corrective_failure", 10000.0)),
        cost_production_loss=float(costs.get("production_loss_per_cycle", 100.0)),
        cost_spare=float(costs.get("spare_part", 300.0)),
        preventive_duration=d_pm,
        corrective_duration=int(cfg_yaml.get("operations", {}).get("corrective_duration", 6)),
    )
    print(f"  Cost Parameters    : C_pm=${cost_agent.cost_preventive:,.0f}, C_fail=${cost_agent.cost_corrective:,.0f}, "
          f"C_down=${cost_agent.cost_production_loss:,.0f}/cyc, C_spare=${cost_agent.cost_spare:,.0f}")

    # 4. COORDINATOR AGENT
    print("--- 4. COORDINATOR AGENT (Collaborative Decision Making) ---")
    coordinator = CoordinatorAgent(
        prognostic=prognostic,
        resource=resource,
        cost=cost_agent,
        replan_k=5,
        horizon_W=60,
        spare_lead_time=lead_time,
    )
    coordinator.reset(total_horizon=current_cycle + 60, capacity_roster=roster)

    obs = EngineObservation(
        slot_id=0,
        unit_number=unit_id,
        current_cycle=current_cycle,
        age=current_cycle,
        pred_rul=pred_rul,
        lower_rul=lower_rul,
        upper_rul=upper_rul,
        order_already_placed=False,
    )

    coordinator.plan_fleet(current_cycle=current_cycle, observations=[obs])
    planning_horizon = lead_time + 60

    if not coordinator.decision_records:
        print("\n" + "=" * 90)
        print("FLEET COORDINATOR STATUS: OUTSIDE PLANNING HORIZON")
        print("=" * 90)
        print(f"No maintenance needed yet. Predicted RUL ({pred_rul:.1f} cycles) is above the planning")
        print(f"horizon ({planning_horizon} cycles = lead time {lead_time} + window 60).")
        print("The agents start planning when it drops below this.")
        print("-" * 90)
        print(f"  Live Predicted RUL      : {pred_rul:6.2f} cycles")
        print(f"  90% Pred Interval       : [{lower_rul:6.2f}, {upper_rul:6.2f}] cycles (width: {upper_rul - lower_rul:5.2f} cycles)")
        print(f"  Predicted Failure Cycle : Cycle {current_cycle + pred_rul:.1f}")
        print("=" * 90)
        if agentic:
            print("\n" + "=" * 90)
            print(f"LANGGRAPH MULTI-AGENT ORCHESTRATION PIPELINE (LLM BACKEND: {llm_mode.upper()})")
            print("=" * 90)
            print(f"No maintenance needed yet. Predicted RUL ({pred_rul:.1f} cycles) is above the planning")
            print(f"horizon ({planning_horizon} cycles = lead time {lead_time} + window 60).")
            print("The agents start planning when it drops below this.")
            print("=" * 90)
        return

    rec = coordinator.decision_records[0]

    # Print Candidate Evaluation Table
    print("\nCandidate Start Windows Evaluated by Multi-Agent System:")
    print(f"{'Start Cycle':12s} | {'Offset s':10s} | {'Spare Ready':12s} | {'Tech Ready':12s} | {'Feasible':10s} | {'P(Fail)':10s} | {'Cost Rate':14s}")
    print("-" * 90)

    # Show candidates around feasible cycles and spare lead time
    shown = 0
    for c in rec["candidates"]:
        cyc = c["start_cycle"]
        # Filter for concise display
        if c["feasible"] or (c["offset_s"] in [lead_time, lead_time + 4, int(round(pred_rul))]) or shown < 8:
            pr_str = "READY" if c["part_ready"] else "TRANSIT"
            tr_str = "AVAILABLE" if c["tech_ready"] else "FULL"
            feas_str = "FEASIBLE" if c["feasible"] else "BLOCKED"
            pf_str = f"{c['prob_fail'] * 100:5.2f}%" if c["prob_fail"] is not None else "N/A"
            cr_str = f"${c['cost_rate']:6.2f}/cyc" if c["cost_rate"] is not None else "N/A"
            chosen_mark = " <=== [RECOMMENDED OVERHAUL]" if cyc == rec["chosen_cycle"] else ""
            print(f"Cycle {cyc:5d} | {c['offset_s']:10d} | {pr_str:12s} | {tr_str:12s} | {feas_str:10s} | {pf_str:10s} | {cr_str:14s}{chosen_mark}")
            shown += 1

    print("-" * 90)
    print(f"RECOMMENDED START CYCLE : Cycle {rec['chosen_cycle']} (Offset +{rec['chosen_cycle'] - current_cycle} cycles from now)")
    print(f"SPARE PART DISPATCH     : Cycle {rec['spare_order_cycle']} (Warehouse arrival: Cycle {rec['spare_order_cycle'] + lead_time})")
    print(f"EMERGENCY ESCALATION    : {'YES (Risk threshold crossed)' if rec['escalated'] else 'NO (Normal planning window)'}")
    print(f"OPERATIONAL RATIONALE   :")
    print(f"  \"{rec['reason']}\"")

    if agentic:
        tech_cycles = [int(c.strip()) for c in tech_str.split(",") if c.strip()]
        run_agentic_demo_section(
            engine_id=unit_id,
            current_cycle=current_cycle,
            age=current_cycle,
            lead_time=lead_time,
            technician_cycles=tech_cycles,
            is_fixture=False,
            rule_rec=rec,
            llm_mode=llm_mode,
        )


def main():
    parser = argparse.ArgumentParser(description="Live Multi-Agent Predictive Maintenance Demonstration")
    parser.add_argument("--engine", type=str, default="24", help="Engine ID (e.g. 24 or 17)")
    parser.add_argument("--lead-time", type=int, default=8, help="Spare part lead time in cycles (default: 8)")
    parser.add_argument("--technicians", type=str, default="16,22", help="Comma-separated cycles with technician capacity (default: '16,22')")
    parser.add_argument("--cycle", type=int, default=None, help="Operating cycle (optional, defaults to last cycle)")
    parser.add_argument("--brief-example", action="store_true", help="Run the Section 14 E-017 brief fixture")
    parser.add_argument("--agentic", action="store_true", help="Execute the LangGraph multi-agent orchestrator")
    parser.add_argument("--llm", type=str, choices=["groq", "fake"], default=None, help="LLM backend for agentic mode: groq or fake (default: groq if GROQ_API_KEY and GROQ_MODEL set, else fake)")

    args = parser.parse_args()

    # Determine LLM backend
    has_groq = bool(os.environ.get("GROQ_API_KEY", "").strip() and os.environ.get("GROQ_MODEL", "").strip())
    if args.llm is None:
        selected_llm = "groq" if has_groq else "fake"
        if not has_groq and args.agentic:
            print("[NOTICE] GROQ_API_KEY or GROQ_MODEL is not set. Defaulting --llm to fake.")
    else:
        selected_llm = args.llm
        if selected_llm == "groq" and not has_groq:
            print("[NOTICE] GROQ_API_KEY or GROQ_MODEL is empty or missing. Falling back to deterministic FakeChatModel.")
            selected_llm = "fake"

    # Clean engine ID
    engine_raw = args.engine.upper().replace("E-", "").replace("#", "")
    unit_id = int(engine_raw)

    if args.brief_example or unit_id == 17:
        run_e017_brief_fixture(agentic=args.agentic, llm_mode=selected_llm)

    if not args.brief_example:
        run_live_engine_demo(
            unit_id=unit_id,
            lead_time=args.lead_time,
            tech_str=args.technicians,
            cycle=args.cycle,
            agentic=args.agentic,
            llm_mode=selected_llm,
        )


if __name__ == "__main__":
    main()
