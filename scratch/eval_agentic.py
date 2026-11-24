"""
eval_agentic.py - Comprehensive Evaluation of LangGraph Agentic Maintenance Pipeline

Evaluates the LangGraph multi-agent decision pipeline against the rule-based Coordinator baseline across:
- Section 14 E-017 benchmark fixture
- 30 FD001 test fleet engines (Units 1..30)

Computes per-engine:
- rule-based cycle
- graph cycle
- agree?
- fallback?
- retries
- tool calls count
- latency (s)

Outputs summary metrics (agreement rate, fallback rate, mean latency) and saves to outputs/agentic_eval.json.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=False)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def evaluate_e017_fixture(llm_mode: str) -> Dict[str, Any]:
    """Evaluate E-017 benchmark fixture."""
    # 1. Rule-based Coordinator
    prognostic = PrognosticAgent(use_intervals=True)
    resource = ResourceAgent(spare_lead_time=8, preventive_duration=2, horizon_W=60)
    cost = CostAgent()
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
    rule_rec = coordinator.decision_records[0]
    rule_cycle = int(rule_rec["chosen_cycle"])

    # 2. LangGraph Agentic Pipeline
    t0 = time.perf_counter()
    graph_res = run_agentic_graph(
        engine_id=17,
        current_cycle=0,
        age=120,
        lead_time=8,
        technician_cycles=[16, 22],
        llm_type=llm_mode,
        is_fixture=True,
    )
    latency = time.perf_counter() - t0

    g_rec = graph_res["final_recommendation"]
    graph_cycle = int(g_rec["chosen_cycle"])
    tool_calls = sum(1 for t in graph_res.get("trace", []) if "tool" in t)
    enforced_steps = sum(1 for t in graph_res.get("trace", []) if t.get("enforced_by") == "graph")
    rationale_source = g_rec.get("rationale_source", "llm")
    agree = (rule_cycle == graph_cycle)
    discrepancy_reason = None if agree else f"Rule chose {rule_cycle}, Graph chose {graph_cycle}"

    return {
        "engine": "E-017 (Fixture)",
        "unit_id": 17,
        "is_fixture": True,
        "current_cycle": 0,
        "pred_rul": 28.0,
        "rule_cycle": rule_cycle,
        "graph_cycle": graph_cycle,
        "agree": bool(agree),
        "fallback": bool(g_rec["fallback"]),
        "retries": int(graph_res.get("retries", 0)),
        "tool_calls": int(tool_calls),
        "enforced_steps": int(enforced_steps),
        "rationale_source": str(rationale_source),
        "discrepancy_reason": discrepancy_reason,
        "latency_s": float(latency),
    }


def evaluate_test_engine(unit_id: int, test_df: pd.DataFrame, prognostic: PrognosticAgent, llm_mode: str) -> Dict[str, Any]:
    """Evaluate FD001 test fleet engine at last observed cycle."""
    eng_df = test_df[test_df["unit_number"] == unit_id].copy()
    if eng_df.empty:
        raise ValueError(f"Unit {unit_id} not found.")

    current_cycle = int(eng_df["time_in_cycles"].max())
    pred_rul, lower_rul, upper_rul = prognostic.predict(eng_df, method="binned_residual")

    lead_time = 8
    d_pm = 2
    W = max(60, int(np.ceil(pred_rul)) + 20)
    resource = ResourceAgent(spare_lead_time=lead_time, preventive_duration=d_pm, horizon_W=W)
    roster_len = current_cycle + W + 50
    roster = np.zeros(roster_len, dtype=int)
    roster[current_cycle + 16 : current_cycle + 18] = 1
    roster[current_cycle + 22 : current_cycle + 24] = 1
    resource.reset(total_horizon=current_cycle + W, capacity_roster=roster)

    cost = CostAgent(preventive_duration=d_pm)
    coordinator = CoordinatorAgent(prognostic, resource, cost, horizon_W=W, spare_lead_time=lead_time)
    coordinator.reset(total_horizon=current_cycle + W, capacity_roster=roster)

    obs = EngineObservation(
        slot_id=0,
        unit_number=unit_id,
        current_cycle=current_cycle,
        age=current_cycle,
        pred_rul=float(pred_rul),
        lower_rul=float(lower_rul),
        upper_rul=float(upper_rul),
        order_already_placed=False,
    )
    coordinator.plan_fleet(current_cycle=current_cycle, observations=[obs])
    rule_rec = coordinator.decision_records[0]
    rule_cycle = int(rule_rec["chosen_cycle"])

    # 2. LangGraph Agentic Pipeline
    t0 = time.perf_counter()
    graph_res = run_agentic_graph(
        engine_id=unit_id,
        current_cycle=current_cycle,
        age=current_cycle,
        lead_time=lead_time,
        technician_cycles=[16, 22],
        llm_type=llm_mode,
        is_fixture=False,
    )
    latency = time.perf_counter() - t0

    g_rec = graph_res["final_recommendation"]
    graph_cycle = int(g_rec["chosen_cycle"])
    tool_calls = sum(1 for t in graph_res.get("trace", []) if "tool" in t)
    enforced_steps = sum(1 for t in graph_res.get("trace", []) if t.get("enforced_by") == "graph")
    rationale_source = g_rec.get("rationale_source", "llm")
    agree = (rule_cycle == graph_cycle)
    discrepancy_reason = None if agree else f"Rule chose {rule_cycle}, Graph chose {graph_cycle}"

    return {
        "engine": f"Unit #{unit_id:03d}",
        "unit_id": unit_id,
        "is_fixture": False,
        "current_cycle": current_cycle,
        "pred_rul": float(pred_rul),
        "rule_cycle": rule_cycle,
        "graph_cycle": graph_cycle,
        "agree": bool(agree),
        "fallback": bool(g_rec["fallback"]),
        "retries": int(graph_res.get("retries", 0)),
        "tool_calls": int(tool_calls),
        "enforced_steps": int(enforced_steps),
        "rationale_source": str(rationale_source),
        "discrepancy_reason": discrepancy_reason,
        "latency_s": float(latency),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LangGraph agentic pipeline against rule-based Coordinator.")
    parser.add_argument("--llm", type=str, choices=["groq", "fake", "auto"], default="auto", help="LLM backend mode (default: auto)")
    parser.add_argument("--num-engines", type=int, default=30, help="Number of test engines to evaluate (default: 30)")
    args = parser.parse_args()

    # Determine LLM mode
    has_api_key = bool(os.environ.get("GROQ_API_KEY", "").strip() and os.environ.get("GROQ_MODEL", "").strip())
    if args.llm == "auto":
        if has_api_key:
            llm_mode = "groq"
        else:
            print("[NOTICE] GROQ_API_KEY or GROQ_MODEL not configured. Running evaluation with deterministic FakeChatModel (scripted tool calling).")
            llm_mode = "fake"
    else:
        llm_mode = args.llm
        if llm_mode == "groq" and not has_api_key:
            print("[WARNING] GROQ_API_KEY or GROQ_MODEL not found in environment. Falling back to FakeChatModel.")
            llm_mode = "fake"

    print("=" * 125)
    print(f"LANGGRAPH MULTI-AGENT EVALUATION: E-017 FIXTURE + {args.num_engines} TEST ENGINES (LLM: {llm_mode.upper()})")
    print("=" * 125)

    # Load data
    _, test_df, _ = load_cmapss(subset="FD001", data_dir=PROJECT_ROOT / "data" / "raw")
    prognostic = PrognosticAgent.from_checkpoint()

    results: List[Dict[str, Any]] = []

    # 1. Evaluate E-017 Fixture
    res_e017 = evaluate_e017_fixture(llm_mode=llm_mode)
    results.append(res_e017)

    # 2. Evaluate Test Engines
    for u in range(1, args.num_engines + 1):
        res_u = evaluate_test_engine(unit_id=u, test_df=test_df, prognostic=prognostic, llm_mode=llm_mode)
        results.append(res_u)

    # Print Table
    print(f"{'Engine':16s} | {'Cycle':6s} | {'Pred RUL':9s} | {'Rule':7s} | {'Graph':7s} | {'Agree?':7s} | {'Fallback?':10s} | {'Enforced':9s} | {'Rationale':10s} | {'Retries':8s} | {'Tools':6s} | {'Latency':9s}")
    print("-" * 125)
    for r in results:
        agree_str = "YES" if r["agree"] else "NO"
        fallback_str = "YES" if r["fallback"] else "NO"
        print(
            f"{r['engine']:16s} | {r['current_cycle']:6d} | {r['pred_rul']:9.1f} | "
            f"C-{r['rule_cycle']:<4d} | C-{r['graph_cycle']:<4d} | "
            f"{agree_str:7s} | {fallback_str:10s} | {r['enforced_steps']:9d} | {r['rationale_source']:10s} | {r['retries']:8d} | {r['tool_calls']:6d} | {r['latency_s']:8.2f}s"
        )
    print("-" * 125)

    # Summary Statistics
    total_count = len(results)
    agree_count = sum(1 for r in results if r["agree"])
    fallback_count = sum(1 for r in results if r["fallback"])
    total_enforced = sum(r["enforced_steps"] for r in results)
    llm_rationale_count = sum(1 for r in results if r["rationale_source"] == "llm")
    template_rationale_count = sum(1 for r in results if r["rationale_source"] == "template")
    latencies = [r["latency_s"] for r in results]
    mean_latency = float(np.mean(latencies))
    agreement_rate = float(agree_count / total_count)
    fallback_rate = float(fallback_count / total_count)

    disagreements = [r for r in results if not r["agree"]]

    summary = {
        "total_engines": total_count,
        "agreement_count": agree_count,
        "agreement_rate": agreement_rate,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_rate,
        "total_enforced_steps": total_enforced,
        "rationale_sources": {
            "llm": llm_rationale_count,
            "template": template_rationale_count,
        },
        "mean_latency_s": mean_latency,
        "median_latency_s": float(np.median(latencies)),
        "min_latency_s": float(np.min(latencies)),
        "max_latency_s": float(np.max(latencies)),
        "disagreements": disagreements,
        "llm_backend": llm_mode,
    }

    print("\nEVALUATION SUMMARY METRICS:")
    print(f"  Total Engines Evaluated : {total_count} (E-017 Benchmark Fixture + {args.num_engines} Test Units)")
    print(f"  Decision Agreement Rate : {agreement_rate * 100:.2f}% ({agree_count}/{total_count} exact cycle matches)")
    print(f"  Fallback Trigger Rate   : {fallback_rate * 100:.2f}% ({fallback_count}/{total_count} triggered)")
    print(f"  Total Enforced Steps    : {total_enforced} deterministic graph enforcements")
    print(f"  Rationale Distribution  : {llm_rationale_count} LLM-generated, {template_rationale_count} Template-guarded")
    print(f"  Mean Pipeline Latency   : {mean_latency:.4f} s (Min: {np.min(latencies):.4f}s, Max: {np.max(latencies):.4f}s)")
    if disagreements:
        print("\nDISCREPANCIES & REASONS:")
        for d in disagreements:
            print(f"  - {d['engine']}: {d['discrepancy_reason']}")
    else:
        print("  Discrepancies Observed  : None (100% agreement across all evaluated units)")
    print("=" * 125)

    # Save to outputs/agentic_eval.json
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "agentic_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"Saved evaluation benchmark results to {out_path}\n")


if __name__ == "__main__":
    main()
