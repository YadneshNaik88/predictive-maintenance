"""
generate_results_final.py

Phase 8b: Single Source of Truth Generator.
Reuses canonical baselines from outputs/results_final_v1.json.
Evaluates agent policies (Agent-Point, Agent-Interval, Agent-Perfect, Agent-Interval-Blind)
across Scenarios A, B, C on the 50-engine Evaluation Pool over 20 evaluation seeds (42..61).
Runs Phase 7 sweeps (Lead Time, Failure Cost, Holding Cost) and Stall Audit.
Writes canonical outputs/results_final.json.
"""

import sys
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import scipy.stats as stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import (
    FleetSimulator,
    SimulationConfig,
    load_engine_trajectories,
    get_tuning_eval_pools,
    EngineObservation,
)
from src.agents import (
    AgenticPolicy,
    PrognosticAgent,
    ResourceAgent,
    CostAgent,
    CoordinatorAgent,
)
from src.policies import NaiveThresholdPolicy, RunToFailurePolicy

OUTPUT_JSON_PATH = PROJECT_ROOT / "outputs" / "results_final.json"
V1_JSON_PATH = PROJECT_ROOT / "outputs" / "results_final_v1.json"
DECISION_LOG_FULL_PATH = PROJECT_ROOT / "outputs" / "decision_log_full.jsonl"
DECISION_LOG_SAMPLE_PATH = PROJECT_ROOT / "outputs" / "decision_log.jsonl"

def run_master_evaluation():
    print("=" * 80)
    print("PHASE 8B MASTER EVALUATION: GENERATING outputs/results_final.json")
    print("=" * 80)

    assert V1_JSON_PATH.exists(), "outputs/results_final_v1.json must exist to reuse baselines"
    with open(V1_JSON_PATH, "r", encoding="utf-8") as f:
        v1 = json.load(f)

    trajectories_real = load_engine_trajectories()
    _, eval_pool = get_tuning_eval_pools(seed=42)
    eval_seeds = [42 + i for i in range(20)]

    # Perfect trajectories for Agent-Perfect reference
    trajectories_perfect = {}
    for u, data in trajectories_real.items():
        trajectories_perfect[u] = {
            "unit_number": data["unit_number"],
            "max_cycles": data["max_cycles"],
            "cycles": data["cycles"],
            "true_rul_raw": data["true_rul_raw"],
            "true_rul_capped": data["true_rul_capped"],
            "pred_rul": data["true_rul_raw"].copy(),
            "lower": data["true_rul_raw"].copy(),
            "upper": data["true_rul_raw"].copy(),
        }

    scenarios_def = [
        {
            "id": "scenario_a",
            "name": "Scenario A: Abundant Capacity (4 Technicians/Cycle)",
            "short_name": "Scenario A",
            "T_star": 125,
            "tau_star": 15,
            "m_star": 2,
            "fixed_infeasible": False,
        },
        {
            "id": "scenario_b",
            "name": "Scenario B: Constrained Capacity (Mean 1.549 Technicians/Cycle)",
            "short_name": "Scenario B",
            "T_star": 75,
            "tau_star": 60,
            "m_star": 40,
            "fixed_infeasible": True,
        },
        {
            "id": "scenario_c",
            "name": "Scenario C: Tight Capacity (Mean 1.487 Technicians/Cycle)",
            "short_name": "Scenario C",
            "T_star": 75,
            "tau_star": 70,
            "m_star": 40,
            "fixed_infeasible": True,
        },
    ]

    final_results: Dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "phase": "Phase 8b (Reservation Fix single source of truth)",
            "evaluation_pool_size": len(eval_pool),
            "num_seeds": len(eval_seeds),
            "seeds": eval_seeds,
            "cost_parameters": v1["metadata"]["cost_parameters"],
            "formula_cost_agent": v1["metadata"].get("formula_cost_agent", {}),
        },
        "scenarios": {},
    }

    # Track stall audit metrics
    stall_audit = {}

    # Clear full decision log before recording seed 42
    if DECISION_LOG_FULL_PATH.exists():
        try:
            DECISION_LOG_FULL_PATH.unlink()
        except Exception:
            pass

    for scen in scenarios_def:
        scen_id = scen["id"]
        scen_name = scen["name"]
        print(f"\nProcessing {scen_name}...")

        cfg = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario=scen_id)
        sim_real = FleetSimulator(config=cfg, trajectories=trajectories_real)
        sim_perf = FleetSimulator(config=cfg, trajectories=trajectories_perfect)

        v1_scen = v1["scenarios"][scen_id]

        scen_res = {
            "name": scen_name,
            "short_name": scen["short_name"],
            "T_star": scen["T_star"],
            "tau_star": scen["tau_star"],
            "m_star": scen["m_star"],
            "fixed_infeasible": scen["fixed_infeasible"],
            "policies": {},
        }

        # 1. Reuse canonical baselines from v1
        baseline_keys = [
            "Run-to-Failure",
            f"Fixed Interval (T={scen['T_star']})",
            f"Naive Threshold (tau={scen['tau_star']})",
            f"Diagnostic Oracle (m={scen['m_star']})",
        ]
        for b_key in baseline_keys:
            assert b_key in v1_scen["policies"], f"Missing baseline {b_key} in v1"
            scen_res["policies"][b_key] = v1_scen["policies"][b_key]

        # 2. Agent policies to simulate under new reservation/dispatch rules
        agent_policies = [
            ("Agent-Point", lambda: AgenticPolicy(use_intervals=False, decision_log_path=None), sim_real),
            ("Agent-Interval", lambda: AgenticPolicy(use_intervals=True, decision_log_path=None), sim_real),
            ("Agent-Interval-Blind", lambda: AgenticPolicy(use_intervals=True, roster_blind=True, decision_log_path=None), sim_real),
            ("Agent-Perfect", lambda: AgenticPolicy(use_intervals=True, decision_log_path=None), sim_perf),
        ]

        for pol_name, pol_factory, sim_engine in agent_policies:
            print(f"  Simulating {pol_name:<24} across {len(eval_seeds)} seeds...")
            runs = []
            for s in eval_seeds:
                # For Scenario B, seed 42 with Agent-Interval: record full decision log
                if scen_id == "scenario_b" and pol_name == "Agent-Interval" and s == 42:
                    pol = AgenticPolicy(use_intervals=True, decision_log_path=DECISION_LOG_FULL_PATH)
                else:
                    pol = pol_factory()

                if pol_name == "Agent-Perfect":
                    for b in pol.prognostic.residuals_by_bin:
                        pol.prognostic.residuals_by_bin[b] = np.array([0.0])

                res = sim_engine.simulate(pol, eval_pool, seed=s)
                runs.append(res)

            costs = np.array([r["total_cost"] for r in runs])
            fails = np.array([r["number_of_failures"] for r in runs])
            prevs = np.array([r["number_of_preventive_actions"] for r in runs])
            downtimes = np.array([r["total_downtime_cycles"] for r in runs])
            leads = np.array([r["average_maintenance_lead"] for r in runs])
            qwaits = np.array([r["mean_queue_wait"] for r in runs])
            utils = np.array([r["utilization"] for r in runs])
            loads = np.array([r["offered_load"] for r in runs])
            escalations = np.array([r["num_escalations"] for r in runs])
            stalls = np.array([r.get("total_stalled_cycles", 0) for r in runs])
            prev_stalls = np.array([r.get("preventive_stalled_cycles", 0) for r in runs])
            q_fails = np.array([r.get("queue_failures", 0) for r in runs])

            # Cost decomposition components
            c_prev = np.array([r["preventive_cost"] for r in runs])
            c_spare = np.array([r["spare_cost"] for r in runs])
            c_down = np.array([r["production_loss"] for r in runs])
            c_fail = np.array([r["corrective_cost"] for r in runs])

            mean_c_tot = float(np.mean(costs))
            mean_c_prev = float(np.mean(c_prev))
            mean_c_spare = float(np.mean(c_spare))
            mean_c_down = float(np.mean(c_down))
            mean_c_fail = float(np.mean(c_fail))

            delta = abs((mean_c_prev + mean_c_spare + mean_c_down + mean_c_fail) - mean_c_tot)
            assert delta < 1e-2, f"Sum check violation in {scen_id} {pol_name}: delta={delta}"

            pol_entry = {
                "policy_name": pol_name,
                "mean_cost": mean_c_tot,
                "std_cost_sample": float(np.std(costs, ddof=1)),
                "std_cost_pop": float(np.std(costs, ddof=0)),
                "mean_failures": float(np.mean(fails)),
                "std_failures": float(np.std(fails, ddof=1)),
                "mean_pm_actions": float(np.mean(prevs)),
                "std_pm_actions": float(np.std(prevs, ddof=1)),
                "mean_downtime_cycles": float(np.mean(downtimes)),
                "downtime_pct": float(np.mean(downtimes) / (cfg.num_slots * cfg.horizon) * 100.0),
                "mean_queue_wait": float(np.mean(qwaits)),
                "mean_lead_time": float(np.mean(leads)),
                "utilization": float(np.mean(utils)),
                "offered_load": float(np.mean(loads)),
                "num_escalations": float(np.mean(escalations)),
                "mean_stalled_cycles": float(np.mean(stalls)),
                "mean_preventive_stalled_cycles": float(np.mean(prev_stalls)),
                "mean_queue_failures": float(np.mean(q_fails)),
                "cost_decomposition": {
                    "preventive_overhaul": mean_c_prev,
                    "spare_parts": mean_c_spare,
                    "downtime_loss": mean_c_down,
                    "failure_penalty": mean_c_fail,
                    "sum_components": mean_c_prev + mean_c_spare + mean_c_down + mean_c_fail,
                    "total_cost": mean_c_tot,
                    "sum_check_delta": delta,
                },
                "raw_costs": costs.tolist(),
                "raw_failures": fails.tolist(),
                "raw_stalls": stalls.tolist(),
                "raw_queue_failures": q_fails.tolist(),
            }
            scen_res["policies"][pol_name] = pol_entry

            if pol_name in ("Agent-Interval", "Agent-Perfect") and scen_id in ("scenario_b", "scenario_c"):
                key = f"{scen_id}_{pol_name}"
                stall_audit[key] = {
                    "scenario": scen_id,
                    "policy": pol_name,
                    "after_mean_stalls": float(np.mean(stalls)),
                    "after_preventive_stalls": float(np.mean(prev_stalls)),
                    "after_queue_failures": float(np.mean(q_fails)),
                }

        # Paired comparison: Agent-Interval minus Naive-Threshold across identical 20 seeds
        naive_key = f"Naive Threshold (tau={scen['tau_star']})"
        agent_key = "Agent-Interval"
        costs_agent = np.array(scen_res["policies"][agent_key]["raw_costs"])
        costs_naive = np.array(scen_res["policies"][naive_key]["raw_costs"])
        diff = costs_agent - costs_naive
        diff_mean = float(np.mean(diff))
        diff_std = float(np.std(diff, ddof=1))
        diff_se = diff_std / np.sqrt(len(eval_seeds))
        t_crit = float(stats.t.ppf(0.975, df=len(eval_seeds) - 1))
        ci_lower = float(diff_mean - t_crit * diff_se)
        ci_upper = float(diff_mean + t_crit * diff_se)
        p_val = float(stats.ttest_rel(costs_agent, costs_naive).pvalue)
        saving_pct = float((np.mean(costs_naive) - np.mean(costs_agent)) / np.mean(costs_naive) * 100.0)

        scen_res["paired_vs_naive"] = {
            "naive_policy": naive_key,
            "agent_interval_mean_cost": float(np.mean(costs_agent)),
            "naive_mean_cost": float(np.mean(costs_naive)),
            "diff_mean_dollar": diff_mean,
            "diff_std_dollar": diff_std,
            "diff_se_dollar": diff_se,
            "ci_95": [ci_lower, ci_upper],
            "p_value": p_val,
            "saving_pct_over_naive": saving_pct,
        }

        # Experiment 2: Agent-Point vs Agent-Interval
        pt_entry = scen_res["policies"]["Agent-Point"]
        it_entry = scen_res["policies"]["Agent-Interval"]
        scen_res["experiment_2"] = {
            "point_cost": pt_entry["mean_cost"],
            "interval_cost": it_entry["mean_cost"],
            "cost_reduction_dollar": pt_entry["mean_cost"] - it_entry["mean_cost"],
            "cost_reduction_pct": (pt_entry["mean_cost"] - it_entry["mean_cost"]) / pt_entry["mean_cost"] * 100.0,
            "point_failures": pt_entry["mean_failures"],
            "interval_failures": it_entry["mean_failures"],
            "failures_eliminated": pt_entry["mean_failures"] - it_entry["mean_failures"],
        }

        final_results["scenarios"][scen_id] = scen_res

    # 3. Phase 7 Sensitivity Sweeps on Scenario B
    print("\nRunning Phase 7 Sensitivity Sweeps on Scenario B...")
    cfg_b = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario="scenario_b")

    # Sweep 1: Lead Time L in {4, 8, 12, 16} with tuned tau* in {50, 60, 65, 70}
    l_sweep_grid = [(4, 50), (8, 60), (12, 65), (16, 70)]
    lead_time_sweep = []
    for l_val, tau_val in l_sweep_grid:
        cfg_l = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario="scenario_b")
        cfg_l.spare_lead_time = l_val
        sim_l = FleetSimulator(config=cfg_l, trajectories=trajectories_real)

        pol_naive = NaiveThresholdPolicy(tau_val)
        pol_agent = AgenticPolicy(use_intervals=True, spare_lead_time=l_val, decision_log_path=None)

        naive_costs = [sim_l.simulate(pol_naive, eval_pool, seed=s)["total_cost"] for s in eval_seeds]
        agent_costs = [sim_l.simulate(pol_agent, eval_pool, seed=s)["total_cost"] for s in eval_seeds]

        diff = np.array(agent_costs) - np.array(naive_costs)
        t_crit = float(stats.t.ppf(0.975, df=len(eval_seeds) - 1))
        se = float(np.std(diff, ddof=1) / np.sqrt(len(eval_seeds)))
        ci = [float(np.mean(diff) - t_crit * se), float(np.mean(diff) + t_crit * se)]
        p_val = float(stats.ttest_rel(agent_costs, naive_costs).pvalue)

        lead_time_sweep.append({
            "lead_time": l_val,
            "tau_star": tau_val,
            "naive_mean_cost": float(np.mean(naive_costs)),
            "agent_mean_cost": float(np.mean(agent_costs)),
            "diff_mean": float(np.mean(diff)),
            "saving_pct": float((np.mean(naive_costs) - np.mean(agent_costs)) / np.mean(naive_costs) * 100.0),
            "ci_95": ci,
            "p_value": p_val,
        })

    # Sweep 2: Failure Cost Penalty C_fail in {10k, 20k, 40k}
    cfail_sweep = []
    for cf in [10000.0, 20000.0, 40000.0]:
        cfg_cf = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario="scenario_b")
        cfg_cf.cost_corrective = cf
        sim_cf = FleetSimulator(config=cfg_cf, trajectories=trajectories_real)

        rtf_costs = [sim_cf.simulate(RunToFailurePolicy(), eval_pool, seed=s)["total_cost"] for s in eval_seeds]
        pt_costs = [sim_cf.simulate(AgenticPolicy(use_intervals=False, decision_log_path=None), eval_pool, seed=s)["total_cost"] for s in eval_seeds]
        it_costs = [sim_cf.simulate(AgenticPolicy(use_intervals=True, decision_log_path=None), eval_pool, seed=s)["total_cost"] for s in eval_seeds]

        cfail_sweep.append({
            "c_fail": cf,
            "rtf_mean_cost": float(np.mean(rtf_costs)),
            "point_mean_cost": float(np.mean(pt_costs)),
            "interval_mean_cost": float(np.mean(it_costs)),
            "interval_advantage_over_point": float(np.mean(pt_costs) - np.mean(it_costs)),
        })

    # Holding cost check
    sim_b = FleetSimulator(config=cfg_b, trajectories=trajectories_real)
    holding_check = []
    naive_pol_b = NaiveThresholdPolicy(60)
    agent_pol_b = AgenticPolicy(use_intervals=True, spare_lead_time=8, decision_log_path=None)

    for h_rate in [0.0, 2.0, 5.0]:
        n_costs = []
        a_costs = []
        for s in eval_seeds:
            res_n = sim_b.simulate(naive_pol_b, eval_pool, seed=s)
            res_a = sim_b.simulate(agent_pol_b, eval_pool, seed=s)
            jobs_n = res_n["number_of_preventive_actions"] + res_n["number_of_failures"]
            jobs_a = res_a["number_of_preventive_actions"] + res_a["number_of_failures"]
            h_cost_n = res_n["mean_queue_wait"] * jobs_n * h_rate
            h_cost_a = res_a["mean_queue_wait"] * jobs_a * h_rate
            n_costs.append(res_n["total_cost"] + h_cost_n)
            a_costs.append(res_a["total_cost"] + h_cost_a)

        diff = float(np.mean(a_costs) - np.mean(n_costs))
        pct = float(diff / np.mean(n_costs) * 100.0)
        holding_check.append({
            "holding_rate": h_rate,
            "naive_cost": float(np.mean(n_costs)),
            "agent_cost": float(np.mean(a_costs)),
            "diff": diff,
            "diff_pct": pct,
        })

    final_results["phase7_sweeps"] = {
        "lead_time_sweep": lead_time_sweep,
        "failure_penalty_sweep": cfail_sweep,
        "holding_cost_check": holding_check,
    }

    final_results["stall_audit"] = stall_audit

    # 4. Save new outputs/results_final.json
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n[SUCCESS] Wrote new results to {OUTPUT_JSON_PATH} ({os.path.getsize(OUTPUT_JSON_PATH):,} bytes)")

    # 5. Restore 4 MB sample decision_log.jsonl from decision_log_full.jsonl
    if DECISION_LOG_FULL_PATH.exists():
        full_size = DECISION_LOG_FULL_PATH.stat().st_size
        print(f"Full decision log created: {full_size:,} bytes")
        # Sample ~4 MB
        target_bytes = 4 * 1024 * 1024
        written_bytes = 0
        with open(DECISION_LOG_FULL_PATH, "r", encoding="utf-8") as f_in, open(DECISION_LOG_SAMPLE_PATH, "w", encoding="utf-8") as f_out:
            for line in f_in:
                f_out.write(line)
                written_bytes += len(line.encode("utf-8"))
                if written_bytes >= target_bytes:
                    break
        print(f"[SUCCESS] Wrote 4 MB sample decision log to {DECISION_LOG_SAMPLE_PATH} ({DECISION_LOG_SAMPLE_PATH.stat().st_size:,} bytes)")

    # Print Before / After Table for Agent policies
    print("\n" + "=" * 110)
    print("BEFORE / AFTER TABLE (v1 vs New Reservation Fix)")
    print("=" * 110)
    print(f"{'Scenario':12s} | {'Policy':22s} | {'v1 Cost':15s} | {'New Cost':15s} | {'v1 Fails':10s} | {'New Fails':10s}")
    print("-" * 110)
    for scen in scenarios_def:
        sc_id = scen["id"]
        v1_pol_dict = v1["scenarios"][sc_id]["policies"]
        new_pol_dict = final_results["scenarios"][sc_id]["policies"]
        for p_name in ["Agent-Point", "Agent-Interval", "Agent-Perfect", "Agent-Interval-Blind"]:
            v1_c = f"${v1_pol_dict[p_name]['mean_cost']:,.0f}" if p_name in v1_pol_dict else "N/A"
            new_c = f"${new_pol_dict[p_name]['mean_cost']:,.0f}"
            v1_f = f"{v1_pol_dict[p_name]['mean_failures']:.2f}" if p_name in v1_pol_dict else "N/A"
            new_f = f"{new_pol_dict[p_name]['mean_failures']:.2f}"
            print(f"{scen['short_name']:12s} | {p_name:22s} | {v1_c:15s} | {new_c:15s} | {v1_f:10s} | {new_f:10s}")

    print("\n" + "=" * 110)
    print("STALL AUDIT (Scenario B and C)")
    print("=" * 110)
    for k, v in stall_audit.items():
        print(f"{k}: Mean Total Stalls={v['after_mean_stalls']:.2f}, Preventive Stalls={v['after_preventive_stalls']:.2f}, Queue Failures={v['after_queue_failures']:.2f}")

if __name__ == "__main__":
    run_master_evaluation()
