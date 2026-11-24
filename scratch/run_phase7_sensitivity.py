"""
run_phase7_sensitivity.py

Phase 7: Mandatory Experiment 6 - Sensitivity Study on Scenario B.
Sweeps two parameters independently around the Scenario B baseline:
- Sweep 1: lead time L in {4, 8, 12, 16} cycles (C_fail = $10,000, C_pm = $2,000).
- Sweep 2: failure penalty C_fail in {$10,000, $20,000, $40,000} (L = 8, C_pm = $2,000).

Strict Fairness Rules:
- Re-tune Naive tau on TUNING pool (50 engines, 20 seeds) for EACH (L, C_fail) setting.
- Re-tune Oracle margin m on TUNING pool (50 engines, 20 seeds) for each setting.
- Fixed Interval: keep T=75 (flagged as capacity-infeasible, rho_offered > 1.0).
- Cost Agent: uses the setting's C_fail in its renewal objective C(s)/E(s).
- Evaluate on EVALUATION pool (50 engines, 20 seeds) for:
  RTF, Fixed, Naive (tuned), Agent-Point, Agent-Interval, Oracle (tuned).
- Generates 3 publication-ready plots:
  * 22_sensitivity_cost_vs_lead_time.png
  * 23_sensitivity_cost_vs_failure_penalty.png
  * 24_sensitivity_failures_vs_lead_time.png
- Outputs outputs/phase7_results.json and outputs/phase7_results.md.
"""

import sys
import json
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("C:/Users/naiky/OneDrive/Desktop/Projects/predictive-maintenance")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import (
    FleetSimulator,
    SimulationConfig,
    get_tuning_eval_pools,
    load_engine_trajectories,
)
from src.policies import (
    RunToFailurePolicy,
    FixedIntervalPolicy,
    NaiveThresholdPolicy,
    DiagnosticOraclePolicy,
)
from src.agents import AgenticPolicy, CostAgent


def tune_naive(sim: FleetSimulator, tuning_pool: np.ndarray, seeds: list, tau_grid: list) -> Tuple[int, dict]:
    best_tau = None
    min_cost = float("inf")
    grid_results = {}
    
    for tau in tau_grid:
        costs = []
        fails = []
        pol = NaiveThresholdPolicy(tau=tau)
        for s in seeds:
            res = sim.simulate(pol, tuning_pool, seed=s)
            costs.append(res["total_cost"])
            fails.append(res["number_of_failures"])
        m_cost = float(np.mean(costs))
        s_cost = float(np.std(costs))
        m_fails = float(np.mean(fails))
        grid_results[tau] = {"mean_cost": m_cost, "std_cost": s_cost, "mean_failures": m_fails}
        
        if m_cost < min_cost:
            min_cost = m_cost
            best_tau = tau
            
    return best_tau, grid_results


def tune_oracle(sim: FleetSimulator, tuning_pool: np.ndarray, seeds: list, m_grid: list, lead_time: int) -> Tuple[int, dict]:
    best_m = None
    min_cost = float("inf")
    grid_results = {}
    
    for m in m_grid:
        costs = []
        fails = []
        pol = DiagnosticOraclePolicy(lead_time=lead_time, margin=m)
        for s in seeds:
            res = sim.simulate(pol, tuning_pool, seed=s)
            costs.append(res["total_cost"])
            fails.append(res["number_of_failures"])
        m_cost = float(np.mean(costs))
        s_cost = float(np.std(costs))
        m_fails = float(np.mean(fails))
        grid_results[m] = {"mean_cost": m_cost, "std_cost": s_cost, "mean_failures": m_fails}
        
        if m_cost < min_cost:
            min_cost = m_cost
            best_m = m
            
    return best_m, grid_results


def evaluate_policies_on_eval_pool(
    sim: FleetSimulator,
    eval_pool: np.ndarray,
    seeds: list,
    policies: list,
) -> dict:
    results = {}
    naive_costs = None
    
    for pol_name, pol, is_naive in policies:
        costs, pre_c, sp_c, dn_c, fa_c = [], [], [], [], []
        failures, downtimes, busy_arr, avail_arr, dem_arr = [], [], [], [], []
        
        for s in seeds:
            res = sim.simulate(pol, eval_pool, seed=s)
            costs.append(res["total_cost"])
            pre_c.append(res["preventive_cost"])
            sp_c.append(res["spare_cost"])
            dn_c.append(res["production_loss"])
            fa_c.append(res["corrective_cost"])
            failures.append(res["number_of_failures"])
            downtimes.append(res["total_downtime_cycles"])
            busy_arr.append(res["total_tech_busy"])
            avail_arr.append(res["total_tech_available"])
            dem_arr.append(res["demanded_tech_cycles"])
            
            # Check sum conservation
            s_sum = res["preventive_cost"] + res["spare_cost"] + res["production_loss"] + res["corrective_cost"]
            assert abs(s_sum - res["total_cost"]) < 1e-3, f"Sum mismatch: {s_sum} != {res['total_cost']}"
            
        costs = np.array(costs)
        if is_naive:
            naive_costs = costs
            
        # Total fleet slot cycles = num_slots * horizon = 100 * 1000 = 100,000
        total_slot_cycles = sim.config.num_slots * sim.config.horizon
        
        m_cost = float(np.mean(costs))
        s_cost = float(np.std(costs))
        m_fail = float(np.mean(failures))
        m_down_pct = float(np.mean(downtimes) / total_slot_cycles * 100)
        offered_load = float(np.sum(dem_arr) / np.sum(avail_arr))
        utilization = float(np.sum(busy_arr) / np.sum(avail_arr))
        
        m_prev = float(np.mean(pre_c))
        m_spare = float(np.mean(sp_c))
        m_down = float(np.mean(dn_c))
        m_fa = float(np.mean(fa_c))
        
        results[pol_name] = {
            "mean_cost": m_cost,
            "std_cost": s_cost,
            "mean_failures": m_fail,
            "downtime_pct": m_down_pct,
            "offered_load": offered_load,
            "utilization": utilization,
            "preventive_cost": m_prev,
            "spare_cost": m_spare,
            "downtime_cost": m_down,
            "failure_cost": m_fa,
            "raw_costs": costs.tolist(),
        }
        
    # Compute paired differences vs Naive
    assert naive_costs is not None, "Naive policy must be evaluated"
    for pol_name in results:
        pol_raw = np.array(results[pol_name]["raw_costs"])
        diff = pol_raw - naive_costs
        m_diff = float(np.mean(diff))
        s_diff = float(np.std(diff, ddof=1))
        se_diff = s_diff / np.sqrt(len(seeds))
        ci_low = m_diff - 1.96 * se_diff
        ci_high = m_diff + 1.96 * se_diff
        results[pol_name]["diff_vs_naive_mean"] = m_diff
        results[pol_name]["diff_vs_naive_ci"] = [ci_low, ci_high]
        
    return results


def run_experiment_6():
    print("=" * 80, flush=True)
    print("STARTING PHASE 7: EXPERIMENT 6 SENSITIVITY STUDY (SCENARIO B)", flush=True)
    print("=" * 80, flush=True)

    trajectories = load_engine_trajectories()
    tuning_pool, eval_pool = get_tuning_eval_pools(42)
    seeds = [42 + i for i in range(20)]

    tau_grid_standard = [20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    m_grid_standard = [10, 20, 25, 30, 35, 40, 45, 50, 60]

    # Unique experimental conditions to simulate
    # Baseline Scenario B is (L=8, C_fail=10000)
    sweep_configs = [
        {"lead_time": 4, "c_fail": 10000.0, "label": "L=4, C_fail=$10k"},
        {"lead_time": 8, "c_fail": 10000.0, "label": "L=8, C_fail=$10k (Baseline B)"},
        {"lead_time": 12, "c_fail": 10000.0, "label": "L=12, C_fail=$10k"},
        {"lead_time": 16, "c_fail": 10000.0, "label": "L=16, C_fail=$10k"},
        {"lead_time": 8, "c_fail": 20000.0, "label": "L=8, C_fail=$20k"},
        {"lead_time": 8, "c_fail": 40000.0, "label": "L=8, C_fail=$40k"},
    ]

    all_condition_results = {}

    for cfg_dict in sweep_configs:
        L = cfg_dict["lead_time"]
        cf = cfg_dict["c_fail"]
        cond_label = cfg_dict["label"]
        cond_key = f"L{L}_cf{int(cf)}"

        print(f"\n>>> Running Condition: {cond_label} (L={L}, C_fail=${cf:,.0f})", flush=True)

        # Base config from Scenario B
        cfg = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario="scenario_b")
        cfg.spare_lead_time = L
        cfg.cost_corrective = cf
        sim = FleetSimulator(config=cfg, trajectories=trajectories)

        # 1. Tune Naive on Tuning Pool
        print(f"  [1/3] Tuning Naive Threshold on Tuning Pool (50 engines, 20 seeds)...", flush=True)
        tau_star, tau_grid_res = tune_naive(sim, tuning_pool, seeds, tau_grid_standard)
        # Verify interior optimum check
        if tau_star in (tau_grid_standard[0], tau_grid_standard[-1]):
            print(f"  WARNING: tau*={tau_star} is at grid edge! Extending grid...", flush=True)
            if tau_star == tau_grid_standard[-1]:
                tau_ext = list(range(tau_star + 5, tau_star + 35, 5))
                t_star_ext, ext_res = tune_naive(sim, tuning_pool, seeds, tau_ext)
                tau_grid_res.update(ext_res)
                if ext_res[t_star_ext]["mean_cost"] < tau_grid_res[tau_star]["mean_cost"]:
                    tau_star = t_star_ext
        print(f"  -> Winning tau* = {tau_star} (Tuning Cost: ${tau_grid_res[tau_star]['mean_cost']:,.0f})", flush=True)

        # 2. Tune Oracle on Tuning Pool
        print(f"  [2/3] Tuning Diagnostic Oracle on Tuning Pool (50 engines, 20 seeds)...", flush=True)
        m_star, m_grid_res = tune_oracle(sim, tuning_pool, seeds, m_grid_standard, lead_time=L)
        print(f"  -> Winning m* = {m_star} (Tuning Cost: ${m_grid_res[m_star]['mean_cost']:,.0f})", flush=True)

        # 3. Instantiate Policies for Evaluation Pool
        print(f"  [3/3] Evaluating 6 Policies on Held-out Evaluation Pool (50 engines, 20 seeds)...", flush=True)
        
        # Instantiate Agentic policies with matched lead_time and cost_corrective
        ag_point = AgenticPolicy(use_intervals=False, spare_lead_time=L, decision_log_path=None)
        ag_point.cost = CostAgent(
            cost_preventive=cfg.cost_preventive,
            cost_corrective=cfg.cost_corrective,
            cost_production_loss=cfg.cost_production_loss,
            cost_spare=cfg.cost_spare,
            preventive_duration=cfg.preventive_duration,
            corrective_duration=cfg.corrective_duration,
        )
        ag_point.coordinator.cost = ag_point.cost

        ag_interval = AgenticPolicy(use_intervals=True, spare_lead_time=L, decision_log_path=None)
        ag_interval.cost = CostAgent(
            cost_preventive=cfg.cost_preventive,
            cost_corrective=cfg.cost_corrective,
            cost_production_loss=cfg.cost_production_loss,
            cost_spare=cfg.cost_spare,
            preventive_duration=cfg.preventive_duration,
            corrective_duration=cfg.corrective_duration,
        )
        ag_interval.coordinator.cost = ag_interval.cost

        eval_policies = [
            ("Run-to-Failure", RunToFailurePolicy(), False),
            ("Fixed Interval (T=75)", FixedIntervalPolicy(75, L), False),
            (f"Naive Threshold (tau={tau_star})", NaiveThresholdPolicy(tau_star), True),
            ("Agent-Point", ag_point, False),
            ("Agent-Interval", ag_interval, False),
            (f"Diagnostic Oracle (m={m_star})", DiagnosticOraclePolicy(L, m_star), False),
        ]

        eval_res = evaluate_policies_on_eval_pool(sim, eval_pool, seeds, eval_policies)

        all_condition_results[cond_key] = {
            "lead_time": L,
            "c_fail": cf,
            "label": cond_label,
            "tau_star": tau_star,
            "m_star": m_star,
            "evaluation_results": eval_res,
        }

        print(f"  Completed {cond_label}.", flush=True)

    # =============================================================================
    # GENERATE PLOTS
    # =============================================================================
    plots_dir = PROJECT_ROOT / "outputs" / "eda_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 22: Sensitivity Cost vs Lead Time L in {4, 8, 12, 16} (C_fail = $10k)
    L_vals = [4, 8, 12, 16]
    plt.figure(figsize=(9, 5.5))
    for pol_key, pol_label, color, marker in [
        ("Naive Threshold", "Naive Threshold (tuned tau*)", "#d62728", "s"),
        ("Agent-Interval", "Agent-Interval (OOF calibrated)", "#1f7774", "o"),
        ("Diagnostic Oracle", "Diagnostic Oracle (tuned m*)", "#2ca02c", "^"),
    ]:
        means, errs = [], []
        for L in L_vals:
            cond = all_condition_results[f"L{L}_cf10000"]["evaluation_results"]
            matched_key = [k for k in cond if pol_key in k][0]
            m = cond[matched_key]["mean_cost"] / 1e6
            s = cond[matched_key]["std_cost"] / 1e6
            ci = 1.96 * s / np.sqrt(20)
            means.append(m)
            errs.append(ci)
        plt.errorbar(L_vals, means, yerr=errs, label=pol_label, color=color, marker=marker, linewidth=2, capsize=4)

    plt.xlabel("Spare Part Lead Time $L$ (flight cycles)", fontsize=11, fontweight="bold")
    plt.ylabel("Fleet Total Maintenance Cost ($ Millions)", fontsize=11, fontweight="bold")
    plt.title(r"Phase 7 (Exp 6): Total Cost Sensitivity vs Spare Part Lead Time $L$" + "\n" + r"(Scenario B Constrained Capacity, $C_{fail}=\$10k$, 20 Evaluation Seeds)", fontsize=12, fontweight="bold")
    plt.xticks(L_vals)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plot22_path = plots_dir / "22_sensitivity_cost_vs_lead_time.png"
    plt.savefig(plot22_path, dpi=300)
    plt.close()
    print(f"Saved: {plot22_path}", flush=True)

    # Plot 23: Sensitivity Cost vs Failure Penalty C_fail in {10k, 20k, 40k} (L = 8)
    cf_vals = [10000, 20000, 40000]
    cf_labels = ["$10k", "$20k", "$40k"]
    plt.figure(figsize=(9, 5.5))
    for pol_key, pol_label, color, marker in [
        ("Naive Threshold", "Naive Threshold (tuned tau*)", "#d62728", "s"),
        ("Agent-Interval", "Agent-Interval (OOF calibrated)", "#1f7774", "o"),
        ("Diagnostic Oracle", "Diagnostic Oracle (tuned m*)", "#2ca02c", "^"),
    ]:
        means, errs = [], []
        for cf in cf_vals:
            cond = all_condition_results[f"L8_cf{cf}"]["evaluation_results"]
            matched_key = [k for k in cond if pol_key in k][0]
            m = cond[matched_key]["mean_cost"] / 1e6
            s = cond[matched_key]["std_cost"] / 1e6
            ci = 1.96 * s / np.sqrt(20)
            means.append(m)
            errs.append(ci)
        plt.errorbar([cf / 1000 for cf in cf_vals], means, yerr=errs, label=pol_label, color=color, marker=marker, linewidth=2, capsize=4)

    plt.xlabel(r"In-Service Failure Penalty $C_{fail}$ ($ Thousands)", fontsize=11, fontweight="bold")
    plt.ylabel("Fleet Total Maintenance Cost ($ Millions)", fontsize=11, fontweight="bold")
    plt.title(r"Phase 7 (Exp 6): Total Cost Sensitivity vs Failure Penalty $C_{fail}$" + "\n" + r"(Scenario B Constrained Capacity, Lead Time $L=8$, 20 Evaluation Seeds)", fontsize=12, fontweight="bold")
    plt.xticks([10, 20, 40], cf_labels)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plot23_path = plots_dir / "23_sensitivity_cost_vs_failure_penalty.png"
    plt.savefig(plot23_path, dpi=300)
    plt.close()
    print(f"Saved: {plot23_path}", flush=True)

    # Plot 24: Sensitivity Failures vs Lead Time L (Naive vs Agent-Interval)
    plt.figure(figsize=(9, 5.5))
    for pol_key, pol_label, color, marker in [
        ("Naive Threshold", "Naive Threshold (tuned tau*)", "#d62728", "s"),
        ("Agent-Interval", "Agent-Interval (OOF calibrated)", "#1f7774", "o"),
    ]:
        fails = []
        for L in L_vals:
            cond = all_condition_results[f"L{L}_cf10000"]["evaluation_results"]
            matched_key = [k for k in cond if pol_key in k][0]
            fails.append(cond[matched_key]["mean_failures"])
        plt.plot(L_vals, fails, label=pol_label, color=color, marker=marker, linewidth=2, markersize=7)

    plt.xlabel(r"Spare Part Lead Time $L$ (flight cycles)", fontsize=11, fontweight="bold")
    plt.ylabel("Fleet In-Service Catastrophic Failures", fontsize=11, fontweight="bold")
    plt.title(r"Phase 7 (Exp 6): In-Service Failures vs Spare Part Lead Time $L$" + "\n" + r"(Scenario B Constrained Capacity, $C_{fail}=\$10k$, 20 Evaluation Seeds)", fontsize=12, fontweight="bold")
    plt.xticks(L_vals)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plot24_path = plots_dir / "24_sensitivity_failures_vs_lead_time.png"
    plt.savefig(plot24_path, dpi=300)
    plt.close()
    print(f"Saved: {plot24_path}", flush=True)

    # =============================================================================
    # SAVE JSON AND MARKDOWN REPORTS
    # =============================================================================
    # Save JSON
    out_json_path = PROJECT_ROOT / "outputs" / "phase7_results.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(all_condition_results, f, indent=2)
    print(f"Saved: {out_json_path}", flush=True)

    # Format Markdown Tables
    md_lines = [
        "# Phase 7: Mandatory Experiment 6 - Sensitivity Study Results",
        f"**Date/Time:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "**Context:** Scenario B Constrained Capacity (Mean Capacity = 1.549 techs/cycle, 100 slots, H=1000 cycles, 20 Evaluation Seeds)",
        "",
        "---",
        "",
        r"## 1. Sweep 1: Sensitivity to Spare Part Lead Time $L \in \{4, 8, 12, 16\}$ cycles",
        r"*Parameters:* $C_{fail} = \$10,000$, $C_{pm} = \$2,000$, $C_{part} = \$300$, $C_{down} = \$100/\text{cycle}$.",
        r"*Fairness Enforcement:* Naive $\tau^*$ and Oracle $m^*$ re-tuned on the Tuning Pool for each lead time setting.",
        "",
    ]

    for L in [4, 8, 12, 16]:
        cond_data = all_condition_results[f"L{L}_cf10000"]
        eval_data = cond_data["evaluation_results"]
        tau_s = cond_data["tau_star"]
        m_s = cond_data["m_star"]

        md_lines.extend([
            f"### Setting: Lead Time $L = {L}$ cycles (Tuned: Naive $\\tau^*={tau_s}$, Oracle $m^*={m_s}$)",
            "",
            "| Policy | Mean Cost ($) | Std ($) | Failures | Downtime % | Offered Load | Utilization | Diff vs Naive (95% CI) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for pol_name, row in eval_data.items():
            cost_str = f"${row['mean_cost']:,.0f}"
            std_str = f"${row['std_cost']:,.0f}"
            diff_mean = row["diff_vs_naive_mean"]
            ci = row["diff_vs_naive_ci"]
            diff_str = f"${diff_mean:+,.0f} [${ci[0]:+,.0f}, ${ci[1]:+,.0f}]" if "Naive" not in pol_name else "Baseline (Ref)"
            
            p_display = pol_name
            if "Fixed" in pol_name:
                p_display += " *(Infeasible $\\rho > 1$)*"

            md_lines.append(
                f"| {p_display} | {cost_str} | {std_str} | {row['mean_failures']:.2f} | {row['downtime_pct']:.2f}% | {row['offered_load']:.3f} | {row['utilization']:.3f} | {diff_str} |"
            )
        md_lines.append("")

    md_lines.extend([
        "---",
        "",
        r"## 2. Sweep 2: Sensitivity to In-Service Failure Penalty $C_{fail} \in \{\$10k, \$20k, \$40k\}$",
        r"*Parameters:* Spare Lead Time $L = 8$ cycles, $C_{pm} = \$2,000$, $C_{part} = \$300$, $C_{down} = \$100/\text{cycle}$.",
        r"*Fairness Enforcement:* Naive $\tau^*$ and Oracle $m^*$ re-tuned on Tuning Pool; Cost Agent optimizes with setting $C_{fail}$.",
        "",
    ])

    for cf in [10000, 20000, 40000]:
        cond_data = all_condition_results[f"L8_cf{cf}"]
        eval_data = cond_data["evaluation_results"]
        tau_s = cond_data["tau_star"]
        m_s = cond_data["m_star"]

        md_lines.extend([
            f"### Setting: Failure Penalty $C_{{fail}} = \\${cf:,.0f}$ (Tuned: Naive $\\tau^*={tau_s}$, Oracle $m^*={m_s}$)",
            "",
            "| Policy | Mean Cost ($) | Std ($) | Failures | Downtime % | Offered Load | Utilization | Diff vs Naive (95% CI) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for pol_name, row in eval_data.items():
            cost_str = f"${row['mean_cost']:,.0f}"
            std_str = f"${row['std_cost']:,.0f}"
            diff_mean = row["diff_vs_naive_mean"]
            ci = row["diff_vs_naive_ci"]
            diff_str = f"${diff_mean:+,.0f} [${ci[0]:+,.0f}, ${ci[1]:+,.0f}]" if "Naive" not in pol_name else "Baseline (Ref)"
            
            p_display = pol_name
            if "Fixed" in pol_name:
                p_display += " *(Infeasible $\\rho > 1$)*"

            md_lines.append(
                f"| {p_display} | {cost_str} | {std_str} | {row['mean_failures']:.2f} | {row['downtime_pct']:.2f}% | {row['offered_load']:.3f} | {row['utilization']:.3f} | {diff_str} |"
            )
        md_lines.append("")

    # Cost Decomposition Tables
    md_lines.extend([
        "---",
        "",
        "## 3. Cost Decomposition Tables (Preventive, Spare, Downtime, Failure, Total)",
        "",
        "### Sweep 1: Lead Time Variations ($C_{fail}=\\$10k$)",
        "",
        "| Lead Time | Policy | Preventive Overhaul ($) | Spare Parts ($) | Downtime Loss ($) | Failure Penalty ($) | Total Cost ($) | Sum Check |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for L in [4, 8, 12, 16]:
        eval_data = all_condition_results[f"L{L}_cf10000"]["evaluation_results"]
        for pol_name, row in eval_data.items():
            tot = row["preventive_cost"] + row["spare_cost"] + row["downtime_cost"] + row["failure_cost"]
            diff_tot = abs(tot - row["mean_cost"])
            chk = "PASSED" if diff_tot < 0.1 else f"FAIL ({diff_tot})"
            p_display = pol_name
            if "Fixed" in pol_name:
                p_display += " *(Infeasible)*"
            md_lines.append(
                f"| L={L} | {p_display} | ${row['preventive_cost']:,.0f} | ${row['spare_cost']:,.0f} | ${row['downtime_cost']:,.0f} | ${row['failure_cost']:,.0f} | ${row['mean_cost']:,.0f} | {chk} |"
            )

    md_lines.extend([
        "",
        "### Sweep 2: Failure Penalty Variations ($L=8$)",
        "",
        "| Failure Penalty | Policy | Preventive Overhaul ($) | Spare Parts ($) | Downtime Loss ($) | Failure Penalty ($) | Total Cost ($) | Sum Check |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for cf in [10000, 20000, 40000]:
        eval_data = all_condition_results[f"L8_cf{cf}"]["evaluation_results"]
        for pol_name, row in eval_data.items():
            tot = row["preventive_cost"] + row["spare_cost"] + row["downtime_cost"] + row["failure_cost"]
            diff_tot = abs(tot - row["mean_cost"])
            chk = "PASSED" if diff_tot < 0.1 else f"FAIL ({diff_tot})"
            p_display = pol_name
            if "Fixed" in pol_name:
                p_display += " *(Infeasible)*"
            md_lines.append(
                f"| ${cf/1000:,.0f}k | {p_display} | ${row['preventive_cost']:,.0f} | ${row['spare_cost']:,.0f} | ${row['downtime_cost']:,.0f} | ${row['failure_cost']:,.0f} | ${row['mean_cost']:,.0f} | {chk} |"
            )

    out_md_path = PROJECT_ROOT / "outputs" / "phase7_results.md"
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Saved: {out_md_path}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("PHASE 7 SENSITIVITY STUDY EXECUTION COMPLETE!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    run_experiment_6()
