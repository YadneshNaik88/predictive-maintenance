import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binom

PROJECT_ROOT = Path(r"c:\Users\naiky\OneDrive\Desktop\Projects\predictive-maintenance")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import (
    FleetSimulator,
    SimulationConfig,
    load_engine_trajectories,
    get_tuning_eval_pools,
    generate_scenario_b_capacity,
)
from src.policies import (
    RunToFailurePolicy,
    FixedIntervalPolicy,
    NaiveThresholdPolicy,
    DiagnosticOraclePolicy,
)

trajectories = load_engine_trajectories()
tuning_pool, eval_pool = get_tuning_eval_pools(seed=42)
seeds = list(range(42, 62))

def test_distribution(name, probs, target_util_range):
    cap = generate_scenario_b_capacity(seed=42, horizon=1000, values=(0, 1, 2, 3), probabilities=probs)
    mean_cap = np.mean(cap[:1000])
    
    cfg = SimulationConfig(
        num_slots=100,
        horizon=1000,
        num_seeds=20,
        technicians_capacity_array=cap,
        staggered_start=True,
    )
    sim = FleetSimulator(config=cfg, trajectories=trajectories)
    
    # 1. Evaluate Naive Threshold over grid [8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80]
    tau_grid = [8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80]
    tau_results = []
    for tau in tau_grid:
        sim_res = [sim.simulate(NaiveThresholdPolicy(tau), tuning_pool, seed=s) for s in seeds]
        tau_results.append({
            "tau": tau,
            "cost": np.mean([r["total_cost"] for r in sim_res]),
            "std": np.std([r["total_cost"] for r in sim_res]),
            "fails": np.mean([r["number_of_failures"] for r in sim_res]),
            "prev": np.mean([r["number_of_preventive_actions"] for r in sim_res]),
            "dt": np.mean([r["total_downtime_cycles"] for r in sim_res]),
            "util": np.mean([r["utilization"] for r in sim_res]),
            "load": np.mean([r["offered_load"] for r in sim_res]),
            "qwait": np.mean([r["mean_queue_wait"] for r in sim_res]),
        })
    df_tau = pd.DataFrame(tau_results)
    min_cost = df_tau["cost"].min()
    cand = df_tau[df_tau["cost"] <= min_cost + df_tau.loc[df_tau["cost"] == min_cost, "std"].values[0]]
    winner_tau = cand.sort_values(by=["fails", "cost"]).iloc[0]
    
    # Check checks
    check_load = winner_tau["load"] < 1.0
    check_dt = winner_tau["dt"] < 10000.0
    check_tau_edge = winner_tau["tau"] < 80
    check_util_target = target_util_range[0] <= winner_tau["util"] <= target_util_range[1]
    
    print(f"\n========================================================")
    print(f"Results for {name}:")
    print(f"Probabilities: {probs} -> Mean Capacity: {mean_cap:.4f}")
    print(f"Tuned Naive Threshold Winner: tau* = {winner_tau['tau']:.0f}")
    print(f"  Cost: ${winner_tau['cost']:,.0f} +/- ${winner_tau['std']:,.0f}")
    print(f"  Failures: {winner_tau['fails']:.2f}")
    print(f"  Preventive Actions: {winner_tau['prev']:.1f}")
    print(f"  Downtime: {winner_tau['dt']:.1f}c ({winner_tau['dt']/1000:.2f}% of slot-cycles)")
    print(f"  Utilization: {winner_tau['util']:.4f} (Target: {target_util_range[0]}-{target_util_range[1]})")
    print(f"  Offered Load: {winner_tau['load']:.4f}")
    print(f"  Mean Queue Wait: {winner_tau['qwait']:.2f} cycles")
    print(f"Checks:")
    print(f"  1. Offered Load < 1.0: {check_load} ({winner_tau['load']:.4f})")
    print(f"  2. Total Downtime < 10%: {check_dt} ({winner_tau['dt']/1000:.2f}%)")
    print(f"  3. Winning tau not on edge (<80): {check_tau_edge} (tau*={winner_tau['tau']:.0f})")
    print(f"  4. Target Utilization Range: {check_util_target} ({winner_tau['util']:.4f})")
    
    # Let's also check Oracle with m in 1..10 to see Oracle headroom!
    oracle_results = []
    for m in range(1, 11):
        sim_res = [sim.simulate(DiagnosticOraclePolicy(lead_time=8, margin=m), tuning_pool, seed=s) for s in seeds]
        oracle_results.append({
            "m": m,
            "cost": np.mean([r["total_cost"] for r in sim_res]),
            "std": np.std([r["total_cost"] for r in sim_res]),
            "fails": np.mean([r["number_of_failures"] for r in sim_res]),
            "util": np.mean([r["utilization"] for r in sim_res]),
            "load": np.mean([r["offered_load"] for r in sim_res]),
            "dt": np.mean([r["total_downtime_cycles"] for r in sim_res]),
        })
    df_orc = pd.DataFrame(oracle_results)
    min_orc = df_orc["cost"].min()
    cand_orc = df_orc[df_orc["cost"] <= min_orc + df_orc.loc[df_orc["cost"] == min_orc, "std"].values[0]]
    winner_orc = cand_orc.sort_values(by=["fails", "cost"]).iloc[0]
    headroom = (winner_tau["cost"] - winner_orc["cost"]) / winner_tau["cost"] * 100.0
    print(f"Diagnostic Oracle Winner on Tuning Pool: m* = {winner_orc['m']:.0f} | Cost: ${winner_orc['cost']:,.0f} | Fails: {winner_orc['fails']:.2f} | Headroom: {headroom:.2f}%")
    
    return winner_tau, winner_orc, df_tau

print("\n--- Evaluating candidates for Scenario B (target 0.75-0.85) ---")
# Test Binomial with p in [0.51, 0.52, 0.525, 0.53, 0.54]
for p in [0.51, 0.52, 0.525, 0.53]:
    probs = tuple(round(float(binom.pmf(k, 3, p)), 4) for k in range(4))
    # Normalize to exactly 1.0
    probs = list(probs)
    probs[-1] = round(1.0 - sum(probs[:-1]), 4)
    probs = tuple(probs)
    test_distribution(f"Scenario B Candidate p={p}", probs, (0.75, 0.85))

print("\n--- Evaluating candidates for Scenario C (target 0.88-0.92) ---")
# Test Binomial with p in [0.49, 0.495, 0.50, 0.505]
for p in [0.49, 0.495, 0.50, 0.505]:
    probs = tuple(round(float(binom.pmf(k, 3, p)), 4) for k in range(4))
    probs = list(probs)
    probs[-1] = round(1.0 - sum(probs[:-1]), 4)
    probs = tuple(probs)
    test_distribution(f"Scenario C Candidate p={p}", probs, (0.88, 0.92))

