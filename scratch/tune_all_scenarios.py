import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("c:/Users/naiky/OneDrive/Desktop/Projects/predictive-maintenance").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
scratch_dir = PROJECT_ROOT / "scratch"
sys.path.insert(0, str(scratch_dir))

from test_sim_corrected import simulate_corrected
from src.policies import RunToFailurePolicy, FixedIntervalPolicy, NaiveThresholdPolicy, DiagnosticOraclePolicy
from src.simulator import load_engine_trajectories, get_tuning_eval_pools, SimulationConfig

trajectories = load_engine_trajectories()
tuning_pool, eval_pool = get_tuning_eval_pools(seed=42)
eval_seeds = [42 + i for i in range(20)]

# Scenario C schedule: seed 42, p = [0.08, 0.77, 0.13, 0.02] (mean cap = 1.094)
rng = np.random.default_rng(42)
cap_schedule_c = rng.choice([0, 1, 2, 3], size=1020, p=[0.08, 0.77, 0.13, 0.02])
cfg_c = SimulationConfig(technicians_capacity_array=cap_schedule_c)

print("=== SCENARIO C TUNING POOL ===")
print("--- Fixed Interval Grid ---")
T_candidates = [100, 110, 120, 125, 130, 140, 150, 160, 175]
for T in T_candidates:
    res = [simulate_corrected(cfg_c, trajectories, FixedIntervalPolicy(T), tuning_pool, s) for s in eval_seeds]
    costs = [r['total_cost'] for r in res]
    fails = [r['number_of_failures'] for r in res]
    print(f"  T = {T:3d}: Cost = ${np.mean(costs):10,.0f} +/- ${np.std(costs):7,.0f} | Fails: {np.mean(fails):5.1f}")

print("\n--- Naive Threshold Grid ---")
tau_candidates = [8, 10, 12, 15, 20, 25, 35, 50]
for tau in tau_candidates:
    res = [simulate_corrected(cfg_c, trajectories, NaiveThresholdPolicy(tau), tuning_pool, s) for s in eval_seeds]
    costs = [r['total_cost'] for r in res]
    fails = [r['number_of_failures'] for r in res]
    utils = [r['utilization'] for r in res]
    print(f"  tau = {tau:2d}: Cost = ${np.mean(costs):10,.0f} +/- ${np.std(costs):7,.0f} | Fails: {np.mean(fails):5.1f} | Util: {np.mean(utils):.3f}")

print("\n--- Oracle Margin Tuning for Scenarios A, B, C ---")
cfg_a = SimulationConfig() # constant 4
rng = np.random.default_rng(42)
cap_schedule_b = rng.choice([0, 1, 2, 3], size=1020, p=[0.04, 0.71, 0.23, 0.02])
cfg_b = SimulationConfig(technicians_capacity_array=cap_schedule_b)

for scen_name, cfg in [("Scenario A", cfg_a), ("Scenario B", cfg_b), ("Scenario C", cfg_c)]:
    print(f"\n{scen_name} Oracle Margin Sweep on Tuning Pool:")
    for m in [1, 2, 3, 4, 5, 6, 7, 8]:
        res = [simulate_corrected(cfg, trajectories, DiagnosticOraclePolicy(margin=m), tuning_pool, s) for s in eval_seeds]
        costs = [r['total_cost'] for r in res]
        fails = [r['number_of_failures'] for r in res]
        wasted = [r['mean_wasted_rul'] for r in res]
        print(f"  m = {m}: Cost = ${np.mean(costs):10,.0f} +/- ${np.std(costs):7,.0f} | Fails: {np.mean(fails):5.1f} | Wasted: {np.mean(wasted):.2f}c")

