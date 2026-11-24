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

# Scenario B: Seed 42, p = [0.04, 0.71, 0.23, 0.02] (mean cap = 1.231)
rng = np.random.default_rng(42)
cap_schedule_b = rng.choice([0, 1, 2, 3], size=1020, p=[0.04, 0.71, 0.23, 0.02])
cfg_b = SimulationConfig(technicians_capacity_array=cap_schedule_b)

eval_seeds = [42 + i for i in range(20)]

print("Testing Fixed-Interval(T=125) in Scenario B (Corrected Model):")
res_fixed = [simulate_corrected(cfg_b, trajectories, FixedIntervalPolicy(125), eval_pool, s) for s in eval_seeds]
print(f"  Cost: ${np.mean([r['total_cost'] for r in res_fixed]):,.0f}")
print(f"  Failures: {np.mean([r['number_of_failures'] for r in res_fixed]):.2f}")
print(f"  Prev: {np.mean([r['number_of_preventive_actions'] for r in res_fixed]):.1f}")
print(f"  Queue wait: {np.mean([r['mean_queue_wait'] for r in res_fixed]):.2f} cycles")
print(f"  Utilization: {np.mean([r['utilization'] for r in res_fixed]):.3f}")
print(f"  Offered Load: {np.mean([r['offered_load'] for r in res_fixed]):.3f}")
print(f"  Busy techs: {np.mean([r['total_tech_busy'] for r in res_fixed]):.1f} / Available: {np.mean([r['total_tech_available'] for r in res_fixed]):.1f}")
assert all(r['total_tech_busy'] <= r['total_tech_available'] for r in res_fixed), "Conservation failed!"
print("[OK] Conservation strictly holds on all 20 seeds (Utilization <= 1.0)!")

