import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("c:/Users/naiky/OneDrive/Desktop/Projects/predictive-maintenance").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
scratch_dir = PROJECT_ROOT / "scratch"
sys.path.insert(0, str(scratch_dir))

from test_sim_corrected import simulate_corrected
from src.policies import NaiveThresholdPolicy, DiagnosticOraclePolicy
from src.simulator import load_engine_trajectories, get_tuning_eval_pools, SimulationConfig

trajectories = load_engine_trajectories()
tuning_pool, eval_pool = get_tuning_eval_pools(seed=42)
eval_seeds = [42 + i for i in range(20)]

# We want Scenario C: tuned Naive Threshold has utilization in [0.88, 0.92] (target ~0.90)
# Under tau=20, preventive jobs ~495 -> demanded ~990 cycles.
# For util = 0.90, capacity should be ~990 / 0.90 = 1100 cycles (mean cap ~1.10).
# Let's test candidate distributions with seed 42:
candidates = [
    ("C1", [0.08, 0.76, 0.14, 0.02]), # mean ~ 1.10
    ("C2", [0.07, 0.77, 0.14, 0.02]), # mean ~ 1.11
    ("C3", [0.06, 0.78, 0.14, 0.02]), # mean ~ 1.12
    ("C4", [0.06, 0.79, 0.13, 0.02]), # mean ~ 1.11
    ("C5", [0.08, 0.77, 0.13, 0.02]), # mean ~ 1.09
]

for name, p in candidates:
    rng = np.random.default_rng(42)
    cap = rng.choice([0, 1, 2, 3], size=1020, p=p)
    mean_c = np.mean(cap[1:1001])
    cfg = SimulationConfig(technicians_capacity_array=cap)
    
    # Evaluate tau=20 on eval pool across 20 seeds
    res = [simulate_corrected(cfg, trajectories, NaiveThresholdPolicy(20), eval_pool, s) for s in eval_seeds]
    u = np.mean([r['utilization'] for r in res])
    u_std = np.std([r['utilization'] for r in res])
    f = np.mean([r['number_of_failures'] for r in res])
    c = np.mean([r['total_cost'] for r in res])
    qw = np.mean([r['mean_queue_wait'] for r in res])
    print(f"{name} p={p} (mean {mean_c:.3f}): Util = {u:.3f} +/- {u_std:.3f} | Fails: {f:.2f} | Cost: ${c:,.0f} | QWait: {qw:.2f}c")

