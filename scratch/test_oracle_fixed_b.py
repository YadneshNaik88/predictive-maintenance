import sys
from pathlib import Path
import numpy as np
import pandas as pd

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
from src.policies import DiagnosticOraclePolicy, FixedIntervalPolicy

trajectories = load_engine_trajectories()
tuning_pool, eval_pool = get_tuning_eval_pools(seed=42)
seeds = list(range(42, 62))

probs_b = (0.1106, 0.3594, 0.3894, 0.1406)
cap_b = generate_scenario_b_capacity(seed=42, horizon=1000, values=(0, 1, 2, 3), probabilities=probs_b)
cfg_b = SimulationConfig(num_slots=100, horizon=1000, num_seeds=20, technicians_capacity_array=cap_b, staggered_start=True)
sim_b = FleetSimulator(config=cfg_b, trajectories=trajectories)

print("--- Testing Oracle margin m in Scenario B (p=0.52) ---")
for m in [1, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 50]:
    sim_res = [sim_b.simulate(DiagnosticOraclePolicy(lead_time=8, margin=m), tuning_pool, seed=s) for s in seeds]
    cost = np.mean([r["total_cost"] for r in sim_res])
    fails = np.mean([r["number_of_failures"] for r in sim_res])
    prev = np.mean([r["number_of_preventive_actions"] for r in sim_res])
    dt = np.mean([r["total_downtime_cycles"] for r in sim_res])
    qwait = np.mean([r["mean_queue_wait"] for r in sim_res])
    print(f"m={m:2d} | Cost=${cost:10,.0f} | Fails={fails:5.1f} | Prev={prev:5.1f} | DT={dt:6.0f}c | QWait={qwait:5.2f}c")

print("\n--- Testing Fixed Interval T in Scenario B (p=0.52) ---")
for T in [100, 110, 120, 125, 130, 140, 150, 160, 175]:
    sim_res = [sim_b.simulate(FixedIntervalPolicy(T, lead_time=8), tuning_pool, seed=s) for s in seeds]
    cost = np.mean([r["total_cost"] for r in sim_res])
    fails = np.mean([r["number_of_failures"] for r in sim_res])
    prev = np.mean([r["number_of_preventive_actions"] for r in sim_res])
    dt = np.mean([r["total_downtime_cycles"] for r in sim_res])
    qwait = np.mean([r["mean_queue_wait"] for r in sim_res])
    print(f"T={T:3d} | Cost=${cost:10,.0f} | Fails={fails:5.1f} | Prev={prev:5.1f} | DT={dt:6.0f}c | QWait={qwait:5.2f}c")

