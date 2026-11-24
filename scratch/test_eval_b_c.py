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
from src.policies import (
    RunToFailurePolicy,
    FixedIntervalPolicy,
    NaiveThresholdPolicy,
    DiagnosticOraclePolicy,
)

trajectories = load_engine_trajectories()
tuning_pool, eval_pool = get_tuning_eval_pools(seed=42)
seeds = list(range(42, 62))

probs_b = (0.1106, 0.3594, 0.3894, 0.1406)
cap_b = generate_scenario_b_capacity(seed=42, horizon=1000, values=(0, 1, 2, 3), probabilities=probs_b)
cfg_b = SimulationConfig(num_slots=100, horizon=1000, num_seeds=20, technicians_capacity_array=cap_b, staggered_start=True)
sim_b = FleetSimulator(config=cfg_b, trajectories=trajectories)

probs_c = (0.1250, 0.3750, 0.3750, 0.1250)
cap_c = generate_scenario_b_capacity(seed=42, horizon=1000, values=(0, 1, 2, 3), probabilities=probs_c)
cfg_c = SimulationConfig(num_slots=100, horizon=1000, num_seeds=20, technicians_capacity_array=cap_c, staggered_start=True)
sim_c = FleetSimulator(config=cfg_c, trajectories=trajectories)

# Let's test eval pool for B (T=75, tau=60, m=40)
print("Evaluating Scenario B on Held-Out Evaluation Pool (20 seeds):")
pols_b = [
    RunToFailurePolicy(),
    FixedIntervalPolicy(75, lead_time=8),
    NaiveThresholdPolicy(60),
    DiagnosticOraclePolicy(lead_time=8, margin=40),
]
for p in pols_b:
    res = [sim_b.simulate(p, eval_pool, seed=s) for s in seeds]
    c = np.mean([r["total_cost"] for r in res])
    c_std = np.std([r["total_cost"] for r in res])
    fl = np.mean([r["cost_floor"] for r in res])
    fl_pct = np.mean([r["pct_above_floor"] for r in res])
    fa = np.mean([r["number_of_failures"] for r in res])
    pr = np.mean([r["number_of_preventive_actions"] for r in res])
    dt = np.mean([r["total_downtime_cycles"] for r in res])
    ld = np.mean([r["average_maintenance_lead"] for r in res])
    qw = np.mean([r["mean_queue_wait"] for r in res])
    ut = np.mean([r["utilization"] for r in res])
    ol = np.mean([r["offered_load"] for r in res])
    print(f"{p.name:<28}: Cost=${c:10,.0f} +/- ${c_std:6,.0f} | Floor=${fl:10,.0f} ({fl_pct:5.1f}%) | "
          f"Fails={fa:5.2f} | Prev={pr:5.1f} | DT={dt:5.0f}c ({dt/1000:4.2f}%) | "
          f"Lead={ld:5.2f}c | QWait={qw:5.2f}c | Util={ut:.3f} | Load={ol:.3f}")

print("\nEvaluating Scenario C on Held-Out Evaluation Pool (20 seeds):")
pols_c = [
    RunToFailurePolicy(),
    FixedIntervalPolicy(75, lead_time=8),
    NaiveThresholdPolicy(70),
    DiagnosticOraclePolicy(lead_time=8, margin=40),
]
for p in pols_c:
    res = [sim_c.simulate(p, eval_pool, seed=s) for s in seeds]
    c = np.mean([r["total_cost"] for r in res])
    c_std = np.std([r["total_cost"] for r in res])
    fl = np.mean([r["cost_floor"] for r in res])
    fl_pct = np.mean([r["pct_above_floor"] for r in res])
    fa = np.mean([r["number_of_failures"] for r in res])
    pr = np.mean([r["number_of_preventive_actions"] for r in res])
    dt = np.mean([r["total_downtime_cycles"] for r in res])
    ld = np.mean([r["average_maintenance_lead"] for r in res])
    qw = np.mean([r["mean_queue_wait"] for r in res])
    ut = np.mean([r["utilization"] for r in res])
    ol = np.mean([r["offered_load"] for r in res])
    print(f"{p.name:<28}: Cost=${c:10,.0f} +/- ${c_std:6,.0f} | Floor=${fl:10,.0f} ({fl_pct:5.1f}%) | "
          f"Fails={fa:5.2f} | Prev={pr:5.1f} | DT={dt:5.0f}c ({dt/1000:4.2f}%) | "
          f"Lead={ld:5.2f}c | QWait={qw:5.2f}c | Util={ut:.3f} | Load={ol:.3f}")

