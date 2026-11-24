import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("C:/Users/naiky/OneDrive/Desktop/Projects/predictive-maintenance")
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig, get_tuning_eval_pools, load_engine_trajectories
from src.policies import FixedIntervalPolicy

trajectories = load_engine_trajectories()
tuning_pool, eval_pool = get_tuning_eval_pools(42)
seeds = [42 + i for i in range(20)]

cfg_b = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario="scenario_b")
sim = FleetSimulator(config=cfg_b, trajectories=trajectories)

policy = FixedIntervalPolicy(75, lead_time=cfg_b.spare_lead_time)

def evaluate_pool(pool, pool_name):
    rows = []
    for s in seeds:
        res = sim.simulate(policy, pool, seed=s)
        # Definition of collapsed: persistent queue backlog / large failures (>5) or cost explosion
        collapsed = res["number_of_failures"] > 5
        rows.append({
            "Pool": pool_name,
            "Seed": s,
            "Cost": res["total_cost"],
            "Failures": res["number_of_failures"],
            "Offered Load": res["offered_load"],
            "Utilization": res["utilization"],
            "Downtime Pct": (res["total_downtime_cycles"] / (100 * 1000)) * 100.0,
            "Collapsed": collapsed,
        })
    return pd.DataFrame(rows)

df_tuning = evaluate_pool(tuning_pool, "Tuning Pool")
df_eval = evaluate_pool(eval_pool, "Evaluation Pool")

print("=== TUNING POOL: FIXED INTERVAL (T=75, Scenario B) ===")
print(df_tuning.to_string(index=False))
print(f"Tuning Summary: Mean Cost = ${df_tuning['Cost'].mean():,.0f} +/- ${df_tuning['Cost'].std(ddof=1):,.0f} | Collapsed Seeds: {df_tuning['Collapsed'].sum()} / 20")

print("\n=== EVALUATION POOL: FIXED INTERVAL (T=75, Scenario B) ===")
print(df_eval.to_string(index=False))
print(f"Eval Summary: Mean Cost = ${df_eval['Cost'].mean():,.0f} +/- ${df_eval['Cost'].std(ddof=1):,.0f} | Collapsed Seeds: {df_eval['Collapsed'].sum()} / 20")
