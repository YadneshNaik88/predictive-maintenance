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
from src.policies import NaiveThresholdPolicy

# Load trajectories and pools
trajectories = load_engine_trajectories()
tuning_pool, eval_pool = get_tuning_eval_pools(seed=42)

seeds = list(range(42, 62))
tau_grid = [8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80]

print(f"Tuning pool: {len(tuning_pool)} engines. Tau grid: {tau_grid}")

def evaluate_distribution(values, probs):
    # generate capacity schedule
    cap = generate_scenario_b_capacity(seed=42, horizon=1000, values=values, probabilities=probs)
    mean_cap = np.mean(cap[:1000])
    
    cfg = SimulationConfig(
        num_slots=100,
        horizon=1000,
        num_seeds=20,
        technicians_capacity_array=cap,
        staggered_start=True,
    )
    sim = FleetSimulator(config=cfg, trajectories=trajectories)
    
    # Tune tau on tuning pool
    best_tau = None
    best_cost = float("inf")
    best_stats = None
    
    results_per_tau = []
    for tau in tau_grid:
        pol = NaiveThresholdPolicy(tau)
        sim_res = [sim.simulate(pol, tuning_pool, seed=s) for s in seeds]
        mean_cost = np.mean([r["total_cost"] for r in sim_res])
        std_cost = np.std([r["total_cost"] for r in sim_res])
        mean_fails = np.mean([r["number_of_failures"] for r in sim_res])
        mean_dt = np.mean([r["total_downtime_cycles"] for r in sim_res])
        mean_util = np.mean([r["utilization"] for r in sim_res])
        mean_load = np.mean([r["offered_load"] for r in sim_res])
        results_per_tau.append({
            "tau": tau,
            "cost": mean_cost,
            "std": std_cost,
            "fails": mean_fails,
            "dt": mean_dt,
            "util": mean_util,
            "load": mean_load,
        })
    
    df = pd.DataFrame(results_per_tau)
    # Selection rule: lowest mean cost; if within 1 std and fewer failures, prefer it
    min_cost = df["cost"].min()
    cand = df[df["cost"] <= min_cost + df.loc[df["cost"] == min_cost, "std"].values[0]]
    winner_row = cand.sort_values(by=["fails", "cost"]).iloc[0]
    
    # Check validity
    # 1. offered load < 1.0
    # 2. total downtime < 10% of slot cycles (10,000 cycles)
    # 3. winning tau is NOT the largest value in the grid (tau != 80)
    is_valid = (
        winner_row["load"] < 1.0 and
        winner_row["dt"] < 10000.0 and
        winner_row["tau"] < 80
    )
    return {
        "mean_cap": mean_cap,
        "probs": probs,
        "winner_tau": winner_row["tau"],
        "cost": winner_row["cost"],
        "fails": winner_row["fails"],
        "dt": winner_row["dt"],
        "dt_pct": winner_row["dt"] / 1000.0, # out of 100,000 slot-cycles = dt / 1000 %
        "util": winner_row["util"],
        "load": winner_row["load"],
        "is_valid": is_valid,
        "df": df,
    }

# Let's test varying mean capacity by varying probabilities
# e.g., binomial B(3, p) or shifting weights
print("\nTesting Binomial B(3, p) distribution family across p:")
for p in np.linspace(0.35, 0.70, 15):
    from scipy.stats import binom
    probs = tuple(float(binom.pmf(k, 3, p)) for k in range(4))
    res = evaluate_distribution((0, 1, 2, 3), probs)
    print(f"p={p:.3f} | MeanCap={res['mean_cap']:.3f} | WinnerTau={res['winner_tau']} | "
          f"Util={res['util']:.3f} | Load={res['load']:.3f} | DT={res['dt']:.0f}c ({res['dt_pct']:.2f}%) | "
          f"Fails={res['fails']:.2f} | Valid={res['is_valid']}")
