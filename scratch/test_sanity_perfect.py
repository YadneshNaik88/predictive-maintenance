import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools
from src.policies import DiagnosticOraclePolicy
from src.agents import AgenticPolicy

# Load real trajectories and create an injected perfect-prediction copy
trajectories_real = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)

# Create injected perfect trajectories: pred_rul = true_rul_raw, lower = true_rul_raw, upper = true_rul_raw
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

# Unlimited technicians and lead time 0
cfg_perfect = SimulationConfig(
    num_slots=100,
    horizon=1000,
    num_technicians=9999,
    spare_lead_time=0,
    staggered_start=True,
)

sim = FleetSimulator(config=cfg_perfect, trajectories=trajectories_perfect)

# Standard Oracle with lead=0, margin=1
oracle = DiagnosticOraclePolicy(lead_time=0, margin=1)
res_oracle = sim.simulate(oracle, eval_pool, seed=42)

# Agent-Interval with lead_time=0 and zero-width residual spread (perfect prediction injected)
agent_interval = AgenticPolicy(use_intervals=True, spare_lead_time=0, horizon_W=60, replan_k=5)
# Set zero-width residual distribution (all residuals = 0.0)
for b in agent_interval.prognostic.residuals_by_bin:
    agent_interval.prognostic.residuals_by_bin[b] = np.array([0.0])

res_agent = sim.simulate(agent_interval, eval_pool, seed=42)

print("=" * 80)
print("TEST 3: SANITY CHECK WITH PERFECT PREDICTIONS & UNLIMITED RESOURCES")
print("=" * 80)
print(f"Oracle Cost:         ${res_oracle['total_cost']:,.0f} | Fails: {res_oracle['number_of_failures']} | Prev: {res_oracle['number_of_preventive_actions']}")
print(f"Agent-Interval Cost: ${res_agent['total_cost']:,.0f} | Fails: {res_agent['number_of_failures']} | Prev: {res_agent['number_of_preventive_actions']}")

diff_pct = abs(res_agent["total_cost"] - res_oracle["total_cost"]) / res_oracle["total_cost"] * 100.0
print(f"Difference:          {diff_pct:.2f}% (Threshold: <= 3.0%)")
assert diff_pct <= 3.0, f"Difference {diff_pct:.2f}% exceeded 3% threshold!"
print("PASSED!")
