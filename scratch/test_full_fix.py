"""
scratch/test_full_fix.py

Test full simulation run of the reservation fix on Scenario B (Seed 42)
and compare with results_final_v1.json to ensure:
1. Baseline policies are bit-for-bit unchanged.
2. Agent-Interval runs cleanly with no capacity assertion violations.
3. Compare stalls and queue failures.
"""

import sys
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("C:/Users/naiky/OneDrive/Desktop/Projects/predictive-maintenance")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.policies import (
    RunToFailurePolicy,
    FixedIntervalPolicy,
    NaiveThresholdPolicy,
    DiagnosticOraclePolicy,
)
from src.simulator import (
    FleetSimulator,
    SimulationConfig,
    load_engine_trajectories,
    get_tuning_eval_pools,
)
from src.agents import AgenticPolicy


def test_fix():
    print("Testing fix on Scenario B Seed 42...")
    trajectories = load_engine_trajectories()
    _, eval_pool = get_tuning_eval_pools(seed=42)

    cfg = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario="scenario_b")
    sim = FleetSimulator(config=cfg, trajectories=trajectories)

    # 1. Test Naive Threshold (Seed 42)
    pol_naive = NaiveThresholdPolicy(60)
    res_naive = sim.simulate(pol_naive, eval_pool, seed=42)
    print(f"Naive Threshold Cost: ${res_naive['total_cost']:,} (Failures: {res_naive['number_of_failures']})")

    # Load v1 json to verify Naive cost for seed 42
    with open("outputs/results_final_v1.json", "r") as f:
        v1 = json.load(f)
    v1_naive_cost_seed42 = v1["scenarios"]["scenario_b"]["policies"]["Naive Threshold (tau=60)"]["raw_costs"][0]
    print(f"v1 Naive Cost (Seed 42): ${v1_naive_cost_seed42:,}")
    assert res_naive["total_cost"] == v1_naive_cost_seed42, "Naive Threshold must be bit-for-bit identical!"

    # 2. Test Agent-Interval (Seed 42)
    pol_agent = AgenticPolicy(use_intervals=True, decision_log_path=None)
    res_agent = sim.simulate(pol_agent, eval_pool, seed=42)
    print(f"Agent-Interval Cost: ${res_agent['total_cost']:,} (Failures: {res_agent['number_of_failures']})")
    print(f"Agent-Interval Stalls: {res_agent.get('total_stalled_cycles', 'N/A')}")
    print(f"Agent-Interval Queue Failures: {res_agent.get('queue_failures', 'N/A')}")


if __name__ == "__main__":
    test_fix()
