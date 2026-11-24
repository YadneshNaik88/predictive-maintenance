import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools, EngineObservation
from src.agents import AgenticPolicy

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
cfg = SimulationConfig.from_yaml("config/sim_params.yaml", scenario="scenario_b")

# Let's inspect the exact Phase 6 code!
# Did Phase 6 use AgenticPolicy or something else?
# Let's check results_final_v1.json!
import json
with open("outputs/results_final_v1.json") as f:
    v1 = json.load(f)

v1_b_int = v1["scenarios"]["scenario_b"]["policies"]["Agent-Interval"]
print("v1 Agent-Interval B mean cost:", v1_b_int["mean_cost"])
print("v1 Agent-Interval B mean failures:", v1_b_int["mean_failures"])
print("v1 Agent-Interval B raw failures:", v1_b_int["raw_failures"])
print("v1 Agent-Interval B raw costs:", v1_b_int["raw_costs"][:5])
