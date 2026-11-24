import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools
from src.agents import AgenticPolicy

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
cfg = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario="scenario_b")

agent = AgenticPolicy(use_intervals=True, spare_lead_time=8, decision_log_path=None)

# Let's inspect Slot 55 decisions and history up to cycle 142
from scratch.trace_140_155 import agent as tr_agent
for r in tr_agent.coordinator.decision_records:
    if r.get("slot_id") == 55 or r.get("engine") == f"E-{tr_agent.coordinator.slot_plans.get(55, {}).get('unit_number', -1):03d}":
        print(f"Cycle {r['cycle']}: {r['engine']}, chosen={r['chosen_cycle']}, reason={r['reason']}")
