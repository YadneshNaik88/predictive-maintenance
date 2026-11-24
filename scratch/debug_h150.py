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
sim = FleetSimulator(config=cfg, trajectories=trajectories)

# Let's inspect Step B behavior by monkey-patching or printing
# Let's run 150 cycles and see why jobs don't start
res = sim.simulate(agent, eval_pool, seed=42, horizon=150)
print("At horizon 150:")
print("Failures:", res["number_of_failures"])
print("Preventive:", res["number_of_preventive_actions"])
print("Stalls:", res["total_stalled_cycles"])
print("Queue failures:", res["queue_failures"])
