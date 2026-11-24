import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util
spec = importlib.util.spec_from_file_location("clean_v1", "scratch/clean_agents_v1.py")
clean_v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean_v1)

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
cfg = SimulationConfig.from_yaml("config/sim_params.yaml", scenario="scenario_b")

# Let's run clean_v1 under FleetSimulator
sim = FleetSimulator(config=cfg, trajectories=trajectories)
agent = clean_v1.AgenticPolicy(use_intervals=True, spare_lead_time=8, decision_log_path=None)
agent.is_agentic = False
res = sim.simulate(agent, eval_pool, seed=42)
print("V1 total_stalled_cycles:", res["total_stalled_cycles"])
print("V1 failures:", res["number_of_failures"])
print("V1 num_preventive:", res["number_of_preventive_actions"])
