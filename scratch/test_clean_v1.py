import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools
# Import clean_agents_v1
import importlib.util
spec = importlib.util.spec_from_file_location("agents_v1", "scratch/clean_agents_v1.py")
agents_v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agents_v1)

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
cfg = SimulationConfig.from_yaml("config/sim_params.yaml", scenario="scenario_b")
sim = FleetSimulator(config=cfg, trajectories=trajectories)

pol = agents_v1.AgenticPolicy(use_intervals=True, decision_log_path=None)
pol.is_agentic = False # using original dispatch in simulator
res = sim.simulate(pol, eval_pool, seed=42)
print("Cost:", res["total_cost"], "failures:", res["number_of_failures"], "PM:", res["number_of_preventive_actions"])
