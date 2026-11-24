import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools
from src.agents import AgenticPolicy

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
cfg = SimulationConfig.from_yaml("config/sim_params.yaml", scenario="scenario_b")
sim = FleetSimulator(config=cfg, trajectories=trajectories)
pol = AgenticPolicy(use_intervals=True, decision_log_path=None)
res = sim.simulate(pol, eval_pool, seed=42)
print("Result for Agent-Interval B seed 42:", res["total_cost"], "failures:", res["number_of_failures"], "PM:", res["number_of_preventive_actions"])
