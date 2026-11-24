import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools, EngineObservation
import importlib.util

spec = importlib.util.spec_from_file_location("agents_v1", "scratch/clean_agents_v1.py")
agents_v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agents_v1)

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
cfg = SimulationConfig.from_yaml("config/sim_params.yaml", scenario="scenario_b")

# Let's see who has reservations at cycle 87 in clean_v1
agent = agents_v1.AgenticPolicy(use_intervals=True, spare_lead_time=8, decision_log_path=None)
agent.reset()
agent.init_simulation(150, cfg.technicians_capacity_array)

# Let's inspect how reservations were managed in clean_v1
print("ResourceAgent capacity roster 85..95:", cfg.technicians_capacity_array[85:95])
