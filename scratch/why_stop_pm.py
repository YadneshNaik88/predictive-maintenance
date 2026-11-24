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

# Let's inspect cycles 140 to 160
# Why did preventive actions stop?
# Let's see: are orders being placed?
# Let's check how many spares were ordered!
res150 = sim.simulate(agent, eval_pool, seed=42, horizon=150)
print("Spares ordered at 150:", res150["total_spares_ordered"])

agent2 = AgenticPolicy(use_intervals=True, spare_lead_time=8, decision_log_path=None)
res200 = sim.simulate(agent2, eval_pool, seed=42, horizon=200)
print("Spares ordered at 200:", res200["total_spares_ordered"])
print("Preventive actions at 200:", res200["number_of_preventive_actions"])
print("Failures at 200:", res200["number_of_failures"])
