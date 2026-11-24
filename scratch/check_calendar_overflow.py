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

agent = AgenticPolicy(use_intervals=True, spare_lead_time=8, decision_log_path=None)
sim = FleetSimulator(config=cfg, trajectories=trajectories)

# Let's run 150 cycles and check after each cycle:
# is reserved_calendar[c] > capacity_roster[c]?
H = 150
agent.reset()
tech_capacity_arr = cfg.technicians_capacity_array
agent.init_simulation(H, tech_capacity_arr)

# Let's check test_5 from tests/test_agents.py:
# test_5 checks: violations = np.where(reserved[:min_len] > capacity[:min_len])[0]
# assert len(violations) == 0
print("Reserved calendar initial:", np.sum(agent.resource.reserved_calendar))
