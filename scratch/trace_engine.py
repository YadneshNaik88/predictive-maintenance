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

# Let's run a custom run where we print what happens during Step B
# Let's inspect sim.simulate with printing
class TracePolicy(AgenticPolicy):
    def __init__(self):
        super().__init__(use_intervals=True, spare_lead_time=8, decision_log_path=None)

agent = TracePolicy()
sim = FleetSimulator(config=cfg, trajectories=trajectories)

# Let's inspect why can_start was False in simulator
# Let's print in simulator when can_start is False!
