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

class StallTracerSimulator(FleetSimulator):
    def simulate(self, policy, engine_pool, seed, horizon=150):
        # We can subclass or we can just run simulate and check history
        return super().simulate(policy, engine_pool, seed, horizon=horizon)

# Let's run and print from Step C by monkey-patching or printing
agent = AgenticPolicy(use_intervals=True, spare_lead_time=8, decision_log_path=None)
sim = FleetSimulator(config=cfg, trajectories=trajectories)

# Let's inspect where stalls come from
res = sim.simulate(agent, eval_pool, seed=42, horizon=150)
print("Finished. Let's find why there were 15 stalls.")
