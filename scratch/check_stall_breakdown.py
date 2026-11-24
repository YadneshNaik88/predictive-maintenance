import importlib.util
import numpy as np
from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools

spec2 = importlib.util.spec_from_file_location('agents_overdue', 'scratch/test_overdue_mod.py')
agents_overdue = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(agents_overdue)

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
cfg = SimulationConfig.from_yaml('config/sim_params.yaml', scenario='scenario_b')
sim = FleetSimulator(config=cfg, trajectories=trajectories)

# Let's inspect Step C in a modified simulator
class StallTracerSim(FleetSimulator):
    pass

# We can run simulate and see
pol2 = agents_overdue.AgenticPolicy(use_intervals=True, decision_log_path=None)
pol2.is_agentic = False
# In simulate:
res = sim.simulate(pol2, eval_pool, seed=42)
print("Total cost:", res["total_cost"])
print("Failures:", res["number_of_failures"])
