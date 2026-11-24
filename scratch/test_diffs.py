import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools
import importlib.util

spec = importlib.util.spec_from_file_location("agents_v1", "scratch/clean_agents_v1.py")
agents_v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agents_v1)

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
cfg = SimulationConfig.from_yaml("config/sim_params.yaml", scenario="scenario_b")
sim = FleetSimulator(config=cfg, trajectories=trajectories)

# Base run with clean_v1
pol = agents_v1.AgenticPolicy(use_intervals=True, decision_log_path=None)
pol.is_agentic = False
res = sim.simulate(pol, eval_pool, seed=42)
print("Base clean_v1:", res["total_cost"], "fails:", res["number_of_failures"])

# Test 1: Add on_job_start to clean_v1
class PolWithJobStart(agents_v1.AgenticPolicy):
    def on_job_start(self, slot_id, current_cycle):
        self.coordinator.on_job_start(slot_id, current_cycle)

def coordinator_on_job_start(self, slot_id, current_cycle):
    if slot_id in self.slot_plans:
        plan = self.slot_plans[slot_id]
        if plan.get("reserved", False):
            dur = plan.get("duration", self.resource.preventive_duration)
            st = plan.get("planned_start", current_cycle)
            self.resource.release(st, dur)
            plan["reserved"] = False

agents_v1.CoordinatorAgent.on_job_start = coordinator_on_job_start
pol1 = PolWithJobStart(use_intervals=True, decision_log_path=None)
pol1.is_agentic = False
res1 = sim.simulate(pol1, eval_pool, seed=42)
print("With on_job_start:", res1["total_cost"], "fails:", res1["number_of_failures"])
