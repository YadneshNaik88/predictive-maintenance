import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools
from scratch.test_priority_agent import PriorityAgenticPolicy

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
cfg = SimulationConfig.from_yaml("config/sim_params.yaml", scenario="scenario_b")

# Override get_other_reservations on PriorityCoordinator
def custom_get_other_reservations(self, slot_id, cycle, current_cycle=None):
    # If called from simulator with only (slot_id, cycle)
    # count reservations held by jobs with planned_start > cycle - duration?
    # No, count jobs whose planned start is strictly greater than the cycle being dispatched!
    # Wait, how does coordinator know current_cycle?
    # self.current_cycle!
    now = getattr(self, "current_cycle", cycle)
    count = 0
    for s, p in self.slot_plans.items():
        if s == slot_id:
            continue
        if not p.get("reserved", False):
            continue
        pl_st = p.get("planned_start")
        dur = p.get("duration", self.resource.preventive_duration)
        if pl_st is not None and pl_st > now:
            if pl_st <= cycle < pl_st + dur:
                count += 1
    return count

from src.agents import CoordinatorAgent
CoordinatorAgent.get_other_reservations = custom_get_other_reservations

# In before_cycle, record current_cycle
orig_before_cycle = PriorityAgenticPolicy.before_cycle
def custom_before_cycle(self, current_cycle, observations):
    self.coordinator.current_cycle = current_cycle
    orig_before_cycle(self, current_cycle, observations)

PriorityAgenticPolicy.before_cycle = custom_before_cycle

sim = FleetSimulator(config=cfg, trajectories=trajectories)
pol = PriorityAgenticPolicy(use_intervals=True)
res = sim.simulate(pol, eval_pool, seed=42)
print("=== FUTURE OVERRIDE RESULTS ===")
print(f"Total Cost: ${res['total_cost']:,.2f}")
print(f"Failures: {res['number_of_failures']}")
print(f"Preventive Actions: {res['number_of_preventive_actions']}")
print(f"Total Stalls: {res.get('total_stalled_cycles', 0)}")
print(f"Queue Failures: {res.get('queue_failures', 0)}")
