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

# Override simulate or run step by step to find first failure
class FailTracer(FleetSimulator):
    pass

sim = FleetSimulator(config=cfg, trajectories=trajectories)
pol = PriorityAgenticPolicy(use_intervals=True)

# Let's inspect cycles 1 to 150 and print when a failure occurs!
H = 150
pol.reset()
rng = np.random.default_rng(42)
N = cfg.num_slots
slot_unit = np.zeros(N, dtype=int)
slot_cycle_idx = np.zeros(N, dtype=int)
slot_op_cycles = np.zeros(N, dtype=int)
slot_state = np.zeros(N, dtype=int)
slot_order_placed = np.zeros(N, dtype=bool)
slot_order_cycle = np.zeros(N, dtype=int)
slot_part_arrival = np.zeros(N, dtype=int)
slot_earliest_start = np.zeros(N, dtype=int)
slot_priority = np.zeros(N, dtype=float)
slot_maint_remaining = np.zeros(N, dtype=int)
slot_job_type = np.zeros(N, dtype=int)
slot_request_time = np.zeros(N, dtype=int)
slot_start_maint_time = np.zeros(N, dtype=int)

for s in range(N):
    u = int(rng.choice(eval_pool))
    slot_unit[s] = u
    max_c = trajectories[u]["max_cycles"]
    age = int(rng.integers(0, max_c // 2 + 1))
    slot_cycle_idx[s] = age
    slot_op_cycles[s] = age
    slot_state[s] = 0
    slot_order_placed[s] = False
    pol.on_renewal(s)

tech_capacity_arr = cfg.technicians_capacity_array
pol.init_simulation(H, tech_capacity_arr)

# Run sim.simulate but with a hook or run it directly
res = sim.simulate(pol, eval_pool, seed=42, horizon=150)
print("150-cycle run: failures =", res["number_of_failures"])
