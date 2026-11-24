import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools, EngineObservation
import importlib.util
spec = importlib.util.spec_from_file_location('clean_v1', 'scratch/clean_agents_v1.py')
clean_v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean_v1)
agent = clean_v1.AgenticPolicy(use_intervals=True, spare_lead_time=8, decision_log_path=None)

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
cfg = SimulationConfig.from_yaml('config/sim_params.yaml', scenario='scenario_b')

H = 110
N = cfg.num_slots
agent.reset()
rng = np.random.default_rng(42)

slot_unit = np.zeros(N, dtype=int)
slot_cycle_idx = np.zeros(N, dtype=int)
slot_op_cycles = np.zeros(N, dtype=int)
slot_state = np.zeros(N, dtype=int)
slot_order_placed = np.zeros(N, dtype=bool)
slot_order_cycle = np.zeros(N, dtype=int)
slot_part_arrival = np.zeros(N, dtype=int)
slot_earliest_start = np.zeros(N, dtype=int)
slot_priority = np.zeros(N, dtype=float)

for s in range(N):
    u = int(rng.choice(eval_pool))
    slot_unit[s] = u
    max_c = trajectories[u]['max_cycles']
    age = int(rng.integers(0, max_c // 2 + 1))
    slot_cycle_idx[s] = age
    slot_op_cycles[s] = age
    agent.on_renewal(s)

tech_capacity_arr = cfg.technicians_capacity_array
agent.init_simulation(H, tech_capacity_arr)

for t in range(1, 111):
    observations = []
    for s in range(N):
        if slot_state[s] in (0, 1, 2):
            u = slot_unit[s]
            c_idx = slot_cycle_idx[s]
            eng_data = trajectories[u]
            obs = EngineObservation(
                slot_id=s, unit_number=u, current_cycle=t, age=int(slot_op_cycles[s]),
                pred_rul=float(eng_data['pred_rul'][c_idx]),
                lower_rul=float(eng_data['lower'][c_idx]),
                upper_rul=float(eng_data['upper'][c_idx]),
                order_already_placed=bool(slot_order_placed[s]),
                part_arrival_cycle=int(slot_part_arrival[s]) if slot_order_placed[s] else None,
                planned_start_cycle=int(slot_earliest_start[s]) if slot_earliest_start[s] > 0 else None,
            )
            observations.append(obs)
    agent.before_cycle(current_cycle=t, observations=observations)
    if t == 110:
        print('Capacity 110..115:', [int(tech_capacity_arr[c]) for c in range(110, 116)])
        print('Reserved 110..115:', [int(agent.coordinator.resource.reserved_calendar[c]) for c in range(110, 116)])
        for s, p in agent.coordinator.slot_plans.items():
            if p.get('reserved'):
                st = p.get('planned_start')
                if 110 <= st <= 115:
                    u = p['unit_number']
                    print(f"Slot {s} (Unit {u}) reserved at {st}, lower_rul={p['urgency_lower']}")
