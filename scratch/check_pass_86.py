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

# Trace what happened during planning pass at cycle 86
agent = agents_v1.AgenticPolicy(use_intervals=True, spare_lead_time=8, decision_log_path=None)
H = 86
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
    max_c = trajectories[u]["max_cycles"]
    age = int(rng.integers(0, max_c // 2 + 1))
    slot_cycle_idx[s] = age
    slot_op_cycles[s] = age
    slot_state[s] = 0
    slot_order_placed[s] = False
    agent.on_renewal(s)

tech_capacity_arr = cfg.technicians_capacity_array
agent.init_simulation(H, tech_capacity_arr)

for t in range(1, 87):
    observations = []
    for s in range(N):
        if slot_state[s] in (0, 1, 2):
            u = slot_unit[s]
            c_idx = slot_cycle_idx[s]
            eng_data = trajectories[u]
            obs = EngineObservation(
                slot_id=s, unit_number=u, current_cycle=t, age=int(slot_op_cycles[s]),
                pred_rul=float(eng_data["pred_rul"][c_idx]),
                lower_rul=float(eng_data["lower"][c_idx]),
                upper_rul=float(eng_data["upper"][c_idx]),
                order_already_placed=bool(slot_order_placed[s]),
                part_arrival_cycle=int(slot_part_arrival[s]) if slot_order_placed[s] else None,
                planned_start_cycle=int(slot_earliest_start[s]) if slot_earliest_start[s] > 0 else None,
            )
            observations.append(obs)
    
    if t == 86:
        print("\n=== AT CYCLE 86 PLANNING PASS ===")
        # inspect Slot 42 before planning pass
        print("Slot 42 plan before pass:", agent.coordinator.slot_plans.get(42))
        obs42 = [o for o in observations if o.slot_id == 42][0]
        print(f"Slot 42 obs: age={obs42.age}, pred_rul={obs42.pred_rul:.1f}, lower={obs42.lower_rul:.1f}")

    agent.before_cycle(current_cycle=t, observations=observations)

    if t == 86:
        print("Slot 42 plan after pass:", agent.coordinator.slot_plans.get(42))
        # check latest decision record for Slot 42
        for rec in reversed(agent.coordinator.decision_records):
            if rec["engine"] == f"E-{obs42.unit_number:03d}":
                print("Decision record for Slot 42:")
                print("  chosen_cycle:", rec["chosen_cycle"], "escalated:", rec["escalated"])
                print("  reason:", rec["reason"])
                break

    # Step D decide
    for s in range(N):
        if slot_state[s] == 0:
            u = slot_unit[s]
            c_idx = slot_cycle_idx[s]
            eng = trajectories[u]
            c_num = c_idx + 1
            if c_num < eng["max_cycles"]:
                if agent.decide(s, t, slot_op_cycles[s], float(eng["pred_rul"][c_idx]), float(eng["lower"][c_idx]), float(eng["upper"][c_idx]), float(eng["true_rul_raw"][c_idx]), u, c_num, False):
                    slot_order_placed[s] = True
                    slot_order_cycle[s] = t
                    slot_part_arrival[s] = t + cfg.spare_lead_time
                    slot_earliest_start[s] = agent.get_earliest_start(s, slot_part_arrival[s])
                    slot_priority[s] = agent.get_priority(s, 0.0)
                    slot_state[s] = 1
