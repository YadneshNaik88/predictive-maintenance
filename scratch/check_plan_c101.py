import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools, EngineObservation
from src.agents import AgenticPolicy

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
cfg = SimulationConfig.from_yaml("config/sim_params.yaml", scenario="scenario_b")
agent = AgenticPolicy(use_intervals=True, spare_lead_time=8, decision_log_path=None)

H = 101
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

for t in range(1, 102):
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
    
    if t == 101:
        # Check to_plan list
        L = agent.coordinator.spare_lead_time
        W = agent.coordinator.horizon_W
        now = t
        cand_obs = [o for o in observations if o.pred_rul <= (L + W) or (o.slot_id in agent.coordinator.slot_plans and agent.coordinator.slot_plans[o.slot_id].get("planned_start", 999) < now)]
        print(f"Cycle 101: cand_obs count = {len(cand_obs)}")
        # Check who is in to_plan
        to_plan = []
        for o in cand_obs:
            s_id = o.slot_id
            if s_id not in agent.coordinator.slot_plans:
                to_plan.append(o)
            else:
                plan = agent.coordinator.slot_plans[s_id]
                planned_start = plan.get("planned_start", now + 999)
                is_overdue = (planned_start < now)
                is_periodic = (now % agent.coordinator.replan_k == 1)
                lower_dropped = (o.lower_rul < (planned_start - now))
                if is_overdue or is_periodic or lower_dropped:
                    to_plan.append(o)
        print(f"Cycle 101: to_plan count = {len(to_plan)}")
        to_plan.sort(key=lambda o: o.lower_rul)
        print(f"Slot 21 rank in to_plan:")
        for idx, o in enumerate(to_plan):
            if o.slot_id == 21:
                print(f"  Slot 21 is index {idx} out of {len(to_plan)}, lower_rul={o.lower_rul}")
        # Print top 10 in to_plan
        print("Top 5 in to_plan:")
        for idx, o in enumerate(to_plan[:5]):
            print(f"  {idx}: Slot {o.slot_id}, lower_rul={o.lower_rul:.1f}, pred_rul={o.pred_rul:.1f}")

    agent.before_cycle(current_cycle=t, observations=observations)

    for s in range(N):
        slot_earliest_start[s] = agent.get_earliest_start(s, default=slot_earliest_start[s])
        slot_priority[s] = agent.get_priority(s, default=slot_priority[s])

    # Step D decide
    for s in range(N):
        if slot_state[s] == 0:
            u = slot_unit[s]
            c_idx = slot_cycle_idx[s]
            eng = trajectories[u]
            if agent.decide(s, t, slot_op_cycles[s], float(eng["pred_rul"][c_idx]), float(eng["lower"][c_idx]), float(eng["upper"][c_idx]), float(eng["true_rul_raw"][c_idx]), u, c_idx + 1, False):
                slot_order_placed[s] = True
                slot_part_arrival[s] = t + cfg.spare_lead_time
                slot_earliest_start[s] = agent.get_earliest_start(s, slot_part_arrival[s])
                slot_priority[s] = agent.get_priority(s, 0.0)
                slot_state[s] = 1
