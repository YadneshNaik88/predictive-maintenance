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

H = 105
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
    agent.on_renewal(s)

tech_capacity_arr = cfg.technicians_capacity_array
agent.init_simulation(H, tech_capacity_arr)

for t in range(1, 105):
    cap_t = int(tech_capacity_arr[t])
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
    
    if t in [99, 100, 101]:
        obs21 = [o for o in observations if o.slot_id == 21][0]
        print(f"\n--- At Cycle {t}: Slot 21 (Unit {obs21.unit_number}) ---")
        print(f"  state={slot_state[21]}, age={obs21.age}, pred_rul={obs21.pred_rul:.1f}, lower={obs21.lower_rul:.1f}")
        prior_p = agent.coordinator.slot_plans.get(21)
        print(f"  Prior plan before planning pass: {prior_p}")

    agent.before_cycle(current_cycle=t, observations=observations)

    if t in [99, 100, 101]:
        new_p = agent.coordinator.slot_plans.get(21)
        print(f"  New plan after planning pass: {new_p}")
        # Let's inspect the candidate search
        obs21 = [o for o in observations if o.slot_id == 21][0]
        max_s = int(np.floor(obs21.pred_rul))
        cands = np.arange(0, max_s + 1)
        mask, diags = agent.coordinator.resource.filter_candidates(obs21, cands)
        print(f"  Filter candidates ({len(cands)} cands): feasible={np.sum(mask)}")
        for d in diags:
            if d['feasible']:
                print(f"    Feasible cand: start_cycle={d['start_cycle']}, offset={d['offset_s']}")

    for s in range(N):
        slot_earliest_start[s] = agent.get_earliest_start(s, default=slot_earliest_start[s])
        slot_priority[s] = agent.get_priority(s, default=slot_priority[s])

    # Step A
    for s in np.where((slot_state == 1) & (t >= slot_part_arrival))[0]:
        slot_state[s] = 2
        slot_request_time[s] = slot_part_arrival[s]
    for s in np.where((slot_state == 3) & (t >= slot_part_arrival))[0]:
        slot_state[s] = 4
        slot_request_time[s] = slot_part_arrival[s]

    # Step B
    ongoing_maint = np.where(slot_state == 5)[0]
    free_techs = max(0, cap_t - len(ongoing_maint))

    failed_ready = np.where(slot_state == 4)[0]
    to_start_failed = failed_ready[:free_techs]
    for s in to_start_failed:
        slot_state[s] = 5
        slot_job_type[s] = 2
        slot_maint_remaining[s] = cfg.corrective_duration
        slot_start_maint_time[s] = t
        free_techs -= 1

    if free_techs > 0:
        planned_ready = np.where((slot_state == 2) & (t >= slot_earliest_start))[0]
        if len(planned_ready) > 0:
            D_pm = cfg.preventive_duration
            committed = {}
            for c in range(t, t + D_pm):
                in_maint_mask = (slot_state == 5)
                committed[c] = int(np.sum(slot_maint_remaining[in_maint_mask] > (c - t)))

            res_holders = [s for s in planned_ready if agent.holds_active_reservation(s, t)]
            unreserved = [s for s in planned_ready if not agent.holds_active_reservation(s, t)]
            candidate_order = res_holders + unreserved
            for s in candidate_order:
                if free_techs <= 0:
                    break
                can_start = True
                for c in range(t, t + D_pm):
                    cap_c = int(tech_capacity_arr[c]) if c < len(tech_capacity_arr) else 99999
                    comm_c = committed[c]
                    other_res_c = 0
                    for s_oth, p in agent.coordinator.slot_plans.items():
                        if s_oth != s and p.get('reserved') and p.get('planned_start', 0) > t:
                            pl_st = p.get('planned_start')
                            if pl_st <= c < pl_st + 2:
                                other_res_c += 1
                    if cap_c - comm_c - other_res_c < 1:
                        can_start = False
                        break
                if can_start:
                    slot_state[s] = 5
                    slot_job_type[s] = 1
                    slot_maint_remaining[s] = D_pm
                    slot_start_maint_time[s] = t
                    free_techs -= 1
                    for c in range(t, t + D_pm):
                        committed[c] += 1
                    agent.on_job_start(s, t)

    # Step C
    all_in_maint = np.where(slot_state == 5)[0]
    active_worked = all_in_maint[:min(len(all_in_maint), cap_t)]
    for s in active_worked:
        slot_maint_remaining[s] -= 1
        if slot_maint_remaining[s] == 0:
            u_next = int(rng.choice(eval_pool))
            slot_unit[s] = u_next
            slot_cycle_idx[s] = 0
            slot_op_cycles[s] = 0
            slot_state[s] = 0
            slot_order_placed[s] = False
            slot_earliest_start[s] = 0
            agent.on_renewal(s)

    # Step D
    for s in range(N):
        st = slot_state[s]
        if st in (0, 1, 2):
            slot_op_cycles[s] += 1
            u = slot_unit[s]
            c_idx = slot_cycle_idx[s]
            engine_data = trajectories[u]
            max_c = engine_data["max_cycles"]
            c_num = c_idx + 1
            if c_num >= max_c:
                pass
            else:
                slot_cycle_idx[s] += 1
                if st == 0:
                    pred_r = float(engine_data["pred_rul"][c_idx])
                    low_r = float(engine_data["lower"][c_idx])
                    up_r = float(engine_data["upper"][c_idx])
                    true_r = float(engine_data["true_rul_raw"][c_idx])
                    if agent.decide(s, t, slot_op_cycles[s], pred_r, low_r, up_r, true_r, u, c_num, False):
                        slot_order_placed[s] = True
                        slot_part_arrival[s] = t + cfg.spare_lead_time
                        slot_earliest_start[s] = agent.get_earliest_start(s, slot_part_arrival[s])
                        slot_priority[s] = agent.get_priority(s, 0.0)
                        slot_state[s] = 1
