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

# Let's inspect cycles 1 to 100
# We want to see EVERY time free_techs > 0 and len(planned_ready) > 0, what happened in Step B!
H = 150
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

for t in range(1, H + 1):
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
    agent.before_cycle(current_cycle=t, observations=observations)
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
    free_techs = max(0, cap_t - len(ongoing_maint)) if cap_t < 99999 else 99999

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
                rem_dur = slot_maint_remaining[in_maint_mask]
                committed[c] = int(np.sum(rem_dur > (c - t)))

            res_holders = []
            unreserved = []
            for s in planned_ready:
                if agent.holds_active_reservation(s, t):
                    res_holders.append(s)
                else:
                    unreserved.append(s)

            candidate_order = res_holders + unreserved
            started_count = 0
            for s in candidate_order:
                if free_techs <= 0:
                    break
                can_start = True
                fail_c = None
                for c in range(t, t + D_pm):
                    cap_c = int(tech_capacity_arr[c])
                    comm_c = committed[c]
                    other_res = agent.get_other_reservations(s, c)
                    avail = cap_c - comm_c - other_res
                    if avail < 1:
                        can_start = False
                        fail_c = (c, cap_c, comm_c, other_res, avail)
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
                    started_count += 1
                else:
                    is_h = s in res_holders
                    print(f"Cycle {t}: Slot {s} (is_res_holder={is_h}) REJECTED because {fail_c}")

    # Step C
    all_in_maint = np.where(slot_state == 5)[0]
    num_to_work = min(len(all_in_maint), cap_t)
    active_worked = all_in_maint[:num_to_work]
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
            eng_data = trajectories[u]
            max_c = eng_data["max_cycles"]
            c_num = c_idx + 1
            slot_cycle_idx[s] += 1
            if st == 0:
                pred_r = float(eng_data["pred_rul"][c_idx])
                low_r = float(eng_data["lower"][c_idx])
                up_r = float(eng_data["upper"][c_idx])
                true_r = float(eng_data["true_rul_raw"][c_idx])
                if agent.decide(s, t, slot_op_cycles[s], pred_r, low_r, up_r, true_r, u, c_num, False):
                    slot_order_placed[s] = True
                    slot_order_cycle[s] = t
                    slot_part_arrival[s] = t + cfg.spare_lead_time
                    slot_earliest_start[s] = agent.get_earliest_start(s, slot_part_arrival[s])
                    slot_priority[s] = agent.get_priority(s, 0.0)
                    slot_state[s] = 1
