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

sim = FleetSimulator(config=cfg, trajectories=trajectories)
pol = PriorityAgenticPolicy(use_intervals=True)

# Run with simulator and print failures
# In simulator, print when failure occurs:
orig_simulate = sim.simulate
# Let's inspect cycles 1 to 150
H = 300
N = cfg.num_slots
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
    pol.on_renewal(s)

tech_capacity_arr = cfg.technicians_capacity_array
pol.init_simulation(H, tech_capacity_arr)

# Let's run Step A, B, C, D using exactly sim.simulate's logic but print on failure
for t in range(1, H + 1):
    cap_t = int(tech_capacity_arr[t])
    observations = []
    for s in range(N):
        if slot_state[s] in (0, 1, 2):
            u = slot_unit[s]
            c_idx = slot_cycle_idx[s]
            eng_data = trajectories[u]
            from src.simulator import EngineObservation
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
    pol.before_cycle(current_cycle=t, observations=observations)
    for s in range(N):
        slot_earliest_start[s] = pol.get_earliest_start(s, default=slot_earliest_start[s])
        slot_priority[s] = pol.get_priority(s, default=slot_priority[s])

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
    if len(failed_ready) > 1:
        failed_ready = failed_ready[np.argsort(slot_request_time[failed_ready])]
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
                p = pol.coordinator.slot_plans.get(s)
                holds_res = bool(p and p.get("reserved") and p.get("planned_start") == t)
                if holds_res:
                    res_holders.append(s)
                else:
                    unreserved.append(s)

            if len(res_holders) > 1:
                res_holders = sorted(res_holders, key=lambda s: (slot_priority[s] if slot_priority[s] > 0 else 999.0, slot_request_time[s]))
            if len(unreserved) > 1:
                unreserved = sorted(unreserved, key=lambda s: (slot_priority[s] if slot_priority[s] > 0 else 999.0, slot_request_time[s]))

            candidate_order = res_holders + unreserved
            for s in candidate_order:
                if free_techs <= 0:
                    break
                can_start = True
                for c in range(t, t + D_pm):
                    cap_c = int(tech_capacity_arr[c]) if c < len(tech_capacity_arr) else 99999
                    comm_c = committed[c]
                    # other future reservations
                    other_res = 0
                    for s_other, p_other in pol.coordinator.slot_plans.items():
                        if s_other != s and p_other.get("reserved"):
                            pl_st = p_other.get("planned_start")
                            dur = p_other.get("duration", D_pm)
                            if pl_st is not None and pl_st > t:
                                if pl_st <= c < pl_st + dur:
                                    other_res += 1
                    if cap_c - comm_c - other_res < 1:
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
                    p = pol.coordinator.slot_plans.get(s)
                    if p and p.get("reserved"):
                        pol.coordinator.resource.release(p["planned_start"], D_pm)
                        p["reserved"] = False

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
            pol.on_renewal(s)

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
            if c_num >= max_c:
                print(f"FAILURE at cycle {t}: Slot {s} (Unit {u}), state={st}")
                print(f"  Slot plan: {pol.coordinator.slot_plans.get(s)}")
                print(f"  part_arr={slot_part_arrival[s]}, earliest_start={slot_earliest_start[s]}")
            else:
                slot_cycle_idx[s] += 1
                if st == 0:
                    pred_r = float(eng_data["pred_rul"][c_idx])
                    low_r = float(eng_data["lower"][c_idx])
                    up_r = float(eng_data["upper"][c_idx])
                    true_r = float(eng_data["true_rul_raw"][c_idx])
                    if pol.decide(s, t, slot_op_cycles[s], pred_r, low_r, up_r, true_r, u, c_num, False):
                        slot_order_placed[s] = True
                        slot_order_cycle[s] = t
                        slot_part_arrival[s] = t + cfg.spare_lead_time
                        slot_earliest_start[s] = pol.get_earliest_start(s, slot_part_arrival[s])
                        slot_priority[s] = pol.get_priority(s, 0.0)
                        slot_state[s] = 1
