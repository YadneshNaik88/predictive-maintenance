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

# Let's run 1000 cycles with the beautiful dispatch logic:
agent = agents_v1.AgenticPolicy(use_intervals=True, spare_lead_time=8, decision_log_path=None)

H = 1000
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

num_failures = 0
num_preventive = 0
total_downtime = 0
total_spares = 0
total_stalls = 0
preventive_stalls = 0
queue_failures = 0

for t in range(1, H + 1):
    cap_t = int(tech_capacity_arr[t])
    observations = []
    for s in range(N):
        if slot_state[s] in (0, 1, 2):
            u = slot_unit[s]
            c_idx = slot_cycle_idx[s]
            eng_data = trajectories[u]
            obs = EngineObservation(
                slot_id=s,
                unit_number=u,
                current_cycle=t,
                age=int(slot_op_cycles[s]),
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

    # Step B: Priority dispatch
    ongoing_maint = np.where(slot_state == 5)[0]
    free_techs = max(0, cap_t - len(ongoing_maint)) if cap_t < 99999 else 99999

    # 1. Corrective jobs (absolute priority)
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

    # 2. Planned jobs
    if free_techs > 0:
        planned_ready = np.where((slot_state == 2) & (t >= slot_earliest_start))[0]
        if len(planned_ready) > 0:
            D_pm = cfg.preventive_duration
            committed = {}
            for c in range(t, t + D_pm):
                in_maint_mask = (slot_state == 5)
                rem_dur = slot_maint_remaining[in_maint_mask]
                committed[c] = int(np.sum(rem_dur > (c - t)))

            # Partition into active reservation holders vs unreserved/overdue
            res_holders = []
            unreserved = []
            for s in planned_ready:
                p = agent.coordinator.slot_plans.get(s)
                holds_res = bool(p and p.get("reserved") and p.get("planned_start") == t)
                if holds_res:
                    res_holders.append(s)
                else:
                    unreserved.append(s)

            if len(res_holders) > 1:
                res_holders = sorted(res_holders, key=lambda s: (slot_priority[s] if slot_priority[s] > 0 else 999.0, slot_request_time[s]))
            if len(unreserved) > 1:
                unreserved = sorted(unreserved, key=lambda s: (slot_priority[s] if slot_priority[s] > 0 else 999.0, slot_request_time[s]))

            # Dispatch active reservation holders first
            for s in res_holders:
                if free_techs <= 0:
                    break
                # Must have capacity >= 1 above ongoing maintenance for [t, t+D)
                can_start = True
                for c in range(t, t + D_pm):
                    cap_c = int(tech_capacity_arr[c]) if c < len(tech_capacity_arr) else 99999
                    if cap_c - committed[c] < 1:
                        can_start = False
                        break
                if can_start:
                    slot_state[s] = 5
                    slot_job_type[s] = 1
                    slot_maint_remaining[s] = D_pm
                    slot_start_maint_time[s] = t
                    num_preventive += 1
                    free_techs -= 1
                    for c in range(t, t + D_pm):
                        committed[c] += 1
                    # release reservation
                    p = agent.coordinator.slot_plans.get(s)
                    if p and p.get("reserved"):
                        agent.coordinator.resource.release(p["planned_start"], D_pm)
                        p["reserved"] = False

            # Dispatch unreserved / overdue jobs second
            for s in unreserved:
                if free_techs <= 0:
                    break
                # Must have capacity >= 1 above ongoing maintenance AND reservations held by others for [t, t+D)
                can_start = True
                for c in range(t, t + D_pm):
                    cap_c = int(tech_capacity_arr[c]) if c < len(tech_capacity_arr) else 99999
                    comm_c = committed[c]
                    # Future reservations held by scheduled jobs
                    other_res = 0
                    for s_other, p_other in agent.coordinator.slot_plans.items():
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
                    num_preventive += 1
                    free_techs -= 1
                    for c in range(t, t + D_pm):
                        committed[c] += 1
                    p = agent.coordinator.slot_plans.get(s)
                    if p and p.get("reserved"):
                        agent.coordinator.resource.release(p["planned_start"], D_pm)
                        p["reserved"] = False

    # Step C
    all_in_maint = np.where(slot_state == 5)[0]
    if len(all_in_maint) > 1:
        sort_keys = np.lexsort((slot_start_maint_time[all_in_maint], -slot_job_type[all_in_maint]))
        all_in_maint = all_in_maint[sort_keys]

    num_to_work = min(len(all_in_maint), cap_t) if cap_t < 99999 else len(all_in_maint)
    active_worked = all_in_maint[:num_to_work]
    stalled_this_cycle = len(all_in_maint) - len(active_worked)
    total_stalls += stalled_this_cycle
    if stalled_this_cycle > 0:
        stalled_jobs = all_in_maint[num_to_work:]
        preventive_stalls += int(np.sum(slot_job_type[stalled_jobs] == 1))

    for s in all_in_maint:
        total_downtime += 1
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
        if st in (3, 4):
            total_downtime += 1
        elif st in (0, 1, 2):
            slot_op_cycles[s] += 1
            u = slot_unit[s]
            c_idx = slot_cycle_idx[s]
            engine_data = trajectories[u]
            max_c = engine_data["max_cycles"]
            c_num = c_idx + 1
            if c_num >= max_c:
                num_failures += 1
                if st == 2:
                    queue_failures += 1
                if hasattr(agent, "on_engine_failure"):
                    agent.on_engine_failure(s)
                if not slot_order_placed[s]:
                    slot_order_placed[s] = True
                    slot_order_cycle[s] = t
                    total_spares += 1
                    slot_part_arrival[s] = t + cfg.spare_lead_time
                    slot_earliest_start[s] = slot_part_arrival[s]
                    slot_state[s] = 3
                else:
                    slot_state[s] = 4 if st == 2 else 3
            else:
                slot_cycle_idx[s] += 1
                if st == 0:
                    pred_r = float(engine_data["pred_rul"][c_idx])
                    low_r = float(engine_data["lower"][c_idx])
                    up_r = float(engine_data["upper"][c_idx])
                    true_r = float(engine_data["true_rul_raw"][c_idx])
                    if agent.decide(s, t, slot_op_cycles[s], pred_r, low_r, up_r, true_r, u, c_num, False):
                        slot_order_placed[s] = True
                        slot_order_cycle[s] = t
                        total_spares += 1
                        slot_part_arrival[s] = t + cfg.spare_lead_time
                        slot_earliest_start[s] = agent.get_earliest_start(s, slot_part_arrival[s])
                        slot_priority[s] = agent.get_priority(s, 0.0)
                        slot_state[s] = 1

total_cost = (
    num_preventive * cfg.cost_preventive
    + num_failures * cfg.cost_corrective
    + total_downtime * cfg.cost_production_loss
    + total_spares * cfg.cost_spare
)
print("=== RESULTS WITH PRIORITY DISPATCH FIX ===")
print(f"Total Cost: ${total_cost:,.2f}")
print(f"Failures: {num_failures}")
print(f"Preventive Actions: {num_preventive}")
print(f"Total Stalls: {total_stalls}")
print(f"Preventive Stalls: {preventive_stalls}")
print(f"Queue Failures: {queue_failures}")
