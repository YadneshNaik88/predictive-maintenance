import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("c:/Users/naiky/OneDrive/Desktop/Projects/predictive-maintenance").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.policies import RunToFailurePolicy, FixedIntervalPolicy, NaiveThresholdPolicy, DiagnosticOraclePolicy
from src.simulator import load_engine_trajectories, get_tuning_eval_pools, SimulationConfig

trajectories = load_engine_trajectories()
tuning_pool, eval_pool = get_tuning_eval_pools(seed=42)

def simulate_corrected(config, trajectories, policy, engine_pool, seed, horizon=1000):
    H = horizon
    N = config.num_slots
    policy.reset()
    rng = np.random.default_rng(seed)

    slot_unit = np.zeros(N, dtype=int)
    slot_cycle_idx = np.zeros(N, dtype=int)
    slot_op_cycles = np.zeros(N, dtype=int)
    slot_state = np.zeros(N, dtype=int)
    slot_order_placed = np.zeros(N, dtype=bool)
    slot_order_cycle = np.zeros(N, dtype=int)
    slot_part_arrival = np.zeros(N, dtype=int)
    slot_maint_remaining = np.zeros(N, dtype=int)
    slot_job_type = np.zeros(N, dtype=int)
    slot_request_time = np.zeros(N, dtype=int)
    slot_start_maint_time = np.zeros(N, dtype=int)
    slot_history = np.zeros((N, H), dtype=np.int8)

    for s in range(N):
        u = int(rng.choice(engine_pool))
        slot_unit[s] = u
        if config.staggered_start:
            max_c = trajectories[u]["max_cycles"]
            age = int(rng.integers(0, max_c // 2 + 1))
            slot_cycle_idx[s] = age
            slot_op_cycles[s] = age
        else:
            slot_cycle_idx[s] = 0
            slot_op_cycles[s] = 0
        slot_state[s] = 0
        slot_order_placed[s] = False
        policy.on_renewal(s)

    num_failures = 0
    num_preventive = 0
    total_downtime_cycles = 0
    total_operating_cycles = 0
    total_spares_ordered = 0
    completed_engine_lives = 0
    jobs_finished_maintenance = 0
    wasted_rul_list = []
    maintenance_lead_list = []
    queue_wait_list = []
    total_tech_cycles_busy = 0
    total_tech_cycles_available = 0

    if config.technicians_capacity_array is not None:
        tech_capacity_arr = config.technicians_capacity_array
    else:
        tech_capacity_arr = np.full(H + 20, config.num_technicians, dtype=int)

    for t in range(1, H + 1):
        cap_t = int(tech_capacity_arr[t])
        total_tech_cycles_available += cap_t

        # Step A: Check spare part deliveries
        waiting_part_operating = np.where((slot_state == 1) & (t >= slot_part_arrival))[0]
        for s in waiting_part_operating:
            slot_state[s] = 2
            slot_request_time[s] = slot_part_arrival[s]

        waiting_part_failed = np.where((slot_state == 3) & (t >= slot_part_arrival))[0]
        for s in waiting_part_failed:
            slot_state[s] = 4
            slot_request_time[s] = slot_part_arrival[s]

        # Step B: Dispatch Technicians
        # Ongoing jobs from previous cycle
        ongoing_maint = np.where(slot_state == 5)[0]
        free_techs = max(0, cap_t - len(ongoing_maint)) if cap_t < 99999 else 99999

        if free_techs > 0:
            failed_ready = np.where(slot_state == 4)[0]
            planned_ready = np.where(slot_state == 2)[0]
            if len(failed_ready) > 1:
                failed_ready = failed_ready[np.argsort(slot_request_time[failed_ready])]
            if len(planned_ready) > 1:
                planned_ready = planned_ready[np.argsort(slot_request_time[planned_ready])]

            to_start_failed = failed_ready[:free_techs]
            for s in to_start_failed:
                slot_state[s] = 5
                slot_job_type[s] = 2
                slot_maint_remaining[s] = config.corrective_duration
                slot_start_maint_time[s] = t
                lead = t - slot_order_cycle[s]
                q_wait = max(0, t - slot_part_arrival[s])
                maintenance_lead_list.append(lead)
                queue_wait_list.append(q_wait)
                free_techs -= 1

            if free_techs > 0:
                to_start_planned = planned_ready[:free_techs]
                for s in to_start_planned:
                    slot_state[s] = 5
                    slot_job_type[s] = 1
                    slot_maint_remaining[s] = config.preventive_duration
                    slot_start_maint_time[s] = t
                    num_preventive += 1
                    completed_engine_lives += 1
                    u = slot_unit[s]
                    c_idx = slot_cycle_idx[s]
                    wasted_rul_list.append(float(trajectories[u]["true_rul_raw"][c_idx]))
                    lead = t - slot_order_cycle[s]
                    q_wait = max(0, t - slot_part_arrival[s])
                    maintenance_lead_list.append(lead)
                    queue_wait_list.append(q_wait)
                    free_techs -= 1

        # Step C: Progress active maintenance jobs
        all_in_maint = np.where(slot_state == 5)[0]
        # Sort so ongoing jobs get technician priority, then failed over planned
        if len(all_in_maint) > 1:
            # Sort by: corrective (job_type=2) first, then by earliest start_maint_time
            # Priority tuple: (-job_type, start_maint_time)
            sort_keys = np.lexsort((slot_start_maint_time[all_in_maint], -slot_job_type[all_in_maint]))
            all_in_maint = all_in_maint[sort_keys]

        num_to_work = min(len(all_in_maint), cap_t) if cap_t < 99999 else len(all_in_maint)
        active_worked = all_in_maint[:num_to_work]
        
        # Conservation assertion
        if cap_t < 99999:
            assert len(active_worked) <= cap_t, f"Conservation violation at t={t}: {len(active_worked)} > {cap_t}"
        total_tech_cycles_busy += len(active_worked)

        # All slots in maintenance incur downtime
        for s in all_in_maint:
            total_downtime_cycles += 1
            slot_history[s, t - 1] = 2

        # Only worked slots decrement remaining maintenance
        for s in active_worked:
            slot_maint_remaining[s] -= 1
            if slot_maint_remaining[s] == 0:
                jobs_finished_maintenance += 1
                u_next = int(rng.choice(engine_pool))
                slot_unit[s] = u_next
                slot_cycle_idx[s] = 0
                slot_op_cycles[s] = 0
                slot_state[s] = 0
                slot_order_placed[s] = False
                slot_order_cycle[s] = 0
                slot_part_arrival[s] = 0
                slot_job_type[s] = 0
                policy.on_renewal(s)

        # Step D: Process non-maintenance slots
        for s in range(N):
            st = slot_state[s]
            if st == 5:
                continue
            elif st in (3, 4):
                total_downtime_cycles += 1
                slot_history[s, t - 1] = 3
            elif st in (0, 1, 2):
                total_operating_cycles += 1
                slot_op_cycles[s] += 1
                slot_history[s, t - 1] = 0
                u = slot_unit[s]
                c_idx = slot_cycle_idx[s]
                engine_data = trajectories[u]
                max_c = engine_data["max_cycles"]
                c_num = c_idx + 1

                if c_num >= max_c:
                    num_failures += 1
                    completed_engine_lives += 1
                    if not slot_order_placed[s]:
                        slot_order_placed[s] = True
                        slot_order_cycle[s] = t
                        total_spares_ordered += 1
                        slot_part_arrival[s] = t + config.spare_lead_time
                        if config.spare_lead_time == 0:
                            slot_state[s] = 4
                            slot_request_time[s] = t
                        else:
                            slot_state[s] = 3
                            slot_request_time[s] = t
                    else:
                        if st == 2:
                            slot_state[s] = 4
                        else:
                            slot_state[s] = 3
                else:
                    slot_cycle_idx[s] += 1
                    if st == 0:
                        pred_r = float(engine_data["pred_rul"][c_idx])
                        low_r = float(engine_data["lower"][c_idx])
                        up_r = float(engine_data["upper"][c_idx])
                        true_r = float(engine_data["true_rul_raw"][c_idx])

                        if policy.decide(
                            slot_id=s,
                            current_cycle=t,
                            operating_cycles=slot_op_cycles[s],
                            pred_rul=pred_r,
                            lower_rul=low_r,
                            upper_rul=up_r,
                            true_rul=true_r,
                            unit_number=u,
                            engine_cycle=c_num,
                            order_already_placed=slot_order_placed[s],
                        ):
                            slot_order_placed[s] = True
                            slot_order_cycle[s] = t
                            total_spares_ordered += 1
                            slot_part_arrival[s] = t + config.spare_lead_time
                            if config.spare_lead_time == 0:
                                slot_state[s] = 2
                                slot_request_time[s] = t
                            else:
                                slot_state[s] = 1
                                slot_request_time[s] = t

    prev_cost = num_preventive * config.cost_preventive
    corr_cost = num_failures * config.cost_corrective
    prod_loss = total_downtime_cycles * config.cost_production_loss
    spare_cost = total_spares_ordered * config.cost_spare
    total_cost = prev_cost + corr_cost + prod_loss + spare_cost

    cost_per_op = total_cost / total_operating_cycles if total_operating_cycles > 0 else 0.0
    avg_lead = float(np.mean(maintenance_lead_list)) if maintenance_lead_list else 0.0
    mean_q_wait = float(np.mean(queue_wait_list)) if queue_wait_list else 0.0
    mean_wasted = float(np.mean(wasted_rul_list)) if wasted_rul_list else 0.0
    utilization = total_tech_cycles_busy / total_tech_cycles_available if total_tech_cycles_available > 0 else 0.0
    demanded_tech_cycles = (num_preventive * config.preventive_duration) + (num_failures * config.corrective_duration)
    offered_load = demanded_tech_cycles / total_tech_cycles_available if total_tech_cycles_available > 0 else 0.0
    unused_parts = int(np.sum(np.isin(slot_state, [1, 2, 3, 4])))
    
    # Cost floor definition
    cost_floor = (num_preventive + num_failures) * (config.cost_preventive + config.cost_spare) + total_downtime_cycles * config.cost_production_loss
    pct_above_floor = (total_cost - cost_floor) / cost_floor * 100.0 if cost_floor > 0 else 0.0

    return {
        "policy": policy.name,
        "seed": seed,
        "total_cost": total_cost,
        "preventive_cost": prev_cost,
        "corrective_cost": corr_cost,
        "production_loss": prod_loss,
        "spare_cost": spare_cost,
        "cost_floor": cost_floor,
        "pct_above_floor": pct_above_floor,
        "number_of_failures": num_failures,
        "number_of_preventive_actions": num_preventive,
        "completed_engine_lives": completed_engine_lives,
        "jobs_completed": jobs_finished_maintenance,
        "total_downtime_cycles": total_downtime_cycles,
        "total_operating_cycles": total_operating_cycles,
        "total_spares_ordered": total_spares_ordered,
        "unused_spares": unused_parts,
        "cost_per_operating_cycle": cost_per_op,
        "average_maintenance_lead": avg_lead,
        "mean_queue_wait": mean_q_wait,
        "mean_wasted_rul": mean_wasted,
        "utilization": utilization,
        "offered_load": offered_load,
        "total_tech_busy": total_tech_cycles_busy,
        "total_tech_available": total_tech_cycles_available,
        "demanded_tech_cycles": demanded_tech_cycles,
        "slot_history": slot_history,
    }
