import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("C:/Users/naiky/OneDrive/Desktop/Projects/predictive-maintenance")
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig, get_tuning_eval_pools, load_engine_trajectories
from src.agents import AgenticPolicy, EngineObservation

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(42)

cfg_b = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario="scenario_b")

# Auditor for Agent-Interval: count stalled jobs and queue failures
class StallAuditor(FleetSimulator):
    def audit_policy(self, policy, seeds):
        H = self.config.horizon
        N = self.config.num_slots
        tech_capacity_arr = self.config.technicians_capacity_array

        total_stalled_events = []
        total_fail_in_q = []

        for seed in seeds:
            policy.reset()
            rng = np.random.default_rng(seed)
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
                if self.config.staggered_start:
                    max_c = self.trajectories[u]["max_cycles"]
                    age = int(rng.integers(0, max_c // 2 + 1))
                    slot_cycle_idx[s] = age
                    slot_op_cycles[s] = age
                else:
                    slot_cycle_idx[s] = 0
                    slot_op_cycles[s] = 0
                slot_state[s] = 0
                slot_order_placed[s] = False
                policy.on_renewal(s)

            policy.init_simulation(total_horizon=H, capacity_roster=tech_capacity_arr)

            seed_stalls = 0
            seed_fail_in_q = 0

            for t in range(1, H + 1):
                cap_t = int(tech_capacity_arr[t])
                observations = []
                for s in range(N):
                    if slot_state[s] in (0, 1, 2):
                        u = slot_unit[s]
                        c_idx = slot_cycle_idx[s]
                        eng = self.trajectories[u]
                        obs = EngineObservation(
                            slot_id=s, unit_number=u, current_cycle=t,
                            age=int(slot_op_cycles[s]),
                            pred_rul=float(eng["pred_rul"][c_idx]),
                            lower_rul=float(eng["lower"][c_idx]),
                            upper_rul=float(eng["upper"][c_idx]),
                            order_already_placed=bool(slot_order_placed[s]),
                            part_arrival_cycle=int(slot_part_arrival[s]) if slot_order_placed[s] else None,
                            planned_start_cycle=int(slot_earliest_start[s]) if slot_earliest_start[s] > 0 else None,
                        )
                        observations.append(obs)
                policy.before_cycle(current_cycle=t, observations=observations)
                for s in range(N):
                    slot_earliest_start[s] = policy.get_earliest_start(s, default=slot_earliest_start[s])
                    slot_priority[s] = policy.get_priority(s, default=slot_priority[s])

                # Part arrivals
                for s in np.where((slot_state == 1) & (t >= slot_part_arrival))[0]:
                    slot_state[s] = 2
                    slot_request_time[s] = slot_part_arrival[s]
                for s in np.where((slot_state == 3) & (t >= slot_part_arrival))[0]:
                    slot_state[s] = 4
                    slot_request_time[s] = slot_part_arrival[s]

                # Dispatch
                ongoing_maint = np.where(slot_state == 5)[0]
                free_techs = max(0, cap_t - len(ongoing_maint)) if cap_t < 99999 else 99999

                failed_ready = np.where(slot_state == 4)[0]
                planned_ready = np.where((slot_state == 2) & (t >= slot_earliest_start))[0]

                if len(failed_ready) > 1:
                    failed_ready = failed_ready[np.argsort(slot_request_time[failed_ready])]
                if len(planned_ready) > 1:
                    if np.any(slot_priority[planned_ready] > 0.0):
                        sort_keys = np.lexsort((slot_request_time[planned_ready], slot_priority[planned_ready]))
                        planned_ready = planned_ready[sort_keys]
                    else:
                        planned_ready = planned_ready[np.argsort(slot_request_time[planned_ready])]

                to_start_failed = failed_ready[:free_techs]
                for s in to_start_failed:
                    slot_state[s] = 5
                    slot_job_type[s] = 2
                    slot_maint_remaining[s] = self.config.corrective_duration
                    slot_start_maint_time[s] = t
                    free_techs -= 1

                if free_techs > 0:
                    to_start_planned = planned_ready[:free_techs]
                    for s in to_start_planned:
                        slot_state[s] = 5
                        slot_job_type[s] = 1
                        slot_maint_remaining[s] = self.config.preventive_duration
                        slot_start_maint_time[s] = t
                        free_techs -= 1

                # Work
                all_in_maint = np.where(slot_state == 5)[0]
                num_to_work = min(len(all_in_maint), cap_t) if cap_t < 99999 else len(all_in_maint)
                active_worked = all_in_maint[:num_to_work]
                stalled = [s for s in all_in_maint if s not in active_worked]
                if len(stalled) > 0:
                    seed_stalls += len(stalled)

                for s in active_worked:
                    slot_maint_remaining[s] -= 1
                    if slot_maint_remaining[s] == 0:
                        u_next = int(rng.choice(eval_pool))
                        slot_unit[s] = u_next
                        slot_cycle_idx[s] = 0
                        slot_op_cycles[s] = 0
                        slot_state[s] = 0
                        slot_order_placed[s] = False
                        policy.on_renewal(s)

                # Operations
                for s in range(N):
                    st = slot_state[s]
                    if st in (0, 1, 2):
                        slot_op_cycles[s] += 1
                        u = slot_unit[s]
                        c_idx = slot_cycle_idx[s]
                        eng = self.trajectories[u]
                        max_c = eng["max_cycles"]
                        c_num = c_idx + 1
                        if c_num >= max_c:
                            if st == 2:
                                seed_fail_in_q += 1
                            policy.on_engine_failure(s)
                            slot_order_placed[s] = True
                            slot_order_cycle[s] = t
                            slot_part_arrival[s] = t + self.config.spare_lead_time
                            slot_earliest_start[s] = slot_part_arrival[s]
                            slot_priority[s] = 0.0
                            slot_state[s] = 3
                            slot_request_time[s] = t
                        else:
                            slot_cycle_idx[s] += 1
                            if st == 0:
                                should_order = policy.decide(
                                    slot_id=s, current_cycle=t, operating_cycles=slot_op_cycles[s],
                                    pred_rul=float(eng["pred_rul"][c_idx]),
                                    lower_rul=float(eng["lower"][c_idx]),
                                    upper_rul=float(eng["upper"][c_idx]),
                                    true_rul=float(eng["true_rul_raw"][c_idx]),
                                    unit_number=u, engine_cycle=c_num,
                                    order_already_placed=bool(slot_order_placed[s]),
                                )
                                if should_order and not slot_order_placed[s]:
                                    slot_order_placed[s] = True
                                    slot_order_cycle[s] = t
                                    slot_part_arrival[s] = t + self.config.spare_lead_time
                                    slot_earliest_start[s] = slot_part_arrival[s]
                                    slot_priority[s] = 0.0
                                    slot_state[s] = 1

            total_stalled_events.append(seed_stalls)
            total_fail_in_q.append(seed_fail_in_q)

        return total_stalled_events, total_fail_in_q

auditor = StallAuditor(config=cfg_b, trajectories=trajectories)

# Audit Agent-Interval
agent_interval = AgenticPolicy(use_intervals=True, spare_lead_time=8, horizon_W=60, replan_k=5, decision_log_path=None)
seeds = [42 + i for i in range(20)]
stalls_interval, fails_interval = auditor.audit_policy(agent_interval, seeds)

print(f"=== AGENT-INTERVAL AUDIT (Scenario B, 20 Seeds) ===")
print(f"Stalled Job Events per Seed: Mean = {np.mean(stalls_interval):.2f} +/- {np.std(stalls_interval):.2f} (Total across 20 seeds = {sum(stalls_interval)})")
print(f"Failures in Queue (state=2): Mean = {np.mean(fails_interval):.2f} +/- {np.std(fails_interval):.2f} (Total across 20 seeds = {sum(fails_interval)})")
