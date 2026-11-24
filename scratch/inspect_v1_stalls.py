import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util
spec = importlib.util.spec_from_file_location("clean_v1", "scratch/clean_agents_v1.py")
clean_v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean_v1)

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
cfg = SimulationConfig.from_yaml("config/sim_params.yaml", scenario="scenario_b")

# Let's inspect Step C when running V1
# We can subclass FleetSimulator to record stalls
class StallInspector(FleetSimulator):
    def simulate(self, policy, engine_pool, seed, horizon=None):
        # We will run the standard simulate but print stall events
        H = horizon if horizon is not None else self.config.horizon
        N = self.config.num_slots
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
            u = int(rng.choice(engine_pool))
            slot_unit[s] = u
            if self.config.staggered_start:
                max_c = self.trajectories[u]["max_cycles"]
                age = int(rng.integers(0, max_c // 2 + 1))
                slot_cycle_idx[s] = age
                slot_op_cycles[s] = age
            slot_state[s] = 0
            slot_order_placed[s] = False
            policy.on_renewal(s)

        tech_capacity_arr = self.config.technicians_capacity_array
        if hasattr(policy, "init_simulation"):
            policy.init_simulation(total_horizon=H, capacity_roster=tech_capacity_arr)

        stalls_by_type = {1: 0, 2: 0}
        stall_events = []

        for t in range(1, H + 1):
            cap_t = int(tech_capacity_arr[t])
            # Step A
            for s in np.where((slot_state == 1) & (t >= slot_part_arrival))[0]:
                slot_state[s] = 2
                slot_request_time[s] = slot_part_arrival[s]
            for s in np.where((slot_state == 3) & (t >= slot_part_arrival))[0]:
                slot_state[s] = 4
                slot_request_time[s] = slot_part_arrival[s]

            # Step B (V1)
            ongoing_maint = np.where(slot_state == 5)[0]
            free_techs = max(0, cap_t - len(ongoing_maint)) if cap_t < 99999 else 99999
            failed_ready = np.where(slot_state == 4)[0]
            planned_ready = np.where((slot_state == 2) & (t >= slot_earliest_start))[0]

            if free_techs > 0:
                if len(failed_ready) > 1:
                    failed_ready = failed_ready[np.argsort(slot_request_time[failed_ready])]
                to_start_failed = failed_ready[:free_techs]
                for s in to_start_failed:
                    slot_state[s] = 5
                    slot_job_type[s] = 2
                    slot_maint_remaining[s] = self.config.corrective_duration
                    slot_start_maint_time[s] = t
                    free_techs -= 1

                if free_techs > 0:
                    if len(planned_ready) > 1:
                        if np.any(slot_priority[planned_ready] > 0.0):
                            sort_keys = np.lexsort((slot_request_time[planned_ready], slot_priority[planned_ready]))
                            planned_ready = planned_ready[sort_keys]
                        else:
                            planned_ready = planned_ready[np.argsort(slot_request_time[planned_ready])]
                    to_start_planned = planned_ready[:free_techs]
                    for s in to_start_planned:
                        slot_state[s] = 5
                        slot_job_type[s] = 1
                        slot_maint_remaining[s] = self.config.preventive_duration
                        slot_start_maint_time[s] = t
                        free_techs -= 1

            # Step C
            all_in_maint = np.where(slot_state == 5)[0]
            if len(all_in_maint) > 1:
                sort_keys = np.lexsort((slot_start_maint_time[all_in_maint], -slot_job_type[all_in_maint]))
                all_in_maint = all_in_maint[sort_keys]

            num_to_work = min(len(all_in_maint), cap_t)
            active_worked = all_in_maint[:num_to_work]
            stalled_slots = all_in_maint[num_to_work:]
            for s in stalled_slots:
                jtype = slot_job_type[s]
                stalls_by_type[jtype] += 1
                stall_events.append((t, s, jtype, cap_t, len(all_in_maint)))

            for s in active_worked:
                slot_maint_remaining[s] -= 1
                if slot_maint_remaining[s] == 0:
                    u_next = int(rng.choice(engine_pool))
                    slot_unit[s] = u_next
                    slot_cycle_idx[s] = 0
                    slot_op_cycles[s] = 0
                    slot_state[s] = 0
                    slot_order_placed[s] = False
                    slot_earliest_start[s] = 0
                    policy.on_renewal(s)

            # Step D
            for s in range(N):
                st = slot_state[s]
                if st in (0, 1, 2):
                    slot_op_cycles[s] += 1
                    u = slot_unit[s]
                    c_idx = slot_cycle_idx[s]
                    eng_data = self.trajectories[u]
                    max_c = eng_data["max_cycles"]
                    c_num = c_idx + 1
                    slot_cycle_idx[s] += 1
                    if st == 0:
                        pred_r = float(eng_data["pred_rul"][c_idx])
                        low_r = float(eng_data["lower"][c_idx])
                        up_r = float(eng_data["upper"][c_idx])
                        true_r = float(eng_data["true_rul_raw"][c_idx])
                        if policy.decide(s, t, slot_op_cycles[s], pred_r, low_r, up_r, true_r, u, c_num, False):
                            slot_order_placed[s] = True
                            slot_order_cycle[s] = t
                            slot_part_arrival[s] = t + self.config.spare_lead_time
                            slot_earliest_start[s] = policy.get_earliest_start(s, slot_part_arrival[s])
                            slot_priority[s] = policy.get_priority(s, 0.0)
                            slot_state[s] = 1

        print("Stalls by type in V1:", stalls_by_type)
        print("First 10 stall events (cycle, slot, job_type, cap_t, in_maint):", stall_events[:10])

inspector = StallInspector(config=cfg, trajectories=trajectories)
agent = clean_v1.AgenticPolicy(use_intervals=True, spare_lead_time=8, decision_log_path=None)
inspector.simulate(agent, eval_pool, seed=42)
