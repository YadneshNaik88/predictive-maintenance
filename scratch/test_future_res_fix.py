import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools, EngineObservation
from src.policies import BasePolicy
from src.agents import PrognosticAgent, ResourceAgent, CostAgent

class FutureResCoordinator:
    def __init__(self, prognostic, resource, cost, replan_k=5, horizon_W=60, spare_lead_time=8, decision_log_path=None):
        self.prognostic = prognostic
        self.resource = resource
        self.cost = cost
        self.replan_k = replan_k
        self.horizon_W = horizon_W
        self.spare_lead_time = spare_lead_time
        self.decision_log_path = Path(decision_log_path) if decision_log_path else None
        self.slot_plans = {}
        self.decision_records = []

    def reset(self, total_horizon, capacity_roster):
        self.resource.reset(total_horizon, capacity_roster)
        self.slot_plans.clear()
        self.decision_records.clear()

    def on_renewal(self, slot_id):
        if slot_id in self.slot_plans:
            plan = self.slot_plans[slot_id]
            if plan.get("reserved", False) and "planned_start" in plan:
                self.resource.release(plan["planned_start"], self.resource.preventive_duration)
            del self.slot_plans[slot_id]

    def on_engine_failure(self, slot_id):
        if slot_id in self.slot_plans:
            plan = self.slot_plans[slot_id]
            if plan.get("reserved", False) and "planned_start" in plan:
                self.resource.release(plan["planned_start"], self.resource.preventive_duration)
                plan["reserved"] = False

    def holds_active_reservation(self, slot_id, cycle):
        plan = self.slot_plans.get(slot_id)
        if not plan:
            return False
        return bool(plan.get("reserved", False) and plan.get("planned_start") == cycle)

    def get_other_future_reservations(self, slot_id, current_cycle, target_cycle):
        count = 0
        for s, p in self.slot_plans.items():
            if s == slot_id:
                continue
            if not p.get("reserved", False):
                continue
            pl_st = p.get("planned_start")
            dur = p.get("duration", self.resource.preventive_duration)
            if pl_st is not None and pl_st > current_cycle:
                if pl_st <= target_cycle < pl_st + dur:
                    count += 1
        return count

    def on_job_start(self, slot_id, current_cycle):
        if slot_id in self.slot_plans:
            plan = self.slot_plans[slot_id]
            if plan.get("reserved", False):
                dur = plan.get("duration", self.resource.preventive_duration)
                st = plan.get("planned_start", current_cycle)
                self.resource.release(st, dur)
                plan["reserved"] = False

    def plan_fleet(self, current_cycle, observations):
        now = current_cycle
        L = self.spare_lead_time
        W = self.horizon_W
        D = self.resource.preventive_duration

        candidate_engines = []
        for obs in observations:
            s_id = obs.slot_id
            is_overdue = False
            if s_id in self.slot_plans:
                pl_st = self.slot_plans[s_id].get("planned_start")
                if pl_st is not None and pl_st < now:
                    is_overdue = True
            if obs.pred_rul <= (L + W) or is_overdue:
                candidate_engines.append(obs)

        to_plan = []
        for obs in candidate_engines:
            s_id = obs.slot_id
            if s_id not in self.slot_plans:
                to_plan.append(obs)
            else:
                plan = self.slot_plans[s_id]
                planned_start = plan.get("planned_start", now + 999)
                is_overdue = (planned_start < now)
                is_periodic = (now % self.replan_k == 1)
                lower_dropped = (obs.lower_rul < (planned_start - now))
                if is_overdue or is_periodic or lower_dropped:
                    to_plan.append(obs)

        if not to_plan:
            return {}

        to_plan.sort(key=lambda o: o.lower_rul)
        action_orders = {}

        for obs in to_plan:
            s_id = obs.slot_id
            had_prior = s_id in self.slot_plans
            if had_prior:
                if self.slot_plans[s_id].get("reserved", False):
                    self.resource.release(self.slot_plans[s_id]["planned_start"], D)
                    self.slot_plans[s_id]["reserved"] = False

            if not self.prognostic.use_intervals:
                max_s = max(0, int(np.floor(obs.pred_rul - 1.0)))
            else:
                max_s = max(0, int(np.floor(obs.pred_rul)))

            candidate_s = np.arange(0, max_s + 1, dtype=int)
            feasible_mask, diag_list = self.resource.filter_candidates(obs, candidate_s)
            feasible_s = candidate_s[feasible_mask]

            chosen_cycle = None
            escalated = False
            best_cost_rate = 999999.0
            order_spare_now = not obs.order_already_placed

            if len(feasible_s) > 0:
                prob_fail, exp_min_r = self.prognostic.evaluate_risk(obs.pred_rul, feasible_s)
                c_s, e_s, cost_rates = self.cost.evaluate_cost_rates(obs.age, feasible_s, prob_fail, exp_min_r)
                best_idx = int(np.argmin(cost_rates))
                best_s = int(feasible_s[best_idx])
                best_cost_rate = float(cost_rates[best_idx])
                chosen_cycle = now + best_s
                reserved = self.resource.reserve(chosen_cycle, D)
            else:
                escalated = True
                order_spare_now = not obs.order_already_placed
                earliest_part = obs.part_arrival_cycle if obs.order_already_placed and obs.part_arrival_cycle else (now + L)
                search_start = max(now, earliest_part)
                chosen_cycle = self.resource.find_earliest_available(search_start, D)
                reserved = self.resource.reserve(chosen_cycle, D)

            plan_record = {
                "slot_id": s_id,
                "unit_number": obs.unit_number,
                "planned_start": chosen_cycle,
                "duration": D,
                "order_cycle": now if order_spare_now else self.slot_plans.get(s_id, {}).get("order_cycle", now),
                "part_arrival": now + L if order_spare_now else obs.part_arrival_cycle,
                "escalated": escalated,
                "reserved": reserved,
                "cost_rate": best_cost_rate,
                "urgency_lower": obs.lower_rul,
            }
            self.slot_plans[s_id] = plan_record
            action_orders[s_id] = {
                "order_spare": order_spare_now,
                "planned_start": chosen_cycle,
                "priority": float(obs.lower_rul),
                "escalated": escalated,
            }

        return action_orders

class FutureResAgenticPolicy(BasePolicy):
    is_agentic = True
    def __init__(self, use_intervals=True):
        super().__init__(name="Agent-Interval" if use_intervals else "Agent-Point")
        self.prognostic = PrognosticAgent(use_intervals=use_intervals)
        self.resource = ResourceAgent(spare_lead_time=8, preventive_duration=2, horizon_W=60)
        self.cost = CostAgent()
        self.coordinator = FutureResCoordinator(self.prognostic, self.resource, self.cost)
        self.current_action_orders = {}
        self.num_escalations = 0

    def reset(self):
        self.current_action_orders.clear()
        self.num_escalations = 0

    def init_simulation(self, total_horizon, capacity_roster):
        self.reset()
        self.coordinator.reset(total_horizon, capacity_roster)

    def on_renewal(self, slot_id):
        self.coordinator.on_renewal(slot_id)
        if slot_id in self.current_action_orders:
            del self.current_action_orders[slot_id]

    def on_engine_failure(self, slot_id):
        self.coordinator.on_engine_failure(slot_id)

    def holds_active_reservation(self, slot_id, cycle):
        return self.coordinator.holds_active_reservation(slot_id, cycle)

    def get_other_reservations(self, slot_id, cycle, current_cycle=None):
        c_now = current_cycle if current_cycle is not None else cycle
        return self.coordinator.get_other_future_reservations(slot_id, c_now, cycle)

    def on_job_start(self, slot_id, current_cycle):
        self.coordinator.on_job_start(slot_id, current_cycle)

    def before_cycle(self, current_cycle, observations):
        orders = self.coordinator.plan_fleet(current_cycle, observations)
        for s_id, ord_dict in orders.items():
            if ord_dict.get("escalated", False):
                self.num_escalations += 1
            self.current_action_orders[s_id] = ord_dict

    def decide(self, slot_id, current_cycle, operating_cycles, pred_rul, lower_rul, upper_rul, true_rul, unit_number, engine_cycle, order_already_placed, fleet_context=None):
        if order_already_placed:
            return False
        if slot_id in self.current_action_orders:
            return bool(self.current_action_orders[slot_id].get("order_spare", False))
        return False

    def get_earliest_start(self, slot_id, default):
        if slot_id in self.coordinator.slot_plans:
            return int(self.coordinator.slot_plans[slot_id].get("planned_start", default))
        return default

    def get_priority(self, slot_id, default=0.0):
        if slot_id in self.coordinator.slot_plans:
            return float(self.coordinator.slot_plans[slot_id].get("urgency_lower", default))
        return default

# Let's test the simulator with this future reservation dispatch
trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
cfg = SimulationConfig.from_yaml("config/sim_params.yaml", scenario="scenario_b")

# Update FleetSimulator Step B dispatch to pass current_cycle to get_other_reservations
agent = FutureResAgenticPolicy(use_intervals=True)
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
    waiting_part_operating = np.where((slot_state == 1) & (t >= slot_part_arrival))[0]
    for s in waiting_part_operating:
        slot_state[s] = 2
        slot_request_time[s] = slot_part_arrival[s]

    waiting_part_failed = np.where((slot_state == 3) & (t >= slot_part_arrival))[0]
    for s in waiting_part_failed:
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
                if agent.holds_active_reservation(s, t):
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
                    cap_c = int(tech_capacity_arr[c])
                    comm_c = committed[c]
                    other_res_c = agent.coordinator.get_other_future_reservations(s, t, c)
                    avail = cap_c - comm_c - other_res_c
                    if avail < 1:
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
                    agent.on_job_start(s, t)

    # Step C
    all_in_maint = np.where(slot_state == 5)[0]
    num_to_work = min(len(all_in_maint), cap_t)
    active_worked = all_in_maint[:num_to_work]
    total_stalls += len(all_in_maint) - len(active_worked)
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
                if not slot_order_placed[s]:
                    slot_order_placed[s] = True
                    slot_order_cycle[s] = t
                    total_spares += 1
                    slot_part_arrival[s] = t + cfg.spare_lead_time
                    slot_earliest_start[s] = slot_part_arrival[s]
                    slot_state[s] = 3
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
print(f"\nFinal 1000-cycle results for Future Reservations Dispatch:")
print(f"Total Cost: ${total_cost:,.2f}")
print(f"Failures: {num_failures}")
print(f"Preventive Actions: {num_preventive}")
print(f"Total Stalls: {total_stalls}")
print(f"Queue Failures: {queue_failures}")
