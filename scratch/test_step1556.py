import sys
from pathlib import Path
import numpy as np
import yaml
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools, EngineObservation
from src.policies import BasePolicy
from src.agents import PrognosticAgent, ResourceAgent, CostAgent

# Let's recreate Step 1556 CoordinatorAgent and AgenticPolicy
class Step1556Coordinator:
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

    def plan_fleet(self, current_cycle, observations):
        now = current_cycle
        L = self.spare_lead_time
        W = self.horizon_W
        D = self.resource.preventive_duration

        candidate_engines = []
        for obs in observations:
            if obs.pred_rul <= (L + W):
                candidate_engines.append(obs)

        to_plan = []
        for obs in candidate_engines:
            s_id = obs.slot_id
            if s_id not in self.slot_plans:
                to_plan.append(obs)
            else:
                plan = self.slot_plans[s_id]
                is_periodic = (now % self.replan_k == 1)
                planned_start = plan.get("planned_start", now + 999)
                lower_dropped = (obs.lower_rul < (planned_start - now))
                if is_periodic or lower_dropped:
                    to_plan.append(obs)

        if not to_plan:
            return {}

        to_plan.sort(key=lambda o: o.lower_rul)
        action_orders = {}

        for obs in to_plan:
            s_id = obs.slot_id
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
                for cost_idx, d in enumerate([x for x in diag_list if x["feasible"]]):
                    d["cost_rate"] = float(cost_rates[cost_idx])
                    d["prob_fail"] = float(prob_fail[cost_idx])
            else:
                escalated = True
                order_spare_now = not obs.order_already_placed
                earliest_part = obs.part_arrival_cycle if obs.order_already_placed and obs.part_arrival_cycle else (now + L)
                chosen_cycle = self.resource.find_earliest_available(earliest_part, D)
                reserved = self.resource.reserve(chosen_cycle, D)

            plan_record = {
                "slot_id": s_id,
                "unit_number": obs.unit_number,
                "planned_start": chosen_cycle,
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

class Step1556AgenticPolicy(BasePolicy):
    is_agentic = False
    def __init__(self, use_intervals=True):
        super().__init__(name="Agent-Interval" if use_intervals else "Agent-Point")
        self.prognostic = PrognosticAgent(use_intervals=use_intervals)
        self.resource = ResourceAgent(spare_lead_time=8, preventive_duration=2, horizon_W=60)
        self.cost = CostAgent()
        self.coordinator = Step1556Coordinator(self.prognostic, self.resource, self.cost)
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

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
cfg = SimulationConfig.from_yaml("config/sim_params.yaml", scenario="scenario_b")
sim = FleetSimulator(config=cfg, trajectories=trajectories)

agent = Step1556AgenticPolicy(use_intervals=True)
res = sim.simulate(agent, eval_pool, seed=42)
print("Step 1556 result on Scenario B seed 42:")
print("Cost:", res["total_cost"])
print("Failures:", res["number_of_failures"])
print("Preventive:", res["number_of_preventive_actions"])
print("Stalls:", res["total_stalled_cycles"])
