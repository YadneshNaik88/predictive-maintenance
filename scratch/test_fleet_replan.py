import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools, EngineObservation
from src.agents import AgenticPolicy, PrognosticAgent, ResourceAgent, CostAgent, CoordinatorAgent

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
cfg = SimulationConfig.from_yaml("config/sim_params.yaml", scenario="scenario_b")

# Let's create an DynamicCoordinator that re-allocates reservations in urgency order
class DynamicCoordinator(CoordinatorAgent):
    def plan_fleet(self, current_cycle, observations):
        now = current_cycle
        L = self.spare_lead_time
        W = self.horizon_W
        D = self.resource.preventive_duration

        # Candidate engines: within horizon or overdue
        candidate_engines = []
        for obs in observations:
            s_id = obs.slot_id
            is_overdue = False
            if s_id in self.slot_plans:
                pl_start = self.slot_plans[s_id].get("planned_start")
                if pl_start is not None and pl_start < now:
                    is_overdue = True
            if obs.pred_rul <= (L + W) or is_overdue:
                candidate_engines.append(obs)

        # Filter engines requiring planning or re-planning
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

        # Release existing reservations of ALL engines being planned/re-planned
        for obs in to_plan:
            s_id = obs.slot_id
            if s_id in self.slot_plans:
                prior = self.slot_plans[s_id]
                if prior.get("reserved") and "planned_start" in prior:
                    self.resource.release(prior["planned_start"], D)
                    prior["reserved"] = False

        # If any engine is overdue, allow it to preempt less urgent reservations
        has_overdue = any(self.slot_plans.get(o.slot_id, {}).get("planned_start", now + 999) < now for o in to_plan)
        if has_overdue or (now % self.replan_k == 1):
            # Re-plan all candidate engines together in strict urgency order
            all_cands = {o.slot_id: o for o in candidate_engines}
            for s_id, o in all_cands.items():
                if s_id in self.slot_plans:
                    p = self.slot_plans[s_id]
                    if p.get("reserved") and "planned_start" in p:
                        self.resource.release(p["planned_start"], D)
                        p["reserved"] = False
            to_plan = list(all_cands.values())

        # Sort by urgency (lowest lower bound first)
        to_plan.sort(key=lambda o: o.lower_rul)

        action_orders = {}
        for obs in to_plan:
            s_id = obs.slot_id
            had_prior_plan = s_id in self.slot_plans
            was_overdue = False
            prior_start = None
            if had_prior_plan:
                prior = self.slot_plans[s_id]
                prior_start = prior.get("planned_start")
                if prior_start is not None and prior_start < now:
                    was_overdue = True

            # Candidate search range
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
                reason = f"Selected cycle {chosen_cycle} with cost rate ${best_cost_rate:.2f}/cycle."
            else:
                escalated = True
                order_spare_now = not obs.order_already_placed
                earliest_part = obs.part_arrival_cycle if obs.order_already_placed and obs.part_arrival_cycle else (now + L)
                search_start = max(now, earliest_part)
                chosen_cycle = self.resource.find_earliest_available(search_start, D)
                reserved = self.resource.reserve(chosen_cycle, D)
                reason = f"ESCALATION: Earliest available cycle {chosen_cycle}."

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

# Build policy using DynamicCoordinator
class DynamicAgenticPolicy(AgenticPolicy):
    def __init__(self, use_intervals=True, spare_lead_time=8):
        super().__init__(use_intervals=use_intervals, spare_lead_time=spare_lead_time, decision_log_path=None)
        self.coordinator = DynamicCoordinator(
            prognostic=self.prognostic,
            resource=self.resource,
            cost=self.cost,
            replan_k=5,
            horizon_W=60,
            spare_lead_time=spare_lead_time,
            decision_log_path=None,
        )

# Run simulation
sim = FleetSimulator(config=cfg, trajectories=trajectories)
pol = DynamicAgenticPolicy(use_intervals=True)
# We test with simulator's is_agentic dispatch
res = sim.simulate(pol, eval_pool, seed=42)
print("=== DYNAMIC COORDINATOR RESULTS ===")
print(f"Total Cost: ${res['total_cost']:,.2f}")
print(f"Failures: {res['number_of_failures']}")
print(f"Preventive Actions: {res['number_of_preventive_actions']}")
print(f"Total Stalls: {res.get('total_stalled_cycles', 0)}")
print(f"Queue Failures: {res.get('queue_failures', 0)}")
