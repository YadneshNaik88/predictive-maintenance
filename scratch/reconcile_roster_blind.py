"""
scratch/reconcile_roster_blind.py

Reconciles the roster-blind ablation for Scenarios A, B, C (20 seeds, evaluation pool).
Compares:
1. Blind-MeanCapacity (Phase 6b definition: float scenario mean for future cycles, true capacity for current cycle)
2. Blind-CurrentOnly / Integer-Mean (src/agents.py roster_blind definition: int-rounded mean filled across roster)
Against canonical baselines from outputs/results_final.json (Naive Threshold, Agent-Interval, Agent-Perfect).
"""

import sys
import json
from pathlib import Path
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import (
    FleetSimulator,
    SimulationConfig,
    load_engine_trajectories,
    get_tuning_eval_pools,
)
from src.agents import (
    AgenticPolicy,
    ResourceAgent,
)

# -----------------------------------------------------------------------------
# Definition 1: Blind-MeanCapacity (Phase 6b run script)
# -----------------------------------------------------------------------------
class BlindResourceAgentMean(ResourceAgent):
    def __init__(self, mean_capacity: float, spare_lead_time: int = 8, preventive_duration: int = 2, horizon_W: int = 60):
        super().__init__(spare_lead_time=spare_lead_time, preventive_duration=preventive_duration, horizon_W=horizon_W)
        self.mean_capacity = float(mean_capacity)
        self.true_capacity_roster = None
        self.current_sim_cycle = 1
        self.current_queue_len = 0

    def reset(self, total_horizon: int, capacity_roster: np.ndarray) -> None:
        super().reset(total_horizon, capacity_roster)
        self.true_capacity_roster = capacity_roster.copy()
        pad_len = max(300, self.horizon_W + 50)
        self.capacity_roster = np.full(len(capacity_roster) + pad_len, self.mean_capacity, dtype=float)
        self.reserved_calendar = np.zeros(len(capacity_roster) + pad_len, dtype=int)

    def set_cycle_and_queue(self, current_cycle: int, queue_len: int = 0) -> None:
        self.current_sim_cycle = current_cycle
        self.current_queue_len = queue_len
        if self.true_capacity_roster is not None and current_cycle < len(self.true_capacity_roster):
            self.capacity_roster[current_cycle] = float(self.true_capacity_roster[current_cycle])

    def is_window_available(self, start_cycle: int, duration: int) -> bool:
        for c in range(start_cycle, start_cycle + duration):
            if c >= len(self.capacity_roster):
                return False
            effective_cap = self.capacity_roster[c]
            if c == self.current_sim_cycle:
                effective_cap = max(0.0, effective_cap - self.current_queue_len)
            if (effective_cap - self.reserved_calendar[c]) < 1.0:
                return False
        return True


class BlindMeanCapacityPolicy(AgenticPolicy):
    def __init__(self, mean_capacity: float, **kwargs):
        super().__init__(use_intervals=True, decision_log_path=None, **kwargs)
        self.name = "Blind-MeanCapacity"
        d_pm = self.cost.preventive_duration
        self.resource = BlindResourceAgentMean(
            mean_capacity=mean_capacity,
            spare_lead_time=self.spare_lead_time,
            preventive_duration=d_pm,
            horizon_W=self.horizon_W,
        )
        self.coordinator.resource = self.resource

    def before_cycle(self, current_cycle: int, observations: list) -> None:
        self.resource.set_cycle_and_queue(current_cycle, queue_len=0)
        super().before_cycle(current_cycle, observations)


# -----------------------------------------------------------------------------
# Definition 2: Blind-CurrentOnly / Integer-Mean (src/agents.py roster_blind=True)
# -----------------------------------------------------------------------------
def make_blind_current_only():
    return AgenticPolicy(use_intervals=True, roster_blind=True, decision_log_path=None)


def run_reconciliation():
    # Load canonical results
    with open(PROJECT_ROOT / "outputs" / "results_final.json", "r", encoding="utf-8") as f:
        results_final = json.load(f)

    trajectories = load_engine_trajectories()
    _, eval_pool = get_tuning_eval_pools(42)
    eval_seeds = results_final["metadata"]["seeds"]

    # 1. Print Agent-Perfect metrics from results_final.json
    print("=" * 100)
    print("AGENT-PERFECT FAILURES AND COSTS FROM outputs/results_final.json")
    print("=" * 100)
    for sc in ["scenario_a", "scenario_b", "scenario_c"]:
        p_data = results_final["scenarios"][sc]["policies"]["Agent-Perfect"]
        print(f"{sc.upper():12s} | Mean Cost: ${p_data['mean_cost']:10,.0f} | Failures: {p_data['mean_failures']:6.2f} "
              f"| Failure Penalty: ${p_data['cost_decomposition']['failure_penalty']:10,.0f}")

    # 2. Storage inspection for Agent-Interval-Blind in results_final.json
    print("\n" + "=" * 100)
    print("STORAGE INSPECTION: WHERE ARE AGENT-INTERVAL-BLIND NUMBERS IN results_final.json?")
    print("=" * 100)
    for sc in ["scenario_a", "scenario_b", "scenario_c"]:
        policies_present = list(results_final["scenarios"][sc]["policies"].keys())
        has_blind = "Agent-Interval-Blind" in policies_present
        print(f"{sc.upper():12s} | 'Agent-Interval-Blind' key present in 'policies': {has_blind}")
        print(f"             | Present policies: {policies_present}")

    # 3. Simulate Scenario B for both definitions
    print("\n" + "=" * 100)
    print("SIMULATING SCENARIO B (20 SEEDS) FOR BOTH BLIND DEFINITIONS")
    print("=" * 100)
    cfg_b = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario="scenario_b")
    sim_b = FleetSimulator(config=cfg_b, trajectories=trajectories)

    mean_cap_b = 1.549  # Scenario B analytical mean capacity

    runs_mean = []
    runs_curr = []

    for s in eval_seeds:
        pol_mean = BlindMeanCapacityPolicy(mean_capacity=mean_cap_b)
        res_mean = sim_b.simulate(pol_mean, eval_pool, seed=s)
        runs_mean.append(res_mean)

        pol_curr = make_blind_current_only()
        res_curr = sim_b.simulate(pol_curr, eval_pool, seed=s)
        runs_curr.append(res_curr)

    # Canonical baselines from results_final.json
    naive_data = results_final["scenarios"]["scenario_b"]["policies"]["Naive Threshold (tau=60)"]
    interval_data = results_final["scenarios"]["scenario_b"]["policies"]["Agent-Interval"]

    naive_costs = naive_data["raw_costs"]
    naive_fails = naive_data["raw_failures"]
    interval_costs = interval_data["raw_costs"]
    interval_fails = interval_data["raw_failures"]

    # 4. Print Per-Seed Cost and Failures for Scenario B
    print("\n" + "=" * 100)
    print("SCENARIO B: PER-SEED COST AND FAILURES TABLE (20 SEEDS)")
    print("=" * 100)
    print(f"{'Seed':5s} | {'Naive (tau=60)':23s} | {'Agent-Interval':23s} | {'Blind-MeanCapacity':23s} | {'Blind-CurrentOnly':23s}")
    print(f"{'':5s} | {'Cost':13s} {'Fails':8s} | {'Cost':13s} {'Fails':8s} | {'Cost':13s} {'Fails':8s} | {'Cost':13s} {'Fails':8s}")
    print("-" * 105)

    for i, s in enumerate(eval_seeds):
        c_n, f_n = naive_costs[i], naive_fails[i]
        c_int, f_int = interval_costs[i], interval_fails[i]
        c_m, f_m = runs_mean[i]["total_cost"], runs_mean[i]["number_of_failures"]
        c_c, f_c = runs_curr[i]["total_cost"], runs_curr[i]["number_of_failures"]
        print(f"{s:5d} | ${c_n:10,.0f}   {f_n:4.0f}   | ${c_int:10,.0f}   {f_int:4.0f}   | "
              f"${c_m:10,.0f}   {f_m:4.0f}   | ${c_c:10,.0f}   {f_c:4.0f}")

    print("-" * 105)
    mean_cn, mean_fn = np.mean(naive_costs), np.mean(naive_fails)
    mean_cint, mean_fint = np.mean(interval_costs), np.mean(interval_fails)
    mean_cm, mean_fm = np.mean([r["total_cost"] for r in runs_mean]), np.mean([r["number_of_failures"] for r in runs_mean])
    mean_cc, mean_fc = np.mean([r["total_cost"] for r in runs_curr]), np.mean([r["number_of_failures"] for r in runs_curr])
    std_cm = np.std([r["total_cost"] for r in runs_mean])
    std_cc = np.std([r["total_cost"] for r in runs_curr])

    print(f"{'MEAN':5s} | ${mean_cn:10,.0f}   {mean_fn:4.2f}   | ${mean_cint:10,.0f}   {mean_fint:4.2f}   | "
          f"${mean_cm:10,.0f}   {mean_fm:4.2f}   | ${mean_cc:10,.0f}   {mean_fc:4.2f}")
    print(f"{'STD':5s}  | ${np.std(naive_costs):10,.0f}   {np.std(naive_fails):4.2f}   | "
          f"${np.std(interval_costs):10,.0f}   {np.std(interval_fails):4.2f}   | "
          f"${std_cm:10,.0f}   {np.std([r['number_of_failures'] for r in runs_mean]):4.2f}   | "
          f"${std_cc:10,.0f}   {np.std([r['number_of_failures'] for r in runs_curr]):4.2f}")

    # 5. Cost Decomposition Table for Scenario B
    print("\n" + "=" * 105)
    print("SCENARIO B: COST DECOMPOSITION TABLE (20 SEEDS MEAN)")
    print("=" * 105)
    print(f"{'Policy':24s} | {'Preventive':14s} | {'Spares':12s} | {'Downtime':14s} | {'Failure Pen':14s} | {'Total Cost':14s}")
    print("-" * 105)

    decomp_policies = [
        ("Naive Threshold (tau=60)", naive_data["cost_decomposition"]),
        ("Agent-Interval", interval_data["cost_decomposition"]),
        ("Blind-MeanCapacity", {
            "preventive_overhaul": float(np.mean([r["preventive_cost"] for r in runs_mean])),
            "spare_parts": float(np.mean([r["spare_cost"] for r in runs_mean])),
            "downtime_loss": float(np.mean([r["production_loss"] for r in runs_mean])),
            "failure_penalty": float(np.mean([r["corrective_cost"] for r in runs_mean])),
            "total_cost": float(np.mean([r["total_cost"] for r in runs_mean])),
        }),
        ("Blind-CurrentOnly", {
            "preventive_overhaul": float(np.mean([r["preventive_cost"] for r in runs_curr])),
            "spare_parts": float(np.mean([r["spare_cost"] for r in runs_curr])),
            "downtime_loss": float(np.mean([r["production_loss"] for r in runs_curr])),
            "failure_penalty": float(np.mean([r["corrective_cost"] for r in runs_curr])),
            "total_cost": float(np.mean([r["total_cost"] for r in runs_curr])),
        }),
    ]

    for p_name, d in decomp_policies:
        print(f"{p_name:24s} | ${d['preventive_overhaul']:12,.0f} | ${d['spare_parts']:10,.0f} | "
              f"${d['downtime_loss']:12,.0f} | ${d['failure_penalty']:12,.0f} | ${d['total_cost']:12,.0f}")

    # 6. Paired Differences and 95% CIs
    print("\n" + "=" * 105)
    print("SCENARIO B: PAIRED DIFFERENCES AND 95% CONFIDENCE INTERVALS (20 SEEDS)")
    print("=" * 105)

    # For Blind-MeanCapacity
    diff_mean_vs_naive = np.array([r["total_cost"] for r in runs_mean]) - np.array(naive_costs)
    diff_mean_vs_int = np.array([r["total_cost"] for r in runs_mean]) - np.array(interval_costs)

    m_mean_nv = float(np.mean(diff_mean_vs_naive))
    sem_mean_nv = float(stats.sem(diff_mean_vs_naive))
    ci_mean_nv = [m_mean_nv - 2.093 * sem_mean_nv, m_mean_nv + 2.093 * sem_mean_nv]

    m_mean_int = float(np.mean(diff_mean_vs_int))
    sem_mean_int = float(stats.sem(diff_mean_vs_int))
    ci_mean_int = [m_mean_int - 2.093 * sem_mean_int, m_mean_int + 2.093 * sem_mean_int]

    # For Blind-CurrentOnly
    diff_curr_vs_naive = np.array([r["total_cost"] for r in runs_curr]) - np.array(naive_costs)
    diff_curr_vs_int = np.array([r["total_cost"] for r in runs_curr]) - np.array(interval_costs)

    m_curr_nv = float(np.mean(diff_curr_vs_naive))
    sem_curr_nv = float(stats.sem(diff_curr_vs_naive))
    ci_curr_nv = [m_curr_nv - 2.093 * sem_curr_nv, m_curr_nv + 2.093 * sem_curr_nv]

    m_curr_int = float(np.mean(diff_curr_vs_int))
    sem_curr_int = float(stats.sem(diff_curr_vs_int))
    ci_curr_int = [m_curr_int - 2.093 * sem_curr_int, m_curr_int + 2.093 * sem_curr_int]

    print(f"1. Blind-MeanCapacity vs Naive Threshold:")
    print(f"   Paired Mean Difference : ${m_mean_nv:+12,.0f}")
    print(f"   95% Confidence Interval: [${ci_mean_nv[0]:+10,.0f}, ${ci_mean_nv[1]:+10,.0f}]")
    print(f"   Relative Saving vs Naive: {(-m_mean_nv / mean_cn) * 100:.2f}%")

    print(f"\n2. Blind-MeanCapacity vs Agent-Interval:")
    print(f"   Paired Mean Difference : ${m_mean_int:+12,.0f}")
    print(f"   95% Confidence Interval: [${ci_mean_int[0]:+10,.0f}, ${ci_mean_int[1]:+10,.0f}]")
    print(f"   Roster Lookahead Value  : ${m_mean_int:10,.0f} (Lookahead saves ${m_mean_int:,.0f})")

    print(f"\n3. Blind-CurrentOnly vs Naive Threshold:")
    print(f"   Paired Mean Difference : ${m_curr_nv:+12,.0f}")
    print(f"   95% Confidence Interval: [${ci_curr_nv[0]:+10,.0f}, ${ci_curr_nv[1]:+10,.0f}]")

    print(f"\n4. Blind-CurrentOnly vs Agent-Interval:")
    print(f"   Paired Mean Difference : ${m_curr_int:+12,.0f}")
    print(f"   95% Confidence Interval: [${ci_curr_int[0]:+10,.0f}, ${ci_curr_int[1]:+10,.0f}]")

    # 7. Scenario A and C Evaluation for Blind-MeanCapacity
    print("\n" + "=" * 105)
    print("SCENARIOS A & C: BLIND-MEANCAPACITY PERFORMANCE (20 SEEDS)")
    print("=" * 105)
    for sc_id, sc_name, mean_cap in [("scenario_a", "Scenario A", 4.0), ("scenario_c", "Scenario C", 1.487)]:
        cfg_sc = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario=sc_id)
        sim_sc = FleetSimulator(config=cfg_sc, trajectories=trajectories)
        runs_sc = [sim_sc.simulate(BlindMeanCapacityPolicy(mean_capacity=mean_cap), eval_pool, seed=s) for s in eval_seeds]
        costs_sc = [r["total_cost"] for r in runs_sc]
        fails_sc = [r["number_of_failures"] for r in runs_sc]
        print(f"{sc_name:12s} | Mean Cost: ${np.mean(costs_sc):10,.0f} +/- ${np.std(costs_sc):6,.0f} | "
              f"Failures: {np.mean(fails_sc):4.2f}")


if __name__ == "__main__":
    run_reconciliation()
