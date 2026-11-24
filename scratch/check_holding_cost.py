import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("C:/Users/naiky/OneDrive/Desktop/Projects/predictive-maintenance")
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig, get_tuning_eval_pools, load_engine_trajectories
from src.policies import NaiveThresholdPolicy
from src.agents import AgenticPolicy

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(42)
seeds = [42 + i for i in range(20)]

cfg_b = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario="scenario_b")

# Simulator tracking part holding cycles
class HoldingCostSimulator(FleetSimulator):
    def simulate_with_holding(self, policy, engine_pool, seed, h_rate=0.0):
        res = self.simulate(policy, engine_pool, seed)
        # queue_wait in sim is max(0, t_start - t_part_arrival)
        # Wait, let's verify what queue_wait_list is in simulator.py:
        # q_wait = max(0, t - slot_part_arrival[s])
        # For operating engines that planned start, they continued operating!
        # The holding duration of the spare part is exactly (t_start - slot_part_arrival[s])!
        total_jobs = res["number_of_preventive_actions"] + res["number_of_failures"]
        holding_cycles = res["mean_queue_wait"] * total_jobs
        holding_cost = holding_cycles * h_rate
        return res["total_cost"] + holding_cost, holding_cycles, holding_cost

sim = HoldingCostSimulator(config=cfg_b, trajectories=trajectories)

naive_pol = NaiveThresholdPolicy(60)
agent_pol = AgenticPolicy(use_intervals=True, spare_lead_time=8, horizon_W=60, replan_k=5, decision_log_path=None)

for h in [0.0, 2.0, 5.0]:
    naive_costs = []
    agent_costs = []
    naive_holds = []
    agent_holds = []
    for s in seeds:
        c_n, h_n, _ = sim.simulate_with_holding(naive_pol, eval_pool, seed=s, h_rate=h)
        c_a, h_a, _ = sim.simulate_with_holding(agent_pol, eval_pool, seed=s, h_rate=h)
        naive_costs.append(c_n)
        agent_costs.append(c_a)
        naive_holds.append(h_n)
        agent_holds.append(h_a)
    
    print(f"\n--- Spare Part Holding Cost Rate h = ${h:.0f}/cycle ---")
    print(f"Naive (tau=60): Cost = ${np.mean(naive_costs):,.0f} +/- ${np.std(naive_costs):,.0f} | Mean Holding Cycles: {np.mean(naive_holds):.1f}")
    print(f"Agent-Interval : Cost = ${np.mean(agent_costs):,.0f} +/- ${np.std(agent_costs):,.0f} | Mean Holding Cycles: {np.mean(agent_holds):.1f}")
    diff = np.mean(agent_costs) - np.mean(naive_costs)
    pct = (diff / np.mean(naive_costs)) * 100.0
    print(f"Difference (Agent - Naive): ${diff:+,.0f} ({pct:+.2f}%)")
