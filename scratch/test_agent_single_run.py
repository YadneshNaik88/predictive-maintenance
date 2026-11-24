import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools
from src.agents import AgenticPolicy

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
cfg_a = SimulationConfig.from_yaml('config/sim_params.yaml', scenario='scenario_a')
sim = FleetSimulator(config=cfg_a, trajectories=trajectories)

for use_intervals in [False, True]:
    policy_name = "Agent-Interval" if use_intervals else "Agent-Point"
    pol = AgenticPolicy(use_intervals=use_intervals)
    t0 = time.time()
    res = sim.simulate(pol, eval_pool, seed=42)
    elapsed = time.time() - t0
    print(f"=== {policy_name} (Scenario A, Seed 42) in {elapsed:.2f}s ===")
    print(f"  Cost: ${res['total_cost']:,.0f} | Fails: {res['number_of_failures']} | Prev: {res['number_of_preventive_actions']}")
    print(f"  Downtime: {res['total_downtime_cycles']}c ({res['total_downtime_cycles']/1000:.2f}%) | Q-Wait: {res['mean_queue_wait']:.2f}c")
    print(f"  Util: {res['utilization']:.3f} | Load: {res['offered_load']:.3f} | Escalations: {res['num_escalations']}")

