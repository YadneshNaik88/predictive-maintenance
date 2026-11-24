import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools
from src.policies import RunToFailurePolicy, FixedIntervalPolicy, NaiveThresholdPolicy, DiagnosticOraclePolicy

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)

for scen in ['scenario_a', 'scenario_b', 'scenario_c']:
    cfg = SimulationConfig.from_yaml('config/sim_params.yaml', scenario=scen)
    sim = FleetSimulator(config=cfg, trajectories=trajectories)
    print(f"=== {scen.upper()} SEED 42 ===")
    pols = [
        RunToFailurePolicy(),
        FixedIntervalPolicy(125 if scen == 'scenario_a' else 75),
        NaiveThresholdPolicy(15 if scen == 'scenario_a' else (60 if scen == 'scenario_b' else 70)),
        DiagnosticOraclePolicy(margin=2 if scen == 'scenario_a' else 40),
    ]
    for pol in pols:
        res = sim.simulate(pol, eval_pool, seed=42)
        print(f"  {pol.name:<28}: Cost=${res['total_cost']:,.0f} | Fails={res['number_of_failures']:3d} | Prev={res['number_of_preventive_actions']:3d} | DT={res['total_downtime_cycles']:5d}c")

