import time
import json
import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools
from src.policies import RunToFailurePolicy, FixedIntervalPolicy, NaiveThresholdPolicy, DiagnosticOraclePolicy

t0 = time.time()
trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
seeds = [42 + i for i in range(20)]

with open("outputs/results_final_v1.json", "r", encoding="utf-8") as f:
    v1 = json.load(f)

for scen_id in ["scenario_a", "scenario_b", "scenario_c"]:
    cfg = SimulationConfig.from_yaml("config/sim_params.yaml", scenario=scen_id)
    sim = FleetSimulator(config=cfg, trajectories=trajectories)
    scen_data = v1["scenarios"][scen_id]
    t_star = scen_data["T_star"]
    tau_star = scen_data["tau_star"]
    m_star = scen_data["m_star"]

    policies = [
        ("Run-to-Failure", RunToFailurePolicy()),
        (f"Fixed Interval (T={t_star})", FixedIntervalPolicy(t_star, lead_time=cfg.spare_lead_time)),
        (f"Naive Threshold (tau={tau_star})", NaiveThresholdPolicy(tau_star)),
        (f"Diagnostic Oracle (m={m_star})", DiagnosticOraclePolicy(lead_time=cfg.spare_lead_time, margin=m_star)),
    ]

    for pol_name, pol in policies:
        v1_pol = scen_data["policies"][pol_name]
        downtimes = []
        for idx, s in enumerate(seeds):
            res = sim.simulate(pol, eval_pool, seed=s)
            assert res["total_cost"] == v1_pol["raw_costs"][idx], (
                f"Cost mismatch in {scen_id} {pol_name} seed {s}: {res['total_cost']} vs {v1_pol['raw_costs'][idx]}"
            )
            assert res["number_of_failures"] == v1_pol["raw_failures"][idx], (
                f"Failure mismatch in {scen_id} {pol_name} seed {s}: {res['number_of_failures']} vs {v1_pol['raw_failures'][idx]}"
            )
            downtimes.append(res["total_downtime_cycles"])
        assert float(np.mean(downtimes)) == pytest.approx(v1_pol["mean_downtime_cycles"], rel=1e-9), (
            f"Mean downtime mismatch in {scen_id} {pol_name}"
        )

print(f"ALL 240 BASELINE RUNS MATCHED BIT-FOR-BIT IN {time.time()-t0:.2f}s!")
