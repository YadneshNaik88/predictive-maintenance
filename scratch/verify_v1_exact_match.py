import sys
from pathlib import Path
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import (
    FleetSimulator,
    SimulationConfig,
    load_engine_trajectories,
    get_tuning_eval_pools,
)
from src.agents import AgenticPolicy
V1_PATH = PROJECT_ROOT / "outputs" / "results_final_v1.json"

def main():
    with open(V1_PATH, "r", encoding="utf-8") as f:
        v1 = json.load(f)

    trajectories_real = load_engine_trajectories()
    trajectories_perf = load_engine_trajectories()
    for u, data in trajectories_perf.items():
        data["pred_rul"] = data["true_rul_capped"].copy()
        data["lower"] = data["true_rul_capped"].copy()
        data["upper"] = data["true_rul_capped"].copy()

    _, eval_pool = get_tuning_eval_pools(seed=42)
    eval_seeds = v1["metadata"]["seeds"]

    scenarios = ["scenario_a", "scenario_b", "scenario_c"]
    policies = [
        ("Agent-Point", lambda: AgenticPolicy(use_intervals=False, decision_log_path=None), False),
        ("Agent-Interval", lambda: AgenticPolicy(use_intervals=True, decision_log_path=None), False),
        ("Agent-Interval-Blind", lambda: AgenticPolicy(use_intervals=True, roster_blind=True, decision_log_path=None), False),
        ("Agent-Perfect", lambda: AgenticPolicy(use_intervals=True, decision_log_path=None), True),
    ]

    print("=" * 95)
    print("VERIFICATION OF EXACT MATCH AGAINST results_final_v1.json (20 seeds)")
    print("=" * 95)
    print(f"{'Scenario':12s} | {'Policy':22s} | {'Run Cost':15s} | {'v1 Cost':15s} | {'Match?':8s}")
    print("-" * 95)

    all_matched = True
    results_summary = []

    for sc_id in scenarios:
        cfg = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario=sc_id)
        sim_real = FleetSimulator(config=cfg, trajectories=trajectories_real)
        sim_perf = FleetSimulator(config=cfg, trajectories=trajectories_perf)

        v1_scen = v1["scenarios"][sc_id]["policies"]

        for pol_name, pol_factory, is_perf in policies:
            sim = sim_perf if is_perf else sim_real
            costs = []
            fails = []
            for s in eval_seeds:
                pol = pol_factory()
                if is_perf:
                    for b in pol.prognostic.residuals_by_bin:
                        pol.prognostic.residuals_by_bin[b] = np.array([0.0])
                res = sim.simulate(pol, eval_pool, seed=s)
                costs.append(res["total_cost"])
                fails.append(res["number_of_failures"])

            mean_c = float(np.mean(costs))
            mean_f = float(np.mean(fails))

            if pol_name in v1_scen:
                v1_c = float(v1_scen[pol_name]["mean_cost"])
                v1_f = float(v1_scen[pol_name]["mean_failures"])
                diff = abs(mean_c - v1_c)
                matched = diff < 1e-3
                match_str = "EXACT" if matched else f"DIFF ${diff:,.0f}"
                if not matched:
                    all_matched = False
                v1_c_str = f"${v1_c:,.0f}"
            else:
                v1_c_str = "N/A (new)"
                match_str = "NEW"

            c_str = f"${mean_c:,.0f}"
            print(f"{sc_id:12s} | {pol_name:22s} | {c_str:15s} | {v1_c_str:15s} | {match_str:8s}")
            results_summary.append({
                "scenario": sc_id,
                "policy": pol_name,
                "cost": mean_c,
                "failures": mean_f,
                "v1_cost": v1_c if pol_name in v1_scen else None,
                "matched": matched if pol_name in v1_scen else True,
            })

    print("-" * 95)
    if all_matched:
        print("[SUCCESS] All policies in all scenarios match results_final_v1.json bit-for-bit!")
    else:
        print("[FAILURE] Some policies differed from results_final_v1.json!")

if __name__ == "__main__":
    main()
