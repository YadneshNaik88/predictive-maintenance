"""
test_simulator.py

Pytest suite for Phase 5 & 5b Discrete-Event Fleet Maintenance Simulator and Baselines.
Verifies all 6 mandatory sanity checks:
1. Total cost == sum of its four components (preventive, corrective, downtime, spare).
2. Run-to-Failure failures == completed engine lives and preventive actions == 0.
3. Ideal downtime (unlimited tech, zero lead time) == preventive_actions * duration + failures * duration.
4. Diagnostic Oracle strictly outperforms all three baseline policies.
5. Deterministic reproducibility: identical seeds produce identical results.
6. Unused spare parts count is small (only a few per run, <= 10 parts across all 100 slots).
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.policies import (
    RunToFailurePolicy,
    FixedIntervalPolicy,
    NaiveThresholdPolicy,
    DiagnosticOraclePolicy,
)
from src.simulator import (
    FleetSimulator,
    SimulationConfig,
    load_engine_trajectories,
    get_tuning_eval_pools,
    simulate_technician_queue,
)


@pytest.fixture(scope="module")
def sim_env():
    """Load trajectories, pools, and default simulator instance."""
    trajectories = load_engine_trajectories()
    tuning_pool, eval_pool = get_tuning_eval_pools(seed=42)
    config = SimulationConfig()
    simulator = FleetSimulator(config=config, trajectories=trajectories)
    return {
        "simulator": simulator,
        "config": config,
        "trajectories": trajectories,
        "tuning_pool": tuning_pool,
        "eval_pool": eval_pool,
    }


def test_sanity_check_1_cost_components_sum(sim_env):
    """
    Sanity Check 1:
    Total cost must strictly equal the sum of its four cost components:
    total_cost == preventive_cost + corrective_cost + production_loss + spare_cost
    for every simulated run.
    """
    sim = sim_env["simulator"]
    pool = sim_env["eval_pool"]
    policies = [
        RunToFailurePolicy(),
        FixedIntervalPolicy(140),
        NaiveThresholdPolicy(15),
        DiagnosticOraclePolicy(margin=1),
    ]

    for pol in policies:
        for seed in [42, 100]:
            res = sim.simulate(pol, pool, seed=seed, horizon=500)
            expected_sum = (
                res["preventive_cost"]
                + res["corrective_cost"]
                + res["production_loss"]
                + res["spare_cost"]
            )
            assert res["total_cost"] == expected_sum, (
                f"Cost mismatch for {pol.name}: total={res['total_cost']} != sum={expected_sum}"
            )


def test_sanity_check_2_run_to_failure_properties(sim_env):
    """
    Sanity Check 2:
    Under Run-to-Failure:
    - Number of preventive actions must be strictly 0.
    - Number of failures must equal completed engine lives.
    """
    sim = sim_env["simulator"]
    pool = sim_env["eval_pool"]
    res = sim.simulate(RunToFailurePolicy(), pool, seed=42, horizon=1000)

    assert res["number_of_preventive_actions"] == 0, (
        f"RTF had non-zero preventive actions: {res['number_of_preventive_actions']}"
    )
    assert res["number_of_failures"] == res["completed_engine_lives"], (
        f"RTF failures ({res['number_of_failures']}) != completed lives ({res['completed_engine_lives']})"
    )


def test_sanity_check_3_ideal_downtime(sim_env):
    """
    Sanity Check 3:
    With unlimited technicians (e.g. 99,999) and zero spare-part lead time (0 cycles),
    total downtime cycles must exactly equal:
    (preventive_actions * preventive_duration) + (failures * corrective_duration).
    """
    trajectories = sim_env["trajectories"]
    pool = sim_env["eval_pool"]
    ideal_cfg = SimulationConfig(
        spare_lead_time=0,
        num_technicians=99999,
        preventive_duration=2,
        corrective_duration=6,
    )
    ideal_sim = FleetSimulator(config=ideal_cfg, trajectories=trajectories)

    res = ideal_sim.simulate(FixedIntervalPolicy(140, lead_time=0), pool, seed=42, horizon=1000)

    expected_downtime = (
        res["number_of_preventive_actions"] * ideal_cfg.preventive_duration
        + res["number_of_failures"] * ideal_cfg.corrective_duration
    )
    assert res["total_downtime_cycles"] == expected_downtime, (
        f"Ideal downtime mismatch: observed {res['total_downtime_cycles']} != expected {expected_downtime}"
    )


def test_sanity_check_4_oracle_beats_baselines(sim_env):
    """
    Sanity Check 4:
    A diagnostic ORACLE policy (using ground-truth RUL) must achieve lower total cost
    than all three baseline policies (RTF, Fixed Interval, Naive Threshold).
    """
    sim = sim_env["simulator"]
    pool = sim_env["eval_pool"]
    seed = 42

    res_rtf = sim.simulate(RunToFailurePolicy(), pool, seed=seed)
    res_fixed = sim.simulate(FixedIntervalPolicy(140), pool, seed=seed)
    res_thresh = sim.simulate(NaiveThresholdPolicy(15), pool, seed=seed)
    res_oracle = sim.simulate(DiagnosticOraclePolicy(margin=1), pool, seed=seed)

    oracle_cost = res_oracle["total_cost"]

    assert oracle_cost < res_rtf["total_cost"], (
        f"Oracle (${oracle_cost:,.0f}) did not beat RTF (${res_rtf['total_cost']:,.0f})"
    )
    assert oracle_cost < res_fixed["total_cost"], (
        f"Oracle (${oracle_cost:,.0f}) did not beat Fixed Interval (${res_fixed['total_cost']:,.0f})"
    )
    assert oracle_cost < res_thresh["total_cost"], (
        f"Oracle (${oracle_cost:,.0f}) did not beat Naive Threshold (${res_thresh['total_cost']:,.0f})"
    )


def test_sanity_check_5_seed_reproducibility(sim_env):
    """
    Sanity Check 5:
    Simulations run with the exact same seed must produce strictly identical metrics.
    """
    sim = sim_env["simulator"]
    pool = sim_env["eval_pool"]
    policy = NaiveThresholdPolicy(15)

    run1 = sim.simulate(policy, pool, seed=777, horizon=1000)
    run2 = sim.simulate(policy, pool, seed=777, horizon=1000)

    for metric in [
        "total_cost",
        "preventive_cost",
        "corrective_cost",
        "production_loss",
        "spare_cost",
        "number_of_failures",
        "number_of_preventive_actions",
        "total_downtime_cycles",
        "total_operating_cycles",
        "average_maintenance_lead",
        "mean_wasted_rul",
        "utilization",
        "unused_spares",
    ]:
        assert run1[metric] == run2[metric], (
            f"Reproducibility failure on {metric}: {run1[metric]} != {run2[metric]}"
        )


def test_sanity_check_6_unused_spare_parts_count(sim_env):
    """
    Sanity Check 6:
    Unused spare parts count (ordered but not yet consumed/started at horizon H)
    must be small (only a few per run, <= 10 parts across all 100 slots).
    """
    sim = sim_env["simulator"]
    pool = sim_env["eval_pool"]
    policies = [
        RunToFailurePolicy(),
        FixedIntervalPolicy(140),
        NaiveThresholdPolicy(15),
        DiagnosticOraclePolicy(margin=1),
    ]

    for pol in policies:
        for seed in [42, 100, 2026]:
            res = sim.simulate(pol, pool, seed=seed, horizon=1000)
            unused = res["unused_spares"]
            spares_ordered = res["total_spares_ordered"]
            lives = res["completed_engine_lives"]

            assert 0 <= unused <= 10, (
                f"Unused spare parts count ({unused}) out of expected range [0, 10] for {pol.name} (seed {seed})"
            )
            # Spares ordered should closely match completed lives + in-flight slots
            assert abs(spares_ordered - lives) <= 15, (
                f"Spares ordered ({spares_ordered}) diverges substantially from completed lives ({lives})"
            )


# =============================================================================
# Phase 5c: Technician Model Audit Micro-Tests & Conservation Suite
# =============================================================================

def test_technician_audit_2a_micro_test():
    """
    Micro-test 2a:
    Capacity 1 per cycle, 3 jobs of duration 2 requested at cycle 0.
    Hand-checkable result:
    Jobs must start at t=0, t=2, t=4 with queue waits 0, 2, 4.
    """
    starts, waits = simulate_technician_queue(
        capacities=[1] * 10,
        jobs_requested_cycle=[0, 0, 0],
        job_durations=[2, 2, 2],
    )
    assert starts == [0, 2, 4], f"Starts mismatch: {starts} != [0, 2, 4]"
    assert waits == [0, 2, 4], f"Waits mismatch: {waits} != [0, 2, 4]"


def test_technician_audit_2b_micro_test():
    """
    Micro-test 2b:
    Capacity 2 per cycle, 4 jobs of duration 2 requested together at cycle 0.
    Hand-checkable result:
    Two start at t=0, two start at t=2 with queue waits 0, 0, 2, 2.
    """
    starts, waits = simulate_technician_queue(
        capacities=[2] * 10,
        jobs_requested_cycle=[0, 0, 0, 0],
        job_durations=[2, 2, 2, 2],
    )
    assert starts == [0, 0, 2, 2], f"Starts mismatch: {starts} != [0, 0, 2, 2]"
    assert waits == [0, 0, 2, 2], f"Waits mismatch: {waits} != [0, 0, 2, 2]"


def test_technician_audit_2c_micro_test():
    """
    Micro-test 2c:
    Capacity schedule [0, 0, 1, 1, ...], a job requested at cycle 0.
    Hand-checkable result:
    Must start at cycle 2 with queue wait 2 cycles.
    """
    starts, waits = simulate_technician_queue(
        capacities=[0, 0, 1, 1, 1, 1],
        jobs_requested_cycle=[0],
        job_durations=[2],
    )
    assert starts == [2], f"Starts mismatch: {starts} != [2]"
    assert waits == [2], f"Waits mismatch: {waits} != [2]"


def test_technician_audit_2d_conservation(sim_env):
    """
    Audit 2d: Conservation Law.
    In every run across scenarios and policies:
    Total technician-busy cycles <= total available technician-cycles.
    Assert it.
    """
    trajectories = sim_env["trajectories"]
    pool = sim_env["eval_pool"]

    cfg_a = SimulationConfig.from_yaml(scenario="scenario_a")
    cfg_b = SimulationConfig.from_yaml(scenario="scenario_b")
    cfg_c = SimulationConfig.from_yaml(scenario="scenario_c")

    policies = [
        RunToFailurePolicy(),
        FixedIntervalPolicy(125),
        NaiveThresholdPolicy(20),
        DiagnosticOraclePolicy(margin=1),
    ]

    for cfg, scen_name in [(cfg_a, "A"), (cfg_b, "B"), (cfg_c, "C")]:
        sim = FleetSimulator(config=cfg, trajectories=trajectories)
        for pol in policies:
            res = sim.simulate(pol, pool, seed=42, horizon=200)
            assert res["total_tech_busy"] <= res["total_tech_available"], (
                f"Conservation violation in Scenario {scen_name} ({pol.name}): "
                f"busy ({res['total_tech_busy']}) > available ({res['total_tech_available']})"
            )
            assert res["utilization"] <= 1.0, (
                f"Utilization > 1.0 in Scenario {scen_name} ({pol.name}): {res['utilization']}"
            )


def test_technician_audit_2e_utilization_vs_offered_load(sim_env):
    """
    Audit 2e: Utilization vs Offered Load Separation.
    utilization = busy technician-cycles / available technician-cycles (must be <= 1.0)
    offered load = demanded technician-cycles / available technician-cycles (may exceed 1.0)
    Under Fixed-Interval T=125 in Scenario B, offered load exceeds 1.0 while utilization <= 1.0.
    """
    trajectories = sim_env["trajectories"]
    pool = sim_env["eval_pool"]
    cfg_b = SimulationConfig.from_yaml(scenario="scenario_b")
    sim = FleetSimulator(config=cfg_b, trajectories=trajectories)

    res = sim.simulate(FixedIntervalPolicy(125), pool, seed=42, horizon=1000)

    assert res["utilization"] <= 1.0, f"Utilization exceeded 1.0: {res['utilization']}"
    assert res["offered_load"] > 1.0, f"Offered load did not exceed 1.0: {res['offered_load']}"
    assert res["utilization"] <= res["offered_load"], (
        f"Utilization ({res['utilization']}) should be <= offered load ({res['offered_load']})"
    )
