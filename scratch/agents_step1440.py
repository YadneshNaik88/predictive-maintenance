"""
test_agents.py

Unit tests and sanity checks for Phase 6 Multi-Agent Maintenance System.
Tests:
1. Information rule: agent modules cannot access true RUL.
2. Baselines unchanged: RTF/Fixed/Naive reproduce Phase 5d numbers exactly for fixed seed 42.
3. Sanity: with unlimited technicians, lead time 0 and perfect predictions injected
   (pred = true RUL, zero-width interval), Agent-Interval's cost is within 3% of the oracle.
4. The brief's example (E-017: predicted RUL 28, interval 21-35, lead time 8,
   technicians only at cycles 16 and 22) as a fixture. Prints decision log and reports
   which cycle chosen and why.
5. Reservation consistency: reserved technician-cycles never exceed roster capacity.
6. Reproducibility with a fixed seed.
"""

from dataclasses import fields
import inspect
from pathlib import Path
import pytest
import numpy as np

from src.simulator import (
    FleetSimulator,
    SimulationConfig,
    EngineObservation,
    load_engine_trajectories,
    get_tuning_eval_pools,
)
from src.policies import (
    RunToFailurePolicy,
    FixedIntervalPolicy,
    NaiveThresholdPolicy,
    DiagnosticOraclePolicy,
)
from src.agents import (
    AgenticPolicy,
    PrognosticAgent,
    ResourceAgent,
    CostAgent,
    CoordinatorAgent,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Test 1: Information Rule - Agent modules cannot access true RUL
# -----------------------------------------------------------------------------
def test_1_information_rule_no_true_rul():
    """Verify agent observation object strictly excludes true RUL and future predictions."""
    obs_fields = [f.name for f in fields(EngineObservation)]
    assert "true_rul" not in obs_fields, "Leakage violation: 'true_rul' found in EngineObservation!"
    assert "true_rul_raw" not in obs_fields, "Leakage violation: 'true_rul_raw' found in EngineObservation!"
    assert "true_rul_capped" not in obs_fields, "Leakage violation: 'true_rul_capped' found in EngineObservation!"

    obs = EngineObservation(
        slot_id=0,
        unit_number=1,
        current_cycle=10,
        age=50,
        pred_rul=45.0,
        lower_rul=38.0,
        upper_rul=52.0,
        order_already_placed=False,
    )
    with pytest.raises(AttributeError):
        _ = obs.true_rul

    with pytest.raises(AttributeError):
        _ = obs.true_rul_raw

    # Check method signatures of agents to ensure none take true_rul
    for cls in [PrognosticAgent, ResourceAgent, CostAgent, CoordinatorAgent]:
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue
            sig = inspect.signature(method)
            assert "true_rul" not in sig.parameters, (
                f"Leakage violation: Method {cls.__name__}.{name} accepts true_rul!"
            )


# -----------------------------------------------------------------------------
# Test 2: Baselines Unchanged - Exact Phase 5d Reproduction for Seed 42
# -----------------------------------------------------------------------------
def test_2_baselines_unchanged():
    """Verify simulator changes preserve exact bit-for-bit baseline metrics for seed 42."""
    trajectories = load_engine_trajectories()
    _, eval_pool = get_tuning_eval_pools(seed=42)
    cfg = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario="scenario_a")
    sim = FleetSimulator(config=cfg, trajectories=trajectories)

    # 1. Run-to-Failure
    res_rtf = sim.simulate(RunToFailurePolicy(), eval_pool, seed=42)
    assert res_rtf["total_cost"] == 5073000.0
    assert res_rtf["number_of_failures"] == 427
    assert res_rtf["number_of_preventive_actions"] == 0
    assert res_rtf["total_downtime_cycles"] == 6749

    # 2. Fixed-Interval (T=125)
    res_fixed = sim.simulate(FixedIntervalPolicy(125), eval_pool, seed=42)
    assert res_fixed["total_cost"] == 1999900.0
    assert res_fixed["number_of_failures"] == 0
    assert res_fixed["number_of_preventive_actions"] == 800
    assert res_fixed["total_downtime_cycles"] == 1599

    # 3. Naive-Threshold (tau=15)
    res_naive = sim.simulate(NaiveThresholdPolicy(15), eval_pool, seed=42)
    assert res_naive["total_cost"] == 1190900.0
    assert res_naive["number_of_failures"] == 0
    assert res_naive["number_of_preventive_actions"] == 476
    assert res_naive["total_downtime_cycles"] == 952

    # 4. Diagnostic-Oracle (margin=2)
    res_oracle = sim.simulate(DiagnosticOraclePolicy(margin=2), eval_pool, seed=42)
    assert res_oracle["total_cost"] == 1149000.0
    assert res_oracle["number_of_failures"] == 0
    assert res_oracle["number_of_preventive_actions"] == 459
    assert res_oracle["total_downtime_cycles"] == 918


# -----------------------------------------------------------------------------
# Test 3: Sanity Check - Perfect Predictions & Unlimited Resources within 3% of Oracle
# -----------------------------------------------------------------------------
def test_3_sanity_perfect_predictions_near_oracle():
    """Verify that with injected perfect predictions, zero spread, and unlimited resources, Agent is within 3% of Oracle."""
    trajectories_real = load_engine_trajectories()
    _, eval_pool = get_tuning_eval_pools(seed=42)

    trajectories_perfect = {}
    for u, data in trajectories_real.items():
        trajectories_perfect[u] = {
            "unit_number": data["unit_number"],
            "max_cycles": data["max_cycles"],
            "cycles": data["cycles"],
            "true_rul_raw": data["true_rul_raw"],
            "true_rul_capped": data["true_rul_capped"],
            "pred_rul": data["true_rul_raw"].copy(),
            "lower": data["true_rul_raw"].copy(),
            "upper": data["true_rul_raw"].copy(),
        }

    cfg_perfect = SimulationConfig(
        num_slots=100,
        horizon=1000,
        num_technicians=9999,
        spare_lead_time=0,
        staggered_start=True,
    )
    sim = FleetSimulator(config=cfg_perfect, trajectories=trajectories_perfect)

    oracle = DiagnosticOraclePolicy(lead_time=0, margin=1)
    res_oracle = sim.simulate(oracle, eval_pool, seed=42)

    agent_interval = AgenticPolicy(use_intervals=True, spare_lead_time=0, horizon_W=60, replan_k=5)
    # Zero-width residual distribution
    for b in agent_interval.prognostic.residuals_by_bin:
        agent_interval.prognostic.residuals_by_bin[b] = np.array([0.0])

    res_agent = sim.simulate(agent_interval, eval_pool, seed=42)

    diff_pct = abs(res_agent["total_cost"] - res_oracle["total_cost"]) / res_oracle["total_cost"] * 100.0
    print(f"\n[Test 3 Sanity] Oracle Cost: ${res_oracle['total_cost']:,.0f} | Agent Cost: ${res_agent['total_cost']:,.0f} | Diff: {diff_pct:.2f}%")
    assert diff_pct <= 3.0, f"Difference {diff_pct:.2f}% exceeded 3% threshold!"


# -----------------------------------------------------------------------------
# Test 4: Brief Example Fixture (E-017)
# -----------------------------------------------------------------------------
def test_4_brief_example_fixture_e017():
    """Verify Coordinator and Cost Agent evaluation for E-017 fixture (technicians at 16 and 22)."""
    prognostic = PrognosticAgent(use_intervals=True)
    resource = ResourceAgent(spare_lead_time=8, preventive_duration=2, horizon_W=60)
    cost = CostAgent()
    coordinator = CoordinatorAgent(
        prognostic=prognostic,
        resource=resource,
        cost=cost,
        replan_k=5,
        horizon_W=60,
        spare_lead_time=8,
    )

    # Roster has technician capacity only at cycles 16-17 and 22-23
    roster = np.zeros(65, dtype=int)
    roster[16:18] = 1
    roster[22:24] = 1

    coordinator.reset(total_horizon=60, capacity_roster=roster)

    obs_e017 = EngineObservation(
        slot_id=0,
        unit_number=17,
        current_cycle=0,
        age=120,
        pred_rul=28.0,
        lower_rul=21.0,
        upper_rul=35.0,
        order_already_placed=False,
    )

    orders = coordinator.plan_fleet(current_cycle=0, observations=[obs_e017])
    assert len(coordinator.decision_records) > 0
    rec = coordinator.decision_records[0]

    print("\n" + "=" * 60)
    print("FIXTURE E-017 DECISION RECORD:")
    print(f"Engine: {rec['engine']}, Pred RUL: {rec['predicted_rul']}, Interval: {rec['interval']}")
    print(f"Chosen Cycle: {rec['chosen_cycle']}, Escalated: {rec['escalated']}")
    print(f"Reason: {rec['reason']}")
    print("=" * 60)

    # Check candidates evaluated
    feasible_cycles = [c["start_cycle"] for c in rec["candidates"] if c["feasible"]]
    assert 16 in feasible_cycles, "Cycle 16 should be feasible"
    assert 22 in feasible_cycles, "Cycle 22 should be feasible"
    assert rec["chosen_cycle"] in [16, 22], f"Chosen cycle {rec['chosen_cycle']} must be 16 or 22"
    assert rec["escalated"] is False


# -----------------------------------------------------------------------------
# Test 5: Reservation Consistency - Reserved Never Exceeds Roster Capacity
# -----------------------------------------------------------------------------
def test_5_reservation_consistency():
    """Verify that in every cycle of simulation, reserved technicians never exceed roster capacity."""
    trajectories = load_engine_trajectories()
    _, eval_pool = get_tuning_eval_pools(seed=42)
    cfg = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario="scenario_b")
    sim = FleetSimulator(config=cfg, trajectories=trajectories)

    agent = AgenticPolicy(use_intervals=True, spare_lead_time=8)
    _ = sim.simulate(agent, eval_pool, seed=42)

    reserved = agent.resource.reserved_calendar
    capacity = agent.resource.capacity_roster
    min_len = min(len(reserved), len(capacity))

    violations = np.where(reserved[:min_len] > capacity[:min_len])[0]
    assert len(violations) == 0, (
        f"Reservation consistency violated at cycles: {violations}, "
        f"reserved={reserved[violations]}, capacity={capacity[violations]}"
    )


# -----------------------------------------------------------------------------
# Test 6: Reproducibility with Fixed Seed
# -----------------------------------------------------------------------------
def test_6_agent_reproducibility():
    """Verify that two simulation runs with identical seed produce identical costs and decisions."""
    trajectories = load_engine_trajectories()
    _, eval_pool = get_tuning_eval_pools(seed=42)
    cfg = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario="scenario_a")
    sim = FleetSimulator(config=cfg, trajectories=trajectories)

    agent1 = AgenticPolicy(use_intervals=True)
    res1 = sim.simulate(agent1, eval_pool, seed=42)

    agent2 = AgenticPolicy(use_intervals=True)
    res2 = sim.simulate(agent2, eval_pool, seed=42)

    assert res1["total_cost"] == res2["total_cost"]
    assert res1["number_of_failures"] == res2["number_of_failures"]
    assert res1["number_of_preventive_actions"] == res2["number_of_preventive_actions"]
    assert res1["total_downtime_cycles"] == res2["total_downtime_cycles"]
    assert np.array_equal(res1["slot_history"], res2["slot_history"])
