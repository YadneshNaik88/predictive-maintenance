"""
test_reservation_fix.py

Phase 8b Unit Tests:
1. Micro-test reproducing E-075: capacity schedule [2, 2, 1, 1, 3], two overdue
   unreserved jobs and one reserved job (reserved for cycles 145-146).
   At cycle 145, the reserved job must start, not the overdue jobs.
2. Invariant assertion on Scenario B run: assert no agent-planned job ever starts
   without technician capacity available for all D cycles of the job.
3. Baselines unchanged assertion: assert Run-to-Failure, Fixed Interval, Naive Threshold
   and Diagnostic Oracle give the exact same numbers across 20 seeds (A, B, C) as in
   outputs/results_final_v1.json (zero diff).
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.simulator import (
    FleetSimulator,
    SimulationConfig,
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
# Test 1: Micro-test Reproducing E-075 Dispatch Priority
# -----------------------------------------------------------------------------
def test_micro_e075_reservation_priority():
    """
    Micro-test reproducing E-075 scenario:
    Capacity schedule: [2, 2, 1, 1, 3] for cycles 143 to 147.
    Slot 86 holds a reservation for cycles 145-146.
    Slots 41 and 89 are overdue unreserved jobs ready since cycle 143/144.
    At cycle 145, the reserved job (Slot 86) must start, NOT the overdue unreserved jobs.
    """
    prognostic = PrognosticAgent()
    resource = ResourceAgent(spare_lead_time=8, preventive_duration=2, horizon_W=60)
    cost = CostAgent()
    coordinator = CoordinatorAgent(prognostic=prognostic, resource=resource, cost=cost)

    # Capacity schedule for cycles 0..150
    roster = np.full(160, 4, dtype=int)
    roster[143] = 2
    roster[144] = 2
    roster[145] = 1
    roster[146] = 1
    roster[147] = 3

    coordinator.reset(150, roster)

    # Slot 86 reserved for cycles 145-146
    assert coordinator.resource.reserve(145, 2) is True
    coordinator.slot_plans[86] = {
        "slot_id": 86,
        "planned_start": 145,
        "duration": 2,
        "reserved": True,
        "urgency_lower": 1.0,
    }

    # Slots 41 and 89 are overdue unreserved jobs (planned_start in past, unreserved)
    coordinator.slot_plans[41] = {
        "slot_id": 41,
        "planned_start": 143,
        "duration": 2,
        "reserved": False,
        "urgency_lower": 5.0,
    }
    coordinator.slot_plans[89] = {
        "slot_id": 89,
        "planned_start": 144,
        "duration": 2,
        "reserved": False,
        "urgency_lower": 6.0,
    }

    # At cycle 144 (capacity=2, reserved=0):
    # Overdue slots 41 and 89 check if they can start for window [144, 146).
    # For cycle 145, capacity=1, but Slot 86 holds a reservation (other_res=1).
    # Available = capacity(1) - committed(0) - other_res(1) = 0 < 1.
    # Therefore, neither unreserved slot may start at 144!
    for s_overdue in [41, 89]:
        can_start_144 = True
        for c in range(144, 146):
            cap_c = roster[c]
            comm_c = 0
            other_res = coordinator.get_other_reservations(s_overdue, c)
            if cap_c - comm_c - other_res < 1:
                can_start_144 = False
                break
        assert can_start_144 is False, f"Slot {s_overdue} should NOT be allowed to start at cycle 144"

    # At cycle 145 (capacity=1):
    # Slot 86 (holding active reservation at 145) has priority over overdue unreserved slots.
    assert coordinator.holds_active_reservation(86, 145) is True
    assert coordinator.holds_active_reservation(41, 145) is False
    assert coordinator.holds_active_reservation(89, 145) is False

    # Check Slot 86 multi-cycle availability for [145, 147):
    # Cycle 145: cap=1, other_res=0 -> avail=1
    # Cycle 146: cap=1, other_res=0 -> avail=1
    can_start_86 = True
    for c in range(145, 147):
        cap_c = roster[c]
        comm_c = 0
        other_res = coordinator.get_other_reservations(86, c)
        if cap_c - comm_c - other_res < 1:
            can_start_86 = False
            break
    assert can_start_86 is True, "Reserved Slot 86 MUST be allowed to start at cycle 145"

    # Once Slot 86 starts at 145, capacity at 145 is fully committed
    # Neither unreserved slot can start at 145
    committed_after_86 = {145: 1, 146: 1}
    for s_overdue in [41, 89]:
        can_start_145 = True
        for c in range(145, 147):
            cap_c = roster[c]
            comm_c = committed_after_86[c]
            other_res = coordinator.get_other_reservations(s_overdue, c)
            if cap_c - comm_c - other_res < 1:
                can_start_145 = False
                break
        assert can_start_145 is False, f"Slot {s_overdue} cannot start at cycle 145 when capacity is committed to 86"


# -----------------------------------------------------------------------------
# Test 2: Invariant Check on Scenario B Run
# -----------------------------------------------------------------------------
def test_scenario_b_planned_job_capacity_invariant():
    """
    Assert that in a Scenario B simulation run, no agent-planned job ever starts
    without technician capacity available for all D cycles of the job.
    Also verifies that stalled cycles drop to 0.
    """
    trajectories = load_engine_trajectories()
    _, eval_pool = get_tuning_eval_pools(seed=42)
    cfg = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario="scenario_b")
    sim = FleetSimulator(config=cfg, trajectories=trajectories)

    agent = AgenticPolicy(use_intervals=True, spare_lead_time=8, decision_log_path=None)
    res = sim.simulate(agent, eval_pool, seed=42)

    # Invariant: no agent-planned job starts without technician capacity available for all D cycles
    assert res.get("unsupported_planned_starts", 0) == 0, (
        f"Found {res.get('unsupported_planned_starts')} planned jobs started without full D-cycle capacity"
    )
    # Under the multi-cycle check, preventive stalls must be 0
    assert res.get("preventive_stalled_cycles", 0) == 0, (
        f"Preventive stalled cycles should be 0 under reservation fix, got {res.get('preventive_stalled_cycles')}"
    )


# -----------------------------------------------------------------------------
# Test 3: Baselines Unchanged Across 20 Seeds in Scenarios A, B, C
# -----------------------------------------------------------------------------
def test_baselines_identical_to_v1():
    """
    Assert that Run-to-Failure, Fixed Interval, Naive Threshold, and Diagnostic Oracle
    produce the EXACT same results as stored in outputs/results_final_v1.json across
    all 20 evaluation seeds in Scenarios A, B, and C (zero diff allowed).
    """
    v1_path = PROJECT_ROOT / "outputs" / "results_final_v1.json"
    assert v1_path.exists(), "outputs/results_final_v1.json must exist as canonical benchmark"

    with open(v1_path, "r", encoding="utf-8") as f:
        v1 = json.load(f)

    trajectories = load_engine_trajectories()
    _, eval_pool = get_tuning_eval_pools(seed=42)
    seeds = [42 + i for i in range(20)]

    for scen_id in ["scenario_a", "scenario_b", "scenario_c"]:
        cfg = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario=scen_id)
        sim = FleetSimulator(config=cfg, trajectories=trajectories)
        scen_data = v1["scenarios"][scen_id]
        t_star = scen_data["T_star"]
        tau_star = scen_data["tau_star"]
        m_star = scen_data["m_star"]

        baseline_specs = [
            ("Run-to-Failure", RunToFailurePolicy()),
            (f"Fixed Interval (T={t_star})", FixedIntervalPolicy(t_star, lead_time=cfg.spare_lead_time)),
            (f"Naive Threshold (tau={tau_star})", NaiveThresholdPolicy(tau_star)),
            (f"Diagnostic Oracle (m={m_star})", DiagnosticOraclePolicy(lead_time=cfg.spare_lead_time, margin=m_star)),
        ]

        for pol_name, pol in baseline_specs:
            v1_pol = scen_data["policies"][pol_name]
            for idx, s in enumerate(seeds):
                res = sim.simulate(pol, eval_pool, seed=s)
                assert res["total_cost"] == v1_pol["raw_costs"][idx], (
                    f"Cost mismatch in {scen_id} {pol_name} seed {s}: {res['total_cost']} vs {v1_pol['raw_costs'][idx]}"
                )
                assert res["number_of_failures"] == v1_pol["raw_failures"][idx], (
                    f"Failure mismatch in {scen_id} {pol_name} seed {s}: {res['number_of_failures']} vs {v1_pol['raw_failures'][idx]}"
                )
