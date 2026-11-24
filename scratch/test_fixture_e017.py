import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.simulator import EngineObservation
from src.agents import PrognosticAgent, ResourceAgent, CostAgent, CoordinatorAgent

# Setup agents
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

# Setup capacity roster: technicians only at cycles 16 and 22 (duration 2 -> 16-17 and 22-23)
roster = np.zeros(65, dtype=int)
roster[16:18] = 1
roster[22:24] = 1

coordinator.reset(total_horizon=60, capacity_roster=roster)

# Engine observation for E-017 at current_cycle = 0
obs_e017 = EngineObservation(
    slot_id=0,
    unit_number=17,
    current_cycle=0,
    age=120,
    pred_rul=28.0,
    lower_rul=21.0,
    upper_rul=35.0,
    order_already_placed=False,
    part_arrival_cycle=None,
    planned_start_cycle=None,
)

# Run planning pass
orders = coordinator.plan_fleet(current_cycle=0, observations=[obs_e017])
record = coordinator.decision_records[0]

print("=" * 80)
print("TEST 4: BRIEF EXAMPLE FIXTURE (E-017) DECISION LOG")
print("=" * 80)
print(f"Engine:             {record['engine']}")
print(f"Current Cycle:      {record['cycle']}")
print(f"Predicted RUL:      {record['predicted_rul']} (90% Interval: {record['interval']})")
print(f"Spare Order Cycle:  {record['spare_order_cycle']}")
print(f"Chosen Cycle:       {record['chosen_cycle']}")
print(f"Escalated:          {record['escalated']}")
print(f"\nPlain-English Reason:\n{record['reason']}")

print("\nCandidate Evaluations:")
for c in record["candidates"]:
    if c["feasible"]:
        print(f"  Cycle {c['start_cycle']:2d} (offset +{c['offset_s']:2d}): Feasible={c['feasible']} | "
              f"Cost Rate=${c['cost_rate']:.2f}/cycle | Fail Prob={c['prob_fail']*100:.2f}% | "
              f"Part Ready={c['part_ready']} | Tech Ready={c['tech_ready']}")
    elif c["start_cycle"] in [8, 16, 22, 28]:
        print(f"  Cycle {c['start_cycle']:2d} (offset +{c['offset_s']:2d}): Feasible={c['feasible']} | "
              f"Part Ready={c['part_ready']} | Tech Ready={c['tech_ready']}")

