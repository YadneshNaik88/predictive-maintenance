import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.data_loader import load_cmapss
from src.agents import PrognosticAgent, ResourceAgent, CostAgent, CoordinatorAgent
from src.simulator import EngineObservation
from demo import parse_technician_cycles

_, test_df, _ = load_cmapss(subset="FD001", data_dir=Path("data/raw"))
eng24 = test_df[test_df["unit_number"] == 24].copy()
max_c = int(eng24["time_in_cycles"].max())
print(f"Engine 24 max cycle: {max_c}")

prognostic = PrognosticAgent.from_checkpoint()
pred, low, up = prognostic.predict(eng24, method="binned_residual")
print(f"Engine 24 live prediction: pred={pred:.4f}, low={low:.4f}, up={up:.4f}")

lead_time = 8
d_pm = 2
resource = ResourceAgent(spare_lead_time=lead_time, preventive_duration=d_pm, horizon_W=60)
roster = parse_technician_cycles("16,22", current_cycle=max_c, horizon_W=60)
resource.reset(total_horizon=max_c + 60, capacity_roster=roster)
cost = CostAgent()
coordinator = CoordinatorAgent(prognostic, resource, cost, spare_lead_time=lead_time)
coordinator.reset(total_horizon=max_c + 60, capacity_roster=roster)

obs = EngineObservation(
    slot_id=0,
    unit_number=24,
    current_cycle=max_c,
    age=max_c,
    pred_rul=float(pred),
    lower_rul=float(low),
    upper_rul=float(up),
    order_already_placed=False,
)
res = coordinator.plan_fleet(current_cycle=max_c, observations=[obs])
rec = coordinator.decision_records[0]
print(f"Coordinator output: chosen_cycle={rec['chosen_cycle']}, escalated={rec['escalated']}")
for c in rec["candidates"]:
    if c["start_cycle"] == rec["chosen_cycle"]:
        print(f"Chosen candidate: start={c['start_cycle']}, pf={c['prob_fail']}, cr={c['cost_rate']}")
