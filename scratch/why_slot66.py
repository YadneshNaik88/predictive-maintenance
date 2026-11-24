import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scratch.debug_c71 import agent

for r in agent.coordinator.decision_records:
    if "Slot 66" in r.get("reason", "") or r.get("engine") == f"E-{agent.coordinator.slot_plans[66]['unit_number']:03d}":
        print(f"Cycle {r['cycle']}: {r['engine']}, chosen={r['chosen_cycle']}, reason={r['reason']}")
