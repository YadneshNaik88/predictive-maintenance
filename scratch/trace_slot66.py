import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Let's inspect CoordinatorAgent.plan_fleet at cycle 72 for Slot 66
from scratch.debug_c71 import agent, observations
# Let's see: what did plan_fleet do at cycle 72?
print("Let's check Slot 66 decision records:")
for rec in agent.coordinator.decision_records:
    if rec.get("slot_id") == 66:
        print(rec)
