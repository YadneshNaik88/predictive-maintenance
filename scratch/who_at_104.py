import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Let's inspect cycle 102 and 103 from first_rejection
from scratch.first_rejection import agent, slot_state, slot_job_type, slot_maint_remaining, slot_start_maint_time
in_maint = np.where(slot_state == 5)[0]
print("Slots in maintenance at cycle 103:")
for s in in_maint:
    print(f"  Slot {s}: type={slot_job_type[s]}, rem_dur={slot_maint_remaining[s]}, start_time={slot_start_maint_time[s]}")

print("\nWho has reservations at cycle 104?")
for s, p in agent.coordinator.slot_plans.items():
    st = p.get("planned_start")
    dur = p.get("duration", 2)
    if st is not None and st <= 104 < st + dur and p.get("reserved"):
        print(f"  Slot {s}: plan={p}")
