"""
scratch/test_fix_micro_e075.py

Test implementation of the reservation/dispatch fix:
1. Micro-test reproducing E-075:
   Capacity 2, 2, 1, 1, 3 at cycles 143-147.
   Two overdue unreserved jobs and one reserved job at 145-146.
   The reserved job must start at 145.
2. Check that at cycle 144, neither overdue job starts because starting at 144 would
   require capacity at 145 which is only 1 and reserved by E-075.
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("C:/Users/naiky/OneDrive/Desktop/Projects/predictive-maintenance")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import SimulationConfig, FleetSimulator, load_engine_trajectories, get_tuning_eval_pools
from src.agents import AgenticPolicy, ResourceAgent, CoordinatorAgent, PrognosticAgent, CostAgent


def test_micro_e075():
    print("=" * 70)
    print("RUNNING MICRO-TEST: E-075 REPRODUCTION")
    print("=" * 70)

    # Roster capacity at cycles 143..147:
    # 143: 2, 144: 2, 145: 1, 146: 1, 147: 3
    cap_schedule = np.array([4] * 150)
    cap_schedule[143] = 2
    cap_schedule[144] = 2
    cap_schedule[145] = 1
    cap_schedule[146] = 1
    cap_schedule[147] = 3

    resource = ResourceAgent(spare_lead_time=8, preventive_duration=2, horizon_W=60)
    resource.reset(total_horizon=150, capacity_roster=cap_schedule)

    # Slot 86 (E-075) reserves cycles 145 and 146
    assert resource.reserve(145, 2), "Slot 86 reservation should succeed"
    print("Reservation calendar after Slot 86 reservation:")
    for c in [143, 144, 145, 146, 147]:
        print(f"  Cycle {c}: Capacity = {cap_schedule[c]}, Reserved = {resource.reserved_calendar[c]}")

    # Now simulate cycle 144:
    # Two overdue unreserved jobs: Slot 41 and Slot 89
    # Can Slot 41 or Slot 89 start at cycle 144?
    # Duration D = 2 -> cycles 144 and 145
    t = 144
    D = 2
    committed = np.zeros(160, dtype=int)

    def can_start(slot_id, start_cycle, duration, is_reserved_for_start):
        for c in range(start_cycle, start_cycle + duration):
            c_cap = cap_schedule[c]
            comm = committed[c]
            # If this slot holds reservation at c
            self_res = 1 if is_reserved_for_start and (145 <= c < 147 and slot_id == 86) else 0
            other_res = max(0, resource.reserved_calendar[c] - self_res)
            avail = c_cap - comm - other_res
            if avail < 1:
                return False
        return True

    # At cycle 144:
    can_start_41 = can_start(slot_id=41, start_cycle=144, duration=D, is_reserved_for_start=False)
    can_start_89 = can_start(slot_id=89, start_cycle=144, duration=D, is_reserved_for_start=False)
    print(f"\nAt cycle 144: Can Slot 41 start? {can_start_41} (Expected: False)")
    print(f"At cycle 144: Can Slot 89 start? {can_start_89} (Expected: False)")
    assert not can_start_41, "Slot 41 must not start at 144 because it would conflict with 145 reservation"
    assert not can_start_89, "Slot 89 must not start at 144 because it would conflict with 145 reservation"

    # At cycle 145:
    # Slot 86 has an active reservation at 145.
    # Slot 41 and 89 are unreserved / overdue.
    # Priority: Slot 86 first!
    can_start_86 = can_start(slot_id=86, start_cycle=145, duration=D, is_reserved_for_start=True)
    print(f"\nAt cycle 145: Can Slot 86 start? {can_start_86} (Expected: True)")
    assert can_start_86, "Slot 86 must start at 145"

    # Once Slot 86 starts at 145:
    committed[145 : 145 + D] += 1
    # Slot 86 releases its reservation because it is now in maintenance
    resource.release(145, D)

    # Now can Slot 41 or Slot 89 start at 145?
    can_start_41_at_145 = can_start(slot_id=41, start_cycle=145, duration=D, is_reserved_for_start=False)
    print(f"At cycle 145: After Slot 86 starts, can Slot 41 start? {can_start_41_at_145} (Expected: False)")
    assert not can_start_41_at_145, "Slot 41 cannot start at 145 because Slot 86 occupies capacity"

    # At cycle 147:
    # Slot 86 finished at end of 146!
    # Cap(147) = 3. Can Slot 41 start at 147?
    can_start_41_at_147 = can_start(slot_id=41, start_cycle=147, duration=D, is_reserved_for_start=False)
    print(f"\nAt cycle 147: Can Slot 41 start? {can_start_41_at_147} (Expected: True)")
    assert can_start_41_at_147, "Slot 41 can start at 147"

    print("\n[PASSED] Micro-test E-075 reproduction verified successfully!")


if __name__ == "__main__":
    test_micro_e075()
