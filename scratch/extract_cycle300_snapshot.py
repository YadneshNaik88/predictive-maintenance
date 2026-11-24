"""
scratch/extract_cycle300_snapshot.py

Extracts the exact 100-slot state at cycle 300 for Scenario B, seed 42.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig
from src.agents import AgenticPolicy, PrognosticAgent

def main():
    sim_cfg = SimulationConfig.from_yaml(scenario="scenario_b")
    sim_cfg.horizon = 301
    sim = FleetSimulator(config=sim_cfg)
    policy = AgenticPolicy(use_intervals=True)
    pool = np.arange(1, 101)

    # We want the state at cycle 300
    # Let's run simulation step-by-step or check slot states
    # In FleetSimulator.simulate, slot_cycle_idx and slot_unit are updated each cycle.
    # Let's inspect simulate loop in simulator.py
    with open(PROJECT_ROOT / "src" / "simulator.py") as f:
        code = f.read()
    print("Simulator file read, length:", len(code))

if __name__ == "__main__":
    main()
