import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scratch.debug_c71 import agent
print("Slot 66 plan:", agent.coordinator.slot_plans[66])
