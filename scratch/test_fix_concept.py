import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Let's test the fix on max(now, earliest_part) first in trace_dispatch
from src.agents import CoordinatorAgent
# Look at line 483 in src/agents.py:
# earliest_part = obs.part_arrival_cycle if obs.order_already_placed and obs.part_arrival_cycle else (now + L)
# If earliest_part < now:
# earliest_search = max(now, earliest_part)
