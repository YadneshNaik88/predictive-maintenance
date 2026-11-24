import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Let's inspect what happened to Slot 81 when it was planned at cycle 98
# In scratch/test_perfect.py
