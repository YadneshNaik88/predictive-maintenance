import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools
from src.agents import AgenticPolicy

trajectories = load_engine_trajectories()
eng = trajectories[70]
print("Unit 70 max cycles:", eng["max_cycles"])
print("Unit 70 true rul raw [:10]:", eng["true_rul_raw"][:10])
print("Unit 70 pred rul [:10]:", eng["pred_rul"][:10])
print("Unit 70 lower [:10]:", eng["lower"][:10])
