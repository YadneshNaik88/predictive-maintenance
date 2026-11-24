import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import FleetSimulator, SimulationConfig, load_engine_trajectories, get_tuning_eval_pools
import importlib.util

spec = importlib.util.spec_from_file_location("agents_v1", "scratch/clean_agents_v1.py")
agents_v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agents_v1)

trajectories = load_engine_trajectories()
_, eval_pool = get_tuning_eval_pools(seed=42)
cfg = SimulationConfig.from_yaml("config/sim_params.yaml", scenario="scenario_b")
sim = FleetSimulator(config=cfg, trajectories=trajectories)

# Let's test difference 1: is_overdue in plan_fleet
import re
with open("scratch/clean_agents_v1.py") as f:
    code = f.read()

# Replace Step 1 & 2 with overdue logic
code_overdue = code.replace(
"""        # Step 1: Identify engines within planning horizon (predicted RUL <= L + W)
        candidate_engines: List[EngineObservation] = []
        for obs in observations:
            if obs.pred_rul <= (L + W):
                candidate_engines.append(obs)""",
"""        # Step 1: Identify engines within planning horizon or overdue
        candidate_engines: List[EngineObservation] = []
        for obs in observations:
            s_id = obs.slot_id
            is_overdue = False
            if s_id in self.slot_plans:
                pl_start = self.slot_plans[s_id].get("planned_start")
                if pl_start is not None and pl_start < now:
                    is_overdue = True
            if obs.pred_rul <= (L + W) or is_overdue:
                candidate_engines.append(obs)"""
)

# Also in step 2:
code_overdue = code_overdue.replace(
"""                planned_start = plan.get("planned_start", now + 999)
                lower_dropped = (obs.lower_rul < (planned_start - now))
                if is_periodic or lower_dropped:
                    to_plan.append(obs)""",
"""                planned_start = plan.get("planned_start", now + 999)
                is_overdue = (planned_start < now)
                is_periodic = (now % self.replan_k == 1)
                lower_dropped = (obs.lower_rul < (planned_start - now))
                if is_overdue or is_periodic or lower_dropped:
                    to_plan.append(obs)"""
)

with open("scratch/test_overdue_mod.py", "w") as f:
    f.write(code_overdue)

spec2 = importlib.util.spec_from_file_location("agents_overdue", "scratch/test_overdue_mod.py")
agents_overdue = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(agents_overdue)

pol2 = agents_overdue.AgenticPolicy(use_intervals=True, decision_log_path=None)
pol2.is_agentic = False
res2 = sim.simulate(pol2, eval_pool, seed=42)
print("With overdue replan:", res2["total_cost"], "fails:", res2["number_of_failures"])
