import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.agents import AgenticPolicy
agent = AgenticPolicy()
agent.init_simulation(100, [4]*120)
agent.resource.reserve(71, 2)
print("After reserve:", agent.resource.reserved_calendar[70:74])
agent.on_job_start(0, 71)
print("After on_job_start with no plan:", agent.resource.reserved_calendar[70:74])
agent.coordinator.slot_plans[0] = {"planned_start": 71, "duration": 2, "reserved": True}
agent.on_job_start(0, 71)
print("After on_job_start with plan:", agent.resource.reserved_calendar[70:74])
