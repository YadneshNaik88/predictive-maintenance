import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import (
    FleetSimulator,
    SimulationConfig,
    load_engine_trajectories,
    get_tuning_eval_pools,
)
from src.agents import AgenticPolicy
from app.app import E017_FIXTURE

FULL_LOG = PROJECT_ROOT / "outputs" / "decision_log_full.jsonl"
SAMPLE_LOG = PROJECT_ROOT / "outputs" / "decision_log.jsonl"

def main():
    if FULL_LOG.exists():
        FULL_LOG.unlink()

    trajectories = load_engine_trajectories()
    _, eval_pool = get_tuning_eval_pools(seed=42)
    cfg = SimulationConfig.from_yaml(PROJECT_ROOT / "config" / "sim_params.yaml", scenario="scenario_b")
    sim = FleetSimulator(config=cfg, trajectories=trajectories)

    print("Running Scenario B seed 42 to generate full decision log...")
    agent = AgenticPolicy(use_intervals=True, spare_lead_time=8, decision_log_path=FULL_LOG)
    res = sim.simulate(agent, eval_pool, seed=42)
    print(f"Scenario B seed 42 complete: cost=${res['total_cost']:,.0f}, failures={res['number_of_failures']}")
    print(f"Full decision log written: {FULL_LOG.stat().st_size:,} bytes")

    # Rebuild sample decision log: E-017 fixture first, then records from seed 42
    target_bytes = 4 * 1024 * 1024
    written_bytes = 0
    count = 0

    with open(SAMPLE_LOG, "w", encoding="utf-8") as f_out:
        # Write E-017 fixture first
        line_fixture = json.dumps(E017_FIXTURE) + "\n"
        f_out.write(line_fixture)
        written_bytes += len(line_fixture.encode("utf-8"))

        with open(FULL_LOG, "r", encoding="utf-8") as f_in:
            for line in f_in:
                f_out.write(line)
                written_bytes += len(line.encode("utf-8"))
                count += 1
                if count >= 500 and written_bytes >= target_bytes:
                    break

    print(f"Sample decision log written: {SAMPLE_LOG.stat().st_size:,} bytes ({count} records + E-017 fixture)")

if __name__ == "__main__":
    main()
