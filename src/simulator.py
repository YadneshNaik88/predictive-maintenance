"""
simulator.py

Discrete-Event Fleet Maintenance Simulator for NASA C-MAPSS FD001 Telemetry.
Phase 5 & 5b: Simulates a multi-slot aircraft engine fleet across operational flight cycles
under logistics constraints, queueing dynamics, technician bottlenecks, staggered starts,
and financial maintenance costs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml

from src.policies import BasePolicy

# Default paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOF_PATH = PROJECT_ROOT / "outputs" / "oof_predictions_L30.csv"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "sim_params.yaml"


def generate_scenario_b_capacity(
    seed: int = 42,
    horizon: int = 1000,
    values: Tuple[int, ...] = (0, 1, 2, 3),
    probabilities: Tuple[float, ...] = (0.1106, 0.3594, 0.3894, 0.1406),
) -> np.ndarray:
    """
    Generate deterministic per-cycle technician capacity schedule for Scenario B.
    Mean capacity = 0*0.1106 + 1*0.3594 + 2*0.3894 + 3*0.1406 = 1.549 technicians/cycle.
    """
    rng = np.random.default_rng(seed)
    return rng.choice(values, size=horizon + 20, p=probabilities)


def simulate_technician_queue(
    capacities: Union[List[int], np.ndarray],
    jobs_requested_cycle: List[int],
    job_durations: List[int],
) -> Tuple[List[int], List[int]]:
    """
    Micro-test reference technician queue model (Phase 5c audit).
    Simulates cycle-by-cycle dispatch and conservation of technician capacity.

    Parameters
    ----------
    capacities : list or array of int
        Per-cycle available technician capacity array.
    jobs_requested_cycle : list of int
        Cycle at which each job enters the technician queue (parts arrived).
    job_durations : list of int
        Maintenance work duration required for each job.

    Returns
    -------
    start_times : list of int
        Cycle at which each job starts maintenance.
    queue_waits : list of int
        Cycles each job spent waiting in the queue before maintenance started.
    """
    N = len(jobs_requested_cycle)
    H = len(capacities)
    start_times = [-1] * N
    queue_waits = [-1] * N
    in_service: List[List[int]] = []  # list of [job_idx, remaining_duration]
    queue: List[int] = []  # list of job_indices

    for t in range(H):
        cap_t = int(capacities[t])
        # Jobs arriving at queue at cycle t
        for j in range(N):
            if jobs_requested_cycle[j] == t:
                queue.append(j)

        # Technicians occupied by ongoing jobs entering cycle t
        ongoing_count = len(in_service)
        free_techs = max(0, cap_t - ongoing_count)

        # Dispatch queued jobs to available technicians
        while free_techs > 0 and len(queue) > 0:
            j = queue.pop(0)
            start_times[j] = t
            queue_waits[j] = t - jobs_requested_cycle[j]
            in_service.append([j, job_durations[j]])
            free_techs -= 1

        # Work active maintenance jobs
        num_to_work = min(len(in_service), cap_t)
        worked = in_service[:num_to_work]
        remaining_in_service: List[List[int]] = []
        for item in in_service:
            if item in worked:
                item[1] -= 1
                if item[1] > 0:
                    remaining_in_service.append(item)
            else:
                remaining_in_service.append(item)
        in_service = remaining_in_service

    return start_times, queue_waits


@dataclass(frozen=True)
class EngineObservation:
    """
    Strictly restricted observation exposed to autonomous maintenance agents.
    Adheres strictly to Information Rules:
    - Current simulation cycle and operating engine age
    - Prognostic model outputs: predicted RUL, lower (90%), upper (90%)
    - Spare part arrival / order status
    - Planned maintenance start cycle
    NEVER contains ground-truth RUL, future predictions, or simulator random state.
    """
    slot_id: int
    unit_number: int
    current_cycle: int
    age: int
    pred_rul: float
    lower_rul: float
    upper_rul: float
    order_already_placed: bool
    part_arrival_cycle: Optional[int] = None
    planned_start_cycle: Optional[int] = None


@dataclass
class SimulationConfig:
    """
    Configuration parameters for fleet maintenance simulation.
    Note: These are simulation parameters, not claims about real airline costs.
    """
    num_slots: int = 100
    horizon: int = 1000
    num_seeds: int = 20
    cost_preventive: float = 2000.0
    cost_corrective: float = 10000.0
    cost_production_loss: float = 100.0
    cost_spare: float = 300.0
    spare_lead_time: int = 8
    num_technicians: int = 4
    preventive_duration: int = 2
    corrective_duration: int = 6
    staggered_start: bool = True
    technicians_capacity_array: Optional[np.ndarray] = None

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
        scenario: Optional[str] = None,
    ) -> "SimulationConfig":
        """
        Load configuration parameters from YAML file.
        Optionally initialize specific scenario ('scenario_a' or 'scenario_b').
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        fleet_cfg = data.get("fleet", {})
        costs_cfg = data.get("costs", {})
        ops_cfg = data.get("operations", {})
        scenarios_cfg = data.get("scenarios", {})

        num_slots = fleet_cfg.get("num_slots", 100)
        horizon = fleet_cfg.get("horizon", 1000)
        num_seeds = fleet_cfg.get("num_seeds", 20)
        staggered_start = fleet_cfg.get("staggered_start", True)

        cost_preventive = costs_cfg.get("preventive_maintenance", 2000.0)
        cost_corrective = costs_cfg.get("corrective_failure", 10000.0)
        cost_production_loss = costs_cfg.get("production_loss_per_cycle", 100.0)
        cost_spare = costs_cfg.get("spare_part", 300.0)

        spare_lead_time = ops_cfg.get("spare_lead_time", 8)
        num_technicians = ops_cfg.get("num_technicians", 4)
        preventive_duration = ops_cfg.get("preventive_duration", 2)
        corrective_duration = ops_cfg.get("corrective_duration", 6)

        tech_capacity_array = None

        if scenario is not None:
            scen_key = scenario.lower()
            if scen_key in scenarios_cfg:
                scen = scenarios_cfg[scen_key]
                spare_lead_time = scen.get("spare_lead_time", spare_lead_time)
                if scen.get("capacity_type") == "seeded_schedule":
                    sched_seed = scen.get("schedule_seed", 42)
                    dist = scen.get("capacity_distribution", {})
                    vals = tuple(dist.get("values", [0, 1, 2, 3]))
                    probs = tuple(dist.get("probabilities", [0.04, 0.71, 0.23, 0.02]))
                    tech_capacity_array = generate_scenario_b_capacity(
                        seed=sched_seed,
                        horizon=horizon,
                        values=vals,
                        probabilities=probs,
                    )
                else:
                    num_technicians = int(scen.get("num_technicians", 4))

        return cls(
            num_slots=num_slots,
            horizon=horizon,
            num_seeds=num_seeds,
            cost_preventive=cost_preventive,
            cost_corrective=cost_corrective,
            cost_production_loss=cost_production_loss,
            cost_spare=cost_spare,
            spare_lead_time=spare_lead_time,
            num_technicians=num_technicians,
            preventive_duration=preventive_duration,
            corrective_duration=corrective_duration,
            staggered_start=staggered_start,
            technicians_capacity_array=tech_capacity_array,
        )


def load_engine_trajectories(
    oof_csv_path: Union[str, Path] = DEFAULT_OOF_PATH
) -> Dict[int, Dict[str, Any]]:
    """
    Load pre-computed cross-validated out-of-fold (L=30) predictions into memory.
    Returns dictionary mapping engine unit_number -> pre-extracted numpy arrays.
    """
    df = pd.read_csv(oof_csv_path)
    trajectories: Dict[int, Dict[str, Any]] = {}

    for unit_id, group in df.groupby("unit_number"):
        group = group.sort_values("cycle")
        trajectories[int(unit_id)] = {
            "unit_number": int(unit_id),
            "max_cycles": int(len(group)),
            "cycles": group["cycle"].values.astype(int),
            "true_rul_raw": group["true_rul_raw"].values.astype(float),
            "true_rul_capped": group["true_rul_capped"].values.astype(float),
            "pred_rul": group["pred_rul"].values.astype(float),
            "lower": group["lower"].values.astype(float) if "lower" in group else np.zeros(len(group)),
            "upper": group["upper"].values.astype(float) if "upper" in group else np.zeros(len(group)),
        }

    return trajectories


def get_tuning_eval_pools(seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministically partition the 100 C-MAPSS training engines into
    a Tuning pool (50 engines) and an Evaluation pool (50 engines).
    """
    rng = np.random.default_rng(seed)
    all_units = np.arange(1, 101)
    shuffled = rng.permutation(all_units)
    tuning_units = np.sort(shuffled[:50])
    eval_units = np.sort(shuffled[50:])
    return tuning_units, eval_units


class FleetSimulator:
    """
    Discrete-event cycle-by-cycle fleet maintenance simulator.
    Tracks slots, queues, lead times, technicians, downtime, and financial costs.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        trajectories: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> None:
        self.config = config if config is not None else SimulationConfig()
        if trajectories is None:
            self.trajectories = load_engine_trajectories()
        else:
            self.trajectories = trajectories

    def simulate(
        self,
        policy: BasePolicy,
        engine_pool: np.ndarray,
        seed: int,
        horizon: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute fleet simulation for given policy and engine pool.

        Parameters
        ----------
        policy : BasePolicy
            Maintenance decision policy instance.
        engine_pool : np.ndarray
            Array of allowable engine unit numbers (e.g. tuning or eval pool).
        seed : int
            Random seed for engine renewal draws and staggered starts.
        horizon : int, optional
            Simulation horizon in operational cycles (defaults to config.horizon).

        Returns
        -------
        results : dict
            Comprehensive dictionary of fleet performance metrics and event history.
        """
        H = horizon if horizon is not None else self.config.horizon
        N = self.config.num_slots
        policy.reset()

        rng = np.random.default_rng(seed)

        # Slot state tracking
        slot_unit = np.zeros(N, dtype=int)
        slot_cycle_idx = np.zeros(N, dtype=int)
        slot_op_cycles = np.zeros(N, dtype=int)

        # States:
        # 0: OPERATING (Normal, no order placed)
        # 1: ORDERED_WAITING_PART (Spare ordered, engine still operating)
        # 2: PART_ARRIVED_WAITING_TECH (Spare arrived, engine still operating, queued for tech)
        # 3: FAILED_WAITING_PART (Engine failed, unavailable, spare ordered/en route)
        # 4: FAILED_WAITING_TECH (Engine failed, unavailable, spare arrived, queued for tech)
        # 5: IN_MAINTENANCE (Job actively occupying a technician, unavailable)
        slot_state = np.zeros(N, dtype=int)
        slot_order_placed = np.zeros(N, dtype=bool)
        slot_order_cycle = np.zeros(N, dtype=int)
        slot_part_arrival = np.zeros(N, dtype=int)
        slot_earliest_start = np.zeros(N, dtype=int)
        slot_priority = np.zeros(N, dtype=float)
        slot_maint_remaining = np.zeros(N, dtype=int)
        slot_job_type = np.zeros(N, dtype=int)  # 1: PREVENTIVE, 2: CORRECTIVE
        slot_request_time = np.zeros(N, dtype=int)  # FCFS queue order
        slot_start_maint_time = np.zeros(N, dtype=int)

        # History tracking (0: Operating, 1: Waiting/Queued, 2: Maintenance, 3: Failed downtime)
        slot_history = np.zeros((N, H), dtype=np.int8)

        # Initialize slots with engines from the pool (with staggered start if enabled)
        for s in range(N):
            u = int(rng.choice(engine_pool))
            slot_unit[s] = u
            if self.config.staggered_start:
                max_c = self.trajectories[u]["max_cycles"]
                age = int(rng.integers(0, max_c // 2 + 1))
                slot_cycle_idx[s] = age
                slot_op_cycles[s] = age
            else:
                slot_cycle_idx[s] = 0
                slot_op_cycles[s] = 0
            slot_state[s] = 0
            slot_order_placed[s] = False
            policy.on_renewal(s)

        # Metrics trackers
        num_failures = 0
        num_preventive = 0
        total_downtime_cycles = 0
        total_operating_cycles = 0
        total_spares_ordered = 0
        completed_engine_lives = 0
        jobs_finished_maintenance = 0
        wasted_rul_list: List[float] = []
        maintenance_lead_list: List[int] = []
        queue_wait_list: List[int] = []
        total_tech_cycles_busy = 0
        total_tech_cycles_available = 0
        total_stalled_cycles = 0
        queue_failures = 0

        # Technician capacity per cycle
        if self.config.technicians_capacity_array is not None:
            tech_capacity_arr = self.config.technicians_capacity_array
        else:
            tech_capacity_arr = np.full(H + 20, self.config.num_technicians, dtype=int)

        if hasattr(policy, "init_simulation"):
            policy.init_simulation(total_horizon=H, capacity_roster=tech_capacity_arr)

        # Cycle-by-cycle simulation loop
        for t in range(1, H + 1):
            cap_t = int(tech_capacity_arr[t])
            total_tech_cycles_available += cap_t

            # Fleet-level multi-agent planning pass (if policy supports before_cycle)
            if hasattr(policy, "before_cycle"):
                observations = []
                for s in range(N):
                    if slot_state[s] in (0, 1, 2):
                        u = slot_unit[s]
                        c_idx = slot_cycle_idx[s]
                        eng_data = self.trajectories[u]
                        obs = EngineObservation(
                            slot_id=s,
                            unit_number=u,
                            current_cycle=t,
                            age=int(slot_op_cycles[s]),
                            pred_rul=float(eng_data["pred_rul"][c_idx]),
                            lower_rul=float(eng_data["lower"][c_idx]),
                            upper_rul=float(eng_data["upper"][c_idx]),
                            order_already_placed=bool(slot_order_placed[s]),
                            part_arrival_cycle=int(slot_part_arrival[s]) if slot_order_placed[s] else None,
                            planned_start_cycle=int(slot_earliest_start[s]) if slot_earliest_start[s] > 0 else None,
                        )
                        observations.append(obs)
                policy.before_cycle(current_cycle=t, observations=observations)
                for s in range(N):
                    if hasattr(policy, "get_earliest_start"):
                        slot_earliest_start[s] = policy.get_earliest_start(s, default=slot_earliest_start[s])
                    if hasattr(policy, "get_priority"):
                        slot_priority[s] = policy.get_priority(s, default=slot_priority[s])

            # -------------------------------------------------------------
            # Step A: Check Spare Part Arrivals
            # -------------------------------------------------------------
            # Operating engines whose spare part has arrived -> ready for tech
            waiting_part_operating = np.where((slot_state == 1) & (t >= slot_part_arrival))[0]
            for s in waiting_part_operating:
                slot_state[s] = 2  # PART_ARRIVED_WAITING_TECH
                slot_request_time[s] = slot_part_arrival[s]

            # Failed engines whose spare part has arrived -> ready for tech
            waiting_part_failed = np.where((slot_state == 3) & (t >= slot_part_arrival))[0]
            for s in waiting_part_failed:
                slot_state[s] = 4  # FAILED_WAITING_TECH
                slot_request_time[s] = slot_part_arrival[s]

            # -------------------------------------------------------------
            # Step B: Dispatch Technicians to Queued Jobs
            # -------------------------------------------------------------
            ongoing_maint = np.where(slot_state == 5)[0]
            free_techs = max(0, cap_t - len(ongoing_maint)) if cap_t < 99999 else 99999

            if free_techs > 0:
                # Priority 1: Failed engines ready for tech
                failed_ready = np.where(slot_state == 4)[0]
                # Priority 2: Planned preventive engines ready for tech (and reached planned start)
                planned_ready = np.where((slot_state == 2) & (t >= slot_earliest_start))[0]

                if len(failed_ready) > 1:
                    failed_ready = failed_ready[np.argsort(slot_request_time[failed_ready])]
                if len(planned_ready) > 1:
                    if np.any(slot_priority[planned_ready] > 0.0):
                        sort_keys = np.lexsort((slot_request_time[planned_ready], slot_priority[planned_ready]))
                        planned_ready = planned_ready[sort_keys]
                    else:
                        planned_ready = planned_ready[np.argsort(slot_request_time[planned_ready])]

                to_start_failed = failed_ready[:free_techs]
                for s in to_start_failed:
                    slot_state[s] = 5
                    slot_job_type[s] = 2  # CORRECTIVE
                    slot_maint_remaining[s] = self.config.corrective_duration
                    slot_start_maint_time[s] = t
                    lead = t - slot_order_cycle[s]
                    q_wait = max(0, t - slot_part_arrival[s])
                    maintenance_lead_list.append(lead)
                    queue_wait_list.append(q_wait)
                    free_techs -= 1

                if free_techs > 0:
                    to_start_planned = planned_ready[:free_techs]
                    for s in to_start_planned:
                        slot_state[s] = 5
                        slot_job_type[s] = 1  # PREVENTIVE
                        slot_maint_remaining[s] = self.config.preventive_duration
                        slot_start_maint_time[s] = t
                        num_preventive += 1
                        completed_engine_lives += 1
                        u = slot_unit[s]
                        c_idx = slot_cycle_idx[s]
                        wasted_rul_list.append(float(self.trajectories[u]["true_rul_raw"][c_idx]))
                        lead = t - slot_order_cycle[s]
                        q_wait = max(0, t - slot_part_arrival[s])
                        maintenance_lead_list.append(lead)
                        queue_wait_list.append(q_wait)
                        free_techs -= 1

            # -------------------------------------------------------------
            # Step C: Progress Active Maintenance Jobs
            # -------------------------------------------------------------
            all_in_maint = np.where(slot_state == 5)[0]
            if len(all_in_maint) > 1:
                # Priority: Corrective (2) before Preventive (1), then earliest start time
                sort_keys = np.lexsort((slot_start_maint_time[all_in_maint], -slot_job_type[all_in_maint]))
                all_in_maint = all_in_maint[sort_keys]

            num_to_work = min(len(all_in_maint), cap_t) if cap_t < 99999 else len(all_in_maint)
            active_worked = all_in_maint[:num_to_work]
            stalled_this_cycle = len(all_in_maint) - len(active_worked)
            total_stalled_cycles += stalled_this_cycle

            # Conservation assertion
            if cap_t < 99999:
                assert len(active_worked) <= cap_t, (
                    f"Conservation violation at cycle {t}: {len(active_worked)} > {cap_t}"
                )
            total_tech_cycles_busy += len(active_worked)

            # All slots in maintenance incur downtime this cycle
            for s in all_in_maint:
                total_downtime_cycles += 1
                slot_history[s, t - 1] = 2  # Maintenance active

            # Only actively worked slots decrement remaining duration
            for s in active_worked:
                slot_maint_remaining[s] -= 1
                if slot_maint_remaining[s] == 0:
                    jobs_finished_maintenance += 1
                    # Renew slot with a fresh engine (starts at cycle 1, age 0)
                    u_next = int(rng.choice(engine_pool))
                    slot_unit[s] = u_next
                    slot_cycle_idx[s] = 0
                    slot_op_cycles[s] = 0
                    slot_state[s] = 0  # Ready to operate next cycle
                    slot_order_placed[s] = False
                    slot_order_cycle[s] = 0
                    slot_part_arrival[s] = 0
                    slot_job_type[s] = 0
                    slot_earliest_start[s] = 0
                    slot_priority[s] = 0.0
                    policy.on_renewal(s)

            # -------------------------------------------------------------
            # Step D: Process Non-Maintenance Slots
            # -------------------------------------------------------------
            for s in range(N):
                st = slot_state[s]

                if st == 5:
                    continue
                elif st in (3, 4):
                    # FAILED waiting for spare or technician
                    total_downtime_cycles += 1
                    slot_history[s, t - 1] = 3  # Failed / Downtime
                elif st in (0, 1, 2):
                    # Engine OPERATES this cycle
                    total_operating_cycles += 1
                    slot_op_cycles[s] += 1
                    slot_history[s, t - 1] = 0  # Operating normally

                    u = slot_unit[s]
                    c_idx = slot_cycle_idx[s]
                    engine_data = self.trajectories[u]
                    max_c = engine_data["max_cycles"]
                    c_num = c_idx + 1  # 1-based cycle

                    if c_num >= max_c:
                        # Engine reached end of life and FAILS at this cycle!
                        num_failures += 1
                        completed_engine_lives += 1
                        if st == 2:
                            queue_failures += 1
                        if hasattr(policy, "on_engine_failure"):
                            policy.on_engine_failure(s)
                        if not slot_order_placed[s]:
                            slot_order_placed[s] = True
                            slot_order_cycle[s] = t
                            total_spares_ordered += 1
                            slot_part_arrival[s] = t + self.config.spare_lead_time
                            slot_earliest_start[s] = slot_part_arrival[s]
                            slot_priority[s] = 0.0
                            if self.config.spare_lead_time == 0:
                                slot_state[s] = 4  # FAILED_WAITING_TECH
                                slot_request_time[s] = t
                            else:
                                slot_state[s] = 3  # FAILED_WAITING_PART
                                slot_request_time[s] = t
                        else:
                            if st == 2:
                                slot_state[s] = 4  # FAILED_WAITING_TECH
                                slot_request_time[s] = t
                            else:
                                slot_state[s] = 3  # FAILED_WAITING_PART
                    else:
                        slot_cycle_idx[s] += 1
                        if st == 0:
                            pred_r = float(engine_data["pred_rul"][c_idx])
                            low_r = float(engine_data["lower"][c_idx])
                            up_r = float(engine_data["upper"][c_idx])
                            true_r = float(engine_data["true_rul_raw"][c_idx])

                            should_order = policy.decide(
                                slot_id=s,
                                current_cycle=t,
                                operating_cycles=slot_op_cycles[s],
                                pred_rul=pred_r,
                                lower_rul=low_r,
                                upper_rul=up_r,
                                true_rul=true_r,
                                unit_number=u,
                                engine_cycle=c_num,
                                order_already_placed=slot_order_placed[s],
                            )

                            if should_order:
                                slot_order_placed[s] = True
                                slot_order_cycle[s] = t
                                total_spares_ordered += 1
                                slot_part_arrival[s] = t + self.config.spare_lead_time
                                if hasattr(policy, "get_earliest_start"):
                                    slot_earliest_start[s] = policy.get_earliest_start(s, default=slot_part_arrival[s])
                                else:
                                    slot_earliest_start[s] = slot_part_arrival[s]
                                if hasattr(policy, "get_priority"):
                                    slot_priority[s] = policy.get_priority(s, default=0.0)
                                else:
                                    slot_priority[s] = 0.0

                                if self.config.spare_lead_time == 0 and t >= slot_earliest_start[s]:
                                    slot_state[s] = 2  # PART_ARRIVED_WAITING_TECH
                                    slot_request_time[s] = t
                                else:
                                    slot_state[s] = 1  # ORDERED_WAITING_PART
                                    slot_request_time[s] = t

        # Global conservation check
        if total_tech_cycles_available < 99999 * H:
            assert total_tech_cycles_busy <= total_tech_cycles_available, (
                f"Global conservation violation: busy={total_tech_cycles_busy} > available={total_tech_cycles_available}"
            )

        # Compute cost components
        prev_cost = num_preventive * self.config.cost_preventive
        corr_cost = num_failures * self.config.cost_corrective
        prod_loss = total_downtime_cycles * self.config.cost_production_loss
        spare_cost = total_spares_ordered * self.config.cost_spare
        total_cost = prev_cost + corr_cost + prod_loss + spare_cost

        cost_per_op = total_cost / total_operating_cycles if total_operating_cycles > 0 else 0.0
        avg_lead = float(np.mean(maintenance_lead_list)) if maintenance_lead_list else 0.0
        mean_q_wait = float(np.mean(queue_wait_list)) if queue_wait_list else 0.0
        mean_wasted = float(np.mean(wasted_rul_list)) if wasted_rul_list else 0.0
        utilization = total_tech_cycles_busy / total_tech_cycles_available if total_tech_cycles_available > 0 else 0.0
        demanded_tech_cycles = (num_preventive * self.config.preventive_duration) + (num_failures * self.config.corrective_duration)
        offered_load = demanded_tech_cycles / total_tech_cycles_available if total_tech_cycles_available > 0 else 0.0
        unused_parts = int(np.sum(np.isin(slot_state, [1, 2, 3, 4])))

        # Cost floor definition:
        # floor = completed jobs x (preventive cost + spare cost) + downtime cycles x loss rate
        cost_floor = (num_preventive + num_failures) * (self.config.cost_preventive + self.config.cost_spare) + total_downtime_cycles * self.config.cost_production_loss
        pct_above_floor = (total_cost - cost_floor) / cost_floor * 100.0 if cost_floor > 0 else 0.0

        return {
            "policy": policy.name,
            "seed": seed,
            "total_cost": total_cost,
            "preventive_cost": prev_cost,
            "corrective_cost": corr_cost,
            "production_loss": prod_loss,
            "spare_cost": spare_cost,
            "cost_floor": cost_floor,
            "pct_above_floor": pct_above_floor,
            "number_of_failures": num_failures,
            "number_of_preventive_actions": num_preventive,
            "completed_engine_lives": completed_engine_lives,
            "jobs_completed": jobs_finished_maintenance,
            "total_downtime_cycles": total_downtime_cycles,
            "total_operating_cycles": total_operating_cycles,
            "total_spares_ordered": total_spares_ordered,
            "unused_spares": unused_parts,
            "total_stalled_cycles": int(total_stalled_cycles),
            "queue_failures": int(queue_failures),
            "num_escalations": int(getattr(policy, "num_escalations", 0)),
            "cost_per_operating_cycle": cost_per_op,
            "average_maintenance_lead": avg_lead,
            "mean_queue_wait": mean_q_wait,
            "mean_wasted_rul": mean_wasted,
            "utilization": utilization,
            "offered_load": offered_load,
            "total_tech_busy": total_tech_cycles_busy,
            "total_tech_available": total_tech_cycles_available,
            "demanded_tech_cycles": demanded_tech_cycles,
            "slot_history": slot_history,
        }
