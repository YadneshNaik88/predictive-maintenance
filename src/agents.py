"""
agents.py

Phase 6: Multi-Agent Collaborative Predictive Maintenance System.
Implements the 4 collaborative agents adhering strictly to the PS-06 brief:
1. Prognostic Agent (RUL predictions, uncertainty bounds, and empirical residual risk table)
2. Resource Agent (spare part lead time tracking, technician roster & reservation calendar)
3. Cost Agent (rate-based renewal cost optimization C(s)/E(s))
4. Scheduling/Coordinator Agent (fleet-level planning pass, priority dispatch, escalation, JSONL decision logging)

Includes two variants for Mandatory Experiment 2:
- Agent-Interval (uncertainty-aware empirical residual risk distribution)
- Agent-Point (deterministic point-estimate baseline, s < pred_rul, R = pred_rul)
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml

from src.policies import BasePolicy
from src.simulator import (
    DEFAULT_OOF_PATH,
    DEFAULT_CONFIG_PATH,
    EngineObservation,
    get_tuning_eval_pools,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECISION_LOG_PATH = PROJECT_ROOT / "outputs" / "decision_log.jsonl"


class PrognosticAgent:
    """
    Agent 1: Prognostic Agent.
    Provides predicted RUL point estimates, 90% uncertainty intervals, and
    an empirical residual risk table built strictly from TUNING-pool out-of-fold predictions.
    Computes P(R <= s) and E[min(s, R)] where R = pred_rul + e.
    """

    def __init__(
        self,
        oof_csv_path: Union[str, Path] = DEFAULT_OOF_PATH,
        tuning_units: Optional[np.ndarray] = None,
        use_intervals: bool = True,
    ) -> None:
        self.use_intervals = use_intervals
        if tuning_units is None:
            tuning_units, _ = get_tuning_eval_pools(seed=42)
        self.tuning_units = tuning_units
        self.residuals_by_bin = self._build_risk_table(oof_csv_path, self.tuning_units)
        self.model = None
        self.scaler = None
        self.target_scale = 125.0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path] = PROJECT_ROOT / "outputs" / "best_lstm_model.pt",
        scaler_path: Union[str, Path] = PROJECT_ROOT / "outputs" / "scaler.pkl",
        calib_path: Union[str, Path] = PROJECT_ROOT / "outputs" / "residual_interval_calibration.json",
        oof_csv_path: Union[str, Path] = DEFAULT_OOF_PATH,
        tuning_units: Optional[np.ndarray] = None,
        use_intervals: bool = True,
        device: Optional[Any] = None,
    ) -> "PrognosticAgent":
        """
        Instantiate PrognosticAgent with live PyTorch LSTM model and fitted MinMaxScaler.
        Enables live inference on raw sensor sequences or preprocessed windows.
        """
        import joblib
        from src.uncertainty import load_uncertainty_model_and_calib

        agent = cls(oof_csv_path=oof_csv_path, tuning_units=tuning_units, use_intervals=use_intervals)
        model, _, target_scale = load_uncertainty_model_and_calib(
            checkpoint_path=checkpoint_path,
            calib_path=calib_path,
            device=device,
        )
        agent.model = model
        agent.target_scale = target_scale
        agent.scaler = joblib.load(scaler_path)
        agent.checkpoint_path = Path(checkpoint_path)
        agent.scaler_path = Path(scaler_path)
        agent.calib_path = Path(calib_path)
        return agent

    def predict(
        self,
        sensor_data: Union[np.ndarray, pd.DataFrame],
        scaled: bool = False,
        method: str = "binned_residual",
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Live prediction of RUL point estimate and 90% uncertainty interval.

        Parameters
        ----------
        sensor_data : np.ndarray or pd.DataFrame
            Can be:
            - DataFrame with sensor measurements (takes 14 retained sensors, scales, left-pads or slices to 30)
            - 2D numpy array (T, 14) of raw or pre-scaled sensor readings
            - 3D numpy array (N, 30, 14) or (1, 30, 14) of pre-scaled sequences
        scaled : bool, default=False
            Whether sensor_data is already MinMax-scaled into [0, 1].
            If False and self.scaler is available, applies self.scaler.transform.
        method : str, default="binned_residual"
            Uncertainty quantification method. If "binned_residual", uses the Tuning-pool
            OOF residual table (residuals_by_bin), matching evaluate_risk.

        Returns
        -------
        pred_rul, lower_rul, upper_rul : float or np.ndarray
        """
        import torch
        from src.preprocessing import RETAINED_SENSOR_COLUMNS

        if self.model is None:
            raise RuntimeError(
                "Live inference model not loaded. Instantiate PrognosticAgent with PrognosticAgent.from_checkpoint()."
            )

        is_single = False
        if isinstance(sensor_data, pd.DataFrame):
            cols = [c for c in RETAINED_SENSOR_COLUMNS if c in sensor_data.columns]
            if len(cols) != 14:
                raise ValueError(f"Expected 14 retained sensor columns, found {len(cols)}")
            df_sub = sensor_data[cols]
            if not scaled and self.scaler is not None:
                vals = self.scaler.transform(df_sub)
            else:
                vals = df_sub.values.astype(np.float32)
            T = len(vals)
            if T < 30:
                pad = np.repeat(vals[0:1, :], 30 - T, axis=0)
                window = np.vstack([pad, vals])
            else:
                window = vals[-30:, :]
            X_in = np.expand_dims(window, axis=0).astype(np.float32)
            is_single = True

        elif isinstance(sensor_data, np.ndarray):
            if sensor_data.ndim == 2:
                T, F = sensor_data.shape
                if F != 14:
                    raise ValueError(f"Expected 14 features, got {F}")
                vals = sensor_data
                if not scaled and self.scaler is not None:
                    vals = self.scaler.transform(vals)
                else:
                    vals = vals.astype(np.float32)
                if T < 30:
                    pad = np.repeat(vals[0:1, :], 30 - T, axis=0)
                    window = np.vstack([pad, vals])
                else:
                    window = vals[-30:, :]
                X_in = np.expand_dims(window, axis=0).astype(np.float32)
                is_single = True
            elif sensor_data.ndim == 3:
                N, T, F = sensor_data.shape
                if F != 14:
                    raise ValueError(f"Expected 14 features, got {F}")
                vals = sensor_data
                if not scaled and self.scaler is not None:
                    vals = self.scaler.transform(vals.reshape(-1, 14)).reshape(N, T, 14)
                else:
                    vals = vals.astype(np.float32)
                X_in = vals.astype(np.float32)
                is_single = (N == 1)
            else:
                raise ValueError(f"Unsupported array dimension: {sensor_data.ndim}")
        else:
            raise TypeError(f"Unsupported input type: {type(sensor_data)}")

        # 1. Point prediction via LSTM regressor
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_in, dtype=torch.float32, device=device)
            y_norm = self.model(X_tensor).squeeze(-1).cpu().numpy()
        y_pred = np.atleast_1d((y_norm * self.target_scale).astype(np.float32))

        # 2. 90% Uncertainty interval via Tuning-pool OOF residual table (same as evaluate_risk)
        if method == "binned_residual" and self.use_intervals:
            lowers = []
            uppers = []
            for p in y_pred:
                b = self.get_bin(float(p))
                residuals = self.residuals_by_bin[b]
                R = np.maximum(0.0, float(p) + residuals)
                lowers.append(float(np.percentile(R, 5.0)))
                uppers.append(float(np.percentile(R, 95.0)))
            lower_arr = np.array(lowers, dtype=np.float32)
            upper_arr = np.array(uppers, dtype=np.float32)
        elif not self.use_intervals:
            lower_arr = y_pred.copy()
            upper_arr = y_pred.copy()
        else:
            from src.uncertainty import predict_rul_with_interval
            _, lower_arr, upper_arr = predict_rul_with_interval(
                X_in, model=self.model, method=method, target_scale=self.target_scale
            )
            if np.isscalar(lower_arr):
                lower_arr = np.array([lower_arr], dtype=np.float32)
                upper_arr = np.array([upper_arr], dtype=np.float32)

        if is_single:
            return float(y_pred[0]), float(lower_arr[0]), float(upper_arr[0])
        return y_pred, lower_arr, upper_arr

    def _build_risk_table(
        self, oof_csv_path: Union[str, Path], tuning_units: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Construct the empirical residual distribution table from Tuning pool engines only.
        Residual e = true_rul_raw - pred_rul.
        Bins are of width 10: [0, 10), [10, 20), ..., [120, 130).
        """
        df = pd.read_csv(oof_csv_path)
        df_tune = df[df["unit_number"].isin(tuning_units)].copy()
        df_tune["residual"] = df_tune["true_rul_raw"] - df_tune["pred_rul"]
        df_tune["bin"] = np.clip((df_tune["pred_rul"] // 10).astype(int), 0, 12)

        table: Dict[int, np.ndarray] = {}
        for b in range(13):
            sub = df_tune[df_tune["bin"] == b]["residual"].values
            if len(sub) == 0:
                sub = np.array([0.0])
            table[b] = sub
        return table

    def get_bin(self, pred_rul: float) -> int:
        return int(np.clip(pred_rul // 10, 0, 12))

    def evaluate_risk(
        self, pred_rul: float, candidate_starts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute failure probability P(R <= s) and expected operating cycles E[min(s, R)]
        for candidate start offsets s (s >= 0 relative to now).
        """
        if len(candidate_starts) == 0:
            return np.array([]), np.array([])

        if not self.use_intervals:
            # Agent-Point: deterministic R = pred_rul, no residual spread
            prob_fail = np.where(candidate_starts >= pred_rul, 1.0, 0.0)
            exp_min_r = np.minimum(candidate_starts, pred_rul).astype(float)
            return prob_fail, exp_min_r

        # Agent-Interval: empirical residual distribution
        b = self.get_bin(pred_rul)
        residuals = self.residuals_by_bin[b]  # shape (K,)
        # R = max(0.0, pred_rul + e)
        R = np.maximum(0.0, pred_rul + residuals)[:, np.newaxis]  # shape (K, 1)
        s = candidate_starts[np.newaxis, :]  # shape (1, M)

        prob_fail = np.mean(R <= s, axis=0)  # shape (M,)
        exp_min_r = np.mean(np.minimum(s, R), axis=0)  # shape (M,)
        return prob_fail, exp_min_r


class ResourceAgent:
    """
    Agent 2: Resource Agent.
    Manages spare parts inventory logistics and technician roster calendars.
    Ensures reservation consistency: reserved technician-cycles never exceed roster capacity.
    """

    def __init__(
        self,
        spare_lead_time: int = 8,
        preventive_duration: int = 2,
        horizon_W: int = 60,
    ) -> None:
        self.spare_lead_time = spare_lead_time
        self.preventive_duration = preventive_duration
        self.horizon_W = horizon_W
        self.capacity_roster: np.ndarray = np.array([], dtype=int)
        self.reserved_calendar: np.ndarray = np.array([], dtype=int)

    def reset(self, total_horizon: int, capacity_roster: np.ndarray) -> None:
        """Initialize technician capacity roster and clear reservation calendar."""
        cal_len = max(len(capacity_roster), total_horizon + 300)
        pad_len = cal_len - len(capacity_roster)
        if pad_len > 0:
            pad_val = int(capacity_roster[-1]) if len(capacity_roster) > 0 else 4
            self.capacity_roster = np.pad(capacity_roster, (0, pad_len), constant_values=pad_val)
        else:
            self.capacity_roster = capacity_roster.copy()
        self.reserved_calendar = np.zeros(len(self.capacity_roster) + 100, dtype=int)

    def is_window_available(self, start_cycle: int, duration: int) -> bool:
        """Return True if capacity - reserved >= 1 on all duration consecutive cycles."""
        for c in range(start_cycle, start_cycle + duration):
            if c >= len(self.capacity_roster):
                return False
            if (self.capacity_roster[c] - self.reserved_calendar[c]) < 1:
                return False
        return True

    def filter_candidates(
        self,
        obs: EngineObservation,
        candidate_s: np.ndarray,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Evaluate feasibility of candidate start offsets s relative to current cycle (now).
        A candidate is feasible if:
        1. Spare part can arrive in time: (now + s) >= part_arrival
        2. Technician capacity is available for full duration D.
        """
        now = obs.current_cycle
        L = self.spare_lead_time
        D = self.preventive_duration

        if obs.order_already_placed and obs.part_arrival_cycle is not None:
            part_arrival = obs.part_arrival_cycle
        else:
            part_arrival = now + L

        feasible_mask = np.zeros(len(candidate_s), dtype=bool)
        diagnostics: List[Dict[str, Any]] = []

        for idx, s in enumerate(candidate_s):
            target_cycle = now + s
            part_ready = bool(target_cycle >= part_arrival)
            tech_ready = bool(self.is_window_available(target_cycle, D))
            is_feasible = bool(part_ready and tech_ready)
            feasible_mask[idx] = is_feasible

            diagnostics.append({
                "start_cycle": int(target_cycle),
                "offset_s": int(s),
                "feasible": is_feasible,
                "part_ready": part_ready,
                "part_arrival": int(part_arrival),
                "tech_ready": tech_ready,
            })

        return feasible_mask, diagnostics

    def reserve(self, start_cycle: int, duration: int) -> bool:
        """Reserve 1 technician for duration consecutive cycles starting at start_cycle."""
        if start_cycle + duration > len(self.capacity_roster):
            pad_val = int(self.capacity_roster[-1]) if len(self.capacity_roster) > 0 else 4
            pad_len = (start_cycle + duration + 100) - len(self.capacity_roster)
            self.capacity_roster = np.pad(self.capacity_roster, (0, pad_len), constant_values=pad_val)
            self.reserved_calendar = np.pad(self.reserved_calendar, (0, pad_len), constant_values=0)

        # Invariant check
        for c in range(start_cycle, start_cycle + duration):
            if (self.capacity_roster[c] - self.reserved_calendar[c]) < 1:
                return False

        self.reserved_calendar[start_cycle : start_cycle + duration] += 1
        return True

    def release(self, start_cycle: int, duration: int) -> None:
        """Release a previously reserved technician window."""
        end_cycle = min(len(self.reserved_calendar), start_cycle + duration)
        if start_cycle < len(self.reserved_calendar):
            self.reserved_calendar[start_cycle : end_cycle] = np.maximum(
                0, self.reserved_calendar[start_cycle : end_cycle] - 1
            )

    def find_earliest_available(self, earliest_cycle: int, duration: int) -> int:
        """Find earliest cycle >= earliest_cycle with sufficient technician capacity."""
        c = earliest_cycle
        max_search = len(self.capacity_roster) - duration
        while c <= max_search:
            if self.is_window_available(c, duration):
                return c
            c += 1
        return earliest_cycle


class CostAgent:
    """
    Agent 3: Cost Agent.
    Evaluates expected cost per operating cycle C(s)/E(s) for candidate start cycles.
    The fleet pays for renewals; comparing C(s) alone would always pick the earliest window.
    The rate C(s)/E(s) properly accounts for wasted engine useful life.
    """

    def __init__(
        self,
        cost_preventive: float = 2000.0,
        cost_corrective: float = 10000.0,
        cost_production_loss: float = 100.0,
        cost_spare: float = 300.0,
        preventive_duration: int = 2,
        corrective_duration: int = 6,
    ) -> None:
        self.cost_preventive = cost_preventive
        self.cost_corrective = cost_corrective
        self.cost_production_loss = cost_production_loss
        self.cost_spare = cost_spare
        self.preventive_duration = preventive_duration
        self.corrective_duration = corrective_duration

        self.c_pm_total = self.cost_preventive + self.cost_spare + self.cost_production_loss * self.preventive_duration
        self.c_fail_total = self.cost_corrective + self.cost_spare + self.cost_production_loss * self.corrective_duration

    def evaluate_cost_rates(
        self,
        age: int,
        candidate_s: np.ndarray,
        prob_fail: np.ndarray,
        exp_min_r: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute C(s), E(s), and C(s)/E(s) for candidate start offsets.
        """
        c_s = (1.0 - prob_fail) * self.c_pm_total + prob_fail * self.c_fail_total
        e_s = float(age) + exp_min_r
        denom = np.maximum(e_s, 1e-6)
        cost_rates = c_s / denom
        return c_s, e_s, cost_rates


class CoordinatorAgent:
    """
    Agent 4: Scheduling / Coordinator Agent.
    Coordinates fleet-level planning pass, orchestrates Prognostic, Resource, and Cost agents,
    manages priority queues, escalation procedures, and decision logging to JSONL.
    """

    def __init__(
        self,
        prognostic: PrognosticAgent,
        resource: ResourceAgent,
        cost: CostAgent,
        replan_k: int = 5,
        horizon_W: int = 60,
        spare_lead_time: int = 8,
        decision_log_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.prognostic = prognostic
        self.resource = resource
        self.cost = cost
        self.replan_k = replan_k
        self.horizon_W = horizon_W
        self.spare_lead_time = spare_lead_time
        self.decision_log_path = Path(decision_log_path) if decision_log_path else None

        # Slot tracking: slot_id -> dict with current plan
        self.slot_plans: Dict[int, Dict[str, Any]] = {}
        self.decision_records: List[Dict[str, Any]] = []

    def reset(self, total_horizon: int, capacity_roster: np.ndarray) -> None:
        self.resource.reset(total_horizon, capacity_roster)
        self.slot_plans.clear()
        self.decision_records.clear()
        if self.decision_log_path and self.decision_log_path.exists():
            try:
                self.decision_log_path.unlink()
            except Exception:
                pass

    def on_renewal(self, slot_id: int) -> None:
        """Callback when an engine renewal occurs; clear reservation and plan for slot."""
        if slot_id in self.slot_plans:
            plan = self.slot_plans[slot_id]
            if plan.get("reserved", False) and "planned_start" in plan:
                self.resource.release(plan["planned_start"], self.resource.preventive_duration)
            del self.slot_plans[slot_id]

    def on_engine_failure(self, slot_id: int) -> None:
        """Callback if an engine fails before planned start; release reserved tech window."""
        if slot_id in self.slot_plans:
            plan = self.slot_plans[slot_id]
            if plan.get("reserved", False) and "planned_start" in plan:
                self.resource.release(plan["planned_start"], self.resource.preventive_duration)
                plan["reserved"] = False

    def plan_fleet(
        self,
        current_cycle: int,
        observations: List[EngineObservation],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Execute fleet-wide planning pass at current_cycle.
        Returns dictionary of updated action orders per slot.
        """
        now = current_cycle
        L = self.spare_lead_time
        W = self.horizon_W
        D = self.resource.preventive_duration

        # Step 1: Identify engines within planning horizon (predicted RUL <= L + W)
        candidate_engines: List[EngineObservation] = []
        for obs in observations:
            if obs.pred_rul <= (L + W):
                candidate_engines.append(obs)

        # Step 2: Filter engines requiring planning or re-planning
        to_plan: List[EngineObservation] = []
        for obs in candidate_engines:
            s_id = obs.slot_id
            if s_id not in self.slot_plans:
                to_plan.append(obs)
            else:
                plan = self.slot_plans[s_id]
                # Re-plan every k cycles
                is_periodic = (now % self.replan_k == 1)
                # Re-plan immediately if lower bound falls below planned start
                planned_start = plan.get("planned_start", now + 999)
                lower_dropped = (obs.lower_rul < (planned_start - now))
                if is_periodic or lower_dropped:
                    to_plan.append(obs)

        if not to_plan:
            return {}

        # Step 3: Sort by urgency (lowest lower bound first)
        to_plan.sort(key=lambda o: o.lower_rul)

        action_orders: Dict[int, Dict[str, Any]] = {}

        # Step 4: Plan each engine in urgency order
        for obs in to_plan:
            s_id = obs.slot_id
            # If engine already had a reservation, temporarily release it during re-plan
            had_prior_plan = s_id in self.slot_plans
            if had_prior_plan:
                prior_plan = self.slot_plans[s_id]
                if prior_plan.get("reserved", False) and "planned_start" in prior_plan:
                    self.resource.release(prior_plan["planned_start"], D)
                    prior_plan["reserved"] = False

            # Candidate search range:
            # Agent-Point: s < pred_rul (hard limit)
            # Agent-Interval: s <= pred_rul (hard limit)
            if not self.prognostic.use_intervals:
                max_s = max(0, int(np.floor(obs.pred_rul - 1.0)))
            else:
                max_s = max(0, int(np.floor(obs.pred_rul)))

            candidate_s = np.arange(0, max_s + 1, dtype=int)

            # Drop cycles where spare cannot arrive or no technician capacity
            feasible_mask, diag_list = self.resource.filter_candidates(obs, candidate_s)
            feasible_s = candidate_s[feasible_mask]

            chosen_cycle: Optional[int] = None
            escalated = False
            best_cost_rate = 999999.0
            order_spare_now = not obs.order_already_placed

            if len(feasible_s) > 0:
                # Ask Cost Agent for cost rates of remaining cycles
                prob_fail, exp_min_r = self.prognostic.evaluate_risk(obs.pred_rul, feasible_s)
                c_s, e_s, cost_rates = self.cost.evaluate_cost_rates(obs.age, feasible_s, prob_fail, exp_min_r)

                best_idx = int(np.argmin(cost_rates))
                best_s = int(feasible_s[best_idx])
                best_cost_rate = float(cost_rates[best_idx])
                chosen_cycle = now + best_s

                # Reserve technician capacity
                reserved = self.resource.reserve(chosen_cycle, D)

                # Attach cost rates to diagnostics for JSONL logging
                cost_idx = 0
                for d in diag_list:
                    if d["feasible"]:
                        d["cost_rate"] = float(cost_rates[cost_idx])
                        d["prob_fail"] = float(prob_fail[cost_idx])
                        cost_idx += 1
                    else:
                        d["cost_rate"] = None
                        d["prob_fail"] = None

                reason = (
                    f"Selected cycle {chosen_cycle} (offset +{best_s} cycles) with lowest expected cost rate "
                    f"${best_cost_rate:.2f}/cycle (prob_fail={prob_fail[best_idx]*100:.1f}%, age={obs.age})."
                )
            else:
                # ESCALATION: No feasible window in candidate set
                escalated = True
                order_spare_now = not obs.order_already_placed
                earliest_part = obs.part_arrival_cycle if obs.order_already_placed and obs.part_arrival_cycle else (now + L)
                chosen_cycle = self.resource.find_earliest_available(earliest_part, D)
                reserved = self.resource.reserve(chosen_cycle, D)

                for d in diag_list:
                    d["cost_rate"] = None
                    d["prob_fail"] = None

                reason = (
                    f"ESCALATION: No feasible maintenance window within predicted RUL {obs.pred_rul:.1f}. "
                    f"Secured earliest feasible window at cycle {chosen_cycle} (part arrives cycle {earliest_part})."
                )

            # Store plan
            plan_record = {
                "slot_id": s_id,
                "unit_number": obs.unit_number,
                "planned_start": chosen_cycle,
                "order_cycle": now if order_spare_now else self.slot_plans.get(s_id, {}).get("order_cycle", now),
                "part_arrival": now + L if order_spare_now else obs.part_arrival_cycle,
                "escalated": escalated,
                "reserved": reserved,
                "cost_rate": best_cost_rate,
                "urgency_lower": obs.lower_rul,
            }
            self.slot_plans[s_id] = plan_record

            # Decision JSONL record
            decision_entry = {
                "engine": f"E-{obs.unit_number:03d}",
                "cycle": now,
                "predicted_rul": float(obs.pred_rul),
                "interval": [float(obs.lower_rul), float(obs.upper_rul)],
                "candidates": diag_list,
                "chosen_cycle": int(chosen_cycle),
                "escalated": bool(escalated),
                "spare_order_cycle": int(plan_record["order_cycle"]),
                "reason": reason,
            }
            self.decision_records.append(decision_entry)
            if self.decision_log_path:
                with open(self.decision_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(decision_entry) + "\n")

            action_orders[s_id] = {
                "order_spare": order_spare_now,
                "planned_start": chosen_cycle,
                "priority": float(obs.lower_rul),
                "escalated": escalated,
            }

        return action_orders


class AgenticPolicy(BasePolicy):
    """
    Agentic Maintenance Decision Policy for Turbofan Fleet Simulation.
    Wraps the 4 collaborative agents (Prognostic, Resource, Cost, Coordinator)
    into the FleetSimulator policy interface.
    """

    def __init__(
        self,
        use_intervals: bool = True,
        replan_k: int = 5,
        horizon_W: int = 60,
        spare_lead_time: int = 8,
        oof_csv_path: Union[str, Path] = DEFAULT_OOF_PATH,
        config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
        decision_log_path: Optional[Union[str, Path]] = DEFAULT_DECISION_LOG_PATH,
        roster_blind: bool = False,
    ) -> None:
        if roster_blind:
            name = "Agent-Interval-Blind"
        else:
            name = "Agent-Interval" if use_intervals else "Agent-Point"
        super().__init__(name=name)
        self.use_intervals = use_intervals
        self.replan_k = replan_k
        self.horizon_W = horizon_W
        self.spare_lead_time = spare_lead_time
        self.roster_blind = roster_blind

        # Load parameters from YAML
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        costs = data.get("costs", {})
        ops = data.get("operations", {})

        c_pm = float(costs.get("preventive_maintenance", 2000.0))
        c_fail = float(costs.get("corrective_failure", 10000.0))
        c_loss = float(costs.get("production_loss_per_cycle", 100.0))
        c_part = float(costs.get("spare_part", 300.0))
        d_pm = int(ops.get("preventive_duration", 2))
        d_fail = int(ops.get("corrective_duration", 6))

        # Instantiate sub-agents
        self.prognostic = PrognosticAgent(oof_csv_path=oof_csv_path, use_intervals=use_intervals)
        self.resource = ResourceAgent(spare_lead_time=spare_lead_time, preventive_duration=d_pm, horizon_W=horizon_W)
        self.cost = CostAgent(
            cost_preventive=c_pm,
            cost_corrective=c_fail,
            cost_production_loss=c_loss,
            cost_spare=c_part,
            preventive_duration=d_pm,
            corrective_duration=d_fail,
        )
        self.coordinator = CoordinatorAgent(
            prognostic=self.prognostic,
            resource=self.resource,
            cost=self.cost,
            replan_k=replan_k,
            horizon_W=horizon_W,
            spare_lead_time=spare_lead_time,
            decision_log_path=decision_log_path,
        )

        self.current_action_orders: Dict[int, Dict[str, Any]] = {}
        self.num_escalations = 0

    def reset(self) -> None:
        self.current_action_orders.clear()
        self.num_escalations = 0

    def init_simulation(self, total_horizon: int, capacity_roster: np.ndarray) -> None:
        """Called by simulator before cycle loop to initialize roster calendars."""
        self.reset()
        if self.roster_blind:
            mean_c = int(np.round(np.mean(capacity_roster[1 : total_horizon + 1])))
            blind_roster = np.full_like(capacity_roster, fill_value=mean_c)
            self.coordinator.reset(total_horizon, blind_roster)
        else:
            self.coordinator.reset(total_horizon, capacity_roster)

    def on_renewal(self, slot_id: int) -> None:
        self.coordinator.on_renewal(slot_id)
        if slot_id in self.current_action_orders:
            del self.current_action_orders[slot_id]

    def on_engine_failure(self, slot_id: int) -> None:
        self.coordinator.on_engine_failure(slot_id)

    def before_cycle(
        self,
        current_cycle: int,
        observations: List[EngineObservation],
    ) -> None:
        """Execute fleet planning pass and record decisions."""
        orders = self.coordinator.plan_fleet(current_cycle, observations)
        for s_id, ord_dict in orders.items():
            if ord_dict.get("escalated", False):
                self.num_escalations += 1
            self.current_action_orders[s_id] = ord_dict

    def decide(
        self,
        slot_id: int,
        current_cycle: int,
        operating_cycles: int,
        pred_rul: float,
        lower_rul: float,
        upper_rul: float,
        true_rul: float,
        unit_number: int,
        engine_cycle: int,
        order_already_placed: bool,
        fleet_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if an order was placed by the Coordinator for this slot at this cycle.
        """
        if order_already_placed:
            return False

        if slot_id in self.current_action_orders:
            ord_dict = self.current_action_orders[slot_id]
            if ord_dict.get("order_spare", False):
                return True
        return False

    def get_earliest_start(self, slot_id: int, default: int) -> int:
        if slot_id in self.coordinator.slot_plans:
            return int(self.coordinator.slot_plans[slot_id].get("planned_start", default))
        return default

    def get_priority(self, slot_id: int, default: float = 0.0) -> float:
        if slot_id in self.coordinator.slot_plans:
            return float(self.coordinator.slot_plans[slot_id].get("urgency_lower", default))
        return default
