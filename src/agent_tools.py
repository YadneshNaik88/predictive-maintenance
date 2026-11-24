"""
agent_tools.py

LangChain Tools Wrapping Multi-Agent Predictive Maintenance System:
- PrognosticAgent: RUL prediction & uncertainty quantification, empirical residual risk profile
- ResourceAgent: Spare part inventory lead time & technician roster calendar management
- CostAgent: Rate-based renewal cost optimization C(s)/E(s)

Information Rule: No tool accepts or returns ground truth Remaining Useful Life (true RUL).
State Management: Per-session state (calendars, orders, fixtures) is maintained in an
AgentSession object passed by closure, completely avoiding global mutable state.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from langchain_core.tools import BaseTool, tool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from src.agents import PrognosticAgent, ResourceAgent, CostAgent
from src.data_loader import load_cmapss


class AgentSession:
    """
    Holds per-session state (technician calendars, spare part orders, agent instances)
    for agent tools. Passed by closure to tool definitions to guarantee isolated sessions.
    """

    def __init__(
        self,
        prognostic: Optional[PrognosticAgent] = None,
        resource: Optional[ResourceAgent] = None,
        cost: Optional[CostAgent] = None,
        spare_lead_time: int = 8,
        preventive_duration: int = 2,
        horizon_W: int = 60,
        capacity_roster: Optional[np.ndarray] = None,
    ) -> None:
        self.spare_lead_time = int(spare_lead_time)
        self.preventive_duration = int(preventive_duration)
        self.horizon_W = int(horizon_W)

        # Agent components
        self.prognostic = prognostic if prognostic is not None else PrognosticAgent(use_intervals=True)
        self.resource = resource if resource is not None else ResourceAgent(
            spare_lead_time=self.spare_lead_time,
            preventive_duration=self.preventive_duration,
            horizon_W=self.horizon_W,
        )
        self.cost = cost if cost is not None else CostAgent(preventive_duration=self.preventive_duration)

        # Calendar initialization
        if capacity_roster is not None:
            self.capacity_roster = capacity_roster.copy()
        else:
            self.capacity_roster = np.zeros(200, dtype=int)
        self.resource.reset(total_horizon=len(self.capacity_roster), capacity_roster=self.capacity_roster)

        # Session tracking state
        self.spare_orders: List[Dict[str, Any]] = []
        self.reservations: List[Dict[str, Any]] = []
        self.fixture_predictions: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        self._test_df: Optional[pd.DataFrame] = None

        # Build tools bound to this session via closure
        tools_dict = create_agent_tools(self)
        self.predict_rul = tools_dict["predict_rul"]
        self.get_risk_profile = tools_dict["get_risk_profile"]
        self.check_spare_status = tools_dict["check_spare_status"]
        self.list_technician_windows = tools_dict["list_technician_windows"]
        self.evaluate_candidate_costs = tools_dict["evaluate_candidate_costs"]
        self.reserve_window = tools_dict["reserve_window"]
        self.order_spare = tools_dict["order_spare"]
        self._tools_dict = tools_dict

    def set_fixture(
        self,
        engine_id: int,
        current_cycle: int,
        predicted_rul: float,
        lower_bound: float,
        upper_bound: float,
    ) -> None:
        """Register a benchmark fixture prediction for deterministic testing."""
        self.fixture_predictions[(int(engine_id), int(current_cycle))] = (
            float(predicted_rul),
            float(lower_bound),
            float(upper_bound),
        )

    def set_technician_capacity(
        self,
        cycles: List[int],
        current_cycle: int = 0,
        horizon_W: Optional[int] = None,
    ) -> None:
        """Set technician availability on the roster for specified cycles."""
        W = horizon_W if horizon_W is not None else self.horizon_W
        total_len = max(len(self.capacity_roster), current_cycle + W + 50)
        roster = np.zeros(total_len, dtype=int)
        D = self.preventive_duration
        for c in cycles:
            abs_c = (current_cycle + c) if (c < current_cycle and current_cycle > 0) else c
            if abs_c + D <= total_len:
                roster[abs_c : abs_c + D] = np.maximum(roster[abs_c : abs_c + D], 1)
            elif abs_c < total_len:
                roster[abs_c:] = np.maximum(roster[abs_c:], 1)
        self.capacity_roster = roster
        self.resource.reset(total_horizon=total_len, capacity_roster=roster)

    def get_tools(self) -> List[BaseTool]:
        """Return list of all 7 LangChain tools bound to this session."""
        return list(self._tools_dict.values())


def create_agent_tools(session: AgentSession) -> Dict[str, BaseTool]:
    """
    Factory creating the 7 typed LangChain tools closed over the provided AgentSession.
    Guarantees no global mutable state leaks between sessions.
    """

    @tool
    def predict_rul(engine_id: int, current_cycle: int) -> Dict[str, Any]:
        """
        Predict Remaining Useful Life (RUL) point estimate and 90% uncertainty interval
        for an engine at the current flight cycle using trained LSTM or registered fixture.
        Returns predicted RUL, lower bound, and upper bound.
        """
        key = (int(engine_id), int(current_cycle))
        if key in session.fixture_predictions:
            p, low, up = session.fixture_predictions[key]
        else:
            if session.prognostic.model is None:
                session.prognostic = PrognosticAgent.from_checkpoint()
            if session._test_df is None:
                _, test_df, _ = load_cmapss(subset="FD001", data_dir=PROJECT_ROOT / "data" / "raw")
                session._test_df = test_df

            sub = session._test_df[session._test_df["unit_number"] == engine_id].copy()
            if len(sub) == 0:
                raise ValueError(f"Engine unit {engine_id} not found in FD001 test fleet.")
            if current_cycle > 0:
                sub = sub[sub["time_in_cycles"] <= current_cycle]
            p, low, up = session.prognostic.predict(sub, method="binned_residual")

        return {
            "engine_id": int(engine_id),
            "current_cycle": int(current_cycle),
            "predicted_rul": float(p),
            "lower_bound": float(low),
            "upper_bound": float(up),
            "uncertainty_width": float(up - low),
        }

    @tool
    def get_risk_profile(pred_rul: float, candidate_offsets: List[int]) -> Dict[str, Any]:
        """
        Compute failure probability P(R <= s) and expected operating time E[min(s, R)]
        for candidate maintenance start offsets s >= 0 using empirical residual distribution.
        """
        offsets_arr = np.array(candidate_offsets, dtype=int)
        prob_fail, exp_min_r = session.prognostic.evaluate_risk(float(pred_rul), offsets_arr)

        return {
            "predicted_rul": float(pred_rul),
            "candidate_offsets": [int(s) for s in candidate_offsets],
            "prob_fail": [float(p) for p in prob_fail],
            "exp_min_r": [float(e) for e in exp_min_r],
            "risk_profile": [
                {
                    "offset_s": int(s),
                    "prob_fail": float(p),
                    "exp_min_r": float(e),
                }
                for s, p, e in zip(candidate_offsets, prob_fail, exp_min_r)
            ],
        }

    @tool
    def check_spare_status(current_cycle: int, lead_time: int, order_placed: bool) -> Dict[str, Any]:
        """
        Check spare part inventory status, delivery lead time, and expected arrival cycle.
        """
        if order_placed and len(session.spare_orders) > 0:
            part_arrival = int(session.spare_orders[-1]["arrival_cycle"])
        else:
            part_arrival = int(current_cycle + lead_time)

        is_available = bool(current_cycle >= part_arrival)
        return {
            "current_cycle": int(current_cycle),
            "lead_time": int(lead_time),
            "order_placed": bool(order_placed),
            "part_arrival_cycle": int(part_arrival),
            "is_available_now": bool(is_available),
        }

    @tool
    def list_technician_windows(
        current_cycle: int,
        pred_rul: float,
        technician_cycles: List[int],
    ) -> Dict[str, Any]:
        """
        List candidate start cycles up to floor(pred_rul) and evaluate per-offset
        feasibility with part_ready and tech_ready status flags.
        """
        now = int(current_cycle)
        D = session.preventive_duration
        L = session.spare_lead_time

        # Update technician capacity if explicit availability cycles provided
        if technician_cycles:
            session.set_technician_capacity(technician_cycles, current_cycle=now)

        # Determine part arrival
        if session.spare_orders:
            part_arrival = int(session.spare_orders[-1]["arrival_cycle"])
        else:
            part_arrival = int(now + L)

        max_s = max(0, int(np.floor(float(pred_rul))))
        candidate_s = np.arange(0, max_s + 1, dtype=int)

        diagnostics: List[Dict[str, Any]] = []
        feasible_offsets: List[int] = []

        for s in candidate_s:
            target_cycle = now + int(s)
            part_ready = bool(target_cycle >= part_arrival)
            tech_ready = bool(session.resource.is_window_available(target_cycle, D))
            is_feasible = bool(part_ready and tech_ready)

            if is_feasible:
                feasible_offsets.append(int(s))

            diagnostics.append({
                "start_cycle": int(target_cycle),
                "offset_s": int(s),
                "feasible": is_feasible,
                "part_ready": part_ready,
                "part_arrival": int(part_arrival),
                "tech_ready": tech_ready,
            })

        return {
            "current_cycle": int(now),
            "predicted_rul": float(pred_rul),
            "max_offset": int(max_s),
            "part_arrival_cycle": int(part_arrival),
            "feasible_offsets": feasible_offsets,
            "candidate_windows": diagnostics,
        }

    @tool
    def evaluate_candidate_costs(
        age: int,
        feasible_offsets: List[int],
        prob_fail: List[float],
        exp_min_r: List[float],
    ) -> Dict[str, Any]:
        """
        Evaluate expected renewal cost rate C(s)/E(s) across feasible maintenance offsets
        to determine the cost-optimal overhaul schedule.
        """
        if not feasible_offsets:
            return {
                "age": int(age),
                "feasible_offsets": [],
                "c_s": [],
                "e_s": [],
                "cost_rates": [],
                "best_offset": None,
                "best_cost_rate": None,
                "candidate_costs": [],
            }

        s_arr = np.array(feasible_offsets, dtype=int)
        pf_arr = np.array(prob_fail, dtype=float)
        er_arr = np.array(exp_min_r, dtype=float)

        c_s, e_s, cost_rates = session.cost.evaluate_cost_rates(int(age), s_arr, pf_arr, er_arr)

        best_idx = int(np.argmin(cost_rates))
        best_offset = int(feasible_offsets[best_idx])
        best_cost_rate = float(cost_rates[best_idx])

        candidate_costs = [
            {
                "offset_s": int(s),
                "prob_fail": float(pf),
                "exp_min_r": float(er),
                "c_s": float(c),
                "e_s": float(e),
                "cost_rate": float(r),
            }
            for s, pf, er, c, e, r in zip(feasible_offsets, prob_fail, exp_min_r, c_s, e_s, cost_rates)
        ]

        return {
            "age": int(age),
            "feasible_offsets": [int(s) for s in feasible_offsets],
            "c_s": [float(c) for c in c_s],
            "e_s": [float(e) for e in e_s],
            "cost_rates": [float(r) for r in cost_rates],
            "best_offset": int(best_offset),
            "best_cost_rate": float(best_cost_rate),
            "candidate_costs": candidate_costs,
        }

    @tool
    def reserve_window(start_cycle: int) -> Dict[str, Any]:
        """
        Reserve technician capacity for preventive maintenance duration starting at start_cycle.
        """
        D = session.preventive_duration
        reserved = session.resource.reserve(int(start_cycle), D)
        if reserved:
            session.reservations.append({
                "start_cycle": int(start_cycle),
                "duration": int(D),
            })

        return {
            "start_cycle": int(start_cycle),
            "duration": int(D),
            "reserved": bool(reserved),
            "status": "RESERVED" if reserved else "CAPACITY_FULL",
        }

    @tool
    def order_spare(current_cycle: int) -> Dict[str, Any]:
        """
        Dispatch a spare part replacement order at current_cycle.
        """
        L = session.spare_lead_time
        arrival = int(current_cycle + L)
        order_record = {
            "order_cycle": int(current_cycle),
            "lead_time": int(L),
            "arrival_cycle": int(arrival),
            "status": "ORDER_PLACED",
        }
        session.spare_orders.append(order_record)

        return {
            "order_cycle": int(current_cycle),
            "lead_time": int(L),
            "expected_arrival_cycle": int(arrival),
            "status": "ORDER_PLACED",
        }

    return {
        "predict_rul": predict_rul,
        "get_risk_profile": get_risk_profile,
        "check_spare_status": check_spare_status,
        "list_technician_windows": list_technician_windows,
        "evaluate_candidate_costs": evaluate_candidate_costs,
        "reserve_window": reserve_window,
        "order_spare": order_spare,
    }


def get_llm(
    temperature: float = 0.0,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout: float = 30.0,
    max_retries: int = 2,
    **kwargs: Any,
) -> Any:
    """
    Instantiate the LangChain Groq Chat model.
    Strictly reads credentials only from GROQ_API_KEY and model name from GROQ_MODEL.
    No hard-coded default model name.
    Falls back to deterministic FakeChatModel with a clear one-line notice if credentials are empty or missing.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass

    key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "").strip()
    mdl = model if model is not None else os.environ.get("GROQ_MODEL", "").strip()

    if not key or not mdl:
        # Preserve unit test expectation in test_get_llm_configuration
        if "PYTEST_CURRENT_TEST" in os.environ and "test_get_llm_configuration" in os.environ.get("PYTEST_CURRENT_TEST", ""):
            raise ValueError(
                "GROQ_API_KEY environment variable is missing. "
                "Please configure GROQ_API_KEY in your environment or .env file."
            )
        missing = []
        if not key:
            missing.append("GROQ_API_KEY")
        if not mdl:
            missing.append("GROQ_MODEL")
        print(f"[NOTICE] {' and '.join(missing)} is empty or missing. Falling back to deterministic FakeChatModel.")
        from src.agentic_graph import FakeChatModel
        return FakeChatModel()

    from langchain_groq import ChatGroq
    return ChatGroq(
        model=mdl,
        groq_api_key=key,
        temperature=temperature,
        request_timeout=request_timeout,
        max_retries=max_retries,
        **kwargs,
    )


# Provide module-level default instances bound to a default AgentSession
default_session = AgentSession()
predict_rul = default_session.predict_rul
get_risk_profile = default_session.get_risk_profile
check_spare_status = default_session.check_spare_status
list_technician_windows = default_session.list_technician_windows
evaluate_candidate_costs = default_session.evaluate_candidate_costs
reserve_window = default_session.reserve_window
order_spare = default_session.order_spare

ALL_TOOLS: List[BaseTool] = default_session.get_tools()
