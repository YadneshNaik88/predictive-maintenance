"""
agentic_graph.py

LangGraph Multi-Agent Predictive Maintenance Orchestrator with Groq Backend Support.
Coordinates specialized agents using a typed shared state, required-step enforcement, and conditional routing:
- supervisor: Orchestrates fleet planning pass, selects next node via LLM under Groq (rule-based under fake).
- prognostic_agent: Predicts RUL point estimates & 90% uncertainty intervals; computes empirical residual failure risk.
- resource_agent: Evaluates spare part logistics and technician roster window feasibility; commits reservations after validation.
- cost_agent: Computes rate-based renewal costs C(s)/E(s) to determine the cost-optimal overhaul schedule.
- validator: Deterministic non-LLM guardrail verifying feasibility, cost table membership, resource availability, and escalation validity.
- escalation: Deterministic fallback securing earliest available window when no feasible slot exists before predicted RUL.
- report: Assembles final recommendation strictly from tool outputs; rejects and replaces LLM rationale if contradiction or hallucinated numbers found.

Information Rule: Strict zero-leakage prohibition on true RUL / ground truth in states, messages, tools, or nodes.
Robustness: Backoff retries (2s, 5s, 10s) on 429/5xx/timeout, GROQ_FALLBACK_MODEL switch, 120s graph timeout cap,
and fallback to deterministic CoordinatorAgent decision.
"""

import os
import re
import time
import json
import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union, TypedDict

import numpy as np
import pandas as pd

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_tools import AgentSession, create_agent_tools, get_llm
from src.agents import (
    PrognosticAgent,
    ResourceAgent,
    CostAgent,
    CoordinatorAgent,
)
from src.simulator import EngineObservation
from src.data_loader import load_cmapss


# =============================================================================
# 1. SHARED STATE SPECIFICATION (TypedDict)
# =============================================================================

class AgenticState(TypedDict, total=False):
    """
    Shared typed state flowing across all nodes of the LangGraph predictive maintenance graph.
    Maintains strict zero-leakage information rules (no true RUL).
    """
    engine_id: int
    current_cycle: int
    inputs: Dict[str, Any]  # lead_time, technician_cycles, cost_params, age
    prognosis: Optional[Dict[str, Any]]  # predicted_rul, lower_bound, upper_bound
    feasible_windows: Optional[List[int]]  # Feasible candidate offsets s (None if not computed)
    candidate_windows: Optional[List[Dict[str, Any]]]  # Full diagnostic window table
    part_arrival_cycle: Optional[int]
    risk_profile: Optional[Dict[str, Any]]  # prob_fail, exp_min_r per offset
    cost_table: Optional[List[Dict[str, Any]]]  # C(s), E(s), cost_rate per offset
    proposed_cycle: Optional[int]
    best_cost_rate: Optional[float]
    escalated: bool
    retries: int
    messages: List[BaseMessage]
    trace: List[Dict[str, Any]]
    validated: bool
    fallback: bool
    fallback_reason: Optional[str]
    validation_rejected: Optional[bool]
    validation_status: Optional[str]
    reservations_done: Optional[bool]
    supervisor_next_node: Optional[str]
    rationale_source: Optional[str]  # "llm" or "template"
    start_time: Optional[float]
    final_recommendation: Optional[Dict[str, Any]]


# =============================================================================
# 2. UTILITIES: SANITIZATION, NUMBER VALIDATION & TOOL PARSING
# =============================================================================

def sanitize_key_from_text(text: str) -> str:
    """Ensure no API key appears in errors, traces, or logs."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key and key in text:
        text = text.replace(key, "[REDACTED]")
    text = re.sub(r'gsk_[A-Za-z0-9]{20,}', '[REDACTED]', text)
    return text


def parse_and_validate_tool_args(raw_args: Any) -> Dict[str, Any]:
    """Handle Groq quirk where tool arguments arrive as a JSON string."""
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    elif isinstance(raw_args, dict):
        return dict(raw_args)
    return {}


def deduplicate_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate identical repeated tool calls in a single turn.
    Preserves the first occurrence while dropping identical subsequent calls.
    """
    seen: Set[Tuple[str, str]] = set()
    unique_calls: List[Dict[str, Any]] = []
    for tc in tool_calls:
        name = tc.get("name", "")
        raw_args = tc.get("args")
        if isinstance(raw_args, dict):
            norm_args = json.dumps(raw_args, sort_keys=True)
        elif isinstance(raw_args, str):
            try:
                norm_args = json.dumps(json.loads(raw_args), sort_keys=True)
            except Exception:
                norm_args = raw_args
        else:
            norm_args = str(raw_args)

        sig = (name, norm_args)
        if sig in seen:
            continue
        seen.add(sig)
        unique_calls.append(tc)
    return unique_calls


def extract_numbers_from_text(text: str) -> List[str]:
    """Extract numeric tokens from plain-English rationale."""
    return re.findall(r'\b\d+(?:\.\d+)?\b', text)


def validate_rationale_guardrail(
    rationale: str,
    allowed_numbers: Set[str],
    escalated: bool = False,
) -> Tuple[bool, str]:
    """
    Guardrail:
    1. Rejects if rationale claims escalation when escalated=False (or the reverse).
    2. Rejects if rationale includes any number not present in tool outputs.
    Returns (is_valid, failure_reason_or_PASSED).
    """
    text_lower = rationale.lower()

    # 1. Check escalation consistency
    escalation_indicators = [
        "escalat",
        "no viable window",
        "no feasible window",
        "emergency",
        "cannot be scheduled",
        "could not be scheduled",
        "unschedulable",
    ]
    normal_indicators = [
        "lowest expected cost rate",
        "cost-optimal",
        "lowest cost rate",
        "cost rate $",
        "cost rate of $",
        "prob_fail=",
    ]

    if not escalated:
        for ind in escalation_indicators:
            if ind in text_lower:
                return False, f"Contradiction: claims escalation/unavailability ('{ind}') when escalated=False"
    else:
        for ind in normal_indicators:
            if ind in text_lower:
                return False, f"Contradiction: claims normal cost rate optimization ('{ind}') when escalated=True"

    # 2. Check for hallucinated numbers
    found = extract_numbers_from_text(rationale)
    unrecognized = []
    for token in found:
        if token in allowed_numbers:
            continue
        try:
            val = float(token)
            matched = False
            for allowed in allowed_numbers:
                try:
                    if abs(val - float(allowed)) < 1e-3:
                        matched = True
                        break
                except ValueError:
                    pass
            if not matched:
                unrecognized.append(token)
        except ValueError:
            unrecognized.append(token)

    if unrecognized:
        return False, f"Hallucinated numbers detected: {unrecognized}"

    return True, "PASSED"


# =============================================================================
# 3. ROBUST LLM INVOCATION WRAPPER (BACKOFF, FALLBACK MODEL, 120s CAP)
# =============================================================================

def invoke_llm_resilient(
    llm: Any,
    messages: List[BaseMessage],
    agent_name: str,
    call_num: int = 1,
    state: Optional[AgenticState] = None,
    primary_model_name: Optional[str] = None,
    fallback_model_name: Optional[str] = None,
    bound_tools: Optional[Sequence[Any]] = None,
    timeout_cap: float = 120.0,
) -> Tuple[Optional[AIMessage], Optional[str]]:
    """
    Invokes LLM with:
    - Progress line on every call and retry (e.g. "[cost_agent] call 2: 429, retrying in 5s (attempt 2/3)")
    - Backoff retries (2s, 5s, 10s) on 429, 5xx, or timeout
    - Switch to GROQ_FALLBACK_MODEL if primary model fails
    - 120s graph timeout cap
    - Returns (AIMessage, error_class_and_message_or_None)
    """
    start_t = state.get("start_time", time.time()) if state is not None else time.time()

    is_fake = isinstance(llm, FakeChatModel) or getattr(llm, "_llm_type", "") == "fake-predictive-maintenance-chat"
    backoffs = [0.01, 0.02, 0.03] if is_fake else [2, 5, 10]

    if is_fake:
        last_err = None
        for attempt, backoff in enumerate(backoffs, start=1):
            try:
                res = llm.invoke(messages)
                return res, None
            except Exception as e:
                last_err = e
                err_str = str(e)
                is_429 = "429" in err_str or "rate_limit" in err_str.lower()
                err_label = "429" if is_429 else "error"
                print(f"[{agent_name}] call {call_num}: {err_label}, retrying in {backoff}s (attempt {attempt}/3)")
                time.sleep(backoff)
        err_cls = last_err.__class__.__name__ if last_err else "RuntimeError"
        clean_msg = sanitize_key_from_text(str(last_err) if last_err else "Error")
        return None, f"{err_cls}: {clean_msg}"

    models_to_try = [primary_model_name or os.environ.get("GROQ_MODEL", "")]
    fb_model = fallback_model_name or os.environ.get("GROQ_FALLBACK_MODEL", "").strip()
    if fb_model and fb_model not in models_to_try:
        models_to_try.append(fb_model)

    last_err: Optional[Exception] = None

    for m_idx, current_model in enumerate(models_to_try):
        if not current_model:
            continue
        if m_idx > 0:
            print(f"[{agent_name}] Switching to fallback model '{current_model}' after exhaustion on primary model.")

        from langchain_groq import ChatGroq
        api_key = os.environ.get("GROQ_API_KEY", "").strip()

        model_llm = ChatGroq(
            model=current_model,
            groq_api_key=api_key,
            temperature=0.0,
            request_timeout=30.0,
            max_retries=2,
        )
        if bound_tools:
            model_llm = model_llm.bind_tools(bound_tools)

        for attempt, backoff in enumerate(backoffs, start=1):
            if (time.time() - start_t) > timeout_cap:
                return None, "TimeoutError: Graph execution exceeded 120s limit"

            try:
                res = model_llm.invoke(messages)
                return res, None
            except Exception as e:
                last_err = e
                err_str = str(e)
                is_429 = "429" in err_str or "rate_limit" in err_str.lower()
                is_5xx = any(code in err_str for code in ["500", "502", "503", "504", "server_error"])
                is_timeout = "timeout" in err_str.lower() or isinstance(e, TimeoutError)

                err_label = "429" if is_429 else ("timeout" if is_timeout else ("5xx" if is_5xx else "error"))
                print(f"[{agent_name}] call {call_num}: {err_label}, retrying in {backoff}s (attempt {attempt}/3)")
                time.sleep(backoff)

    err_cls = last_err.__class__.__name__ if last_err else "RuntimeError"
    clean_msg = sanitize_key_from_text(str(last_err) if last_err else "All retry attempts exhausted")
    return None, f"{err_cls}: {clean_msg}"


# =============================================================================
# 4. SCRIPTED FAKE CHAT MODEL FOR OFFLINE & REGRESSION TESTING
# =============================================================================

class FakeChatModel(BaseChatModel):
    """
    Deterministic scripted chat model for testing and offline demos without external API keys.
    Emits valid tool calls matching the agent's role and bound tools.
    Supports injecting invalid proposals, skipping required tools, and simulating errors.
    """
    bound_tools: List[Any] = []
    inject_invalid_cycles: int = 0
    _invalid_injected_count: int = 0
    simulate_429_count: int = 0
    _simulated_429_so_far: int = 0
    simulate_malformed_tool_call: bool = False
    _malformed_delivered: bool = False
    simulate_skip_tech_windows: bool = False
    simulate_contradictory_rationale: bool = False

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeChatModel":
        clone = FakeChatModel()
        clone.bound_tools = list(tools)
        clone.inject_invalid_cycles = self.inject_invalid_cycles
        clone._invalid_injected_count = self._invalid_injected_count
        clone.simulate_429_count = self.simulate_429_count
        clone._simulated_429_so_far = self._simulated_429_so_far
        clone.simulate_malformed_tool_call = self.simulate_malformed_tool_call
        clone._malformed_delivered = self._malformed_delivered
        clone.simulate_skip_tech_windows = self.simulate_skip_tech_windows
        clone.simulate_contradictory_rationale = self.simulate_contradictory_rationale
        return clone

    @property
    def _llm_type(self) -> str:
        return "fake-predictive-maintenance-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_names = [getattr(t, "name", str(t)) for t in self.bound_tools]
        last_msg_text = messages[-1].content if messages else ""

        # Simulated 429 for testing
        if self.simulate_429_count > 0 and self._simulated_429_so_far < self.simulate_429_count:
            self._simulated_429_so_far += 1
            raise RuntimeError(f"429 Too Many Requests (Rate limit reached) [simulated {self._simulated_429_so_far}]")

        # Simulated malformed tool call on first attempt
        if self.simulate_malformed_tool_call and not self._malformed_delivered:
            self._malformed_delivered = True
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="I suggest doing maintenance soon.", tool_calls=[]))])

        # 1. Prognostic Agent Tool Calls
        if "predict_rul" in tool_names and "get_risk_profile" in tool_names:
            if "risk" in last_msg_text.lower() or "candidate_offsets" in last_msg_text.lower() or "offsets=" in last_msg_text.lower():
                pred_rul = 28.0
                offsets = [16, 22]
                for m in reversed(messages):
                    if isinstance(m.content, str) and "pred_rul=" in m.content:
                        try:
                            part = m.content.split("pred_rul=")[1].split()[0].replace(",", "")
                            pred_rul = float(part)
                        except Exception:
                            pass
                    if isinstance(m.content, str) and ("offsets=" in m.content or "candidate_offsets=" in m.content):
                        try:
                            if "candidate_offsets=" in m.content:
                                part = m.content.split("candidate_offsets=")[1].split("]")[0] + "]"
                            else:
                                part = m.content.split("offsets=")[1].split("]")[0] + "]"
                            offsets = json.loads(part)
                        except Exception:
                            pass
                tool_call = {
                    "name": "get_risk_profile",
                    "args": {"pred_rul": pred_rul, "candidate_offsets": offsets},
                    "id": "call_risk_profile_1",
                }
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[tool_call]))])
            else:
                eng_id = 17
                cyc = 0
                for m in reversed(messages):
                    if isinstance(m.content, str):
                        if "engine_id=" in m.content:
                            try:
                                eng_id = int(m.content.split("engine_id=")[1].split()[0].replace(",", "").rstrip("."))
                            except Exception:
                                pass
                        if "current_cycle=" in m.content:
                            try:
                                cyc = int(m.content.split("current_cycle=")[1].split()[0].replace(",", "").rstrip("."))
                            except Exception:
                                pass
                tool_call = {
                    "name": "predict_rul",
                    "args": {"engine_id": eng_id, "current_cycle": cyc},
                    "id": "call_predict_rul_1",
                }
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[tool_call]))])

        # 2. Resource Agent Tool Calls
        elif "list_technician_windows" in tool_names:
            # Check if this is Pass 2: after validation passes (committing reservations)
            if "reserve technician window" in last_msg_text.lower() or "validation has passed" in last_msg_text.lower():
                prop_c = 16
                cur_c = 0
                for m in reversed(messages):
                    if isinstance(m.content, str):
                        if "start_cycle=" in m.content:
                            try:
                                prop_c = int(m.content.split("start_cycle=")[1].split()[0].replace(",", ""))
                            except Exception:
                                pass
                        if "current_cycle=" in m.content:
                            try:
                                cur_c = int(m.content.split("current_cycle=")[1].split()[0].replace(",", ""))
                            except Exception:
                                pass
                tool_calls = [
                    {"name": "reserve_window", "args": {"start_cycle": prop_c}, "id": "call_reserve_1"},
                    {"name": "order_spare", "args": {"current_cycle": cur_c}, "id": "call_order_1"},
                ]
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=tool_calls))])

            cyc = 0
            p_rul = 28.0
            tech_cycles = [16, 22]
            lead_time = 8
            for m in reversed(messages):
                if "current_cycle=" in m.content:
                    try:
                        cyc = int(m.content.split("current_cycle=")[1].split()[0].replace(",", ""))
                    except Exception:
                        pass
                if "pred_rul=" in m.content:
                    try:
                        p_rul = float(m.content.split("pred_rul=")[1].split()[0].replace(",", ""))
                    except Exception:
                        pass
                if "technician_cycles=" in m.content:
                    try:
                        part = m.content.split("technician_cycles=")[1].split("]")[0] + "]"
                        tech_cycles = json.loads(part)
                    except Exception:
                        pass
                if "lead_time=" in m.content:
                    try:
                        lead_time = int(m.content.split("lead_time=")[1].split()[0].replace(",", "").rstrip("."))
                    except Exception:
                        pass

            args_spare = {"current_cycle": cyc, "lead_time": lead_time, "order_placed": False}
            args_tech = {"current_cycle": cyc, "pred_rul": p_rul, "technician_cycles": tech_cycles}

            # Regression Test (a): Model calls check_spare_status twice and skips list_technician_windows
            if self.simulate_skip_tech_windows:
                tool_calls = [
                    {"name": "check_spare_status", "args": args_spare, "id": "call_spare_status_1"},
                    {"name": "check_spare_status", "args": args_spare, "id": "call_spare_status_2"},
                ]
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=tool_calls))])

            tool_calls = [
                {
                    "name": "check_spare_status",
                    "args": args_spare,
                    "id": "call_spare_status_1",
                },
                {
                    "name": "list_technician_windows",
                    "args": args_tech,
                    "id": "call_tech_windows_1",
                },
            ]
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=tool_calls))])

        # 3. Cost Agent Tool Calls
        elif "evaluate_candidate_costs" in tool_names:
            age = 120
            offsets = [16, 22]
            pf = [0.0, 0.0722]
            exp_r = [16.0, 21.8]
            for m in reversed(messages):
                if "age=" in m.content:
                    try:
                        age = int(m.content.split("age=")[1].split()[0].replace(",", ""))
                    except Exception:
                        pass
                if "feasible_offsets=" in m.content:
                    try:
                        part = m.content.split("feasible_offsets=")[1].split("]")[0] + "]"
                        offsets = json.loads(part)
                    except Exception:
                        pass
                if "prob_fail=" in m.content:
                    try:
                        part = m.content.split("prob_fail=")[1].split("]")[0] + "]"
                        pf = json.loads(part)
                    except Exception:
                        pass
                if "exp_min_r=" in m.content:
                    try:
                        part = m.content.split("exp_min_r=")[1].split("]")[0] + "]"
                        exp_r = json.loads(part)
                    except Exception:
                        pass

            args_cost = {
                "age": age,
                "feasible_offsets": offsets,
                "prob_fail": pf,
                "exp_min_r": exp_r,
            }

            tool_call = {
                "name": "evaluate_candidate_costs",
                "args": args_cost,
                "id": "call_cost_rates_1",
            }
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[tool_call]))])

        # 4. Supervisor Routing or Report Node
        else:
            if "determine which specialized node runs next" in last_msg_text.lower():
                if "prognosis_available=false" in last_msg_text.lower():
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="prognostic_agent"))])
                elif "feasible_windows_count=none" in last_msg_text.lower():
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="resource_agent"))])
                elif "feasible_windows_count=0" in last_msg_text.lower():
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="escalation"))])
                elif "risk_profile_available=false" in last_msg_text.lower():
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="prognostic_agent"))])
                elif "cost_table_available=false" in last_msg_text.lower():
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="cost_agent"))])
                elif "validated=false" in last_msg_text.lower():
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="validator"))])
                elif "reservations_done=false" in last_msg_text.lower():
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="resource_agent"))])
                else:
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="report"))])

            # Regression Test (c): Contradictory rationale simulation
            if self.simulate_contradictory_rationale:
                if "status: scheduled" in last_msg_text.lower():
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="EMERGENCY ESCALATION: No viable window was available, emergency dispatch required."))])
                else:
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Selected cost-optimal maintenance cycle minimizing renewal cost rate."))])

            # Report rationale generation
            reason = "Selected cost-optimal maintenance window minimizing renewal cost rate while preserving engine useful life."
            if "escalation" in last_msg_text.lower() or "status: escalation" in last_msg_text.lower():
                cyc_val = "100"
                part_val = "8"
                for m in reversed(messages):
                    if isinstance(m.content, str):
                        if "chosen_cycle:" in m.content:
                            try:
                                cyc_val = m.content.split("chosen_cycle:")[1].split()[0]
                            except Exception:
                                pass
                        if "spare_arrival:" in m.content:
                            try:
                                part_val = m.content.split("spare_arrival:")[1].split()[0]
                            except Exception:
                                pass
                reason = f"ESCALATION: No feasible maintenance window within predicted RUL. Secured earliest window at cycle {cyc_val} (part arrives cycle {part_val})."
            else:
                for m in reversed(messages):
                    if isinstance(m.content, str) and "chosen_cycle=" in m.content:
                        try:
                            cyc_val = m.content.split("chosen_cycle=")[1].split()[0].replace(",", "")
                            offset_part = m.content.split("offset=")[1].split()[0].replace(",", "")
                            offset_val = offset_part.replace("+", "")
                            rate_val = m.content.split("cost_rate=")[1].split()[0].replace(",", "")
                            reason = f"Selected cycle {cyc_val} (offset +{offset_val} cycles) with lowest expected cost rate ${float(rate_val):.2f}/cycle."
                        except Exception:
                            pass
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reason))])


# =============================================================================
# 5. GRAPH BUILDER & NODES IMPLEMENTATION
# =============================================================================

def build_predictive_maintenance_graph(
    session: AgentSession,
    llm_type: str = "fake",
    inject_invalid_cycles: int = 0,
    simulate_429_count: int = 0,
    simulate_malformed: bool = False,
    simulate_skip_tech_windows: bool = False,
    simulate_contradictory_rationale: bool = False,
) -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph with supervisor, specialized agent nodes,
    deterministic validator, escalation procedure, resource reservation, and report assembler.
    """
    primary_model_name = os.environ.get("GROQ_MODEL", "").strip()
    fallback_model_name = os.environ.get("GROQ_FALLBACK_MODEL", "").strip()

    # Resolve LLM instance
    if llm_type == "groq":
        base_llm = get_llm(temperature=0.0)
    else:
        fake_model = FakeChatModel()
        fake_model.inject_invalid_cycles = inject_invalid_cycles
        fake_model.simulate_429_count = simulate_429_count
        fake_model.simulate_malformed_tool_call = simulate_malformed
        fake_model.simulate_skip_tech_windows = simulate_skip_tech_windows
        fake_model.simulate_contradictory_rationale = simulate_contradictory_rationale
        base_llm = fake_model

    # Bind tools to individual agent nodes
    prognostic_tools = [session.predict_rul, session.get_risk_profile]
    resource_tools = [session.check_spare_status, session.list_technician_windows, session.reserve_window, session.order_spare]
    cost_tools = [session.evaluate_candidate_costs]

    prognostic_llm = base_llm.bind_tools(prognostic_tools)
    resource_llm = base_llm.bind_tools(resource_tools)
    cost_llm = base_llm.bind_tools(cost_tools)

    tool_map: Dict[str, BaseTool] = {
        "predict_rul": session.predict_rul,
        "get_risk_profile": session.get_risk_profile,
        "check_spare_status": session.check_spare_status,
        "list_technician_windows": session.list_technician_windows,
        "evaluate_candidate_costs": session.evaluate_candidate_costs,
        "reserve_window": session.reserve_window,
        "order_spare": session.order_spare,
    }

    # Helper: trigger Coordinator deterministic fallback
    def trigger_coordinator_fallback(state: AgenticState, reason_str: str) -> Dict[str, Any]:
        cur_c = state["current_cycle"]
        eng_id = state["engine_id"]
        inputs = state.get("inputs", {})
        lead_time = inputs.get("lead_time", session.spare_lead_time)
        tech_cycles = inputs.get("technician_cycles", [16, 22])
        age = inputs.get("age", cur_c)

        # 1. Ensure prognosis
        if not state.get("prognosis"):
            res_pred = session.predict_rul.invoke({"engine_id": eng_id, "current_cycle": cur_c})
            state["prognosis"] = res_pred

        p_rul = state["prognosis"]["predicted_rul"]

        # 2. Ensure resource feasibility
        if state.get("feasible_windows") is None:
            res_tech = session.list_technician_windows.invoke({
                "current_cycle": cur_c,
                "pred_rul": p_rul,
                "technician_cycles": tech_cycles,
            })
            state["feasible_windows"] = res_tech["feasible_offsets"]
            state["candidate_windows"] = res_tech["candidate_windows"]
            state["part_arrival_cycle"] = res_tech["part_arrival_cycle"]

        feas = state.get("feasible_windows", [])

        # 3. If no feasible windows -> escalation fallback
        if len(feas) == 0:
            chosen_cycle = session.resource.find_earliest_available(cur_c + lead_time, session.preventive_duration)
            state["proposed_cycle"] = chosen_cycle
            state["escalated"] = True
            state["fallback"] = True
            state["fallback_reason"] = reason_str
            state["validated"] = True
            return state

        # 4. Ensure cost table
        if not state.get("cost_table"):
            res_risk = session.get_risk_profile.invoke({"pred_rul": p_rul, "candidate_offsets": feas})
            state["risk_profile"] = res_risk
            res_cost = session.evaluate_candidate_costs.invoke({
                "age": age,
                "feasible_offsets": feas,
                "prob_fail": res_risk["prob_fail"],
                "exp_min_r": res_risk["exp_min_r"],
            })
            state["cost_table"] = res_cost["candidate_costs"]
            state["best_cost_rate"] = res_cost["best_cost_rate"]
            best_offset = res_cost["best_offset"]
        else:
            c_table = state["cost_table"]
            rates = [c["cost_rate"] for c in c_table]
            best_idx = int(np.argmin(rates))
            best_offset = c_table[best_idx]["offset_s"]

        state["proposed_cycle"] = cur_c + best_offset
        state["fallback"] = True
        state["fallback_reason"] = reason_str
        state["validated"] = True
        return state

    # -------------------------------------------------------------------------
    # Node 1: Supervisor (Coordinator)
    # -------------------------------------------------------------------------
    def supervisor_node(state: AgenticState) -> Dict[str, Any]:
        """
        Coordinator supervisor node: inspects state and logs trajectory progress.
        Under --llm groq: invokes supervisor LLM to select the next node.
        Under --llm fake: delegates routing to deterministic route_supervisor.
        """
        # Timeout cap check
        if state.get("start_time") and (time.time() - state["start_time"] > 120.0):
            trigger_coordinator_fallback(state, "TimeoutError: Graph execution exceeded 120s limit")

        next_decision = None
        if llm_type == "groq" and not state.get("fallback"):
            eng_id = state["engine_id"]
            cur_c = state["current_cycle"]
            prompt = (
                f"SYSTEM: You are the fleet maintenance supervisor in a predictive maintenance system.\n"
                f"Determine which specialized node runs next based on strict dependencies:\n"
                f"1. prognostic_agent (if prognosis is None)\n"
                f"2. resource_agent (if feasible_windows is None)\n"
                f"3. escalation (ONLY if feasible_windows is an empty list [])\n"
                f"4. prognostic_agent (if risk_profile is None)\n"
                f"5. cost_agent (if cost_table is None or validation_rejected is True)\n"
                f"6. validator (if validated is False and proposed_cycle is set)\n"
                f"7. resource_agent (if validated is True and reservations_done is False)\n"
                f"8. report (if validated is True and reservations_done is True)\n\n"
                f"Current State: engine_id={eng_id}, current_cycle={cur_c}, "
                f"prognosis_available={bool(state.get('prognosis'))}, "
                f"feasible_windows={state.get('feasible_windows')}, "
                f"risk_profile_available={bool(state.get('risk_profile'))}, "
                f"cost_table_available={bool(state.get('cost_table'))}, "
                f"proposed_cycle={state.get('proposed_cycle')}, "
                f"validated={bool(state.get('validated'))}, "
                f"reservations_done={bool(state.get('reservations_done'))}, "
                f"escalated={bool(state.get('escalated'))}, "
                f"retries={state.get('retries', 0)}.\n\n"
                f"Reply ONLY with the exact node name from the allowed list."
            )
            ai_msg, err = invoke_llm_resilient(
                llm=base_llm,
                messages=[HumanMessage(content=prompt)],
                agent_name="supervisor",
                call_num=1,
                state=state,
                primary_model_name=primary_model_name,
                fallback_model_name=fallback_model_name,
            )
            if ai_msg and ai_msg.content:
                chosen = ai_msg.content.strip().lower()
                valid_nodes = ["prognostic_agent", "resource_agent", "cost_agent", "validator", "escalation", "report"]
                for v in valid_nodes:
                    if v in chosen:
                        next_decision = v
                        break

        trace_entry = {
            "node": "supervisor",
            "prognosis_available": bool(state.get("prognosis")),
            "feasible_windows_count": len(state.get("feasible_windows")) if state.get("feasible_windows") is not None else None,
            "cost_table_available": bool(state.get("cost_table")),
            "validated": bool(state.get("validated")),
            "reservations_done": bool(state.get("reservations_done")),
            "escalated": bool(state.get("escalated")),
            "retries": int(state.get("retries", 0)),
            "supervisor_choice": next_decision,
        }
        state["trace"].append(trace_entry)
        return {
            "supervisor_next_node": next_decision,
            "trace": state["trace"],
        }

    # -------------------------------------------------------------------------
    # Node 2: Prognostic Agent
    # -------------------------------------------------------------------------
    def prognostic_agent_node(state: AgenticState) -> Dict[str, Any]:
        """
        Executes Prognostic Agent tasks with required-step enforcement:
        - Pass 1: Required tool predict_rul.
        - Pass 2: Required tool get_risk_profile.
        """
        eng_id = state["engine_id"]
        cur_c = state["current_cycle"]

        if not state.get("prognosis"):
            # Pass 1: Predict RUL
            prompt = (
                f"SYSTEM: You are the Prognostic Agent in a turbofan predictive maintenance system.\n"
                f"Required tool order:\n"
                f"1. Call 'predict_rul' with engine_id={eng_id} and current_cycle={cur_c}.\n"
                f"Predict Remaining Useful Life point estimate and 90% uncertainty bounds. Zero true RUL leakage allowed."
            )
            ai_msg, err = invoke_llm_resilient(
                llm=prognostic_llm,
                messages=[HumanMessage(content=prompt)],
                agent_name="prognostic_agent",
                call_num=1,
                state=state,
                primary_model_name=primary_model_name,
                fallback_model_name=fallback_model_name,
                bound_tools=prognostic_tools,
            )
            if err:
                trigger_coordinator_fallback(state, err)
                return {
                    "fallback": True,
                    "fallback_reason": err,
                    "prognosis": state.get("prognosis"),
                    "feasible_windows": state.get("feasible_windows"),
                    "candidate_windows": state.get("candidate_windows"),
                    "part_arrival_cycle": state.get("part_arrival_cycle"),
                    "cost_table": state.get("cost_table"),
                    "best_cost_rate": state.get("best_cost_rate"),
                    "proposed_cycle": state.get("proposed_cycle"),
                    "escalated": state.get("escalated", False),
                    "validated": True,
                    "messages": state["messages"],
                    "trace": state["trace"],
                }

            state["messages"].append(ai_msg)

            prognosis_dict = None
            executed_tools = set()
            tool_calls = deduplicate_tool_calls(ai_msg.tool_calls) if ai_msg and ai_msg.tool_calls else []

            for tc in tool_calls:
                t_name = tc.get("name")
                if t_name in tool_map:
                    t_args = parse_and_validate_tool_args(tc["args"])
                    res = tool_map[t_name].invoke(t_args)
                    state["messages"].append(ToolMessage(content=json.dumps(res), tool_call_id=tc.get("id", f"tc_{t_name}")))
                    state["trace"].append({"node": "prognostic_agent", "tool": t_name, "args": t_args, "result": res, "enforced_by": "agent"})
                    executed_tools.add(t_name)
                    if t_name == "predict_rul":
                        prognosis_dict = res

            # Required-step enforcement for predict_rul
            if "predict_rul" not in executed_tools:
                corr_msg, _ = invoke_llm_resilient(
                    llm=prognostic_llm,
                    messages=state["messages"] + [HumanMessage(content="Required tool 'predict_rul' was not called. You MUST call predict_rul(engine_id=..., current_cycle=...).")],
                    agent_name="prognostic_agent",
                    call_num=2,
                    state=state,
                    primary_model_name=primary_model_name,
                    fallback_model_name=fallback_model_name,
                    bound_tools=prognostic_tools,
                )
                if corr_msg and corr_msg.tool_calls:
                    state["messages"].append(corr_msg)
                    for tc in deduplicate_tool_calls(corr_msg.tool_calls):
                        t_name = tc.get("name")
                        if t_name == "predict_rul":
                            t_args = parse_and_validate_tool_args(tc["args"])
                            res = tool_map[t_name].invoke(t_args)
                            state["messages"].append(ToolMessage(content=json.dumps(res), tool_call_id=tc.get("id", "tc_pred_corr")))
                            state["trace"].append({"node": "prognostic_agent", "tool": t_name, "args": t_args, "result": res, "enforced_by": "agent"})
                            executed_tools.add(t_name)
                            prognosis_dict = res

            if "predict_rul" not in executed_tools:
                res = session.predict_rul.invoke({"engine_id": eng_id, "current_cycle": cur_c})
                prognosis_dict = res
                state["trace"].append({
                    "node": "prognostic_agent",
                    "tool": "predict_rul",
                    "args": {"engine_id": eng_id, "current_cycle": cur_c},
                    "result": res,
                    "enforced_by": "graph",
                })

            return {"prognosis": prognosis_dict, "messages": state["messages"], "trace": state["trace"]}

        else:
            # Pass 2: Evaluate risk profile for feasible offsets
            p_rul = state["prognosis"]["predicted_rul"]
            offsets = state.get("feasible_windows", [])
            prompt = (
                f"SYSTEM: You are the Prognostic Agent.\n"
                f"Required tool order:\n"
                f"1. Call 'get_risk_profile' with pred_rul={p_rul:.4f} and candidate_offsets={offsets}.\n"
                f"Compute empirical residual failure risk profile P(R <= s) and expected run duration."
            )
            ai_msg, err = invoke_llm_resilient(
                llm=prognostic_llm,
                messages=[HumanMessage(content=prompt)],
                agent_name="prognostic_agent",
                call_num=1,
                state=state,
                primary_model_name=primary_model_name,
                fallback_model_name=fallback_model_name,
                bound_tools=prognostic_tools,
            )
            if err:
                trigger_coordinator_fallback(state, err)
                return {
                    "fallback": True,
                    "fallback_reason": err,
                    "prognosis": state.get("prognosis"),
                    "feasible_windows": state.get("feasible_windows"),
                    "candidate_windows": state.get("candidate_windows"),
                    "part_arrival_cycle": state.get("part_arrival_cycle"),
                    "cost_table": state.get("cost_table"),
                    "best_cost_rate": state.get("best_cost_rate"),
                    "proposed_cycle": state.get("proposed_cycle"),
                    "escalated": state.get("escalated", False),
                    "validated": True,
                    "messages": state["messages"],
                    "trace": state["trace"],
                }

            state["messages"].append(ai_msg)

            risk_dict = None
            executed_tools = set()
            tool_calls = deduplicate_tool_calls(ai_msg.tool_calls) if ai_msg and ai_msg.tool_calls else []

            for tc in tool_calls:
                t_name = tc.get("name")
                if t_name in tool_map:
                    t_args = parse_and_validate_tool_args(tc["args"])
                    res = tool_map[t_name].invoke(t_args)
                    state["messages"].append(ToolMessage(content=json.dumps(res), tool_call_id=tc.get("id", f"tc_{t_name}")))
                    state["trace"].append({"node": "prognostic_agent", "tool": t_name, "args": t_args, "result": res, "enforced_by": "agent"})
                    executed_tools.add(t_name)
                    if t_name == "get_risk_profile":
                        risk_dict = res

            # Required-step enforcement for get_risk_profile
            if "get_risk_profile" not in executed_tools:
                corr_msg, _ = invoke_llm_resilient(
                    llm=prognostic_llm,
                    messages=state["messages"] + [HumanMessage(content="Required tool 'get_risk_profile' was not called. You MUST call get_risk_profile(pred_rul=..., candidate_offsets=...).")],
                    agent_name="prognostic_agent",
                    call_num=2,
                    state=state,
                    primary_model_name=primary_model_name,
                    fallback_model_name=fallback_model_name,
                    bound_tools=prognostic_tools,
                )
                if corr_msg and corr_msg.tool_calls:
                    state["messages"].append(corr_msg)
                    for tc in deduplicate_tool_calls(corr_msg.tool_calls):
                        t_name = tc.get("name")
                        if t_name == "get_risk_profile":
                            t_args = parse_and_validate_tool_args(tc["args"])
                            res = tool_map[t_name].invoke(t_args)
                            state["messages"].append(ToolMessage(content=json.dumps(res), tool_call_id=tc.get("id", "tc_risk_corr")))
                            state["trace"].append({"node": "prognostic_agent", "tool": t_name, "args": t_args, "result": res, "enforced_by": "agent"})
                            executed_tools.add(t_name)
                            risk_dict = res

            if "get_risk_profile" not in executed_tools:
                res = session.get_risk_profile.invoke({"pred_rul": p_rul, "candidate_offsets": offsets})
                risk_dict = res
                state["trace"].append({
                    "node": "prognostic_agent",
                    "tool": "get_risk_profile",
                    "args": {"pred_rul": p_rul, "candidate_offsets": offsets},
                    "result": res,
                    "enforced_by": "graph",
                })

            return {"risk_profile": risk_dict, "messages": state["messages"], "trace": state["trace"]}

    # -------------------------------------------------------------------------
    # Node 3: Resource Agent
    # -------------------------------------------------------------------------
    def resource_agent_node(state: AgenticState) -> Dict[str, Any]:
        """
        Executes Resource Agent tasks with required-step enforcement:
        - Pass 1: Required tools check_spare_status then list_technician_windows.
        - Pass 2: Required tools reserve_window and order_spare post-validation.
        """
        cur_c = state["current_cycle"]
        inputs = state.get("inputs", {})
        lead_time = inputs.get("lead_time", session.spare_lead_time)

        # Pass 2: Validation has passed -> Commit technician window and spare order
        if state.get("validated") and not state.get("reservations_done"):
            prop_cycle = state["proposed_cycle"]
            prompt = (
                f"SYSTEM: You are the Resource Agent. Validation has passed.\n"
                f"Required tool order:\n"
                f"1. Call 'reserve_window' with start_cycle={prop_cycle}.\n"
                f"2. Call 'order_spare' with current_cycle={cur_c}."
            )
            ai_msg, err = invoke_llm_resilient(
                llm=resource_llm,
                messages=[HumanMessage(content=prompt)],
                agent_name="resource_agent",
                call_num=1,
                state=state,
                primary_model_name=primary_model_name,
                fallback_model_name=fallback_model_name,
                bound_tools=resource_tools,
            )
            executed_res_tools = set()
            if ai_msg and ai_msg.tool_calls:
                for tc in deduplicate_tool_calls(ai_msg.tool_calls):
                    t_name = tc["name"]
                    if t_name in tool_map:
                        t_args = parse_and_validate_tool_args(tc["args"])
                        res = tool_map[t_name].invoke(t_args)
                        state["messages"].append(ToolMessage(content=json.dumps(res), tool_call_id=tc.get("id", f"tc_{t_name}")))
                        state["trace"].append({"node": "resource_agent", "tool": t_name, "args": t_args, "result": res, "enforced_by": "agent"})
                        executed_res_tools.add(t_name)

            # Required enforcement for post-validation reservations
            if "reserve_window" not in executed_res_tools:
                res_w = session.reserve_window.invoke({"start_cycle": prop_cycle})
                state["trace"].append({"node": "resource_agent", "tool": "reserve_window", "args": {"start_cycle": prop_cycle}, "result": res_w, "enforced_by": "graph"})
            if "order_spare" not in executed_res_tools:
                res_o = session.order_spare.invoke({"current_cycle": cur_c})
                state["trace"].append({"node": "resource_agent", "tool": "order_spare", "args": {"current_cycle": cur_c}, "result": res_o, "enforced_by": "graph"})

            return {
                "reservations_done": True,
                "messages": state["messages"],
                "trace": state["trace"],
            }

        # Pass 1: Evaluates feasibility windows (Required: check_spare_status, list_technician_windows)
        p_rul = state["prognosis"]["predicted_rul"]
        tech_cycles = inputs.get("technician_cycles", [16, 22])

        prompt = (
            f"SYSTEM: You are the Resource Agent managing turbofan spare parts and technician rosters.\n"
            f"Required tool order:\n"
            f"1. Call 'check_spare_status' with current_cycle={cur_c}, lead_time={lead_time}, order_placed=False.\n"
            f"2. Call 'list_technician_windows' with current_cycle={cur_c}, pred_rul={p_rul:.4f}, technician_cycles={tech_cycles}.\n"
            f"You MUST execute both tools in this exact order. Do NOT repeat tool calls."
        )
        ai_msg, err = invoke_llm_resilient(
            llm=resource_llm,
            messages=[HumanMessage(content=prompt)],
            agent_name="resource_agent",
            call_num=1,
            state=state,
            primary_model_name=primary_model_name,
            fallback_model_name=fallback_model_name,
            bound_tools=resource_tools,
        )
        if err:
            trigger_coordinator_fallback(state, err)
            return {
                "fallback": True,
                "fallback_reason": err,
                "prognosis": state.get("prognosis"),
                "feasible_windows": state.get("feasible_windows"),
                "candidate_windows": state.get("candidate_windows"),
                "part_arrival_cycle": state.get("part_arrival_cycle"),
                "cost_table": state.get("cost_table"),
                "best_cost_rate": state.get("best_cost_rate"),
                "proposed_cycle": state.get("proposed_cycle"),
                "escalated": state.get("escalated", False),
                "validated": True,
                "messages": state["messages"],
                "trace": state["trace"],
            }

        state["messages"].append(ai_msg)

        executed_tools = set()
        feasible_offsets = None
        candidate_windows = None
        part_arrival = cur_c + lead_time
        tool_calls = deduplicate_tool_calls(ai_msg.tool_calls) if ai_msg and ai_msg.tool_calls else []

        for tc in tool_calls:
            t_name = tc.get("name")
            if t_name in tool_map:
                t_args = parse_and_validate_tool_args(tc["args"])
                if t_name == "list_technician_windows":
                    t_args["technician_cycles"] = tech_cycles
                res = tool_map[t_name].invoke(t_args)
                state["messages"].append(ToolMessage(content=json.dumps(res), tool_call_id=tc.get("id", f"tc_{t_name}")))
                state["trace"].append({"node": "resource_agent", "tool": t_name, "args": t_args, "result": res, "enforced_by": "agent"})
                executed_tools.add(t_name)
                if t_name == "list_technician_windows":
                    feasible_offsets = res["feasible_offsets"]
                    candidate_windows = res["candidate_windows"]
                    part_arrival = res["part_arrival_cycle"]
                elif t_name == "check_spare_status":
                    part_arrival = res["part_arrival_cycle"]

        # Required-step enforcement: both check_spare_status and list_technician_windows must run
        missing_tools = [t for t in ["check_spare_status", "list_technician_windows"] if t not in executed_tools]
        if missing_tools:
            miss = missing_tools[0]
            if miss == "list_technician_windows":
                corr_prompt = (
                    f"Required tool 'list_technician_windows' was not called. "
                    f"You MUST call 'list_technician_windows' with current_cycle={cur_c}, "
                    f"pred_rul={p_rul:.4f}, technician_cycles={tech_cycles}. Output only the tool call."
                )
            else:
                corr_prompt = (
                    f"Required tool 'check_spare_status' was not called. "
                    f"You MUST call 'check_spare_status' with current_cycle={cur_c}, "
                    f"lead_time={lead_time}, order_placed=False. Output only the tool call."
                )
            corr_msg, _ = invoke_llm_resilient(
                llm=resource_llm,
                messages=state["messages"] + [HumanMessage(content=corr_prompt)],
                agent_name="resource_agent",
                call_num=2,
                state=state,
                primary_model_name=primary_model_name,
                fallback_model_name=fallback_model_name,
                bound_tools=resource_tools,
            )
            if corr_msg and corr_msg.tool_calls:
                state["messages"].append(corr_msg)
                for tc in deduplicate_tool_calls(corr_msg.tool_calls):
                    t_name = tc.get("name")
                    if t_name in tool_map and t_name not in executed_tools:
                        t_args = parse_and_validate_tool_args(tc["args"])
                        if t_name == "list_technician_windows":
                            t_args["technician_cycles"] = tech_cycles
                        res = tool_map[t_name].invoke(t_args)
                        state["messages"].append(ToolMessage(content=json.dumps(res), tool_call_id=tc.get("id", f"tc_corr_{t_name}")))
                        state["trace"].append({"node": "resource_agent", "tool": t_name, "args": t_args, "result": res, "enforced_by": "agent"})
                        executed_tools.add(t_name)
                        if t_name == "list_technician_windows":
                            feasible_offsets = res["feasible_offsets"]
                            candidate_windows = res["candidate_windows"]
                            part_arrival = res["part_arrival_cycle"]
                        elif t_name == "check_spare_status":
                            part_arrival = res["part_arrival_cycle"]

        # If still missing, enforce deterministically with enforced_by="graph"
        if "check_spare_status" not in executed_tools:
            res_spare = session.check_spare_status.invoke({"current_cycle": cur_c, "lead_time": lead_time, "order_placed": False})
            part_arrival = res_spare["part_arrival_cycle"]
            state["trace"].append({
                "node": "resource_agent",
                "tool": "check_spare_status",
                "args": {"current_cycle": cur_c, "lead_time": lead_time, "order_placed": False},
                "result": res_spare,
                "enforced_by": "graph",
            })
            executed_tools.add("check_spare_status")

        if "list_technician_windows" not in executed_tools:
            res_tech = session.list_technician_windows.invoke({"current_cycle": cur_c, "pred_rul": p_rul, "technician_cycles": tech_cycles})
            feasible_offsets = res_tech["feasible_offsets"]
            candidate_windows = res_tech["candidate_windows"]
            part_arrival = res_tech["part_arrival_cycle"]
            state["trace"].append({
                "node": "resource_agent",
                "tool": "list_technician_windows",
                "args": {"current_cycle": cur_c, "pred_rul": p_rul, "technician_cycles": tech_cycles},
                "result": res_tech,
                "enforced_by": "graph",
            })
            executed_tools.add("list_technician_windows")

        return {
            "feasible_windows": feasible_offsets,
            "candidate_windows": candidate_windows,
            "part_arrival_cycle": part_arrival,
            "messages": state["messages"],
            "trace": state["trace"],
        }

    # -------------------------------------------------------------------------
    # Node 4: Cost Agent
    # -------------------------------------------------------------------------
    def cost_agent_node(state: AgenticState) -> Dict[str, Any]:
        """
        Executes Cost Agent tasks with required-step enforcement:
        - Evaluates expected cost per operating cycle C(s)/E(s) across feasible offsets.
        - Required tool: evaluate_candidate_costs.
        """
        inputs = state.get("inputs", {})
        age = inputs.get("age", state["current_cycle"])
        feasible_offsets = state["feasible_windows"]
        risk_profile = state["risk_profile"]

        prompt = (
            f"SYSTEM: You are the Cost Agent.\n"
            f"Required tool order:\n"
            f"1. Call 'evaluate_candidate_costs' with age={age}, feasible_offsets={feasible_offsets}, "
            f"prob_fail={risk_profile['prob_fail']}, exp_min_r={risk_profile['exp_min_r']}."
        )
        ai_msg, err = invoke_llm_resilient(
            llm=cost_llm,
            messages=[HumanMessage(content=prompt)],
            agent_name="cost_agent",
            call_num=1,
            state=state,
            primary_model_name=primary_model_name,
            fallback_model_name=fallback_model_name,
            bound_tools=cost_tools,
        )
        if err:
            trigger_coordinator_fallback(state, err)
            return {
                "fallback": True,
                "fallback_reason": err,
                "prognosis": state.get("prognosis"),
                "feasible_windows": state.get("feasible_windows"),
                "candidate_windows": state.get("candidate_windows"),
                "part_arrival_cycle": state.get("part_arrival_cycle"),
                "cost_table": state.get("cost_table"),
                "best_cost_rate": state.get("best_cost_rate"),
                "proposed_cycle": state.get("proposed_cycle"),
                "escalated": state.get("escalated", False),
                "validated": True,
                "messages": state["messages"],
                "trace": state["trace"],
            }

        state["messages"].append(ai_msg)

        cost_res = None
        executed_tools = set()
        tool_calls = deduplicate_tool_calls(ai_msg.tool_calls) if ai_msg and ai_msg.tool_calls else []

        for tc in tool_calls:
            t_name = tc.get("name")
            if t_name in tool_map:
                t_args = parse_and_validate_tool_args(tc["args"])
                res = tool_map[t_name].invoke(t_args)
                state["messages"].append(ToolMessage(content=json.dumps(res), tool_call_id=tc.get("id", f"tc_{t_name}")))
                state["trace"].append({"node": "cost_agent", "tool": t_name, "args": t_args, "result": res, "enforced_by": "agent"})
                executed_tools.add(t_name)
                if t_name == "evaluate_candidate_costs":
                    cost_res = res

        # Required-step enforcement for evaluate_candidate_costs
        if "evaluate_candidate_costs" not in executed_tools:
            corr_msg, _ = invoke_llm_resilient(
                llm=cost_llm,
                messages=state["messages"] + [HumanMessage(content="Required tool 'evaluate_candidate_costs' was not called. You MUST call evaluate_candidate_costs with feasible offsets.")],
                agent_name="cost_agent",
                call_num=2,
                state=state,
                primary_model_name=primary_model_name,
                fallback_model_name=fallback_model_name,
                bound_tools=cost_tools,
            )
            if corr_msg and corr_msg.tool_calls:
                state["messages"].append(corr_msg)
                for tc in deduplicate_tool_calls(corr_msg.tool_calls):
                    t_name = tc.get("name")
                    if t_name == "evaluate_candidate_costs":
                        t_args = parse_and_validate_tool_args(tc["args"])
                        res = tool_map[t_name].invoke(t_args)
                        state["messages"].append(ToolMessage(content=json.dumps(res), tool_call_id=tc.get("id", "tc_cost_corr")))
                        state["trace"].append({"node": "cost_agent", "tool": t_name, "args": t_args, "result": res, "enforced_by": "agent"})
                        executed_tools.add(t_name)
                        cost_res = res

        if "evaluate_candidate_costs" not in executed_tools:
            cost_res = session.evaluate_candidate_costs.invoke({
                "age": age,
                "feasible_offsets": feasible_offsets,
                "prob_fail": risk_profile["prob_fail"],
                "exp_min_r": risk_profile["exp_min_r"],
            })
            state["trace"].append({
                "node": "cost_agent",
                "tool": "evaluate_candidate_costs",
                "args": {"age": age, "feasible_offsets": feasible_offsets},
                "result": cost_res,
                "enforced_by": "graph",
            })

        best_offset = cost_res["best_offset"]
        best_cost_rate = cost_res["best_cost_rate"]
        proposed_cycle = state["current_cycle"] + best_offset

        # Check for simulated invalid injection for test cases
        if getattr(base_llm, "inject_invalid_cycles", 0) > 0:
            if base_llm._invalid_injected_count < base_llm.inject_invalid_cycles:
                base_llm._invalid_injected_count += 1
                proposed_cycle = 999  # Injected invalid cycle outside feasible set

        return {
            "cost_table": cost_res["candidate_costs"],
            "proposed_cycle": proposed_cycle,
            "best_cost_rate": best_cost_rate,
            "validation_rejected": False,
            "messages": state["messages"],
            "trace": state["trace"],
        }

    # -------------------------------------------------------------------------
    # Node 5: Validator (Deterministic Non-LLM Guardrail)
    # -------------------------------------------------------------------------
    def validator_node(state: AgenticState) -> Dict[str, Any]:
        """
        Deterministic non-LLM validation guardrail.
        Verifies:
        1. All prerequisites exist:
           - prognosis must exist
           - feasible_windows must exist (list_technician_windows must have run)
        2. If escalated:
           - feasible_windows must be computed and empty ([])
           - proposed_cycle must equal find_earliest_available(part_arrival, D)
        3. If not escalated:
           - risk_profile and cost_table must exist
           - proposed_cycle in feasible set
           - proposed_cycle in cost table
           - resources available (part_ready and tech_ready)
        If valid: sets validated=True.
        If invalid: returns to supervisor with rejection reason (max 2 retries).
        If max retries exceeded: falls back to deterministic CoordinatorAgent decision (fallback=True).
        """
        prop_cycle = state.get("proposed_cycle")
        cur_c = state["current_cycle"]
        is_escalated = bool(state.get("escalated"))

        has_prognosis = state.get("prognosis") is not None
        has_windows = state.get("feasible_windows") is not None

        is_valid = False
        rejection_reasons: List[str] = []

        if not has_prognosis:
            rejection_reasons.append("Missing prognosis prerequisite.")
        if not has_windows:
            rejection_reasons.append("Missing feasible_windows prerequisite: list_technician_windows has not run.")

        if has_prognosis and has_windows:
            if is_escalated:
                inputs = state.get("inputs", {})
                lead_time = inputs.get("lead_time", session.spare_lead_time)
                D = session.preventive_duration
                part_arrival = state.get("part_arrival_cycle", cur_c + lead_time)
                expected_escalation_cycle = session.resource.find_earliest_available(part_arrival, D)

                feas_empty = (len(state.get("feasible_windows", [])) == 0)
                cycle_matches = (prop_cycle == expected_escalation_cycle)

                if not feas_empty:
                    rejection_reasons.append("Escalation claimed but feasible windows exist before predicted RUL.")
                if not cycle_matches:
                    rejection_reasons.append(f"Escalation proposed cycle {prop_cycle} != earliest available window {expected_escalation_cycle}.")

                is_valid = bool(feas_empty and cycle_matches)
            else:
                has_risk = state.get("risk_profile") is not None
                has_cost = bool(state.get("cost_table"))

                if not has_risk:
                    rejection_reasons.append("Missing risk_profile prerequisite.")
                if not has_cost:
                    rejection_reasons.append("Missing cost_table prerequisite.")

                if has_risk and has_cost and prop_cycle is not None:
                    prop_offset = prop_cycle - cur_c
                    feasible_offsets = state.get("feasible_windows", [])
                    cost_table = state.get("cost_table", [])
                    cand_windows = state.get("candidate_windows", [])

                    is_in_feasible = prop_offset in feasible_offsets
                    is_in_cost_table = any(c.get("offset_s") == prop_offset for c in cost_table)
                    matching_windows = [w for w in cand_windows if w.get("offset_s") == prop_offset]
                    resources_available = bool(matching_windows and matching_windows[0].get("part_ready") and matching_windows[0].get("tech_ready"))

                    if not is_in_feasible:
                        rejection_reasons.append(f"Offset +{prop_offset} (cycle {prop_cycle}) is outside feasible set {feasible_offsets}.")
                    if not is_in_cost_table:
                        rejection_reasons.append("Proposed cycle does not exist in renewal cost table.")
                    if not resources_available:
                        rejection_reasons.append("Proposed window violates technician capacity or spare part arrival constraints.")

                    is_valid = bool(is_in_feasible and is_in_cost_table and resources_available)
                else:
                    is_valid = False

        if is_valid:
            trace_entry = {
                "node": "validator",
                "validation": "PASSED",
                "proposed_cycle": prop_cycle,
                "escalated": is_escalated,
            }
            state["trace"].append(trace_entry)
            return {
                "validated": True,
                "validation_rejected": False,
                "fallback": False,
                "validation_status": "PASSED",
                "trace": state["trace"],
            }
        else:
            err_msg = " | ".join(rejection_reasons)
            current_retries = state.get("retries", 0) + 1

            if current_retries <= 2:
                trace_entry = {
                    "node": "validator",
                    "validation": "REJECTED",
                    "reason": err_msg,
                    "retry_attempt": current_retries,
                    "action": "return_to_supervisor_for_retry",
                }
                state["trace"].append(trace_entry)
                return {
                    "validated": False,
                    "validation_rejected": True,
                    "retries": current_retries,
                    "validation_status": f"REJECTED: {err_msg}",
                    "trace": state["trace"],
                }
            else:
                trigger_coordinator_fallback(state, f"Max retries (2) exceeded. Rejection: {err_msg}")
                trace_entry = {
                    "node": "validator",
                    "validation": "FALLBACK_TRIGGERED",
                    "reason": f"Max retries (2) exceeded. Rejection: {err_msg}",
                    "fallback_cycle": state["proposed_cycle"],
                    "fallback": True,
                }
                state["trace"].append(trace_entry)
                return {
                    "validated": True,
                    "validation_rejected": False,
                    "fallback": True,
                    "proposed_cycle": state["proposed_cycle"],
                    "retries": current_retries,
                    "validation_status": f"FALLBACK_COMMITTED: Cycle {state['proposed_cycle']}",
                    "trace": state["trace"],
                }

    # -------------------------------------------------------------------------
    # Node 6: Escalation (Deterministic Fallback when No Feasible Window Exists)
    # -------------------------------------------------------------------------
    def escalation_node(state: AgenticState) -> Dict[str, Any]:
        """
        Escalation procedure matching CoordinatorAgent.plan_fleet when feasible_windows is empty.
        Secures earliest available technician cycle after spare arrival.
        """
        cur_c = state["current_cycle"]
        inputs = state.get("inputs", {})
        lead_time = inputs.get("lead_time", session.spare_lead_time)
        D = session.preventive_duration
        earliest_part = cur_c + lead_time

        chosen_cycle = session.resource.find_earliest_available(earliest_part, D)
        p_rul = state["prognosis"]["predicted_rul"] if state.get("prognosis") else 0.0

        trace_entry = {
            "node": "escalation",
            "escalated": True,
            "earliest_part_arrival": earliest_part,
            "chosen_cycle": chosen_cycle,
            "reason": f"ESCALATION: No feasible maintenance window within predicted RUL {p_rul:.1f}. Secured earliest window at cycle {chosen_cycle}.",
        }
        state["trace"].append(trace_entry)

        return {
            "escalated": True,
            "proposed_cycle": chosen_cycle,
            "validated": False,  # Passes through validator node for formal prerequisite audit
            "trace": state["trace"],
        }

    # -------------------------------------------------------------------------
    # Node 7: Report Assembler (Numbers strictly from tools; Rationale guardrail)
    # -------------------------------------------------------------------------
    def report_node(state: AgenticState) -> Dict[str, Any]:
        """
        Assembles final structured decision report strictly from validated tool outputs.
        The LLM receives only the final validated recommendation facts.
        Rejects and replaces with template if rationale contradicts state or hallucinates numbers.
        For escalated decisions, displays N/A for prob_fail and cost_rate.
        """
        cur_c = state["current_cycle"]
        prop_cycle = state["proposed_cycle"]
        chosen_offset = (prop_cycle - cur_c) if prop_cycle is not None else None
        p_dict = state["prognosis"]
        inputs = state.get("inputs", {})
        age = inputs.get("age", cur_c)
        is_escalated = bool(state.get("escalated"))

        if is_escalated:
            chosen_pf: Optional[float] = None
            chosen_cr: Optional[float] = None
        else:
            matching_costs = [c for c in (state.get("cost_table") or []) if c.get("offset_s") == chosen_offset]
            if matching_costs:
                chosen_pf = float(matching_costs[0]["prob_fail"])
                chosen_cr = float(matching_costs[0]["cost_rate"])
            else:
                chosen_pf = 0.0
                chosen_cr = float(state.get("best_cost_rate") or 0.0)

        # Build allowed numbers set for rationale guardrail
        allowed_nums: Set[str] = {
            str(state["engine_id"]),
            str(cur_c),
            str(age),
            str(inputs.get("lead_time", session.spare_lead_time)),
            str(prop_cycle),
            str(chosen_offset),
            str(session.preventive_duration),
            "0", "1", "2",
        }
        for v in [p_dict["predicted_rul"], p_dict["lower_bound"], p_dict["upper_bound"]]:
            allowed_nums.add(str(v))
            allowed_nums.add(f"{v:.1f}")
            allowed_nums.add(f"{v:.2f}")
            allowed_nums.add(str(int(np.floor(v))))
            allowed_nums.add(str(int(np.ceil(v))))

        if chosen_pf is not None:
            allowed_nums.add(str(chosen_pf))
            allowed_nums.add(f"{chosen_pf:.1f}")
            allowed_nums.add(f"{chosen_pf:.2f}")
            allowed_nums.add(f"{chosen_pf:.3f}")
            allowed_nums.add(f"{chosen_pf*100:.1f}")
            allowed_nums.add(f"{chosen_pf*100:.0f}")

        if chosen_cr is not None:
            allowed_nums.add(str(chosen_cr))
            allowed_nums.add(f"{chosen_cr:.1f}")
            allowed_nums.add(f"{chosen_cr:.2f}")
            allowed_nums.add(str(int(chosen_cr)))

        for w in (state.get("candidate_windows") or []):
            allowed_nums.add(str(w.get("start_cycle", "")))
            allowed_nums.add(str(w.get("offset_s", "")))
            allowed_nums.add(str(w.get("part_arrival", "")))

        # Formulate prompt for LLM plain-English reason using only validated recommendation facts
        template_reason = ""
        if is_escalated:
            earliest_part = cur_c + inputs.get("lead_time", session.spare_lead_time)
            template_reason = (
                f"ESCALATION: No feasible maintenance window within predicted RUL {p_dict['predicted_rul']:.1f}. "
                f"Secured earliest window at cycle {prop_cycle} (spare part arrives cycle {earliest_part})."
            )
            prompt = (
                f"SYSTEM: You are the Report Agent in a predictive maintenance system.\n"
                f"The maintenance decision has been validated and committed.\n"
                f"Formulate a single-sentence plain-English maintenance escalation rationale.\n"
                f"Validated facts only:\n"
                f"- engine_id: {state['engine_id']}\n"
                f"- current_cycle: {cur_c}\n"
                f"- predicted_rul: {p_dict['predicted_rul']:.1f}\n"
                f"- chosen_cycle: {prop_cycle}\n"
                f"- spare_arrival: {earliest_part}\n"
                f"- status: ESCALATION (no feasible technician slots prior to predicted RUL).\n"
                f"Output only the plain-English rationale sentence. Do not invent unlisted numbers."
            )
        else:
            template_reason = (
                f"Selected cycle {prop_cycle} (offset +{chosen_offset} cycles) with lowest expected cost rate "
                f"${chosen_cr:.2f}/cycle (prob_fail={chosen_pf*100:.1f}%, age={age})."
            )
            prompt = (
                f"SYSTEM: You are the Report Agent in a predictive maintenance system.\n"
                f"The maintenance decision has been validated and committed.\n"
                f"Formulate a single-sentence plain-English maintenance scheduling rationale.\n"
                f"Validated facts only:\n"
                f"- engine_id: {state['engine_id']}\n"
                f"- current_cycle: {cur_c}\n"
                f"- age: {age}\n"
                f"- predicted_rul: {p_dict['predicted_rul']:.1f}\n"
                f"- chosen_cycle: {prop_cycle} (offset +{chosen_offset} cycles)\n"
                f"- cost_rate: ${chosen_cr:.2f}/cycle\n"
                f"- prob_fail: {chosen_pf*100:.1f}%\n"
                f"- status: SCHEDULED (cost-optimal feasible window).\n"
                f"Output only the plain-English rationale sentence. Do not invent unlisted numbers."
            )

        ai_msg, err = invoke_llm_resilient(
            llm=base_llm,
            messages=[HumanMessage(content=prompt)],
            agent_name="report",
            call_num=1,
            state=state,
            primary_model_name=primary_model_name,
            fallback_model_name=fallback_model_name,
        )

        rationale_text = ai_msg.content.strip() if ai_msg and ai_msg.content else ""
        if not rationale_text or err:
            reason = template_reason
            rationale_source = "template"
        else:
            # Audit rationale for state contradictions and hallucinated numbers
            is_valid_rat, rej_reason = validate_rationale_guardrail(
                rationale=rationale_text,
                allowed_numbers=allowed_nums,
                escalated=is_escalated,
            )
            if not is_valid_rat:
                reason = template_reason
                rationale_source = "template"
            else:
                reason = rationale_text
                rationale_source = "llm"

        final_rec = {
            "engine_id": int(state["engine_id"]),
            "current_cycle": int(cur_c),
            "predicted_rul": float(p_dict["predicted_rul"]),
            "lower_bound": float(p_dict["lower_bound"]),
            "upper_bound": float(p_dict["upper_bound"]),
            "chosen_cycle": int(prop_cycle),
            "chosen_offset": int(chosen_offset) if chosen_offset is not None else None,
            "prob_fail": chosen_pf,
            "cost_rate": chosen_cr,
            "escalated": is_escalated,
            "fallback": bool(state.get("fallback", False)),
            "fallback_reason": state.get("fallback_reason"),
            "retries": int(state.get("retries", 0)),
            "reason": reason,
            "rationale_source": rationale_source,
        }

        trace_entry = {
            "node": "report",
            "final_recommendation": final_rec,
        }
        state["trace"].append(trace_entry)

        return {
            "final_recommendation": final_rec,
            "rationale_source": rationale_source,
            "trace": state["trace"],
        }

    # -------------------------------------------------------------------------
    # Conditional Routing Logic
    # -------------------------------------------------------------------------
    def route_supervisor(state: AgenticState) -> str:
        """
        Supervisor routing logic based on current shared state.
        Under Groq: if supervisor LLM made a valid choice, follow it.
        Under Fake / Fallback: deterministic closed-form routing.
        """
        if state.get("fallback"):
            if not state.get("reservations_done"):
                return "resource_agent"
            return "report"

        # 1. Need RUL prediction
        if not state.get("prognosis"):
            return "prognostic_agent"

        # 2. Need resource feasibility windows (must distinguish None from [])
        if state.get("feasible_windows") is None:
            return "resource_agent"

        # 3. Escalation allowed ONLY when feasible_windows has run and returned empty list []
        if state.get("feasible_windows") == []:
            if not state.get("escalated"):
                return "escalation"
            if not state.get("validated"):
                return "validator"
            if not state.get("reservations_done"):
                return "resource_agent"
            return "report"

        # 4. Need empirical risk profile for feasible windows
        if state.get("risk_profile") is None:
            return "prognostic_agent"

        # 5. Check validation retry status: if rejected, re-run cost_agent
        if state.get("validation_rejected"):
            return "cost_agent"

        # 6. Need cost table evaluation
        if not state.get("cost_table"):
            return "cost_agent"

        # 7. Need validation check
        if not state.get("validated"):
            return "validator"

        # 8. Post-validation: commit reservations in resource_agent
        if state.get("validated") and not state.get("reservations_done"):
            return "resource_agent"

        # 9. Complete: route to report
        return "report"

    def conditional_router(state: AgenticState) -> str:
        choice = state.get("supervisor_next_node")
        if choice in ["prognostic_agent", "resource_agent", "cost_agent", "validator", "escalation", "report"]:
            # Guard against invalid transitions
            if choice == "escalation" and (state.get("feasible_windows") is None or len(state.get("feasible_windows", [])) > 0):
                return route_supervisor(state)
            if choice == "report" and not (state.get("validated") and state.get("reservations_done")):
                return route_supervisor(state)
            if choice == "validator" and state.get("proposed_cycle") is None:
                return route_supervisor(state)
            return choice
        return route_supervisor(state)

    # -------------------------------------------------------------------------
    # Assemble & Compile StateGraph
    # -------------------------------------------------------------------------
    workflow = StateGraph(AgenticState)

    # Register Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("prognostic_agent", prognostic_agent_node)
    workflow.add_node("resource_agent", resource_agent_node)
    workflow.add_node("cost_agent", cost_agent_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("escalation", escalation_node)
    workflow.add_node("report", report_node)

    # Edge from START to supervisor
    workflow.add_edge(START, "supervisor")

    # Conditional Routing from Supervisor
    workflow.add_conditional_edges(
        "supervisor",
        conditional_router,
        {
            "prognostic_agent": "prognostic_agent",
            "resource_agent": "resource_agent",
            "cost_agent": "cost_agent",
            "validator": "validator",
            "escalation": "escalation",
            "report": "report",
        },
    )

    # Return edges to supervisor
    workflow.add_edge("prognostic_agent", "supervisor")
    workflow.add_edge("resource_agent", "supervisor")
    workflow.add_edge("cost_agent", "supervisor")
    workflow.add_edge("validator", "supervisor")
    workflow.add_edge("escalation", "supervisor")
    workflow.add_edge("report", END)

    app = workflow.compile()
    return app


# =============================================================================
# 6. CLI RUNNER & ENTRYPOINT
# =============================================================================

def run_agentic_graph(
    engine_id: int = 17,
    current_cycle: int = 0,
    age: int = 120,
    lead_time: int = 8,
    technician_cycles: Optional[List[int]] = None,
    llm_type: Optional[str] = None,
    is_fixture: bool = False,
    fixture_rul: float = 28.0,
    fixture_low: float = 21.0,
    fixture_up: float = 35.0,
    inject_invalid_cycles: int = 0,
    simulate_429_count: int = 0,
    simulate_malformed: bool = False,
    simulate_skip_tech_windows: bool = False,
    simulate_contradictory_rationale: bool = False,
) -> Dict[str, Any]:
    """
    Execute the LangGraph multi-agent maintenance decision pipeline for a target engine.
    """
    if technician_cycles is None:
        technician_cycles = [16, 22]

    # Resolve LLM type default: groq if credentials set, else fake with notice
    has_credentials = bool(os.environ.get("GROQ_API_KEY", "").strip() and os.environ.get("GROQ_MODEL", "").strip())
    if llm_type is None:
        if has_credentials:
            selected_llm = "groq"
        else:
            print("[NOTICE] GROQ_API_KEY or GROQ_MODEL is not set. Defaulting to deterministic FakeChatModel.")
            selected_llm = "fake"
    elif llm_type == "groq" and not has_credentials:
        print("[NOTICE] GROQ_API_KEY or GROQ_MODEL is empty or missing. Falling back to deterministic FakeChatModel.")
        selected_llm = "fake"
    else:
        selected_llm = llm_type

    session = AgentSession(spare_lead_time=lead_time, preventive_duration=2, horizon_W=150)
    if is_fixture:
        session.set_fixture(engine_id, current_cycle, fixture_rul, fixture_low, fixture_up)

    app = build_predictive_maintenance_graph(
        session=session,
        llm_type=selected_llm,
        inject_invalid_cycles=inject_invalid_cycles,
        simulate_429_count=simulate_429_count,
        simulate_malformed=simulate_malformed,
        simulate_skip_tech_windows=simulate_skip_tech_windows,
        simulate_contradictory_rationale=simulate_contradictory_rationale,
    )

    initial_state: AgenticState = {
        "engine_id": int(engine_id),
        "current_cycle": int(current_cycle),
        "inputs": {
            "lead_time": int(lead_time),
            "technician_cycles": technician_cycles,
            "cost_params": {
                "c_pm": 2000.0,
                "c_fail": 10000.0,
                "c_down": 100.0,
                "c_part": 300.0,
            },
            "age": int(age),
        },
        "prognosis": None,
        "feasible_windows": None,
        "candidate_windows": None,
        "part_arrival_cycle": None,
        "risk_profile": None,
        "cost_table": None,
        "proposed_cycle": None,
        "best_cost_rate": None,
        "escalated": False,
        "retries": 0,
        "messages": [],
        "trace": [],
        "validated": False,
        "fallback": False,
        "fallback_reason": None,
        "validation_rejected": False,
        "validation_status": None,
        "reservations_done": False,
        "supervisor_next_node": None,
        "rationale_source": None,
        "start_time": time.time(),
        "final_recommendation": None,
    }

    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph Multi-Agent Predictive Maintenance Orchestrator")
    parser.add_argument("--llm", type=str, choices=["groq", "fake"], default=None, help="LLM backend: groq or fake (default: groq if key and model set, else fake)")
    parser.add_argument("--engine", type=int, default=17, help="Engine unit number")
    parser.add_argument("--cycle", type=int, default=0, help="Current operational flight cycle")
    parser.add_argument("--lead-time", type=int, default=8, help="Spare part delivery lead time")
    parser.add_argument("--technicians", type=str, default="16,22", help="Comma-separated technician availability cycles")
    parser.add_argument("--draw", action="store_true", help="Draw Mermaid graph to outputs/agent_graph.md")

    args = parser.parse_args()

    tech_cycles = [int(c.strip()) for c in args.technicians.split(",") if c.strip()]
    is_fix = (args.engine == 17 and args.cycle == 0)

    res_state = run_agentic_graph(
        engine_id=args.engine,
        current_cycle=args.cycle,
        age=120 if is_fix else args.cycle,
        lead_time=args.lead_time,
        technician_cycles=tech_cycles,
        llm_type=args.llm,
        is_fixture=is_fix,
    )

    rec = res_state["final_recommendation"]
    print("=" * 80)
    print("LANGGRAPH MULTI-AGENT MAINTENANCE DECISION REPORT")
    print("=" * 80)
    print(f"Engine Unit Number   : E-{rec['engine_id']:03d}")
    print(f"Current Flight Cycle : Cycle {rec['current_cycle']}")
    print(f"Predicted RUL        : {rec['predicted_rul']:.2f} cycles (90% Interval: [{rec['lower_bound']:.2f}, {rec['upper_bound']:.2f}])")
    print(f"Chosen Start Cycle   : Cycle {rec['chosen_cycle']} (Offset +{rec['chosen_offset']} cycles)")
    
    pf_display = "N/A" if rec.get("escalated") or rec.get("prob_fail") is None else f"{rec['prob_fail']*100:.2f}%"
    cr_display = "N/A" if rec.get("escalated") or rec.get("cost_rate") is None else f"${rec['cost_rate']:.2f}/cycle"
    print(f"Failure Probability  : {pf_display}")
    print(f"Renewal Cost Rate    : {cr_display}")
    print(f"Escalation Triggered : {rec['escalated']}")
    print(f"Fallback Triggered   : {rec['fallback']}")
    if rec.get("fallback_reason"):
        print(f"Fallback Reason      : {rec['fallback_reason']}")
    print(f"Rationale Source     : {rec.get('rationale_source', 'N/A')}")
    print(f"Operational Rationale: \"{rec['reason']}\"")
    print("=" * 80)

    if args.draw:
        sess = AgentSession()
        g_app = build_predictive_maintenance_graph(session=sess, llm_type="fake")
        mermaid_code = g_app.get_graph().draw_mermaid()
        os.makedirs(PROJECT_ROOT / "outputs", exist_ok=True)
        out_path = PROJECT_ROOT / "outputs" / "agent_graph.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# LangGraph Multi-Agent Predictive Maintenance Graph\n\n```mermaid\n" + mermaid_code + "\n```\n")
        print(f"Saved Mermaid architecture graph to {out_path}")
