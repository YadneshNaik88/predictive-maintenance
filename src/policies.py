"""
policies.py

Maintenance Decision Policies for Turbofan Fleet Simulation.
Phase 5: Baseline Policies & Diagnostic Oracle Benchmark.
- Run-to-Failure (Corrective Maintenance baseline)
- Fixed Interval (Time-based preventive maintenance)
- Naive RUL Threshold (Prognostic point-estimate trigger)
- Diagnostic Oracle (Theoretical performance upper bound using ground truth)

Designed with a common plug-in interface for Phase 6 multi-agent scheduling.
"""

from typing import Any, Dict, Optional


class BasePolicy:
    """
    Common abstract interface for fleet maintenance decision policies.
    Evaluated per slot per operational cycle.
    """

    def __init__(self, name: str = "BasePolicy") -> None:
        self.name = name

    def reset(self) -> None:
        """Reset internal policy state before starting a simulation run."""
        pass

    def on_renewal(self, slot_id: int) -> None:
        """Callback invoked when a fresh engine trajectory is installed into slot_id."""
        pass

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
        Decide whether to place a maintenance order for this slot at current_cycle.

        Parameters
        ----------
        slot_id : int
            Fleet slot index (0 .. num_slots - 1).
        current_cycle : int
            Global simulation cycle (1 .. H).
        operating_cycles : int
            Number of flight cycles this engine has operated since last renewal.
        pred_rul : float
            Current model predicted RUL point estimate in cycles.
        lower_rul : float
            Lower bound of calibrated 90% prediction interval.
        upper_rul : float
            Upper bound of calibrated 90% prediction interval.
        true_rul : float
            Ground-truth remaining useful life (accessible ONLY by Oracle).
        unit_number : int
            Underlying C-MAPSS engine unit identifier.
        engine_cycle : int
            Current trajectory cycle within the engine run (1 .. max_cycle).
        order_already_placed : bool
            True if a maintenance spare part has already been ordered for this engine.
        fleet_context : dict, optional
            Global fleet state (available technicians, queued jobs, etc.) for agentic policies.

        Returns
        -------
        bool
            True if a maintenance order should be placed at this cycle; False otherwise.
        """
        raise NotImplementedError("Subclasses must implement decide().")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class RunToFailurePolicy(BasePolicy):
    """
    Policy 1: Run-to-Failure (Corrective Maintenance Baseline).

    Never orders preventive maintenance. Engines operate until in-service
    functional failure occurs, at which point emergency repair is triggered.
    """

    def __init__(self) -> None:
        super().__init__(name="Run-to-Failure")

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
        # Never triggers preventive orders; failure orders are handled by the simulator
        return False


class FixedIntervalPolicy(BasePolicy):
    """
    Policy 2: Fixed Interval Maintenance (Time-Based Preventive Policy).

    Schedules preventive maintenance every T operating cycles since the last renewal.
    Orders replacement spare parts `lead_time` cycles in advance of T so that
    the part arrives exactly at cycle T (planned in advance).
    """

    def __init__(self, T: int, lead_time: int = 8) -> None:
        super().__init__(name=f"Fixed-Interval(T={T})")
        self.T = int(T)
        self.lead_time = int(lead_time)

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
        if order_already_placed:
            return False

        # Place order lead_time cycles before scheduled cycle T
        # (operating_cycles is the count of completed cycles so far)
        order_threshold = max(1, self.T - self.lead_time)
        return operating_cycles >= order_threshold

    def __repr__(self) -> str:
        return f"FixedIntervalPolicy(T={self.T}, lead_time={self.lead_time})"


class NaiveThresholdPolicy(BasePolicy):
    """
    Policy 3: Naive RUL Threshold (Prognostic Point-Estimate Trigger).

    Places a maintenance order when the model predicted RUL point estimate
    falls below or equal to threshold tau. Does NOT account for spare-part lead time.
    """

    def __init__(self, tau: float) -> None:
        tau_val = int(tau) if float(tau).is_integer() else float(tau)
        super().__init__(name=f"Naive-Threshold(tau={tau_val})")
        self.tau = float(tau)

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
        if order_already_placed:
            return False

        # Trigger order as soon as predicted RUL <= tau
        return pred_rul <= self.tau

    def __repr__(self) -> str:
        return f"NaiveThresholdPolicy(tau={self.tau})"


class DiagnosticOraclePolicy(BasePolicy):
    """
    Diagnostic Oracle Policy (Theoretical Performance Upper Bound).

    Uses true ground-truth RUL to trigger spare ordering exactly when
    true_rul <= lead_time + margin.
    NOTE: This is NOT a deployable project policy, but a theoretical diagnostic
    reference upper bound to benchmark the limits of prognostic scheduling.
    """

    def __init__(self, lead_time: int = 8, margin: int = 1) -> None:
        super().__init__(name=f"Diagnostic-Oracle(margin={margin})")
        self.lead_time = int(lead_time)
        self.margin = int(margin)

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
        if order_already_placed:
            return False

        # Order spare when true RUL drops to lead_time + margin
        return true_rul <= (self.lead_time + self.margin)

    def __repr__(self) -> str:
        return f"DiagnosticOraclePolicy(lead_time={self.lead_time}, margin={self.margin})"

