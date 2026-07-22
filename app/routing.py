"""
Routing logic: combines alert type + severity into a decision.

Separated from dispatch.py on purpose: routing is a pure
decision (given inputs, what SHOULD happen) with no side
effects, while dispatch is what actually DOES it (beeps, spawns
processes, etc). Keeping the pure function separate makes it
trivially unit-testable without mocking cv2/threads/multiprocessing.
"""

from dataclasses import dataclass

from app.severity import Severity, ALERT_ACTION_MAP


@dataclass(frozen=True)
class RoutingDecision:
    alert_type: str
    action: str
    severity: Severity
    immediate: bool  # HIGH/CRITICAL bypass the priority queue


def route_event(alert_type: str, severity: Severity) -> RoutingDecision:
    action = ALERT_ACTION_MAP.get(alert_type, "NONE")
    immediate = severity >= Severity.HIGH

    return RoutingDecision(
        alert_type=alert_type,
        action=action,
        severity=severity,
        immediate=immediate
    )