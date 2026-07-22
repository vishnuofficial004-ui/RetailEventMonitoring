"""
Severity levels and alert-type configuration.

This module has ZERO dependencies on FastAPI, cv2, asyncio, or
anything else in this project — it's pure data and pure
functions. That's deliberate: severity resolution is a decision,
not an action, so it should be testable by importing this file
alone, with no mocking required.
"""

from enum import IntEnum


# =========================================================
# Severity Levels
# =========================================================
# IntEnum so severities are naturally orderable/comparable
# (e.g. CRITICAL > HIGH), used by routing.py for immediate-
# dispatch decisions and by dispatch_queue.py for priority
# ordering.

class Severity(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# =========================================================
# Alert → Action Mapping
# =========================================================

ALERT_ACTION_MAP = {
    "identity_mismatch": "TRIGGER_ONLY",
    "missing_from_workstation": "TRIGGER_ONLY",

    "loitering": "VIDEO_ONLY",
    "stealing": "VIDEO_ONLY",

    "unauthorized_access": "TRIGGER_AND_VIDEO"
}

# =========================================================
# Alert → Severity Mapping
# =========================================================
# Kept as a separate map (rather than folded into
# ALERT_ACTION_MAP) so severity and action-routing can
# evolve independently — e.g. two alert types can share an
# action but carry different severities (see "stealing" vs
# "loitering": same VIDEO_ONLY action, very different severity).

ALERT_SEVERITY_MAP = {
    "identity_mismatch": Severity.MEDIUM,
    "missing_from_workstation": Severity.MEDIUM,

    "loitering": Severity.LOW,
    "stealing": Severity.CRITICAL,

    "unauthorized_access": Severity.HIGH
}

DEFAULT_SEVERITY = Severity.LOW


def resolve_severity(alert_type: str, severity_override: str = None) -> Severity:
    """
    Resolve the Severity for a given alert type.

    Priority:
    1. Client-supplied override (validated against Severity enum names)
    2. ALERT_SEVERITY_MAP lookup by alert_type
    3. DEFAULT_SEVERITY fallback for unknown alert types

    An invalid override (typo, unsupported level, wrong type) is
    logged and ignored rather than raising — a malformed override
    from an upstream camera/agent shouldn't be able to crash event
    ingestion.
    """
    if severity_override:
        try:
            return Severity[str(severity_override).upper()]
        except KeyError:
            print(f"[WARN] Invalid severity override '{severity_override}' "
                  f"for alert_type='{alert_type}', falling back to map")

    return ALERT_SEVERITY_MAP.get(alert_type, DEFAULT_SEVERITY)