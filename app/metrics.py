"""
Latency measurement and structured logging.

Everything observability-related lives here: the receipt-time
store, the structured JSON logger, the rolling latency window,
and the stats computation the /metrics endpoint (main.py) reads
from.
"""

import time
import json
import logging
from collections import deque

from app.severity import Severity


# =========================================================
# Structured Event Logger
# =========================================================
# Deliberately separate from the plain print() calls used
# elsewhere in this project (e.g. [ROUTE], [QUEUE], [WORKER] in
# dispatch_queue.py). Those are fine for human-readable console
# debugging, but they're not machine-parseable — you can't
# grep/aggregate them reliably (free-form f-strings, inconsistent
# field order). This logger emits one JSON object per line
# specifically for latency records, so it can later feed a log
# aggregator (ELK/Datadog/CloudWatch) or the metrics endpoint
# without string-parsing print output.
#
# Scope is intentionally narrow: only latency records go through
# this. Migrating every print() call in the project to structured
# logging is a much bigger, separate change.

event_logger = logging.getLogger("event_pipeline")
event_logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))  # message IS the JSON; no extra prefix to keep it valid JSON per line
event_logger.addHandler(_handler)
event_logger.propagate = False


def log_latency_record(event_id: str, alert_type: str, severity: Severity,
                        camera_id: str, latency_ms: float):
    record = {
        "log_type": "dispatch_latency",
        "event_id": event_id,
        "alert_type": alert_type,
        "severity": severity.name,
        "camera_id": camera_id,
        "latency_ms": round(latency_ms, 2),
        "timestamp": time.time()  # wall-clock, for correlating with
                                   # other systems' logs — NOT used
                                   # for the latency math itself,
                                   # which stays on monotonic clocks
    }
    event_logger.info(json.dumps(record))


# =========================================================
# Event Receipt Times
# =========================================================
# Records receipt time (monotonic clock) per event_id so
# record_dispatch_completion() can compute end-to-end handling
# latency (dispatch completion time − receipt time). A plain
# dict is fine for a single-process prototype; a real deployment
# would use a TTL'd store so entries don't leak memory if
# dispatch never completes for some event.
#
# KNOWN LIMITATION: entries are currently never cleaned up (see
# record_dispatch_completion docstring below) — this is a real,
# named, not-yet-fixed memory leak.

EVENT_RECEIPT_TIMES = {}


# =========================================================
# Rolling Latency Window
# =========================================================
# Fixed-size deque of the most recent dispatch latencies
# (milliseconds). maxlen=500 means once full, appending
# automatically drops the oldest value — O(1) eviction, no
# manual trimming needed. This is what the /metrics endpoint
# reads from to report avg/p95.
#
# Deliberately a SEPARATE structure from EVENT_RECEIPT_TIMES:
# that dict is per-event bookkeeping needed to compute a latency
# (and has the known leak above); this deque is just the
# resulting numbers, bounded and leak-free by construction
# because of maxlen. Fixing one doesn't fix the other — they
# solve different problems.

LATENCY_WINDOW_SIZE = 500
recent_latencies_ms = deque(maxlen=LATENCY_WINDOW_SIZE)


def get_latency_stats():
    """
    Compute avg and p95 over the current rolling window.
    Returns None if no data yet (empty deque) — callers must
    handle that, not assume there's always at least one sample.
    """
    if not recent_latencies_ms:
        return None

    samples = sorted(recent_latencies_ms)
    n = len(samples)
    avg = sum(samples) / n

    # p95 via nearest-rank method: index = int(0.95 * n), clamped
    # so a small n (e.g. n=1..20) doesn't index out of range.
    # This is an approximation, not interpolated percentile (e.g.
    # numpy's default) — fine for a rolling operational metric,
    # NOT rigorous enough to cite in a paper or SLA without
    # saying which method was used.
    p95_index = min(n - 1, max(0, int(0.95 * n)))
    p95 = samples[p95_index]

    return {"count": n, "avg_ms": round(avg, 2), "p95_ms": round(p95, 2)}


def record_dispatch_completion(event_id: str, alert_type: str, severity: Severity, camera_id: str):
    """
    Capture dispatch-completion time and compute end-to-end
    handling latency against the receipt time stored in
    EVENT_RECEIPT_TIMES.

    Deliberately uses .get(), not .pop(): a single event_id can
    carry MULTIPLE alerts (e.g. one camera frame flags both
    "loitering" and "identity_mismatch"), and each alert's
    dispatch calls this function independently. Popping on the
    first call would leave every subsequent alert for that same
    event with no receipt time to measure against.

    KNOWN LIMITATION, not yet fixed: because we never pop,
    EVENT_RECEIPT_TIMES entries are never cleaned up at all right
    now — this is a real memory leak over the process lifetime.
    Fixing it properly needs a per-event pending-dispatch counter
    (decrement per alert, delete the entry when it hits zero),
    which is a separate, scoped piece of work, not done here.
    """
    completion_time = time.monotonic()
    receipt_time = EVENT_RECEIPT_TIMES.get(event_id)

    if receipt_time is None:
        print(f"[LATENCY][WARN] No receipt time found for Event={event_id}")
        return None

    latency_ms = (completion_time - receipt_time) * 1000
    recent_latencies_ms.append(latency_ms)
    log_latency_record(event_id, alert_type, severity, camera_id, latency_ms)
    return latency_ms