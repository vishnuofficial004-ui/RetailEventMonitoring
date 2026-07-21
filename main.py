from fastapi import FastAPI, Request
from enum import IntEnum
from dataclasses import dataclass
import threading
import uuid
import asyncio
import cv2
import time
import os
import uvicorn
import platform
import multiprocessing
import itertools


# =========================================================
# Severity Levels
# =========================================================
# IntEnum so severities are naturally orderable/comparable
# (e.g. CRITICAL > HIGH), which we'll use later for routing
# and priority-queue ordering.

class Severity(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# =========================================================
# App Initialization
# =========================================================

app = FastAPI()

RECORDINGS_DIR = "recordings"
DEFAULT_RECORD_DURATION = 25  # seconds
DEFAULT_FPS = 20
FRAME_SIZE = (640, 480)

os.makedirs(RECORDINGS_DIR, exist_ok=True)

# =========================================================
# Camera Configuration
# =========================================================

CAMERA_SOURCES = {
    "pc_cam": [0],
    "mobile_cam": ["http://192.168.1.5:8080/video"]
}

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
# action but carry different severities.

ALERT_SEVERITY_MAP = {
    "identity_mismatch": Severity.MEDIUM,
    "missing_from_workstation": Severity.MEDIUM,

    "loitering": Severity.LOW,
    "stealing": Severity.CRITICAL,

    "unauthorized_access": Severity.HIGH
}

DEFAULT_SEVERITY = Severity.LOW

# =========================================================
# Utility Functions
# =========================================================

def generate_event_id() -> str:
    return f"evt_{uuid.uuid4()}"

def get_camera_sources(camera_id: str):
    return CAMERA_SOURCES.get(camera_id, [])

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

# =========================================================
# Trigger / Alert Logic
# =========================================================

async def trigger_alert(event_id: str, alert_type: str):
    print(f"[ALERT] Event={event_id} | Type={alert_type}")

# =========================================================
# Video Recording Logic
# =========================================================

async def record_live_video(event_id, alert_type, camera_id, duration=DEFAULT_RECORD_DURATION):

    print(f"[VIDEO] Event={event_id} | Type={alert_type} | Recording started")

    caps = []
    writers = []

    sources = get_camera_sources(camera_id)

    for idx, source in enumerate(sources):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera: {source}")
            continue

        caps.append(cap)

        output_path = f"{RECORDINGS_DIR}/{event_id}_{camera_id}_{idx}.mp4"
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            DEFAULT_FPS,
            FRAME_SIZE
        )
        writers.append(writer)

    start_time = time.time()

    while time.time() - start_time < duration:
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, FRAME_SIZE)
                writers[i].write(frame)
                cv2.imshow(f"Recording-{event_id}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap, writer in zip(caps, writers):
        cap.release()
        writer.release()

    cv2.destroyAllWindows()

    print(f"[VIDEO] Event={event_id} | Type={alert_type} | Recording completed")

# =========================================================
# Beep Sound
# =========================================================

def play_beep():
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 800)
        else:
            print("\a", end="", flush=True)
    except Exception as e:
        print(f"[WARN] Beep failed: {e}")

# =========================================================
# Multiprocessing Camera Worker
# =========================================================

def camera_process_worker(event_id, alert_type, camera_id, duration):
    import asyncio
    asyncio.run(record_live_video(event_id, alert_type, camera_id, duration))

def trigger_camera_process(event_id, alert_type, camera_id, duration=15):
    process = multiprocessing.Process(
        target=camera_process_worker,
        args=(event_id, alert_type, camera_id, duration),
        daemon=True
    )
    process.start()
    print(f"[PROCESS] Event={event_id} | Camera={camera_id} | PID={process.pid} started")

# =========================================================
# Routing Logic (severity + type → decision)
# =========================================================
# Separated from dispatch_alert_action on purpose: routing is
# a pure decision (given inputs, what SHOULD happen) with no
# side effects, while dispatch is what actually DOES it (beeps,
# spawns processes, etc). Keeping the pure function separate
# makes it trivially unit-testable without mocking cv2/threads.

@dataclass(frozen=True)
class RoutingDecision:
    alert_type: str
    action: str
    severity: Severity
    immediate: bool  # HIGH/CRITICAL bypass the priority queue — wired in step 8

def route_event(alert_type: str, severity: Severity) -> RoutingDecision:
    action = ALERT_ACTION_MAP.get(alert_type, "NONE")
    immediate = severity >= Severity.HIGH

    return RoutingDecision(
        alert_type=alert_type,
        action=action,
        severity=severity,
        immediate=immediate
    )

# =========================================================
# Dispatch Priority Queue
# =========================================================
# asyncio.PriorityQueue is a min-heap: the LOWEST value comes
# out first. Severity is stored as a NEGATIVE int so CRITICAL
# (4) becomes -4 and is dequeued before LOW (1) → -1.
#
# Queue items are plain tuples: (priority, seq, event_id,
# decision, camera_id). `seq` (a monotonically increasing
# counter) exists purely as a tiebreaker — if two items have
# equal priority, PriorityQueue compares the NEXT tuple element
# to order them, and RoutingDecision isn't orderable. Without
# `seq`, two same-priority items would raise a TypeError trying
# to compare dataclasses. `seq` also happens to preserve FIFO
# order among equal-priority items, which is a nice side effect.
#
# No consumer reads from this queue yet — that's step 7. This
# step only establishes the structure and how items get onto it.

DISPATCH_QUEUE: asyncio.PriorityQueue = asyncio.PriorityQueue()
_dispatch_seq = itertools.count()

async def enqueue_dispatch(event_id: str, decision: RoutingDecision, camera_id: str):
    priority = -int(decision.severity)
    seq = next(_dispatch_seq)
    await DISPATCH_QUEUE.put((priority, seq, event_id, decision, camera_id))
    print(f"[QUEUE] Event={event_id} | Type={decision.alert_type} | "
          f"Severity={decision.severity.name} | priority={priority} | "
          f"qsize={DISPATCH_QUEUE.qsize()}")

# =========================================================
# Dispatcher Logic  ✅ UPDATED HERE
# =========================================================

async def dispatch_alert_action(event_id: str, alert_type: str, camera_id: str):
    action = ALERT_ACTION_MAP.get(alert_type)

    def play_beep_in_thread():
        threading.Thread(target=play_beep, daemon=True).start()

    if action == "TRIGGER_AND_VIDEO":
        play_beep_in_thread()
        asyncio.create_task(trigger_alert(event_id, alert_type))

        # ✅ UPDATED: Now using multiprocessing camera process
        trigger_camera_process(event_id, alert_type, camera_id)

        print("[DISPATCH] Executed TRIGGER + VIDEO (Parallel Process)")

    elif action == "TRIGGER_ONLY":
        asyncio.create_task(trigger_alert(event_id, alert_type))
        print("[DISPATCH] Executed TRIGGER ONLY")

    elif action == "VIDEO_ONLY":
        # ✅ UPDATED: VIDEO_ONLY also uses multiprocessing
        trigger_camera_process(event_id, alert_type, camera_id)
        print("[DISPATCH] Executed VIDEO ONLY (Parallel Process)")

    else:
        print(f"[INFO] No action mapped for alert: {alert_type}")

# =========================================================
# Background Dispatch Worker
# =========================================================
# Consumes DISPATCH_QUEUE and calls dispatch_alert_action for
# each item, highest priority first. Runs as a single
# long-lived asyncio task started at app startup, not per-
# request — one consumer draining a shared queue.
#
# Not fed by anything yet: enqueue_dispatch() is written but
# receive_event() still calls dispatch_alert_action directly
# (unchanged since before step 6). Wiring the endpoint to
# actually go through this queue is steps 8-9, so HIGH/CRITICAL
# vs LOW/MEDIUM can be routed differently rather than all-or-
# nothing.

async def dispatch_worker():
    print("[WORKER] Dispatch worker started")
    while True:
        priority, seq, event_id, decision, camera_id = await DISPATCH_QUEUE.get()
        try:
            print(f"[WORKER] Dequeued Event={event_id} | Type={decision.alert_type} | "
                  f"Severity={decision.severity.name} | qsize={DISPATCH_QUEUE.qsize()}")
            await dispatch_alert_action(event_id, decision.alert_type, camera_id)
            record_dispatch_completion(event_id)
        except Exception as e:
            # A single bad dispatch must not kill the worker loop —
            # every subsequent queued event would silently stop
            # being processed if this exception propagated.
            print(f"[WORKER][ERROR] Event={event_id} failed: {e}")
        finally:
            DISPATCH_QUEUE.task_done()

@app.on_event("startup")
async def start_background_workers():
    asyncio.create_task(dispatch_worker())

# =========================================================
# REST API Endpoint
# =========================================================

# Records receipt time (monotonic clock) per event_id so later
# steps can compute end-to-end handling latency (dispatch
# completion time − receipt time). A plain dict is fine for a
# single-process prototype; a real deployment would use a
# TTL'd store so entries don't leak memory if dispatch never
# completes for some event.
EVENT_RECEIPT_TIMES = {}

def record_dispatch_completion(event_id: str):
    """
    Capture dispatch-completion time and compute end-to-end
    handling latency against the receipt time stored in
    EVENT_RECEIPT_TIMES (step 4).

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
    which is out of scope for this step and left as a named gap.
    """
    completion_time = time.monotonic()
    receipt_time = EVENT_RECEIPT_TIMES.get(event_id)

    if receipt_time is None:
        print(f"[LATENCY][WARN] No receipt time found for Event={event_id}")
        return None

    latency_ms = (completion_time - receipt_time) * 1000
    print(f"[LATENCY] Event={event_id} | handling_latency_ms={latency_ms:.2f}")
    return latency_ms

@app.post("/api/v1/event")
async def receive_event(request: Request):
    # Captured before body parsing so latency reflects true
    # end-to-end handling time, not just processing after
    # the JSON has already been read off the wire.
    receipt_time = time.monotonic()

    data = await request.json()

    event_id = generate_event_id()
    EVENT_RECEIPT_TIMES[event_id] = receipt_time
    print(f"[RECEIVE] Event={event_id} | Received at t={receipt_time:.6f}")

    alerts = data.get("alerts", [])
    camera_id = data.get("camera_id", "pc_cam")

    if not alerts:
        EVENT_RECEIPT_TIMES.pop(event_id, None)
        return {"status": "failed", "message": "No alerts provided"}

    # Each alert can be either a plain string ("loitering") or an
    # object with an optional client-supplied severity override
    # ({"type": "loitering", "severity": "HIGH"}). Normalize both
    # into (alert_type, resolved_severity) pairs up front so the
    # rest of the pipeline only ever deals with one shape.
    resolved_alerts = []
    decisions = []
    for alert in alerts:
        if isinstance(alert, dict):
            alert_type = alert.get("type")
            severity_override = alert.get("severity")
        else:
            alert_type = alert
            severity_override = None

        if not alert_type:
            print(f"[WARN] Skipping malformed alert entry: {alert}")
            continue

        severity = resolve_severity(alert_type, severity_override)
        decision = route_event(alert_type, severity)
        print(f"[ROUTE] Event={event_id} | Type={alert_type} | "
              f"Severity={decision.severity.name} | Action={decision.action} | "
              f"Immediate={decision.immediate}")

        resolved_alerts.append({"type": alert_type, "severity": severity.name})
        decisions.append(decision)

    for decision in decisions:
        if decision.immediate:
            # HIGH/CRITICAL severity: skip the queue entirely and
            # dispatch inline on this request's own coroutine, so
            # urgent alerts (e.g. "stealing") aren't stuck waiting
            # behind whatever the single background worker (step 7)
            # happens to be processing at that moment.
            print(f"[BYPASS] Event={event_id} | Type={decision.alert_type} | "
                  f"Severity={decision.severity.name} — immediate dispatch, skipping queue")
            await dispatch_alert_action(event_id, decision.alert_type, camera_id)
            record_dispatch_completion(event_id)
        else:
            # LOW/MEDIUM severity: don't dispatch inline. Hand off
            # to DISPATCH_QUEUE and let the background worker
            # (step 7) drain it in priority order. This is the
            # actual behavior change deferred from step 8 — routine
            # alerts no longer compete with urgent ones for this
            # request's own execution time, and multiple routine
            # alerts arriving in a burst get smoothed through one
            # consumer instead of all firing dispatch side effects
            # (beeps, camera processes) simultaneously.
            print(f"[QUEUED] Event={event_id} | Type={decision.alert_type} | "
                  f"Severity={decision.severity.name} — enqueued for background dispatch")
            await enqueue_dispatch(event_id, decision, camera_id)

    return {
        "status": "success",
        "event_id": event_id,
        "alerts_received": resolved_alerts
    }

# =========================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)