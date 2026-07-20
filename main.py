from fastapi import FastAPI, Request
from enum import IntEnum
import threading
import uuid
import asyncio
import cv2
import time
import os
import uvicorn
import platform
import multiprocessing


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
# REST API Endpoint
# =========================================================

@app.post("/api/v1/event")
async def receive_event(request: Request):
    data = await request.json()

    event_id = generate_event_id()
    alerts = data.get("alerts", [])
    camera_id = data.get("camera_id", "pc_cam")

    if not alerts:
        return {"status": "failed", "message": "No alerts provided"}

    # Each alert can be either a plain string ("loitering") or an
    # object with an optional client-supplied severity override
    # ({"type": "loitering", "severity": "HIGH"}). Normalize both
    # into (alert_type, resolved_severity) pairs up front so the
    # rest of the pipeline only ever deals with one shape.
    resolved_alerts = []
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
        resolved_alerts.append({"type": alert_type, "severity": severity.name})

    for alert in resolved_alerts:
        await dispatch_alert_action(event_id, alert["type"], camera_id)

    return {
        "status": "success",
        "event_id": event_id,
        "alerts_received": resolved_alerts
    }

# =========================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)