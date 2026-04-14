from fastapi import FastAPI, Request
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
# Utility Functions
# =========================================================

def generate_event_id() -> str:
    return f"evt_{uuid.uuid4()}"

def get_camera_sources(camera_id: str):
    return CAMERA_SOURCES.get(camera_id, [])

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

    for alert in alerts:
        await dispatch_alert_action(event_id, alert, camera_id)

    return {
        "status": "success",
        "event_id": event_id,
        "alerts_received": alerts
    }

# =========================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

