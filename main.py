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
DEFAULT_RECORD_DURATION = 15  # seconds
DEFAULT_FPS = 20
FRAME_SIZE = (640, 480)

os.makedirs(RECORDINGS_DIR, exist_ok=True)

# =========================================================
# Camera Configuration (PC Camera for Now)
# =========================================================

CAMERA_SOURCES = {
    "pc_cam": [0] ,
    "mobile_cam" :["http://192.168.1.5:8080/video"]# Default laptop / PC webcam
}

# =========================================================
# Alert → Action Mapping
# =========================================================

ALERT_ACTION_MAP = {
    # Trigger only
    "identity_mismatch": "TRIGGER_ONLY",
    "missing_from_workstation": "TRIGGER_ONLY",

    # Video only
    "loitering": "VIDEO_ONLY",
    "stealing": "VIDEO_ONLY",

    # Trigger + Video
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
    """
    Displays trigger message (real-time alert simulation)
    """
    print(f"[ALERT] Event={event_id} | Type={alert_type}")

# =========================================================
# Video Recording / Reconstruction Logic
# =========================================================

async def record_live_video(
    event_id: str,
    alert_type: str,
    camera_id: str,
    duration: int = DEFAULT_RECORD_DURATION
):
    """
    Starts recording live camera feed when video reconstruction is required
    """
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

                # Optional live preview (testing only)
                cv2.imshow(f"Recording-{event_id}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap, writer in zip(caps, writers):
        cap.release()
        writer.release()

    cv2.destroyAllWindows()

    print(f"[VIDEO] Event={event_id} | Type={alert_type} | Recording completed")

# =========================================================
# Combined Trigger + Video Logic
# =========================================================

async def trigger_and_record(
    event_id: str,
    alert_type: str,
    camera_id: str
):
    """
    Immediate trigger + background video recording
    """
    await trigger_alert(event_id, alert_type)
    await record_live_video(event_id, alert_type, camera_id)

# =========================================================
# Dispatcher Logic (Corrected Simultaneous Trigger+Video)
# =========================================================

async def dispatch_alert_action(
    event_id: str,
    alert_type: str,
    camera_id: str
):
    action = ALERT_ACTION_MAP.get(alert_type)

    # Threaded beep for TRIGGER+VIDEO
    def play_beep_in_thread():
        threading.Thread(
            target=play_beep,
            daemon=True
        ).start()

    if action == "TRIGGER_AND_VIDEO":
        # 🔹 Start beep
        play_beep_in_thread()
        # 🔹 Start alert async
        asyncio.create_task(trigger_alert(event_id, alert_type))
        # 🔹 Start video recording async
        asyncio.create_task(record_live_video(event_id, alert_type, camera_id))
        print("[DISPATCH] Executed TRIGGER+VIDEO simultaneously")

    elif action == "TRIGGER_ONLY":
        asyncio.create_task(trigger_alert(event_id, alert_type))
        print("[DISPATCH] Executed TRIGGER ONLY")

    elif action == "VIDEO_ONLY":
        asyncio.create_task(record_live_video(event_id, alert_type, camera_id))
        print("[DISPATCH] Executed VIDEO ONLY")

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
        return {
            "status": "failed",
            "message": "No alerts provided"
        }

    for alert in alerts:
        await dispatch_alert_action(event_id, alert, camera_id)

    return {
        "status": "success",
        "event_id": event_id,
        "alerts_received": alerts
    }

def play_beep():
    """
    Produces a beep sound for trigger events
    """
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 5000)  # frequency, duration(ms)
        else:
            # macOS / Linux
            print("\a", end="", flush=True)
    except Exception as e:
        print(f"[WARN] Beep failed: {e}")


# =========================================================
# Multiprocessing Camera Recording Worker (NEW)
# =========================================================

def camera_process_worker(event_id: str, alert_type: str, camera_id: str, duration: int):
    """
    Worker function to be run in a separate process.
    Calls the existing record_live_video() synchronously.
    """
    import asyncio
    asyncio.run(record_live_video(event_id, alert_type, camera_id, duration))

# =========================================================
# Multiprocessing Camera Event Launcher (NEW)
# =========================================================

async def record_camera_event(event_id: str, alert_type: str, camera_id: str, duration: int = 15):
    """
    Launches a separate process to record video from a single camera.
    Non-blocking for the main event pipeline.
    """
    process = multiprocessing.Process(
        target=camera_process_worker,
        args=(event_id, alert_type, camera_id, duration),
        daemon=True  # ensures process exits if main app stops
    )
    process.start()
    print(f"[PROCESS] Event={event_id} | Camera={camera_id} | Process PID={process.pid} started")

# =========================================================
# Start Multiple Cameras Concurrently (NEW)
# =========================================================

async def start_cameras_concurrently(event_id: str, alert_type: str, camera_ids: list, duration: int = 15):
    """
    Starts recording for multiple cameras concurrently.
    Each camera runs in a separate process.
    """
    tasks = []
    for cam_id in camera_ids:
        # Create an async task for each camera
        tasks.append(asyncio.create_task(record_camera_event(event_id, alert_type, cam_id, duration)))

    # Run all camera tasks concurrently
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

