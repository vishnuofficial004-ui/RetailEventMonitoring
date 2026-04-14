import cv2
import numpy as np
from openvino.runtime import Core
from datetime import datetime
from collections import deque
from action_rules import ActionEncoderDecoder

# ---------------- OpenVINO ----------------
ie = Core()

# Person detection
person_model = ie.read_model("models/person-detection-retail-0013/person-detection-retail-0013.xml")
person_compiled = ie.compile_model(person_model, "CPU")
p_in = person_compiled.input(0)
p_out = person_compiled.output(0)
_, _, ph, pw = p_in.shape

# Pose estimation
pose_model = ie.read_model("models/human-pose-estimation-0007/human-pose-estimation-0007.xml")
pose_compiled = ie.compile_model(pose_model, "CPU")
pose_out = pose_compiled.output(0)

# ---------------- Action Engine ----------------
action_engine = ActionEncoderDecoder(window_duration=1.5, event_cooldown_seconds=2.0)

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)

# ---------------- Skeleton ----------------
NUM_KEYPOINTS = 17
SKELETON = [
    (1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),(1,11),(11,12),(12,13),
    (0,14),(0,15)
]

# ---------------- Tracker & smoothing ----------------
tracker = None
tracking_active = False
DETECTION_INTERVAL = 5
frame_count = 0
keypoints_history = deque(maxlen=7)  # for action engine
prev_points = [None]*NUM_KEYPOINTS  # EMA smoothing
EMA_ALPHA = 0.4
DETECTION_THRESHOLD = 0.7

# ---------------- Utilities ----------------
def draw_timestamp(frame):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, ts, (frame.shape[1]-260, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def decode_pose(heatmaps, roi_shape):
    roi_h, roi_w = roi_shape[:2]
    points = []
    for i in range(NUM_KEYPOINTS):
        hm = heatmaps[0, i]
        _, _, _, max_loc = cv2.minMaxLoc(hm)
        x = int(max_loc[0] * roi_w / hm.shape[1])
        y = int(max_loc[1] * roi_h / hm.shape[0])
        points.append((x,y))
    return points

def apply_ema(prev, current, alpha=0.4):
    smoothed = []
    for p, c in zip(prev, current):
        if c is None:
            smoothed.append(p)
        elif p is None:
            smoothed.append(c)
        else:
            x = int(alpha*c[0] + (1-alpha)*p[0])
            y = int(alpha*c[1] + (1-alpha)*p[1])
            smoothed.append((x,y))
    return smoothed

def is_box_valid(box, frame_shape):
    xmin, ymin, xmax, ymax = box
    H, W = frame_shape[:2]
    if xmax <= 0 or ymax <= 0 or xmin >= W or ymin >= H:
        return False
    if (xmax-xmin) < 20 or (ymax-ymin) < 20:
        return False
    return True

# ---------------- Main Loop ----------------
while True:
    ret, frame = cap.read()
    if not ret: break

    H, W = frame.shape[:2]
    frame_count += 1
    bbox_available = False
    best_box = None

    # -------- Person Detection --------
    if frame_count % DETECTION_INTERVAL == 0 or not tracking_active:
        resized = cv2.resize(frame, (pw, ph))
        blob = resized.transpose(2,0,1)[None,:]
        detections = person_compiled([blob])[p_out]

        best_area = 0
        for det in detections[0][0]:
            if det[2] < DETECTION_THRESHOLD: continue
            xmin = int(det[3]*W); ymin = int(det[4]*H)
            xmax = int(det[5]*W); ymax = int(det[6]*H)
            area = (xmax-xmin)*(ymax-ymin)
            if area>best_area:
                best_area = area
                best_box = (xmin, ymin, xmax, ymax)

        if best_box:
            xmin, ymin, xmax, ymax = best_box
            bbox_available = True
            if not tracking_active:
                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, (xmin,ymin,xmax-xmin,ymax-ymin))
                tracking_active = True
                keypoints_history.clear()
                prev_points = [None]*NUM_KEYPOINTS

    # -------- Tracker fallback --------
    if not bbox_available and tracking_active:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int,bbox)
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                tracking_active = False
                keypoints_history.clear()
                prev_points = [None]*NUM_KEYPOINTS
                continue
            # Detector confirmation
            blob = cv2.resize(roi, (pw, ph)).transpose(2,0,1)[None,:]
            detections = person_compiled([blob])[p_out]
            detected = any(d[2]>DETECTION_THRESHOLD for d in detections[0][0])
            if not detected:
                tracking_active = False
                keypoints_history.clear()
                prev_points = [None]*NUM_KEYPOINTS
                continue
            xmin, ymin, xmax, ymax = x, y, x+w, y+h
            bbox_available = True
        else:
            tracking_active = False
            keypoints_history.clear()
            prev_points = [None]*NUM_KEYPOINTS

    if not bbox_available:
        draw_timestamp(frame)
        cv2.imshow("RetailEvent", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
        continue

    # -------- ROI & Pose --------
    xmin, ymin = max(0,xmin), max(0,ymin)
    xmax, ymax = min(W,xmax), min(H,ymax)
    roi = frame[ymin:ymax, xmin:xmax]
    if roi.size==0: continue

    pose_input = cv2.resize(roi, (448,448)).transpose(2,0,1)[None,:]
    heatmaps = pose_compiled([pose_input])[pose_out]
    points = decode_pose(heatmaps, roi.shape)

    # -------- EMA smoothing --------
    points = apply_ema(prev_points, points, EMA_ALPHA)
    prev_points = points.copy()

    # -------- Multi-frame smoothing for action engine --------
    keypoints_history.append(points)

    # -------- Draw Skeleton --------
    for x,y in points:
        if x is not None and y is not None:
            cv2.circle(roi,(x,y),3,(0,0,255),-1)
    for a,b in SKELETON:
        if points[a] and points[b]:
            cv2.line(roi, points[a], points[b], (0,255,0),2)

    frame[ymin:ymax, xmin:xmax] = roi

    # -------- Action Engine --------
    action, event = action_engine.update(points)
    if action != "None":
        cv2.putText(frame,f"Action: {action}",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        if event:
            print(event)  # JSON output

    # -------- UI --------
    cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,255,0),2)
    draw_timestamp(frame)
    cv2.imshow("RetailEvent", frame)

    if cv2.waitKey(1)&0xFF==ord("q"): break

# ---------------- Cleanup ----------------
cap.release()
cv2.destroyAllWindows()



