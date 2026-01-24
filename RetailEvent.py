import cv2
import numpy as np
from collections import deque
from datetime import datetime
from openvino.runtime import Core

# ---------- Load Models ----------
ie = Core()

# Person Detection
det_model = ie.read_model("models/person-detection-retail-0013/person-detection-retail-0013.xml")
det_compiled = ie.compile_model(det_model, "CPU")
det_input = det_compiled.input(0)
det_output = det_compiled.output(0)
_, _, det_h, det_w = det_input.shape

# Pose Estimation
pose_model = ie.read_model("models/human-pose-estimation-0001/human-pose-estimation-0001.xml")
pose_compiled = ie.compile_model(pose_model, "CPU")
pose_input = pose_compiled.input(0)
pose_output = pose_compiled.output(0)
_, _, pose_h, pose_w = pose_input.shape

# ---------- Skeleton Pairs ----------
POSE_PAIRS = [
    (1,2),(1,5),(2,3),(3,4),(5,6),(6,7),
    (1,8),(8,9),(9,10),(1,11),(11,12),(12,13),
    (1,0),(0,14),(14,16),(0,15),(15,17)
]
NUM_JOINTS = 18

# ---------- Heatmap Decoder ----------
def decode_pose(heatmaps):
    joints = []
    for i in range(NUM_JOINTS):
        hmap = heatmaps[i]
        _, conf, _, maxLoc = cv2.minMaxLoc(hmap)
        joints.append((maxLoc[0], maxLoc[1], conf))
    return joints

# ---------- Temporal Smoothing ----------
SMOOTHING_FRAMES = 3
joint_history = [deque(maxlen=SMOOTHING_FRAMES) for _ in range(NUM_JOINTS)]

def smooth_points(points):
    smoothed = []
    for i, p in enumerate(points):
        if p is not None:
            joint_history[i].append(p)
        if len(joint_history[i]) == 0:
            smoothed.append(None)
        else:
            xs = [pt[0] for pt in joint_history[i]]
            ys = [pt[1] for pt in joint_history[i]]
            smoothed.append((int(sum(xs)/len(xs)), int(sum(ys)/len(ys))))
    return smoothed

# ---------- Timestamp Function ----------
def draw_timestamp(frame):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (255, 255, 255)
    thickness = 1
    text_size, _ = cv2.getTextSize(timestamp, font, scale, thickness)
    text_w, text_h = text_size
    x = frame.shape[1] - text_w - 10
    y = frame.shape[0] - 10
    cv2.putText(frame, timestamp, (x, y), font, scale, color, thickness, cv2.LINE_AA)

# ---------- Define Shelf Zone (x1, y1, x2, y2) ----------
SHELF_ZONE = (300, 100, 600, 400)  # example rectangle
cv2_color = (0, 255, 255)  # yellow for zone

# ---------- Hand History for Action Encoder ----------
hand_history = {
    "right": deque(maxlen=5),
    "left": deque(maxlen=5)
}

# ---------- Camera ----------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

    # Draw shelf zone rectangle
    cv2.rectangle(frame, (SHELF_ZONE[0], SHELF_ZONE[1]), (SHELF_ZONE[2], SHELF_ZONE[3]), cv2_color, 2)

    action_detected = "None"

    # ---- Person Detection ----
    det_img = cv2.resize(frame, (det_w, det_h)).transpose(2,0,1)
    det_img = np.expand_dims(det_img, axis=0)
    det_result = det_compiled([det_img])[det_output]

    for det in det_result[0][0]:
        if float(det[2]) < 0.6:
            continue

        pad = 20
        xmin = max(0, int(det[3] * W) - pad)
        ymin = max(0, int(det[4] * H) - pad)
        xmax = min(W, int(det[5] * W) + pad)
        ymax = min(H, int(det[6] * H) + pad)

        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,255,0), 2)

        person = frame[ymin:ymax, xmin:xmax]
        if person.size == 0:
            continue

        # ---- Pose Inference ----
        pose_img = cv2.resize(person, (pose_w, pose_h)).transpose(2,0,1)
        pose_img = np.expand_dims(pose_img, axis=0)

        pose_result = pose_compiled([pose_img])[pose_output][0]
        heatmaps = pose_result[:NUM_JOINTS]

        joints = decode_pose(heatmaps)

        # Map to frame coordinates
        points = []
        for xh, yh, jc in joints:
            if jc < 0.03:
                points.append(None)
                continue
            x = int((xh / 57) * (xmax - xmin)) + xmin
            y = int((yh / 32) * (ymax - ymin)) + ymin
            points.append((x, y))

        points = smooth_points(points)

        # Draw joints & skeleton
        for p in points:
            if p:
                cv2.circle(frame, p, 4, (0,0,255), -1)
        for a,b in POSE_PAIRS:
            if points[a] and points[b]:
                cv2.line(frame, points[a], points[b], (255,0,0), 2)

        # ---------- Action Encoder ----------
        right_hand = points[4]  # right wrist
        left_hand = points[7]   # left wrist

        hand_positions = {"right": right_hand, "left": left_hand}
        for h, pos in hand_positions.items():
            if pos:
                hand_history[h].append(pos)
            else:
                hand_history[h].append(None)

        # ---------- Action Decoder (Rule-Based) ----------
        # Simple logic: hand enters shelf zone → reaching; leaves → pick
        for h, pos in hand_positions.items():
            if pos is None:
                continue
            x, y = pos
            hx1, hy1, hx2, hy2 = SHELF_ZONE
            if hx1 <= x <= hx2 and hy1 <= y <= hy2:
                action_detected = f"{h}_reaching_shelf"
            elif len(hand_history[h]) > 1 and hand_history[h][-2] is not None:
                prev_x, prev_y = hand_history[h][-2]
                if hx1 <= prev_x <= hx2 and hy1 <= prev_y <= hy2:
        # just exited shelf
                  action_detected = f"{h}_picked_object"
    # ---------- Draw timestamp & detected action ----------
    draw_timestamp(frame)
    cv2.putText(frame, f"Action: {action_detected}", (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Pose + Action Encoder/Decoder", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
