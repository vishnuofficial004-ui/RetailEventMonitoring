import cv2
import numpy as np
from collections import deque
from openvino.runtime import Core

# ---------------- Load Models ----------------

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


# ---------------- Skeleton Pairs ----------------

POSE_PAIRS = [
    (1,2),(1,5),(2,3),(3,4),(5,6),(6,7),
    (1,8),(8,9),(9,10),(1,11),(11,12),(12,13),
    (1,0),(0,14),(14,16),(0,15),(15,17)
]

NUM_JOINTS = 18

# ---------------- Simple Heatmap Decoder ----------------

def decode_pose(heatmaps):
    joints = []
    for i in range(NUM_JOINTS):
        hmap = heatmaps[i]
        _, conf, _, maxLoc = cv2.minMaxLoc(hmap)
        joints.append((maxLoc[0], maxLoc[1], conf))
    return joints


# ---------------- Temporal Smoothing Buffer ----------------

SMOOTHING_FRAMES = 5   # number of frames to average

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


# ---------------- Camera ----------------

cap = cv2.VideoCapture(0)

print("Running... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

    # ---- Person Detection ----
    det_img = cv2.resize(frame, (det_w, det_h)).transpose(2,0,1)
    det_img = np.expand_dims(det_img, axis=0)
    det_result = det_compiled([det_img])[det_output]

    for det in det_result[0][0]:
        if float(det[2]) < 0.6:
            continue

        xmin = int(det[3] * W)
        ymin = int(det[4] * H)
        xmax = int(det[5] * W)
        ymax = int(det[6] * H)

        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,255,0), 2)

        person = frame[ymin:ymax, xmin:xmax]
        if person.size == 0:
            continue

        # ---- Pose Inference ----
        pose_img = cv2.resize(person, (pose_w, pose_h)).transpose(2,0,1)
        pose_img = np.expand_dims(pose_img, axis=0)

        pose_result = pose_compiled([pose_img])[pose_output][0]
        heatmaps = pose_result[:18]

        joints = decode_pose(heatmaps)

        # ---- Convert to frame coordinates ----
        points = []
        for xh, yh, jc in joints:
            if jc < 0.05:
                points.append(None)
                continue

            x = int((xh / 57) * (xmax - xmin)) + xmin
            y = int((yh / 32) * (ymax - ymin)) + ymin
            points.append((x,y))

        # ---- Apply Temporal Smoothing ----
        points = smooth_points(points)

        # ---- Draw Joints ----
        for p in points:
            if p:
                cv2.circle(frame, p, 4, (0,0,255), -1)

        # ---- Draw Skeleton ----
        for a,b in POSE_PAIRS:
            if points[a] and points[b]:
                cv2.line(frame, points[a], points[b], (255,0,0), 2)

    cv2.imshow("Real-Time Pose (CPU + Smoothed)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

