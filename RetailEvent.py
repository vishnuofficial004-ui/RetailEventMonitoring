import cv2
import numpy as np
from openvino.runtime import Core

# -------------------------------
# Load OpenVINO Models
# -------------------------------

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


# -------------------------------
# Skeleton Connection Pairs
# -------------------------------
POSE_PAIRS = [
    (1,2), (1,5), (2,3), (3,4), (5,6), (6,7),
    (1,8), (8,9), (9,10), (1,11), (11,12), (12,13),
    (1,0), (0,14), (14,16), (0,15), (15,17)
]


# -------------------------------
# Simple Heatmap Decoder
# -------------------------------
def decode_pose(heatmaps):
    joints = []
    for i in range(18):  # 18 joints
        hmap = heatmaps[i]
        _, conf, _, maxLoc = cv2.minMaxLoc(hmap)
        joints.append((maxLoc[0], maxLoc[1], conf))
    return joints


# -------------------------------
# Open Camera
# -------------------------------
cap = cv2.VideoCapture(0)
print("Camera started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_frame, w_frame = frame.shape[:2]

    # -------------------------------
    # Person Detection
    # -------------------------------
    det_img = cv2.resize(frame, (det_w, det_h))
    det_img = det_img.transpose(2, 0, 1)
    det_img = np.expand_dims(det_img, axis=0)

    det_result = det_compiled([det_img])[det_output]

    for det in det_result[0][0]:
        conf = float(det[2])
        if conf < 0.6:
            continue

        xmin = int(det[3] * w_frame)
        ymin = int(det[4] * h_frame)
        xmax = int(det[5] * w_frame)
        ymax = int(det[6] * h_frame)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

        # -------------------------------
        # Crop Person
        # -------------------------------
        person_crop = frame[ymin:ymax, xmin:xmax]
        if person_crop.size == 0:
            continue

        pose_img = cv2.resize(person_crop, (pose_w, pose_h))
        pose_img = pose_img.transpose(2, 0, 1)
        pose_img = np.expand_dims(pose_img, axis=0)

        # -------------------------------
        # Pose Inference
        # -------------------------------
        pose_result = pose_compiled([pose_img])[pose_output]
        pose_result = pose_result[0]  # (38,32,57)

        heatmaps = pose_result[:18]
        joints = decode_pose(heatmaps)

        # -------------------------------
        # Convert joints to frame coords
        # -------------------------------
        points = []

        for j in joints:
            x_heat, y_heat, j_conf = j

            if j_conf < 0.05:
                points.append(None)
                continue

            x = int((x_heat / 57) * (xmax - xmin)) + xmin
            y = int((y_heat / 32) * (ymax - ymin)) + ymin

            points.append((x, y))
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # -------------------------------
        # Draw Skeleton Lines
        # -------------------------------
        for pair in POSE_PAIRS:
            pA, pB = pair
            if points[pA] and points[pB]:
                cv2.line(frame, points[pA], points[pB], (255, 0, 0), 2)

    # -------------------------------
    # Display
    # -------------------------------
    cv2.imshow("OpenVINO Pose Skeleton", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

