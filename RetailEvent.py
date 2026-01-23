import cv2
import numpy as np
from openvino.runtime import Core

# Load model
ie = Core()
model = ie.read_model(r"models\person-detection-retail-0013\person-detection-retail-0013.xml")
compiled_model = ie.compile_model(model, "CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Get input shape
_, _, h, w = input_layer.shape

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    resized = cv2.resize(frame, (w, h))
    input_image = resized.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)

    # Inference
    result = compiled_model([input_image])[output_layer]

    # Draw detections
    for det in result[0][0]:
        confidence = float(det[2])
        if confidence < 0.7:   # your tuned threshold
            continue

        xmin = int(det[3] * frame.shape[1])
        ymin = int(det[4] * frame.shape[0])
        xmax = int(det[5] * frame.shape[1])
        ymax = int(det[6] * frame.shape[0])

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

    cv2.imshow("Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

