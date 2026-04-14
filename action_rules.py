from collections import deque
from datetime import datetime
import time
import json

class ActionEncoderDecoder:
    def __init__(self, window_duration=2.0, event_cooldown_seconds=2.0):
        """
        window_duration: seconds of pose history to keep
        event_cooldown_seconds: minimum time gap between same events
        """
        self.window_duration = window_duration
        self.event_cooldown = event_cooldown_seconds

        # Stores (timestamp, keypoints) for full pose history
        self.pose_buffer = deque()
        self.last_event_time = {}

        # Hand Y-coordinates history for sequence tracking
        self.hand_history = {
            "left": deque(maxlen=15),
            "right": deque(maxlen=15)
        }

    # ---------------- Public API ----------------
    def update(self, keypoints):
        """
        keypoints: list of 18 (x, y) points
        """
        now = time.time()
        self.pose_buffer.append((now, keypoints))

        # Remove old poses beyond window_duration
        while self.pose_buffer and now - self.pose_buffer[0][0] > self.window_duration:
            self.pose_buffer.popleft()

        if len(self.pose_buffer) < 3:
            return "None", None

        # Decode action based on pose history
        action = self._decode()

        if action == "None":
            return "None", None

        # Event cooldown
        last_time = self.last_event_time.get(action, 0)
        if now - last_time < self.event_cooldown:
            return action, None

        self.last_event_time[action] = now

        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": action,
            "confidence": 0.85
        }

        return action, json.dumps(event)

    # ---------------- Decoder ----------------
    def _decode(self):
        """
        Detects:
        - Taking_Object: hand stretches and returns
        - Interested_In_Object: hand moves toward object repeatedly
        - Loitering: minimal movement
        - Suspicious_Movement: rapid large motion
        """
        # OpenPose-like indices
        RIGHT_HAND = 4
        LEFT_HAND = 7
        TORSO = 1
        STOMACH = 8

        hand_reach_count = 0
        body_motion = 0

        # Estimate ROI height for normalized Y
        roi_h = 1.0
        for _, kp in self.pose_buffer:
            for p in kp:
                if p is not None:
                    roi_h = max(roi_h, p[1])
        if roi_h == 0: roi_h = 1.0

        # ---------------- Frame-to-frame motion ----------------
        for i in range(1, len(self.pose_buffer)):
            prev = self.pose_buffer[i - 1][1]
            curr = self.pose_buffer[i][1]

            if not prev or not curr:
                continue
            if not curr[TORSO] or not curr[STOMACH]:
                continue

            torso_y = curr[TORSO][1] / roi_h
            stomach_y = curr[STOMACH][1] / roi_h

            # Body motion (excluding hands)
            for idx in range(len(curr)):
                if idx in (RIGHT_HAND, LEFT_HAND):
                    continue
                if prev[idx] and curr[idx]:
                    body_motion += abs(curr[idx][0] - prev[idx][0])
                    body_motion += abs(curr[idx][1] - prev[idx][1])

            # ---------------- Hand tracking ----------------
            for hand, idx in [("right", RIGHT_HAND), ("left", LEFT_HAND)]:
                if not curr[idx]:
                    continue
                hand_y = curr[idx][1] / roi_h
                self.hand_history[hand].append(hand_y)

                # Detect hand stretch → return sequence
                positions = list(self.hand_history[hand])
                if len(positions) >= 3:
                    # previous above torso, middle stretched above torso, last back near torso-stomach
                    if (positions[-3] > torso_y and
                        positions[-2] < torso_y - 0.1 and
                        torso_y <= positions[-1] <= stomach_y):
                        hand_reach_count += 1
                        self.hand_history[hand].clear()  # reset for next sequence

        # ---------------- Event Decision ----------------
        if hand_reach_count >= 1:
            return "Taking_Object"

        # Interested in object: multiple short hand movements, low body motion
        if hand_reach_count >= 2 and body_motion < 50:
            return "Interested_In_Object"

        # Loitering: minimal movement
        if body_motion < 20 and hand_reach_count == 0:
            return "Loitering"

        # Suspicious movement: high motion
        if body_motion > 200:
            return "Suspicious_Movement"

        return "None"
