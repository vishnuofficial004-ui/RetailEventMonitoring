# action_rules.py

from collections import deque
from datetime import datetime

class ActionEncoderDecoder:
    def __init__(self, shelf_zone):
        self.shelf_zone = shelf_zone
        self.hand_history = {
            "right": deque(maxlen=5),
            "left": deque(maxlen=5)
        }

    def update(self, points):
        """
        points = list of smoothed pose keypoints from main pipeline
        returns: (action_label, action_json)
        """

        action_detected = "None"
        action_json = None

        # wrist indexes from pose model
        right_hand = points[4]  # right wrist
        left_hand  = points[7]  # left wrist

        hand_positions = {"right": right_hand, "left": left_hand}

        # Store history
        for h, pos in hand_positions.items():
            self.hand_history[h].append(pos)

        hx1, hy1, hx2, hy2 = self.shelf_zone

        # approximate torso center (shoulders)
        torso = None
        if points[1] and points[2]:
            torso = (
                (points[1][0] + points[2][0]) // 2,
                (points[1][1] + points[2][1]) // 2
            )

        # ---- Decoder Rules ----
        for h, pos in hand_positions.items():
            if pos is None:
                continue

            x, y = pos
            prev_pos = None
            if len(self.hand_history[h]) > 1:
                prev_pos = self.hand_history[h][-2]

            # 1. Reaching shelf
            if hx1 <= x <= hx2 and hy1 <= y <= hy2:
                action_detected = f"{h}_reaching_shelf"

            # 2. Pick object (left shelf this frame)
            elif prev_pos:
                px, py = prev_pos
                if hx1 <= px <= hx2 and hy1 <= py <= hy2:
                    action_detected = f"{h}_picked_object"

            # 3. Hide object (hand moves back toward torso)
            if torso and prev_pos:
                tx, ty = torso
                if prev_pos[1] < ty and y >= ty:
                    action_detected = f"{h}_hide_object"

            # Build JSON event if any action happened
            if action_detected != "None":
                action_json = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "person_id": 1,
                    "action": action_detected,
                    "hand": h,
                    "coordinates": [x, y]
                }

        return action_detected, action_json
