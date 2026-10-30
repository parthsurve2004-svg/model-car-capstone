"""
avoidance.py — Comprehensive Obstacle Avoidance Decision Engine
===============================================================
Owned by perception-agent.
Calculates danger scores for Left, Center, and Right zones based on:
  1. Camera Bounding Boxes (YOLO obstacles, humans, potholes)
  2. Ground-plane vertical proximity (bottom edge y2 / frame_h)
  3. Ultrasonic distance fusion (Left, Right, and future Rear HC-SR04P)

Provides progressive avoidance decisions:
  - CLEAR: Path is open, cruise forward.
  - VEER_LEFT / VEER_RIGHT: Gentle proactive veering while maintaining forward speed.
  - AVOID_LEFT / AVOID_RIGHT: Active skid-steer bypass around close obstacles.
  - PIVOT_ESCAPE: Center blocked and no forward bypass corridor; pivot in place.
  - STOP: Emergency stop (imminent bumper impact or all escape corridors blocked).
"""

from typing import List, Dict, Any, Optional

# Zone divider boundaries across horizontal frame width
ZONE_LEFT_END = 0.33
ZONE_RIGHT_START = 0.67

# Weights for obstacle types
OBSTACLE_WEIGHT = 1.0
POTHOLE_WEIGHT = 1.8
HUMAN_WEIGHT = 1.15

# Danger threshold for avoidance triggers
DANGER_THRESHOLD = 0.30
CRITICAL_DANGER_THRESHOLD = 0.85


def compute_zone_scores(
    detections: List[Dict[str, Any]],
    frame_w: int,
    frame_h: int,
    sonar: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Computes danger scores for Left, Center, and Right zones by fusing
    camera detections (with bottom-edge proximity) and ultrasonic distance readings.

    Args:
        detections: List of detection dicts [{"bbox": [x1, y1, x2, y2], "confidence": ..., "type": ...}]
        frame_w: Frame width in pixels
        frame_h: Frame height in pixels
        sonar: Optional dict of ultrasonic distances in cm, e.g. {"left": 25.0, "right": 80.0, "rear": 999.0}

    Returns:
        dict: {"left": score, "center": score, "right": score}
    """
    scores = {"left": 0.0, "center": 0.0, "right": 0.0}
    frame_area = frame_w * frame_h
    if frame_area == 0:
        return scores

    # ── 1. Process Vision Detections (YOLO + Potholes) ────────────────────────
    for det in (detections or []):
        if not isinstance(det, dict):
            continue
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (ValueError, TypeError):
            continue

        cx = (x1 + x2) / 2.0
        conf = float(det.get("confidence", 0.5))
        cls_name = str(det.get("class_name", "")).lower()
        is_pothole = det.get("type") == "negative" or "pothole" in cls_name
        is_human = "person" in cls_name

        if is_pothole:
            wt = POTHOLE_WEIGHT
        elif is_human:
            wt = HUMAN_WEIGHT
        else:
            wt = OBSTACLE_WEIGHT

        # Calculate bounding box area score
        bbox_w = max(x2 - x1, 1.0)
        bbox_h = max(y2 - y1, 1.0)
        bbox_area = bbox_w * bbox_h
        area_score = min(bbox_area / frame_area, 0.5)

        # Ground-plane proximity: in a forward-facing camera, bottom edge (y2)
        # correlates strongly with physical distance to the front bumper.
        y2_frac = y2 / max(frame_h, 1)
        bottom_prox = max(0.0, (y2_frac - 0.25) / 0.75)  # 0.0 at top/horizon, 1.0 at bumper

        # Danger score combines confidence, area, and bottom-edge proximity
        danger = conf * wt * (0.35 + 0.35 * area_score + 0.50 * (bottom_prox ** 1.3))

        # Horizontal corridor assignment
        cx_frac = cx / frame_w
        w_frac = bbox_w / frame_w

        # If obstacle is wide (> 35% of frame width), it partially spans adjacent zones
        if w_frac > 0.35:
            scores["center"] += danger * 0.7
            if cx_frac < 0.50:
                scores["left"] += danger * 0.5
            else:
                scores["right"] += danger * 0.5
            det["zone"] = "center_wide"
        elif cx_frac < ZONE_LEFT_END:
            scores["left"] += danger
            det["zone"] = "left"
        elif cx_frac > ZONE_RIGHT_START:
            scores["right"] += danger
            det["zone"] = "right"
        else:
            scores["center"] += danger
            det["zone"] = "center"

    # ── 2. Fuse Ultrasonic Telemetry (HC-SR04P) ──────────────────────────────
    if sonar and isinstance(sonar, dict):
        us_l = sonar.get("left", 999.0)
        us_r = sonar.get("right", 999.0)

        # Helper to validate distance (reject 0.0 noise & 999 timeout)
        def is_valid(d):
            return isinstance(d, (int, float)) and 1.0 < d < 850.0

        # Left ultrasonic distance fusion
        if is_valid(us_l):
            if us_l < 22.0:
                scores["left"] += 1.8  # Critical proximity
            elif us_l < 42.0:
                scores["left"] += (42.0 - us_l) / 20.0 * 0.9
            elif us_l < 60.0:
                scores["left"] += (60.0 - us_l) / 18.0 * 0.35

        # Right ultrasonic distance fusion
        if is_valid(us_r):
            if us_r < 22.0:
                scores["right"] += 1.8  # Critical proximity
            elif us_r < 42.0:
                scores["right"] += (42.0 - us_r) / 20.0 * 0.9
            elif us_r < 60.0:
                scores["right"] += (60.0 - us_r) / 18.0 * 0.35

        # If both front sensors detect close obstacle, center path is impassable
        if is_valid(us_l) and is_valid(us_r) and us_l < 25.0 and us_r < 25.0:
            scores["center"] += 2.0

    return scores


def decide_avoidance(scores: Dict[str, float]) -> str:
    """
    Evaluates Left, Center, and Right zone danger scores and returns
    a directional avoidance decision:
      - 'CLEAR': Path forward is clear.
      - 'LEFT': Steer/bypass to the left.
      - 'RIGHT': Steer/bypass to the right.
      - 'STOP': All forward paths blocked or critical danger.
    """
    L = scores.get("left", 0.0)
    C = scores.get("center", 0.0)
    R = scores.get("right", 0.0)
    T = DANGER_THRESHOLD

    # All clear
    if L < T and C < T and R < T:
        return "CLEAR"

    # Center is blocked
    if C >= T:
        # Both sides also blocked -> STOP
        if L >= T and R >= T:
            return "STOP"
        # Steer toward whichever side has lower danger score
        return "LEFT" if L <= R else "RIGHT"

    # Center clear, but sides have obstacles
    if L >= T and R < T:
        return "RIGHT"
    if R >= T and L < T:
        return "LEFT"
    if L >= T and R >= T:
        return "STOP"

    return "CLEAR"


def decide_avoidance_detailed(
    scores: Dict[str, float],
    sonar: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Detailed decision engine providing nuanced steering recommendations:
      - action: 'CLEAR', 'VEER_LEFT', 'VEER_RIGHT', 'AVOID_LEFT', 'AVOID_RIGHT', 'PIVOT_ESCAPE', 'STOP'
      - angle: recommended steering angle in degrees (negative = left, positive = right)
      - speed_factor: recommended speed multiplier (0.0 to 1.0)
    """
    L = scores.get("left", 0.0)
    C = scores.get("center", 0.0)
    R = scores.get("right", 0.0)
    T = DANGER_THRESHOLD

    # All clear
    if L < T and C < T and R < T:
        return {"action": "CLEAR", "angle": 0, "speed_factor": 1.0}

    # Critical all-blocked condition
    if (L >= T and C >= T and R >= T) or C >= CRITICAL_DANGER_THRESHOLD:
        # Check if one side has even a slight opening for a pivot escape
        if L < CRITICAL_DANGER_THRESHOLD or R < CRITICAL_DANGER_THRESHOLD:
            turn_dir = -45 if L <= R else 45
            return {"action": "PIVOT_ESCAPE", "angle": turn_dir, "speed_factor": 0.0}
        return {"action": "STOP", "angle": 0, "speed_factor": 0.0}

    # Center obstacle: active bypass
    if C >= T:
        turn_dir = -35 if L <= R else 35
        action = "AVOID_LEFT" if turn_dir < 0 else "AVOID_RIGHT"
        return {"action": action, "angle": turn_dir, "speed_factor": 0.75}

    # Off-center obstacle: gentle veering
    if L >= T and R < T:
        return {"action": "VEER_RIGHT", "angle": 20, "speed_factor": 0.85}
    if R >= T and L < T:
        return {"action": "VEER_LEFT", "angle": -20, "speed_factor": 0.85}

    if L >= T and R >= T:
        return {"action": "STOP", "angle": 0, "speed_factor": 0.0}

    return {"action": "CLEAR", "angle": 0, "speed_factor": 1.0}
