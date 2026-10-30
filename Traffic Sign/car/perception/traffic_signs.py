"""Optional four-class sign detection; no model is loaded until explicitly enabled.

Train/export weights with stop, no_entry, turn_left and turn_right labels. This
module does not download weights, issue movement commands or estimate junctions.
"""

import math
import time
from numbers import Real
from pathlib import Path
from typing import Callable, Optional

from car.perception.config import CAMERA_HFOV_DEG


SIGN_CLASSES = frozenset({"stop", "no_entry", "turn_left", "turn_right"})
_ALIASES = {"compulsory_left": "turn_left", "compulsory_right": "turn_right"}


def _canonical_label(value: str) -> str:
    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(label, label)


def suppress_duplicate_signs(detections, iou_threshold=0.6, confidence_margin=0.1):
    """Remove weaker duplicate boxes for one physical sign, without guessing ties.

    Inputs are validated detection dictionaries, as returned by ``poll``. YOLOv10
    can return overlapping boxes with different labels for the same card. A
    weaker conflicting label is removed only when the confidence advantage is
    at least ``confidence_margin``; close conflicts remain so the decision policy
    can withhold a turn. Spatially separate signs are never merged. This greedy
    pass compares against retained boxes only, preventing a suppressed box from
    suppressing a neighboring sign through an overlap chain. Input is unchanged.
    """
    if (not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1
            or not math.isfinite(confidence_margin) or not 0 <= confidence_margin <= 1):
        raise ValueError("Invalid duplicate-sign overlap threshold or confidence margin")

    def overlap(first, second):
        ax1, ay1, ax2, ay2 = first["bbox"]
        bx1, by1, bx2, by2 = second["bbox"]
        intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0.0, min(ay2, by2) - max(ay1, by1))
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
        return intersection / union if union > 0 else 0.0

    kept = []
    for candidate in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        duplicate = any(
            overlap(candidate, stronger) >= iou_threshold
            and (candidate["class_name"] == stronger["class_name"]
                 or stronger["confidence"] - candidate["confidence"] >= confidence_margin - 1e-12)
            for stronger in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


def sign_geometry(detection, frame_shape, hfov_deg: float = CAMERA_HFOV_DEG,
                  min_height_frac: float = 0.12,
                  max_bearing_deg: float = 22.0) -> dict:
    """Gate actions using apparent sign size and horizontal pinhole bearing.

    Height fraction is a proximity heuristic, not a metric distance estimate.
    The camera's ground pitch cannot locate an elevated sign on the road plane.
    A true ``actionable`` result also does not establish a junction's position.
    Invalid inputs raise ValueError; valid small/peripheral signs remain visible
    to detection consumers but return actionable=False here.
    """
    def finite_number(value):
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)

    if (not isinstance(detection, dict) or detection.get("type") != "traffic_sign"
            or not isinstance(detection.get("class_name"), str)
            or detection.get("class_name") not in SIGN_CLASSES):
        raise ValueError("Expected a supported traffic_sign detection")
    try:
        height, width = frame_shape[:2]
    except (TypeError, ValueError):
        raise ValueError("frame_shape must contain image height and width") from None
    if any(not finite_number(v) or v <= 0 or int(v) != v for v in (height, width)):
        raise ValueError("Image dimensions must be positive finite integers")
    if not finite_number(hfov_deg) or not 0 < hfov_deg < 180:
        raise ValueError("Horizontal field of view must be in (0, 180) degrees")
    if not finite_number(min_height_frac) or not 0 < min_height_frac <= 1:
        raise ValueError("Minimum sign height fraction must be in (0, 1]")
    if not finite_number(max_bearing_deg) or not 0 <= max_bearing_deg < 90:
        raise ValueError("Maximum sign bearing must be in [0, 90) degrees")
    coords = detection.get("bbox")
    try:
        if len(coords) != 4 or not all(finite_number(v) for v in coords):
            raise ValueError("Sign bbox must contain four finite coordinates")
        x1, y1, x2, y2 = (float(v) for v in coords)
    except TypeError:
        raise ValueError("Sign bbox must contain four finite coordinates") from None
    x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
    y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Sign bbox must have positive area within the image")
    focal_x = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    bearing_deg = math.degrees(math.atan(((x1 + x2) / 2.0 - width / 2.0) / focal_x))
    height_frac = (y2 - y1) / height
    return {"height_frac": height_frac, "bearing_deg": bearing_deg,
            "actionable": height_frac >= min_height_frac and abs(bearing_deg) <= max_bearing_deg}


class TrafficSignDetector:
    """Rate-limited Ultralytics detector with a validated four-class vocabulary.

    ``poll`` returns None when throttled, [] for a fresh frame with no accepted
    signs, or detections. Consumers must not count a throttled result as another
    confirming observation. Prediction errors propagate so callers can stop or
    disable the enabled feature explicitly instead of treating failure as clear.
    """

    def __init__(self, model_path, rate_hz: float = 4.0, confidence: float = 0.6,
                 imgsz: int = 320, model=None,
                 clock: Callable[[], float] = time.monotonic):
        if not math.isfinite(rate_hz) or not 0 < rate_hz <= 5:
            raise ValueError("Traffic-sign rate_hz must be greater than 0 and at most 5")
        if not math.isfinite(confidence) or not 0 < confidence <= 1:
            raise ValueError("Traffic-sign confidence must be in (0, 1]")
        if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz < 32 or imgsz % 32:
            raise ValueError("Traffic-sign imgsz must be a positive multiple of 32")
        if model is None:
            if model_path is None or not Path(model_path).exists():
                raise FileNotFoundError("Supply existing trained four-class traffic-sign weights/export")
            from ultralytics import YOLO
            model = YOLO(str(model_path), task="detect")

        names = model.names
        if isinstance(names, (list, tuple)):
            names = dict(enumerate(names))
        if not isinstance(names, dict) or len(names) != len(SIGN_CLASSES):
            raise ValueError("Traffic-sign model must contain exactly the four supported classes")
        self.labels = {int(key): _canonical_label(value) for key, value in names.items()}
        if len(self.labels) != 4 or set(self.labels.values()) != SIGN_CLASSES:
            raise ValueError("Traffic-sign model classes must be stop, no_entry, turn_left, turn_right")
        self.model = model
        self.confidence = confidence
        self.imgsz = imgsz
        self.interval = 1.0 / rate_hz
        self.clock = clock
        self._next_poll = float("-inf")

    def poll(self, frame) -> Optional[list[dict]]:
        if self.clock() < self._next_poll:
            return None
        if frame is None or getattr(frame, "size", 0) == 0 or len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError("Traffic-sign detector needs a nonempty three-channel image")
        height, width = frame.shape[:2]
        try:
            results = self.model.predict(frame, imgsz=self.imgsz,
                                         conf=self.confidence, verbose=False)
            detections = []
            for result in results:
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    raw_class = float(box.cls[0])
                    coords = [float(value) for value in box.xyxy[0].tolist()]
                    if (not math.isfinite(confidence) or not self.confidence <= confidence <= 1
                            or not math.isfinite(raw_class) or not raw_class.is_integer()
                            or int(raw_class) not in self.labels
                            or len(coords) != 4 or not all(math.isfinite(v) for v in coords)):
                        continue
                    x1, y1, x2, y2 = coords
                    x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
                    y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    detections.append({"class_name": self.labels[int(raw_class)],
                                       "confidence": confidence,
                                       "bbox": [x1, y1, x2, y2],
                                       "type": "traffic_sign"})
            return suppress_duplicate_signs(detections)
        finally:
            # Expensive inference is synchronous: avoid immediate catch-up work.
            self._next_poll = self.clock() + self.interval
