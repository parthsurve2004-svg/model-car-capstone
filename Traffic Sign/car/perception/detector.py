"""
detector.py — NCNN Object Detection Wrapper (YOLOv10n + GPU Fine-Tuned Pothole Model)
=====================================================================================
Owned by perception-agent.
Loads model exports from benchmarking/models/ for positive obstacles and the
GPU printout-tuned pothole model (pothole_detection/best.pt) focused on the
lower camera section for negative obstacles (potholes on road printouts or roads).
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from ultralytics import YOLO

from car.perception.config import (
    REPO_ROOT,
    MODEL_PATH,
    POTHOLE_MODEL_PATH,
    IMGSZ,
    POTHOLE_IMGSZ,
    CONFIDENCE_THRESHOLD,
    POTHOLE_CONFIDENCE_THRESHOLD,
    POTHOLE_LOWER_Y_MIN_FRAC,
)


def enhance_road_contrast(bgr_roi: np.ndarray) -> np.ndarray:
    """
    Applies light denoising + LAB L-channel CLAHE contrast enhancement.
    Counteracts contrast washout, indoor lighting glare, and sensor noise when
    detecting potholes on real roads or on printed/screen photos held in front
    of the camera.
    """
    if bgr_roi is None or bgr_roi.size == 0:
        return bgr_roi
    denoised = cv2.GaussianBlur(bgr_roi, (3, 3), 0)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_chan)
    merged = cv2.merge((l_eq, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def cv_detect_potholes(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Classical CV pothole/road-damage fallback detector (only used if YOLO pothole model is absent).

    Scans the bottom 55% of the frame (the road surface) using Laplacian
    edge detection + morphological filtering to find dark, irregular holes
    without triggering false positives on shadows or upper walls.

    Args:
        frame: BGR numpy image array (H, W, 3).

    Returns:
        List of detection dicts compatible with the avoidance engine:
        [{"class_id": -1, "class_name": "pothole", "confidence": 0.42,
          "bbox": [x1, y1, x2, y2], "type": "negative"}, ...]
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return []

    h, w = frame.shape[:2]
    if h < 40 or w < 40:
        return []

    roi_y = int(h * 0.45)
    roi = frame[roi_y:, :]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = cv2.Laplacian(blur, cv2.CV_64F)
    lap_u = np.uint8(np.clip(np.absolute(lap), 0, 255))

    _, thresh = cv2.threshold(lap_u, 28, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    roi_area = roi.shape[0] * roi.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 900 or area > roi_area * 0.25:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        if not (0.25 < cw / max(ch, 1) < 5.0):
            continue
        dets.append({
            "class_id": -1,
            "class_name": "pothole",
            "confidence": 0.42,
            "bbox": [x, y + roi_y, x + cw, y + roi_y + ch],
            "type": "negative",
        })
    return dets


class Detector:
    """Wrapper around YOLOv10n positive obstacle + GPU fine-tuned pothole detection pipeline."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        pothole_model_path: Optional[str] = POTHOLE_MODEL_PATH,
        imgsz: int = IMGSZ,
        conf: float = CONFIDENCE_THRESHOLD,
        pothole_conf: float = POTHOLE_CONFIDENCE_THRESHOLD,
        enable_pothole_cv: bool = False,
    ):
        self.imgsz = imgsz
        self.conf = conf
        self.pothole_conf = pothole_conf
        self.model_path = str(model_path)
        self.pothole_model_path = str(pothole_model_path) if pothole_model_path else None
        self.model = None
        self.pothole_model = None
        self.enable_pothole_cv = enable_pothole_cv
        self._load_model()

    def _load_model(self):
        """Loads the primary obstacle model (with PyTorch fallback if NCNN runtime is absent) and fine-tuned pothole model."""
        model_to_load = self.model_path
        if "ncnn" in model_to_load.lower():
            try:
                import ncnn  # noqa: F401
            except ImportError:
                pt_fallback = REPO_ROOT / "benchmarking" / "models" / "yolov10n.pt"
                if not pt_fallback.exists():
                    pt_fallback = REPO_ROOT / "yolov10n.pt"
                if pt_fallback.exists():
                    model_to_load = str(pt_fallback)

        try:
            self.model = YOLO(model_to_load, task="detect")
        except Exception as e:
            print(f"⚠️ Failed to load model from {model_to_load}: {e}", file=sys.stderr)
            raise

        # Load dedicated GPU fine-tuned pothole model (best.pt / pothole_best.pt)
        candidates = []
        if self.pothole_model_path:
            candidates.append(Path(self.pothole_model_path))
        candidates.extend([
            REPO_ROOT / "pothole_best.pt",
            REPO_ROOT / "pothole_detection" / "best.pt",
            REPO_ROOT / "pothole_detection" / "runs_rdd" / "printout_gpu_pothole_model" / "weights" / "best.pt",
            REPO_ROOT / "pothole_detection" / "rdd2022_yolov8n.pt",
        ])
        for cand in candidates:
            if cand.exists():
                try:
                    self.pothole_model = YOLO(str(cand), task="detect")
                    break
                except Exception:
                    continue

    def _detect_yolo_potholes(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Runs the GPU fine-tuned D40 pothole YOLO model on the full 4:3 frame and filters strictly to the lower camera section."""
        if self.pothole_model is None or frame is None:
            return []

        h, w = frame.shape[:2]
        if h < 40 or w < 40:
            return []

        potholes: List[Dict[str, Any]] = []
        frame_area = float(h * w)
        try:
            res = self.pothole_model.predict(
                frame,
                imgsz=max(self.imgsz, POTHOLE_IMGSZ),
                conf=self.pothole_conf,
                verbose=False,
            )[0]
            for box in res.boxes:
                cls_id = int(box.cls[0])
                raw_conf = float(box.conf[0])
                name = str(res.names.get(cls_id, "pothole")).lower()
                # Strict D40 pothole filter (Class 0: pothole_D40) — ignore cracks, lines, and borders
                if cls_id == 0 or "pothole" in name or "d40" in name:
                    gx1, gy1, gx2, gy2 = [float(v) for v in box.xyxy[0]]
                    bw = gx2 - gx1
                    bh = gy2 - gy1
                    gyc = (gy1 + gy2) / 2.0
                    box_area = bw * bh

                    # 1. Must lie in the lower section of the camera frame
                    if gyc < h * POTHOLE_LOWER_Y_MIN_FRAC or gy2 < h * 0.45:
                        continue
                    # 2. Reject whole-printout border boxes or tiny specks
                    if bw > w * 0.62 or bh > h * 0.48 or box_area > frame_area * 0.22:
                        continue
                    if bw < w * 0.025 or bh < h * 0.02 or box_area < frame_area * 0.0015:
                        continue
                    # 3. Aspect ratio sanity check
                    aspect = bw / max(bh, 1.0)
                    if not (0.25 <= aspect <= 4.5):
                        continue

                    potholes.append({
                        "class_id": -1,
                        "class_name": "pothole",
                        "confidence": raw_conf,
                        "bbox": [gx1, gy1, gx2, gy2],
                        "type": "negative",
                    })
        except Exception:
            pass

        return potholes

    def detect(self, frame) -> List[Dict[str, Any]]:
        """Run object detection on a BGR numpy frame.

        Runs YOLOv10n for positive obstacles (cars, people, etc.) and the
        GPU fine-tuned pothole neural network for negative obstacles (potholes
        in the lower camera section).

        Args:
            frame: BGR numpy image array (H, W, 3).

        Returns:
            List of detection dicts:
            [{"class_id": int, "class_name": str, "confidence": float,
              "bbox": [x1, y1, x2, y2], "type": "positive" | "negative"}, ...]
        """
        if self.model is None or frame is None:
            return []

        results = self.model(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
        detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = results[0].names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()

            detections.append({
                "class_id": cls_id,
                "class_name": name,
                "confidence": conf,
                "bbox": coords,
                "type": "positive",
            })

        # 2. GPU fine-tuned YOLO pothole model detections (lower camera section)
        yolo_potholes = self._detect_yolo_potholes(frame)
        detections.extend(yolo_potholes)

        # 3. Classical CV pothole fallback ONLY when explicitly enabled AND no YOLO pothole model is loaded
        if self.enable_pothole_cv and self.pothole_model is None:
            cv_dets = cv_detect_potholes(frame)
            detections.extend(cv_dets)

        return detections
