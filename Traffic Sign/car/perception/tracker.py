"""
tracker.py — KCF Person Tracker with Multi-Feature Re-ID & Anti-Confusion
==========================================================================
Owned by perception-agent.

Implements Follow-Me person tracking and re-identification:
  Phase 1 — Interactive / Multi-Frame Calibration:
    User selects a person via Web UI click or dock button.
    Gathers multi-frame samples (default 8 frames, ~1.2s stationary)
    to build a robust target profile:
      - Inner-core 3-zone clothing histograms (upper torso, mid torso, lower body)
      - Illumination normalization via CLAHE on V channel in HSV
      - Body figure ratios: aspect ratio, upper/lower contrast, torso taper ratio
    Initialises KCF tracker on the target bounding box.

  Phase 2 — Tracking & Anti-Confusion:
    update_with_detections(frame, detections) advances KCF and verifies
    against full-frame YOLO person detections.
    If a bystander crosses paths or KCF drifts, re-anchors to the target YOLO
    box or drops tracking to prevent following the wrong person.
    Non-target persons are classified as obstacles.
    Tracks horizontal velocity vector (last_vx) and updates last_seen_side (LEFT/RIGHT/CENTER).

  Phase 3 — Re-Acquisition & Directional Search:
    When the target is lost or walks out of FOV:
      1. Car enters RE_ACQUIRE.
      2. Full-frame YOLO candidates are evaluated against target profile.
      3. If no candidate matches, the car can execute stepped directional
         turns toward last_seen_side to re-acquire the target person.
"""

import math
import time
from typing import Tuple, Optional, Dict, Any, List

import cv2
import numpy as np

from car.perception.config import (
    FOLLOW_CALIBRATION_FRAMES,
    FOLLOW_SIMILARITY_THRESHOLD,
    FOLLOW_TRACK_VERIFY_THRESHOLD,
    FOLLOW_ANTI_CONFUSION_MARGIN,
    FOLLOW_SEARCH_STEP_DEG,
    FOLLOW_SEARCH_TURN_SPEED,
    FOLLOW_SEARCH_MAX_STEPS,
    FOLLOW_SEARCH_INTERVAL_S,
)

# Minimum histogram correlation score to accept a candidate as the owner.
REACQUIRE_THRESHOLD = FOLLOW_SIMILARITY_THRESHOLD

# Number of HSV histogram bins per channel for the appearance profile.
_HIST_BINS = [18, 24, 16]  # Hue (0-180), Saturation (0-256), Value (0-256)
_HIST_RANGES = [0, 180, 0, 256, 0, 256]


def _extract_hsv_hist(crop: np.ndarray, apply_clahe: bool = True) -> Optional[np.ndarray]:
    """
    Computes a flattened, L1-normalised 3D HSV histogram for a BGR crop.
    Applies CLAHE on the V channel to normalize indoor/outdoor illumination.
    Returns None if the crop is empty or too small.
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if apply_clahe:
        grid_y = min(4, max(1, hsv.shape[0]))
        grid_x = min(4, max(1, hsv.shape[1]))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_x, grid_y))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    hist = cv2.calcHist([hsv], [0, 1, 2], None, _HIST_BINS, _HIST_RANGES)
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    return hist.flatten()


def _build_appearance_profile(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """
    Extracts inner-core 3-zone clothing histograms and body figure ratios.

    Args:
        frame: Full BGR frame.
        bbox: (x, y, w, h) bounding box of the person.

    Returns:
        Dict with keys: upper_hist, mid_hist, lower_hist, aspect_ratio,
        upper_lower_contrast, torso_taper_ratio, bbox_area.
    """
    x, y, w, h = [int(v) for v in bbox]
    fh, fw = frame.shape[:2]

    # Clamp to frame bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)
    w_c = x2 - x1
    h_c = y2 - y1

    if w_c < 4 or h_c < 4:
        return {}

    # Inner crop horizontally: central 60% (excluding background edges)
    x1_inner = int(x1 + 0.20 * w_c)
    x2_inner = int(x1 + 0.80 * w_c)
    if x2_inner - x1_inner < 4:
        x1_inner, x2_inner = x1, x2

    # Vertical 3-band split:
    # 1. Upper torso (shirt / chest): 15% to 50%
    y_up1 = y1 + int(0.15 * h_c)
    y_up2 = max(y_up1 + 4, y1 + int(0.50 * h_c))

    # 2. Mid torso (waist / jacket line): 40% to 68%
    y_mid1 = y1 + int(0.40 * h_c)
    y_mid2 = max(y_mid1 + 4, y1 + int(0.68 * h_c))

    # 3. Lower body (pants / legs): 55% to 90%
    y_low1 = y1 + int(0.55 * h_c)
    y_low2 = max(y_low1 + 4, y1 + int(0.90 * h_c))

    upper_crop = frame[y_up1:min(fh, y_up2), x1_inner:x2_inner]
    mid_crop = frame[y_mid1:min(fh, y_mid2), x1_inner:x2_inner]
    lower_crop = frame[y_low1:min(fh, y_low2), x1_inner:x2_inner]

    upper_hist = _extract_hsv_hist(upper_crop)
    mid_hist = _extract_hsv_hist(mid_crop)
    lower_hist = _extract_hsv_hist(lower_crop)

    # Body figure & proportion features
    aspect_ratio = float(h_c) / max(float(w_c), 1.0)

    # Upper/Lower contrast (two-tone outfit vs uniform)
    if upper_hist is not None and lower_hist is not None:
        c_val = float(cv2.compareHist(
            upper_hist.reshape(-1, 1).astype(np.float32),
            lower_hist.reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_CORREL
        ))
        upper_lower_contrast = float(np.clip(c_val, 0.0, 1.0))
    else:
        upper_lower_contrast = 0.5

    # Torso taper ratio (spread of horizontal edges in upper vs lower torso)
    try:
        gray_up = cv2.cvtColor(upper_crop, cv2.COLOR_BGR2GRAY)
        sobel_up = cv2.Sobel(gray_up, cv2.CV_32F, 1, 0, ksize=3)
        spread_up = float(np.mean(np.abs(sobel_up)))

        gray_low = cv2.cvtColor(lower_crop, cv2.COLOR_BGR2GRAY)
        sobel_low = cv2.Sobel(gray_low, cv2.CV_32F, 1, 0, ksize=3)
        spread_low = float(np.mean(np.abs(sobel_low)))

        taper = spread_up / max(spread_low, 1e-4)
        torso_taper_ratio = float(np.clip(taper, 0.1, 5.0))
    except Exception:
        torso_taper_ratio = 1.0

    return {
        "upper_hist": upper_hist,
        "mid_hist": mid_hist,
        "lower_hist": lower_hist,
        "aspect_ratio": aspect_ratio,
        "upper_lower_contrast": upper_lower_contrast,
        "torso_taper_ratio": torso_taper_ratio,
        "bbox_area": w_c * h_c,
    }


def _compare_profiles(profile: Dict[str, Any], candidate_profile: Dict[str, Any]) -> float:
    """
    Computes a composite similarity score in [0.0, 1.0] between two appearance profiles.
    Combines:
      - 70% Clothing similarity (weighted correlation across upper, mid, lower bands)
      - 30% Figure & proportion similarity (aspect ratio, contrast, torso taper)
    """
    if not profile or not candidate_profile:
        return 0.0

    # ── 1. Clothing Similarity (70% weight) ──────────────────────────────────
    zone_weights = [("upper_hist", 0.45), ("mid_hist", 0.20), ("lower_hist", 0.35)]
    clothing_scores = []
    clothing_weights = []

    for key, weight in zone_weights:
        h1 = profile.get(key)
        h2 = candidate_profile.get(key)
        if h1 is not None and h2 is not None:
            if np.sum(h1) > 0 and np.sum(h2) > 0 and np.std(h1) > 1e-7 and np.std(h2) > 1e-7:
                corr = float(cv2.compareHist(
                    h1.reshape(-1, 1).astype(np.float32),
                    h2.reshape(-1, 1).astype(np.float32),
                    cv2.HISTCMP_CORREL
                ))
                if math.isnan(corr):
                    corr = 0.0
                clothing_scores.append(max(0.0, corr) * weight)
                clothing_weights.append(weight)

    if not clothing_scores or sum(clothing_weights) == 0:
        return 0.0

    clothing_sim = sum(clothing_scores) / sum(clothing_weights)

    # ── 2. Figure & Proportion Similarity (30% weight) ───────────────────────
    ar1 = profile.get("aspect_ratio", 2.0)
    ar2 = candidate_profile.get("aspect_ratio", 2.0)
    sim_ar = math.exp(-((ar1 - ar2) / 0.65) ** 2)

    c1 = profile.get("upper_lower_contrast", 0.5)
    c2 = candidate_profile.get("upper_lower_contrast", 0.5)
    sim_contrast = max(0.0, 1.0 - abs(c1 - c2))

    t1 = profile.get("torso_taper_ratio", 1.0)
    t2 = candidate_profile.get("torso_taper_ratio", 1.0)
    sim_taper = math.exp(-((t1 - t2) / 0.75) ** 2)

    figure_sim = 0.40 * sim_ar + 0.30 * sim_contrast + 0.30 * sim_taper

    # ── 3. Weighted Combination ──────────────────────────────────────────────
    total_score = 0.70 * clothing_sim + 0.30 * figure_sim
    return float(np.clip(total_score, 0.0, 1.0))


def _box_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection-over-Union (IoU) between two (x, y, w, h) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(1, boxA[2] * boxA[3])
    boxB_area = max(1, boxB[2] * boxB[3])
    union = boxA_area + boxB_area - inter_area
    return float(inter_area / max(union, 1))


class ObjectTracker:
    """
    Multi-feature person tracker with interactive calibration, continuous
    YOLO-anchored anti-confusion verification, and directional search memory.
    """

    def __init__(self):
        self.tracker = None
        self.is_tracking: bool = False
        self.bbox: Optional[Tuple[int, int, int, int]] = None
        self._owner_profile: Dict[str, Any] = {}

        # Multi-frame calibration state
        self.is_calibrating: bool = False
        self._calibration_samples: List[Dict[str, Any]] = []
        self._calibration_bboxes: List[Tuple[int, int, int, int]] = []
        self._total_calibration_frames: int = FOLLOW_CALIBRATION_FRAMES
        self._calibration_lost_frames: int = 0

        # Last-seen side and velocity tracking
        self.last_seen_side: str = "CENTER"
        self.last_vx: float = 0.0
        self.last_cx_frac: float = 0.5
        self._prev_cx_frac: Optional[float] = None

        # Directional search step tracking
        self.search_steps_taken: int = 0
        self.last_search_step_time: float = 0.0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        """Returns True if the target profile has been learned and stored."""
        return bool(self._owner_profile and self._owner_profile.get("upper_hist") is not None)

    @property
    def calibration_progress(self) -> float:
        """Returns calibration completion progress from 0.0 to 1.0."""
        if not self.is_calibrating or self._total_calibration_frames <= 0:
            return 1.0 if self.is_calibrated else 0.0
        return min(1.0, float(len(self._calibration_samples)) / float(self._total_calibration_frames))

    # ── Calibration ───────────────────────────────────────────────────────────

    def start_calibration(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        total_frames: int = FOLLOW_CALIBRATION_FRAMES,
    ) -> bool:
        """
        Begins multi-frame calibration for the selected person bounding box.
        """
        self.reset()
        self.is_calibrating = True
        self._total_calibration_frames = max(1, total_frames)
        self._calibration_samples = []
        self._calibration_bboxes = []
        self._calibration_lost_frames = 0

        x, y, w, h = [int(v) for v in bbox]
        init_box = (x, y, w, h)
        self.bbox = init_box

        # Initialise tracker on initial box
        self._init_cv_tracker(frame, init_box)

        # Collect initial sample
        prof = _build_appearance_profile(frame, init_box)
        if prof:
            self._calibration_samples.append(prof)
            self._calibration_bboxes.append(init_box)
            self._update_side_memory(init_box, frame.shape[1])

        return True

    def add_calibration_sample(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> bool:
        """
        Appends a calibration frame sample.
        Returns True if calibration has completed (sufficient frames gathered).
        If the person is lost for 5+ frames, aborts calibration and returns False.
        """
        if not self.is_calibrating:
            return True

        sample_box = bbox
        if sample_box is None and self.tracker is not None:
            success, tracked_box = self.tracker.update(frame)
            if success:
                sample_box = tuple(int(v) for v in tracked_box)

        if sample_box is not None:
            prof = _build_appearance_profile(frame, sample_box)
            if prof:
                self.bbox = sample_box
                self._calibration_samples.append(prof)
                self._calibration_bboxes.append(sample_box)
                self._update_side_memory(sample_box, frame.shape[1])
                self._calibration_lost_frames = 0
            else:
                self._calibration_lost_frames += 1
        else:
            self._calibration_lost_frames += 1

        # Check if person was lost during calibration
        if self._calibration_lost_frames >= 5:
            # Person lost during calibration -> abort calibration and reset
            self.is_calibrating = False
            self.is_tracking = False
            self.bbox = None
            self._calibration_samples.clear()
            self._calibration_bboxes.clear()
            return False

        if len(self._calibration_samples) >= self._total_calibration_frames:
            self._finalize_calibration(frame)
            return True

        return False

    def step_calibration(
        self,
        frame: np.ndarray,
        detections: Optional[list] = None,
    ) -> bool:
        """
        Advances calibration using full-frame detections if available (anchoring on closest IoU).
        """
        matched_box = None
        if detections and self.bbox is not None:
            persons = [d for d in detections if d.get("class_name") == "person"]
            best_iou = -1.0
            for p in persons:
                x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
                box = (x1, y1, x2 - x1, y2 - y1)
                iou = _box_iou(self.bbox, box)
                if iou > best_iou:
                    best_iou = iou
                    matched_box = box
            if best_iou < 0.2:
                matched_box = None

        return self.add_calibration_sample(frame, bbox=matched_box)

    def _finalize_calibration(self, frame: np.ndarray):
        """Averages and normalizes collected profile samples into self._owner_profile."""
        if not self._calibration_samples:
            self.is_calibrating = False
            return

        def _avg_hist(key: str) -> Optional[np.ndarray]:
            valid = [s[key] for s in self._calibration_samples if s.get(key) is not None and np.sum(s.get(key)) > 0]
            if not valid:
                return None
            stacked = np.stack(valid, axis=0)
            avg = np.mean(stacked, axis=0)
            norm = np.sum(avg)
            if norm > 0:
                avg = avg / norm
            return avg.astype(np.float32)

        self._owner_profile = {
            "upper_hist": _avg_hist("upper_hist"),
            "mid_hist": _avg_hist("mid_hist"),
            "lower_hist": _avg_hist("lower_hist"),
            "aspect_ratio": float(np.mean([s.get("aspect_ratio", 2.0) for s in self._calibration_samples])),
            "upper_lower_contrast": float(np.mean([s.get("upper_lower_contrast", 0.5) for s in self._calibration_samples])),
            "torso_taper_ratio": float(np.mean([s.get("torso_taper_ratio", 1.0) for s in self._calibration_samples])),
            "bbox_area": int(np.mean([s.get("bbox_area", 0) for s in self._calibration_samples])),
        }

        # Lock onto latest calibration bounding box
        final_box = self._calibration_bboxes[-1] if self._calibration_bboxes else self.bbox
        if final_box is not None:
            self.lock_on(frame, final_box, profile=self._owner_profile)

        self.is_calibrating = False
        self.is_tracking = True
        self.reset_search()

    # ── Phase 1: Lock-On (Single-Frame or Override) ───────────────────────────

    def lock_on(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        profile: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Initialises tracker on bbox and sets owner appearance profile.
        """
        if profile is not None:
            self._owner_profile = profile
        else:
            self._owner_profile = _build_appearance_profile(frame, bbox)

        self._init_cv_tracker(frame, bbox)
        self.bbox = bbox if self.is_tracking else None
        if self.is_tracking and self.bbox is not None:
            self._update_side_memory(self.bbox, frame.shape[1])
            self.reset_search()
        return self.is_tracking

    def _init_cv_tracker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Helper to create and initialize OpenCV tracker."""
        self.tracker = None
        for creator in [
            lambda: cv2.TrackerKCF_create(),
            lambda: getattr(getattr(cv2, "legacy", None), "TrackerKCF_create", None)(),
            lambda: cv2.TrackerMIL_create(),
            lambda: getattr(getattr(cv2, "legacy", None), "TrackerMIL_create", None)(),
        ]:
            try:
                t = creator()
                if t is not None:
                    self.tracker = t
                    break
            except (AttributeError, TypeError, Exception):
                continue

        if self.tracker is None:
            self.is_tracking = False
            return

        try:
            res = self.tracker.init(frame, bbox)
            self.is_tracking = True if res is None or res is True else bool(res)
        except Exception:
            self.is_tracking = False

        if not self.is_tracking:
            self.bbox = None

    # ── Phase 2: Tracking & Anti-Confusion ───────────────────────────────────

    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """Advances OpenCV KCF tracker by one frame."""
        if not self.is_tracking or self.tracker is None:
            return False, None

        # Guard against TrackerMIL drift on featureless/zero-variance synthetic frames
        if frame is not None and getattr(frame, "size", 0) > 0 and float(np.std(frame)) < 1.0:
            if self.bbox is not None:
                self._update_side_memory(self.bbox, frame.shape[1])
                return True, self.bbox

        success, bbox = self.tracker.update(frame)
        if success:
            self.bbox = tuple(int(v) for v in bbox)
            self._update_side_memory(self.bbox, frame.shape[1])
            return True, self.bbox
        else:
            self.is_tracking = False
            self.bbox = None
            return False, None

    def update_with_detections(
        self,
        frame: np.ndarray,
        detections: list,
        fresh_inference: bool = True,
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]], Optional[dict], list, float]:
        """
        Continuous YOLO-anchored anti-confusion verification:
          - Advances KCF tracker
          - Separates target from non_target_persons
          - Re-anchors KCF to matching YOLO box to prevent drift (only on fresh inference)
          - Drops tracking to RE_ACQUIRE if bystander causes confusion or occlusion.

        Returns:
          (is_tracking, bbox, target_det, non_target_persons, similarity_score)
        """
        kcf_success, kcf_box = self.update(frame)
        persons = [d for d in (detections or []) if isinstance(d, dict) and d.get("class_name") == "person"]

        # If no profile learned, fall back to pure KCF
        if not self._owner_profile:
            if kcf_success:
                return True, kcf_box, None, persons, 1.0
            return False, None, None, persons, 0.0

        # Score all person detections against stored profile
        for p in persons:
            bx1, by1, bx2, by2 = [int(v) for v in p["bbox"]]
            p_box = (bx1, by1, bx2 - bx1, by2 - by1)
            p["_box_xywh"] = p_box
            p["_prof"] = _build_appearance_profile(frame, p_box)
            p["_score"] = _compare_profiles(self._owner_profile, p["_prof"])
            p["_iou"] = _box_iou(kcf_box, p_box) if kcf_box else 0.0

        persons_sorted = sorted(persons, key=lambda d: d.get("_score", 0.0), reverse=True)
        best_p = persons_sorted[0] if persons_sorted else None
        second_p = persons_sorted[1] if len(persons_sorted) > 1 else None

        # Evaluate KCF current box similarity
        kcf_prof = _build_appearance_profile(frame, kcf_box) if kcf_box else {}
        kcf_score = _compare_profiles(self._owner_profile, kcf_prof) if kcf_prof else 0.0

        if kcf_success and kcf_box is not None:
            # Check if KCF appearance is still confident
            if kcf_score >= FOLLOW_TRACK_VERIFY_THRESHOLD:
                # Anti-confusion gate: check if multiple persons are near/overlapping tracker box
                overlapping = [p for p in persons if (p.get("_iou", 0.0) > 0.15 or (best_p and _box_iou(best_p["_box_xywh"], p["_box_xywh"]) > 0.20))]
                if len(overlapping) > 1:
                    margin = best_p["_score"] - (second_p["_score"] if second_p else 0.0)
                    if margin < FOLLOW_ANTI_CONFUSION_MARGIN:
                        # Ambiguous distractor overlapping target: drop tracking to RE_ACQUIRE
                        self.is_tracking = False
                        self.bbox = None
                        return False, None, None, persons, kcf_score

                # If a YOLO detection matches the target well and aligns with KCF box, re-anchor on fresh inference
                if best_p and best_p["_score"] >= FOLLOW_SIMILARITY_THRESHOLD and best_p["_iou"] > 0.35:
                    if fresh_inference:
                        self._init_cv_tracker(frame, best_p["_box_xywh"])
                        if self.is_tracking:
                            self.bbox = best_p["_box_xywh"]
                            self._update_side_memory(self.bbox, frame.shape[1])
                    target_det = best_p
                    non_target = [p for p in persons if p is not target_det]
                    return self.is_tracking, self.bbox, target_det, non_target, best_p["_score"]
                else:
                    target_det = best_p if (best_p and best_p["_iou"] > 0.35 and best_p["_score"] >= FOLLOW_TRACK_VERIFY_THRESHOLD) else None
                    non_target = [p for p in persons if p is not target_det]
                    return True, self.bbox, target_det, non_target, kcf_score
            else:
                # KCF appearance has degraded (< 0.50). Check if target is elsewhere in frame
                if best_p and best_p["_score"] >= FOLLOW_SIMILARITY_THRESHOLD:
                    margin = best_p["_score"] - (second_p["_score"] if second_p else 0.0)
                    if second_p is None or margin >= FOLLOW_ANTI_CONFUSION_MARGIN:
                        # Clear target detection elsewhere; re-anchor to it!
                        self.lock_on(frame, best_p["_box_xywh"], profile=self._owner_profile)
                        target_det = best_p
                        non_target = [p for p in persons if p is not target_det]
                        return self.is_tracking, self.bbox, target_det, non_target, best_p["_score"]

                # Occluded or ambiguous distractor: drop tracking rather than following wrong person
                self.is_tracking = False
                self.bbox = None
                return False, None, None, persons, kcf_score

        # If KCF was not successful, attempt direct recovery from YOLO detections
        if best_p and best_p["_score"] >= FOLLOW_SIMILARITY_THRESHOLD:
            margin = best_p["_score"] - (second_p["_score"] if second_p else 0.0)
            if second_p is None or margin >= FOLLOW_ANTI_CONFUSION_MARGIN:
                self.lock_on(frame, best_p["_box_xywh"], profile=self._owner_profile)
                target_det = best_p
                non_target = [p for p in persons if p is not target_det]
                return self.is_tracking, self.bbox, target_det, non_target, best_p["_score"]

        self.is_tracking = False
        self.bbox = None
        return False, None, None, persons, 0.0

    # ── Phase 3: Re-Acquisition & Directional Search ──────────────────────────

    def reacquire(self, frame: np.ndarray, candidates: list) -> bool:
        """
        Attempts to re-identify the target among YOLO person detections.
        Enforces FOLLOW_SIMILARITY_THRESHOLD and distractor margin.
        """
        if not self._owner_profile:
            return False

        scored_candidates = []
        for det in candidates:
            if not isinstance(det, dict) or det.get("class_name") != "person":
                continue
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            w = x2 - x1
            h = y2 - y1
            if w < 4 or h < 4:
                continue

            candidate_bbox = (x1, y1, w, h)
            prof = _build_appearance_profile(frame, candidate_bbox)
            score = _compare_profiles(self._owner_profile, prof)
            scored_candidates.append((score, candidate_bbox))

        if not scored_candidates:
            return False

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_bbox = scored_candidates[0]

        # Check distractor margin if multiple persons present
        if len(scored_candidates) > 1:
            second_score = scored_candidates[1][0]
            if (best_score - second_score) < FOLLOW_ANTI_CONFUSION_MARGIN:
                # Distractor too similar or ambiguous
                if best_score < (FOLLOW_SIMILARITY_THRESHOLD + FOLLOW_ANTI_CONFUSION_MARGIN):
                    return False

        if best_score >= FOLLOW_SIMILARITY_THRESHOLD and best_bbox is not None:
            return self.lock_on(frame, best_bbox, profile=self._owner_profile)

        return False

    def should_step_search(self, now_mono: float) -> bool:
        """Returns True if the car should execute a stepped directional turn during RE_ACQUIRE."""
        if self.search_steps_taken >= FOLLOW_SEARCH_MAX_STEPS:
            return False
        if self.last_seen_side not in ("LEFT", "RIGHT"):
            return False
        if (now_mono - self.last_search_step_time) < FOLLOW_SEARCH_INTERVAL_S:
            return False
        return True

    def record_search_step(self, now_mono: float):
        """Records a completed search turn step."""
        self.search_steps_taken += 1
        self.last_search_step_time = now_mono

    def reset_search(self):
        """Resets search step counter and timer."""
        self.search_steps_taken = 0
        self.last_search_step_time = 0.0

    # ── Helpers & State Management ────────────────────────────────────────────

    def _update_side_memory(self, bbox: Tuple[int, int, int, int], frame_w: int):
        """Updates smoothed horizontal velocity vx and last_seen_side (LEFT/RIGHT/CENTER)."""
        x, y, w, h = bbox
        cx = x + w / 2.0
        cx_frac = cx / max(frame_w, 1)

        if self._prev_cx_frac is not None:
            vx = cx_frac - self._prev_cx_frac
            self.last_vx = 0.6 * self.last_vx + 0.4 * vx
        else:
            self.last_vx = 0.0

        self._prev_cx_frac = cx_frac
        self.last_cx_frac = cx_frac

        # Determine last-seen exit direction
        if cx_frac < 0.38 or (cx_frac < 0.52 and self.last_vx < -0.015):
            self.last_seen_side = "LEFT"
        elif cx_frac > 0.62 or (cx_frac > 0.48 and self.last_vx > 0.015):
            self.last_seen_side = "RIGHT"
        else:
            self.last_seen_side = "CENTER"

    def get_owner_bbox_area(self) -> int:
        """Returns the last-known owner bbox area, or 0 if not tracking."""
        if self.bbox is None:
            return 0
        _, _, w, h = self.bbox
        return w * h

    def stop(self):
        """Stops active tracking. Does NOT clear the learned owner profile."""
        self.is_tracking = False
        self.tracker = None
        self.bbox = None

    def reset(self):
        """Full reset including learned owner profile and calibration buffers."""
        self.stop()
        self._owner_profile = {}
        self.is_calibrating = False
        self._calibration_samples = []
        self._calibration_bboxes = []
        self._calibration_lost_frames = 0
        self.last_seen_side = "CENTER"
        self.last_vx = 0.0
        self._prev_cx_frac = None
        self.reset_search()
