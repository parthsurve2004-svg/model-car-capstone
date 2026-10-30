"""
lane_detector.py — Classical Chart-Paper Lane Detection & Pure Pursuit Steering
=================================================================================
Owned by perception-agent.

Implements real-time 2-line chart paper tracking on the Raspberry Pi 5:
  1. ROI scanline / perspective extraction (lower 55% of camera frame).
  2. Morphological black-hat + adaptive contrast thresholding for dark marker/tape
     lines on light chart paper (with optional HSV tape presets for yellow/blue/white).
  3. YOLO obstacle & D40 pothole bounding-box exclusion masking so dark printed
     potholes placed inside the lane do not pull or corrupt curve fitting.
  4. Histogram base peak detection + 9-step sliding window outlier rejection.
  5. Robust 2nd-order polynomial curve fitting (x/W = a*(y/H)^2 + b*(y/H) + c)
     with iterative MAD (Median Absolute Deviation) outlier pruning.
  6. Ego-centerline calculation with single-line fallback recovery (SINGLE_LEFT /
     SINGLE_RIGHT) for sharp turns where one line leaves the camera FOV.
  7. Pure pursuit + tangent heading steering angle calculation (skid-steer degrees).
  8. HUD overlay geometry (drivable green corridor, line polylines, target reticle).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from car.perception.config import (
    LANE_ROI_Y_MIN_FRAC,
    LANE_LOOKAHEAD_Y_FRAC,
    LANE_DEFAULT_WIDTH_FRAC,
    LANE_EMA_ALPHA,
    LANE_MAX_JUMP_FRAC,
    LANE_LOST_MAX_FRAMES,
    LANE_MAX_STEER_DEG,
    LANE_STEER_DEADBAND_DEG,
    LANE_COLOR_MODE,
)


@dataclass
class LaneCurve:
    """Represents a fitted 2nd-order polynomial lane boundary curve."""
    # Normalized model: x/W = a*(y/H)^2 + b*(y/H) + c
    coefficients: np.ndarray  # [a, b, c]
    ymin: float               # Normalized vertical span start in [0, 1]
    ymax: float               # Normalized vertical span end in [0, 1]
    confidence: float = 1.0
    residual_px: float = 0.0

    def x(self, y: float) -> float:
        """Returns normalized x (0..1) for a given normalized y (0..1)."""
        return float(np.polyval(self.coefficients, y))

    def points(self, width: int, height: int, n: int = 40) -> np.ndarray:
        """Returns (N, 2) pixel coordinates [x_px, y_px]."""
        ys = np.linspace(self.ymin, self.ymax, n)
        xs = np.polyval(self.coefficients, ys) * width
        return np.column_stack([xs, ys * height]).astype(np.int32)


@dataclass
class LaneDetectionResult:
    """Complete result of lane analysis on a single frame."""
    status: str                                  # "DUAL_LOCKED", "SINGLE_LEFT", "SINGLE_RIGHT", "LANE_LOST"
    left_line: Optional[LaneCurve] = None
    right_line: Optional[LaneCurve] = None
    center_coeffs: Optional[np.ndarray] = None   # [a, b, c] for centerline x/W = f(y/H)
    steer_angle_deg: float = 0.0                 # Target steering correction (-left, +right)
    lateral_offset_pct: float = 0.0              # Offset from center (-50% to +50%)
    lane_width_frac: float = LANE_DEFAULT_WIDTH_FRAC
    lookahead_point: Tuple[int, int] = (320, 326)
    corridor_pts: Optional[np.ndarray] = None    # Polygon points for cv2.fillPoly
    left_pts: Optional[np.ndarray] = None        # Nx2 pixel points
    right_pts: Optional[np.ndarray] = None       # Nx2 pixel points
    center_pts: Optional[np.ndarray] = None      # Nx2 pixel points
    binary_mask: Optional[np.ndarray] = None     # ROI binary mask
    lost_frames: int = 0


def fit_poly_mad(y_norm: np.ndarray, x_norm: np.ndarray, frame_w: int) -> Optional[Tuple[np.ndarray, float, float, float]]:
    """
    Fits x = a*y^2 + b*y + c with iterative Median Absolute Deviation (MAD) outlier pruning.
    Adapted from Capstone geometry pipeline.
    """
    if len(x_norm) < 8 or not np.isfinite(x_norm).all() or not np.isfinite(y_norm).all():
        return None
    if np.ptp(y_norm) < 0.12:  # Must span at least 12% vertical frame height
        return None

    keep = np.ones(len(x_norm), dtype=bool)
    for _ in range(4):
        if keep.sum() < 8:
            return None
        try:
            c = np.polyfit(y_norm[keep], x_norm[keep], 2)
        except (np.linalg.LinAlgError, ValueError):
            return None
        residual = np.abs(np.polyval(c, y_norm) - x_norm) * frame_w
        mad = float(np.median(residual[keep]))
        keep = residual < max(5.0, 2.8 * mad)

    if keep.sum() < 8:
        return None

    try:
        c = np.polyfit(y_norm[keep], x_norm[keep], 2)
    except (np.linalg.LinAlgError, ValueError):
        return None

    rms = float(np.sqrt(np.mean(((np.polyval(c, y_norm[keep]) - x_norm[keep]) * frame_w) ** 2)))
    if rms > 0.06 * frame_w:  # Reject overly noisy fits
        return None

    return c, float(y_norm[keep].min()), float(y_norm[keep].max()), rms


class ChartLaneDetector:
    """
    Classical vision lane detector tailored for autonomous chart-paper track following.
    Runs at 30 Hz (~2.5ms on Pi 5) in parallel with asynchronous YOLO obstacle/pothole models.
    """

    def __init__(
        self,
        roi_y_min_frac: float = LANE_ROI_Y_MIN_FRAC,
        lookahead_y_frac: float = LANE_LOOKAHEAD_Y_FRAC,
        default_width_frac: float = LANE_DEFAULT_WIDTH_FRAC,
        ema_alpha: float = LANE_EMA_ALPHA,
        max_jump_frac: float = LANE_MAX_JUMP_FRAC,
        lost_max_frames: int = LANE_LOST_MAX_FRAMES,
        max_steer_deg: float = LANE_MAX_STEER_DEG,
        steer_deadband_deg: float = LANE_STEER_DEADBAND_DEG,
        color_mode: str = LANE_COLOR_MODE,
        kp_steer: float = 145.0,
        kd_steer: float = 28.0,
    ):
        self.roi_y_min_frac = roi_y_min_frac
        self.lookahead_y_frac = lookahead_y_frac
        self.default_width_frac = default_width_frac
        self.learned_lane_width = default_width_frac
        self.ema_alpha = ema_alpha
        self.max_jump_frac = max_jump_frac
        self.lost_max_frames = lost_max_frames
        self.max_steer_deg = max_steer_deg
        self.steer_deadband_deg = steer_deadband_deg
        self.color_mode = color_mode.lower()
        self.kp_steer = kp_steer
        self.kd_steer = kd_steer

        # Internal state memory
        self.prev_left_line: Optional[LaneCurve] = None
        self.prev_right_line: Optional[LaneCurve] = None
        self.prev_center_coeffs: Optional[np.ndarray] = None
        self.lost_frames: int = 0

    def reset(self):
        """Resets tracking memory and temporal filters."""
        self.prev_left_line = None
        self.prev_right_line = None
        self.prev_center_coeffs = None
        self.learned_lane_width = self.default_width_frac
        self.lost_frames = 0

    def set_color_mode(self, mode: str):
        """Switches lane line color preset (dark, yellow, blue, white)."""
        valid = {"dark", "yellow", "blue", "white"}
        m = mode.lower().strip()
        if m in valid:
            self.color_mode = m

    def _extract_binary_mask(
        self,
        frame: np.ndarray,
        exclude_boxes: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Extracts binary mask (255 for lane line pixels, 0 elsewhere) in the lower ROI.
        Zeros out bounding boxes of detected obstacles and potholes so they don't corrupt lines.
        """
        h, w = frame.shape[:2]
        roi_y_start = int(h * self.roi_y_min_frac)
        roi = frame[roi_y_start:, :]
        if roi.size == 0:
            return np.zeros((1, 1), dtype=np.uint8), roi_y_start

        roi_h, roi_w = roi.shape[:2]

        if self.color_mode == "dark":
            # Dark/black marker or tape on light/white chart paper
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Morphological black-hat highlights dark structures on bright backgrounds
            k_size = max(9, min(25, int(w * 0.025)) | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
            _, bh_thresh = cv2.threshold(blackhat, 24, 255, cv2.THRESH_BINARY)

            # Adaptive threshold on inverted grayscale for indoor illumination robustness
            adapt = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 9
            )

            # Combined mask ensures strong lines under varying ambient light & shadows
            mask = cv2.bitwise_or(bh_thresh, adapt)

        elif self.color_mode == "yellow":
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([16, 65, 65]), np.array([38, 255, 255]))

        elif self.color_mode == "blue":
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([92, 70, 60]), np.array([130, 255, 255]))

        elif self.color_mode == "white":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(blur, 195, 255, cv2.THRESH_BINARY)

        else:
            # Fallback to dark
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

        # Morphological opening removes small specks
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, clean_kernel)

        # ── Zero out active obstacle and pothole printout bounding boxes ────────
        if exclude_boxes:
            pad = 10
            for det in exclude_boxes:
                if not isinstance(det, dict):
                    continue
                bbox = det.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                bx1, by1, bx2, by2 = [int(v) for v in bbox]
                # Map full-frame Y into ROI coordinates with non-negative bounds
                roi_by1 = max(0, min(roi_h, by1 - roi_y_start - pad))
                roi_by2 = max(0, min(roi_h, by2 - roi_y_start + pad))
                roi_bx1 = max(0, min(roi_w, bx1 - pad))
                roi_bx2 = max(0, min(roi_w, bx2 + pad))

                if roi_by2 > roi_by1 and roi_bx2 > roi_bx1:
                    mask[roi_by1:roi_by2, roi_bx1:roi_bx2] = 0

        return mask, roi_y_start

    def _find_base_peaks(self, mask: np.ndarray, frame_w: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Finds Left and Right lane line base X coordinates using bottom-half histogram peaks.
        Uses full-width peak extraction, temporal memory matching, and geometric lane width constraints
        to reliably disambiguate left/right boundaries even when the car is off-center or turning.
        """
        roi_h = mask.shape[0]
        bottom_half = mask[int(roi_h * 0.45):, :]
        hist = np.sum(bottom_half, axis=0).astype(np.float32)

        # 1D moving average smoothing
        kernel_size = max(7, min(25, int(frame_w * 0.025)) | 1)
        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        smooth_hist = np.convolve(hist, kernel, mode="same")

        min_peak = 255.0 * 2.5
        min_sep = int(frame_w * 0.16)

        # Extract local maxima across entire ROI width
        candidates = []
        x_min = int(0.015 * frame_w)
        x_max = int(0.985 * frame_w)
        for x in range(x_min + 1, x_max - 1):
            val = smooth_hist[x]
            if val > min_peak and val >= smooth_hist[x - 1] and val >= smooth_hist[x + 1]:
                candidates.append((val, x))

        candidates.sort(key=lambda item: item[0], reverse=True)

        # Non-maximum suppression across peak locations
        pruned_peaks: List[int] = []
        for val, x in candidates:
            if not any(abs(x - p) < min_sep for p in pruned_peaks):
                pruned_peaks.append(x)

        pruned_peaks.sort()

        expected_lane_w = int(self.learned_lane_width * frame_w)
        prev_l = int(self.prev_left_line.x(1.0) * frame_w) if self.prev_left_line else None
        prev_r = int(self.prev_right_line.x(1.0) * frame_w) if self.prev_right_line else None

        left_base: Optional[int] = None
        right_base: Optional[int] = None

        if len(pruned_peaks) >= 2:
            # Search for best pair whose separation matches expected lane width
            best_pair = None
            best_score = float("inf")
            for i in range(len(pruned_peaks)):
                for j in range(i + 1, len(pruned_peaks)):
                    p1, p2 = pruned_peaks[i], pruned_peaks[j]
                    sep = p2 - p1
                    if int(0.35 * expected_lane_w) <= sep <= int(1.65 * expected_lane_w):
                        score = abs(sep - expected_lane_w)
                        if prev_l is not None:
                            score += 0.5 * abs(p1 - prev_l)
                        if prev_r is not None:
                            score += 0.5 * abs(p2 - prev_r)
                        if score < best_score:
                            best_score = score
                            best_pair = (p1, p2)

            if best_pair is not None:
                left_base, right_base = best_pair
            else:
                p1, p2 = pruned_peaks[0], pruned_peaks[1]
                if p1 > p2:
                    p1, p2 = p2, p1
                left_base, right_base = p1, p2

        elif len(pruned_peaks) == 1:
            p = pruned_peaks[0]
            if prev_l is not None and prev_r is not None:
                if abs(p - prev_l) <= abs(p - prev_r):
                    left_base = p
                else:
                    right_base = p
            elif prev_l is not None:
                if abs(p - prev_l) <= self.max_jump_frac * frame_w * 1.5:
                    left_base = p
                elif p > prev_l + 0.28 * frame_w:
                    right_base = p
                else:
                    left_base = p
            elif prev_r is not None:
                if abs(p - prev_r) <= self.max_jump_frac * frame_w * 1.5:
                    right_base = p
                elif p < prev_r - 0.28 * frame_w:
                    left_base = p
                else:
                    right_base = p
            else:
                # Cold start single peak: assign based on center half
                if p >= int(0.50 * frame_w):
                    right_base = p
                else:
                    left_base = p

        else:
            if prev_l is not None:
                left_base = prev_l
            if prev_r is not None:
                right_base = prev_r

        return left_base, right_base

    def _sliding_window_tracking(
        self,
        mask: np.ndarray,
        base_x: int,
        frame_w: int,
        frame_h: int,
        roi_y_start: int,
    ) -> Optional[LaneCurve]:
        """
        Executes sliding window search upward from base_x and fits a 2nd-order polynomial.
        Rejects windows where blob width exceeds 12% of frame width.
        """
        roi_h = mask.shape[0]
        n_windows = 9
        win_h = roi_h // n_windows
        win_margin = max(24, int(frame_w * 0.08))
        min_recenter_pix = 35
        max_blob_w = int(frame_w * 0.12)

        curr_x = base_x
        nonzero_y, nonzero_x = mask.nonzero()
        line_inds = []

        for w_idx in range(n_windows):
            y_low = roi_h - (w_idx + 1) * win_h
            y_high = roi_h - w_idx * win_h
            x_low = max(0, curr_x - win_margin)
            x_high = min(frame_w, curr_x + win_margin)

            good_inds = (
                (nonzero_y >= y_low) & (nonzero_y < y_high) &
                (nonzero_x >= x_low) & (nonzero_x < x_high)
            ).nonzero()[0]

            if len(good_inds) > 0:
                xs = nonzero_x[good_inds]
                # Reject window if width is too wide (horizontal boundary or noise block)
                if np.ptp(xs) <= max_blob_w:
                    line_inds.append(good_inds)
                    if len(good_inds) > min_recenter_pix:
                        curr_x = int(np.mean(xs))

        if not line_inds:
            return None

        all_inds = np.concatenate(line_inds)
        if len(all_inds) < 8:
            return None

        # Convert to full-frame normalized coords: y/H in [0, 1], x/W in [0, 1]
        x_norm = nonzero_x[all_inds] / float(frame_w)
        y_norm = (nonzero_y[all_inds] + roi_y_start) / float(frame_h)

        fit = fit_poly_mad(y_norm, x_norm, frame_w)
        if fit is None:
            return None

        c, ymin, ymax, rms = fit
        return LaneCurve(coefficients=c, ymin=ymin, ymax=ymax, confidence=1.0, residual_px=rms)

    def detect(
        self,
        frame: np.ndarray,
        exclude_boxes: Optional[List[Dict[str, Any]]] = None,
    ) -> LaneDetectionResult:
        """
        Runs full lane detection pipeline on frame:
          - Mask extraction + obstacle/pothole exclusion
          - Sliding window polynomial curve fitting
          - Dual-line & single-line fallback recovery
          - Pure pursuit steering angle calculation
          - HUD overlay geometry generation
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return LaneDetectionResult(status="LANE_LOST", lost_frames=self.lost_frames + 1)

        frame_h, frame_w = frame.shape[:2]
        mask, roi_y_start = self._extract_binary_mask(frame, exclude_boxes)

        left_base, right_base = self._find_base_peaks(mask, frame_w)

        left_line: Optional[LaneCurve] = None
        right_line: Optional[LaneCurve] = None

        if left_base is not None:
            left_line = self._sliding_window_tracking(mask, left_base, frame_w, frame_h, roi_y_start)
        if right_base is not None:
            right_line = self._sliding_window_tracking(mask, right_base, frame_w, frame_h, roi_y_start)

        # Validate geometry if both lines are present
        if left_line is not None and right_line is not None:
            y_ref = self.lookahead_y_frac
            x_l = left_line.x(y_ref)
            x_r = right_line.x(y_ref)
            if x_l > x_r:
                left_line, right_line = right_line, left_line
                x_l, x_r = x_r, x_l

            w_meas = x_r - x_l
            # Check for sanity: right must be right of left, and separation reasonable
            if w_meas <= 0.12 or w_meas >= 0.95:
                # One of the lines is invalid; keep the one with lower residual
                if left_line.residual_px <= right_line.residual_px:
                    right_line = None
                else:
                    left_line = None

        # Disambiguate single-line identification on cold start via perspective convergence
        y_bot = 0.95
        y_top = self.roi_y_min_frac + 0.05
        if left_line is not None and right_line is None and self.prev_left_line is None and self.prev_right_line is None:
            dx_dy = (left_line.x(y_bot) - left_line.x(y_top)) / (y_bot - y_top)
            if left_line.x(y_bot) > 0.56 or (left_line.x(y_bot) > 0.40 and dx_dy > 0.04):
                right_line = left_line
                left_line = None
        elif right_line is not None and left_line is None and self.prev_left_line is None and self.prev_right_line is None:
            dx_dy = (right_line.x(y_bot) - right_line.x(y_top)) / (y_bot - y_top)
            if right_line.x(y_bot) < 0.44 or (right_line.x(y_bot) < 0.60 and dx_dy < -0.04):
                left_line = right_line
                right_line = None

        # ── Centerline Calculation & Fallback Logic ──────────────────────────
        status = "LANE_LOST"
        center_coeffs = None
        y_look = self.lookahead_y_frac

        if left_line is not None and right_line is not None:
            status = "DUAL_LOCKED"
            # Midpoint polynomial
            c_mid = 0.5 * (left_line.coefficients + right_line.coefficients)
            w_meas = right_line.x(y_look) - left_line.x(y_look)
            if 0.15 < w_meas < 0.85:
                # Update learned lane width via EMA
                self.learned_lane_width = (1.0 - 0.20) * self.learned_lane_width + 0.20 * w_meas

            # Smooth centerline coeffs with EMA
            if self.prev_center_coeffs is not None:
                center_coeffs = self.ema_alpha * c_mid + (1.0 - self.ema_alpha) * self.prev_center_coeffs
            else:
                center_coeffs = c_mid

            self.prev_left_line = left_line
            self.prev_right_line = right_line
            self.prev_center_coeffs = center_coeffs
            self.lost_frames = 0

        elif left_line is not None:
            status = "SINGLE_LEFT"
            # Reconstruct virtual centerline shifted right by half the learned width
            c_left = left_line.coefficients.copy()
            c_left[2] += 0.5 * self.learned_lane_width
            if self.prev_center_coeffs is not None:
                center_coeffs = self.ema_alpha * c_left + (1.0 - self.ema_alpha) * self.prev_center_coeffs
            else:
                center_coeffs = c_left

            self.prev_left_line = left_line
            self.prev_center_coeffs = center_coeffs
            self.lost_frames = 0

        elif right_line is not None:
            status = "SINGLE_RIGHT"
            # Reconstruct virtual centerline shifted left by half the learned width
            c_right = right_line.coefficients.copy()
            c_right[2] -= 0.5 * self.learned_lane_width
            if self.prev_center_coeffs is not None:
                center_coeffs = self.ema_alpha * c_right + (1.0 - self.ema_alpha) * self.prev_center_coeffs
            else:
                center_coeffs = c_right

            self.prev_right_line = right_line
            self.prev_center_coeffs = center_coeffs
            self.lost_frames = 0

        else:
            self.lost_frames += 1
            if self.lost_frames <= self.lost_max_frames and self.prev_center_coeffs is not None:
                # Coasting memory
                status = "LANE_LOST"
                center_coeffs = self.prev_center_coeffs
            else:
                status = "LANE_LOST"
                center_coeffs = None
                self.prev_left_line = None
                self.prev_right_line = None
                self.prev_center_coeffs = None

        # ── Pure Pursuit & Steering Calculation ──────────────────────────────
        steer_angle_deg = 0.0
        lateral_offset_pct = 0.0
        lookahead_x_px = frame_w // 2
        lookahead_y_px = int(y_look * frame_h)

        if center_coeffs is not None:
            # Lateral position error: target centerline normalized x vs ego vehicle center (0.5)
            x_target = float(np.polyval(center_coeffs, y_look))
            lateral_err = x_target - 0.5  # [-0.5, +0.5]
            lateral_offset_pct = lateral_err * 100.0

            # Tangent heading error: dx/dy = 2*a*y + b
            # Note: image y points down; forward travel is decreasing y
            dx_dy = 2.0 * center_coeffs[0] * y_look + center_coeffs[1]
            tangent_heading_deg = -np.degrees(np.arctan(dx_dy))

            # Steering command: P on lateral error + D on path tangent heading
            raw_steer = (lateral_err * self.kp_steer) + (tangent_heading_deg * (self.kd_steer / 45.0))
            steer_angle_deg = float(np.clip(raw_steer, -self.max_steer_deg, self.max_steer_deg))

            lookahead_x_px = int(np.clip(x_target * frame_w, 0, frame_w - 1))

        # ── HUD Geometry Generation ──────────────────────────────────────────
        left_pts = None
        right_pts = None
        center_pts = None
        corridor_pts = None

        y_eval = np.linspace(self.roi_y_min_frac, 0.98, 30)

        if left_line is not None:
            xs_l = np.clip(np.polyval(left_line.coefficients, y_eval) * frame_w, 0, frame_w - 1)
            left_pts = np.column_stack([xs_l, y_eval * frame_h]).astype(np.int32)

        if right_line is not None:
            xs_r = np.clip(np.polyval(right_line.coefficients, y_eval) * frame_w, 0, frame_w - 1)
            right_pts = np.column_stack([xs_r, y_eval * frame_h]).astype(np.int32)

        if center_coeffs is not None:
            xs_c = np.clip(np.polyval(center_coeffs, y_eval) * frame_w, 0, frame_w - 1)
            center_pts = np.column_stack([xs_c, y_eval * frame_h]).astype(np.int32)

            # Generate drivable ribbon corridor polygon
            if left_pts is not None and right_pts is not None:
                corridor_pts = np.vstack([left_pts, np.flipud(right_pts)])
            elif left_pts is not None:
                # Reconstruct right side from learned width
                xs_r_virt = np.clip((np.polyval(left_line.coefficients, y_eval) + self.learned_lane_width) * frame_w, 0, frame_w - 1)
                right_virt_pts = np.column_stack([xs_r_virt, y_eval * frame_h]).astype(np.int32)
                corridor_pts = np.vstack([left_pts, np.flipud(right_virt_pts)])
            elif right_pts is not None:
                # Reconstruct left side from learned width
                xs_l_virt = np.clip((np.polyval(right_line.coefficients, y_eval) - self.learned_lane_width) * frame_w, 0, frame_w - 1)
                left_virt_pts = np.column_stack([xs_l_virt, y_eval * frame_h]).astype(np.int32)
                corridor_pts = np.vstack([left_virt_pts, np.flipud(right_pts)])
            else:
                # Coasting memory: reconstruct virtual left and right boundaries from center_coeffs
                half_w = 0.5 * self.learned_lane_width
                xs_l_virt = np.clip((np.polyval(center_coeffs, y_eval) - half_w) * frame_w, 0, frame_w - 1)
                xs_r_virt = np.clip((np.polyval(center_coeffs, y_eval) + half_w) * frame_w, 0, frame_w - 1)
                left_pts = np.column_stack([xs_l_virt, y_eval * frame_h]).astype(np.int32)
                right_pts = np.column_stack([xs_r_virt, y_eval * frame_h]).astype(np.int32)
                corridor_pts = np.vstack([left_pts, np.flipud(right_pts)])

        return LaneDetectionResult(
            status=status,
            left_line=left_line,
            right_line=right_line,
            center_coeffs=center_coeffs,
            steer_angle_deg=steer_angle_deg,
            lateral_offset_pct=lateral_offset_pct,
            lane_width_frac=self.learned_lane_width,
            lookahead_point=(lookahead_x_px, lookahead_y_px),
            corridor_pts=corridor_pts,
            left_pts=left_pts,
            right_pts=right_pts,
            center_pts=center_pts,
            binary_mask=mask,
            lost_frames=self.lost_frames,
        )
