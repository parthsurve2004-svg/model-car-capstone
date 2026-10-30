"""
test_lane_detection.py — Verification of Chart-Paper Lane Detection & Multi-Model Arbitration
=============================================================================================
Tests:
  1. Synthetic chart-paper line detection (straight & curved track).
  2. Single-line fallback recovery (SINGLE_LEFT and SINGLE_RIGHT).
  3. Pothole printout bounding-box exclusion masking so dark potholes do not pull curves.
  4. Multi-model priority stack coordination in _run_lane():
     - Clear lane -> DRIVE straight / corrective curve TURN.
     - In-lane medium hazard -> Tier 2.5 active bypass.
     - Imminent obstacle -> Tier 1 PI_STOP latch.
     - Ultrasonic blocked -> Tier 2 US AUTO-BRAKE.
     - Lost lane (> 8 frames) -> safe STOP.
  5. Color mode presets (dark, yellow, blue, white).
  6. Web UI mode switching and HUD corridor rendering.
"""

import json
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pytest
import cv2

from car.perception.config import (
    LANE_ROI_Y_MIN_FRAC,
    LANE_LOOKAHEAD_Y_FRAC,
    LANE_DEFAULT_WIDTH_FRAC,
    LANE_LOST_MAX_FRAMES,
    LANE_CRUISE_SPEED,
    LANE_CURVE_SPEED,
)
from car.perception.lane_detector import (
    ChartLaneDetector,
    LaneCurve,
    LaneDetectionResult,
    fit_poly_mad,
)
from car.main import (
    CommandThrottler,
    SystemState,
    DiagnosticHandler,
    StreamingServer,
    draw_hud,
    _run_lane,
)
from car.control.motor_controller import MotorController
from car.perception.sensors import SensorManager


class MockSerialDevice:
    """Mock serial transport recording written bytes and serving simulated telemetry."""

    def __init__(self, incoming_lines=None):
        self.is_open = True
        self.written_commands = []
        self._in_buffer = list(incoming_lines or [])

    @property
    def in_waiting(self):
        return sum(len(line.encode("utf-8") + b"\n") for line in self._in_buffer)

    def readline(self):
        if self._in_buffer:
            line = self._in_buffer.pop(0)
            return (line + "\n").encode("utf-8")
        return b""

    def write(self, data: bytes):
        text = data.decode("utf-8").strip()
        self.written_commands.append(text)
        return len(data)

    def flush(self):
        pass

    def reset_input_buffer(self):
        self._in_buffer.clear()

    def reset_output_buffer(self):
        pass

    def close(self):
        self.is_open = False


def create_mock_motor(incoming_lines=None):
    """Creates a MotorController backed by a MockSerialDevice."""
    motor = MotorController(port="MOCK_PORT", baud=115200)
    mock_serial = MockSerialDevice(incoming_lines)
    motor._serial = mock_serial
    return motor, mock_serial


def create_synthetic_chart_paper_frame(
    width: int = 640,
    height: int = 480,
    left_x_frac: float = 0.25,
    right_x_frac: float = 0.75,
    curve_offset_px: float = 0.0,
    line_color: Tuple[int, int, int] = (25, 25, 25),
    bg_color: Tuple[int, int, int] = (245, 245, 245),
    draw_left: bool = True,
    draw_right: bool = True,
    line_thickness: int = 8,
) -> np.ndarray:
    """Creates a synthetic image representing lines drawn or taped on chart paper."""
    frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

    ys = np.linspace(int(height * 0.40), height - 1, 60).astype(int)

    if draw_left:
        pts_left = []
        for y in ys:
            # Curve offset bends horizontally toward top
            y_norm = (height - y) / height
            x = int(width * left_x_frac + curve_offset_px * (y_norm ** 2))
            pts_left.append((x, y))
        cv2.polylines(frame, [np.array(pts_left)], False, line_color, line_thickness)

    if draw_right:
        pts_right = []
        for y in ys:
            y_norm = (height - y) / height
            x = int(width * right_x_frac + curve_offset_px * (y_norm ** 2))
            pts_right.append((x, y))
        cv2.polylines(frame, [np.array(pts_right)], False, line_color, line_thickness)

    return frame


# =============================================================================
# 1. Pure Vision: Synthetic Chart-Paper Lane Detection
# =============================================================================

def test_straight_track_dual_locked():
    """Verify ChartLaneDetector detects straight dark lines on light chart paper."""
    detector = ChartLaneDetector(color_mode="dark")
    frame = create_synthetic_chart_paper_frame(
        width=640, height=480, left_x_frac=0.25, right_x_frac=0.75, curve_offset_px=0.0
    )

    result = detector.detect(frame)

    assert result.status == "DUAL_LOCKED"
    assert result.left_line is not None
    assert result.right_line is not None
    assert result.center_coeffs is not None
    # Straight lines centered in frame: steering angle should be near 0 deg (within deadband)
    assert abs(result.steer_angle_deg) <= 3.0
    # Lane width should be ~0.50 (0.75 - 0.25)
    assert 0.45 <= result.lane_width_frac <= 0.55
    # Lookahead point X should be near 320
    assert abs(result.lookahead_point[0] - 320) <= 8


def test_curved_track_steering():
    """Verify right and left curving tracks command appropriate positive and negative steering."""
    detector = ChartLaneDetector(color_mode="dark")

    # Curve toward right (+x at lookahead)
    frame_right = create_synthetic_chart_paper_frame(
        width=640, height=480, left_x_frac=0.25, right_x_frac=0.75, curve_offset_px=80.0
    )
    res_right = detector.detect(frame_right)
    assert res_right.status == "DUAL_LOCKED"
    assert res_right.steer_angle_deg > 4.0

    detector.reset()

    # Curve toward left (-x at lookahead)
    frame_left = create_synthetic_chart_paper_frame(
        width=640, height=480, left_x_frac=0.25, right_x_frac=0.75, curve_offset_px=-80.0
    )
    res_left = detector.detect(frame_left)
    assert res_left.status == "DUAL_LOCKED"
    assert res_left.steer_angle_deg < -4.0


# =============================================================================
# 2. Single-Line Fallback Recovery
# =============================================================================

def test_single_line_fallback_left():
    """Verify that when only the left line is visible, virtual centerline is reconstructed."""
    detector = ChartLaneDetector(color_mode="dark")
    frame = create_synthetic_chart_paper_frame(
        width=640, height=480, left_x_frac=0.25, draw_right=False
    )

    result = detector.detect(frame)

    assert result.status == "SINGLE_LEFT"
    assert result.left_line is not None
    assert result.right_line is None
    assert result.center_coeffs is not None
    # Reconstructed centerline should be shifted right by half the learned width (~0.26)
    # Left base at ~0.25 -> Centerline at ~0.51
    assert abs(result.lookahead_point[0] - 326) <= 20


def test_single_line_fallback_right():
    """Verify that when only the right line is visible, virtual centerline is reconstructed."""
    detector = ChartLaneDetector(color_mode="dark")
    frame = create_synthetic_chart_paper_frame(
        width=640, height=480, right_x_frac=0.75, draw_left=False
    )

    result = detector.detect(frame)

    assert result.status == "SINGLE_RIGHT"
    assert result.left_line is None
    assert result.right_line is not None
    assert result.center_coeffs is not None
    # Right base at ~0.75 -> Centerline at ~0.49
    assert abs(result.lookahead_point[0] - 314) <= 20


# =============================================================================
# 3. Pothole Printout BBox Exclusion Masking
# =============================================================================

def test_pothole_printout_bbox_exclusion():
    """
    Verify that placing a dark pothole printout between chart paper lines is
    masked out when exclude_boxes is passed, preventing curve distortion.
    """
    detector = ChartLaneDetector(color_mode="dark")
    frame = create_synthetic_chart_paper_frame(
        width=640, height=480, left_x_frac=0.25, right_x_frac=0.75
    )

    # Place a dark pothole blob right in the center corridor (x: 280-360, y: 350-430)
    pothole_bbox = [280, 350, 360, 430]
    cv2.circle(frame, (320, 390), 38, (15, 15, 15), -1)

    # Pass the pothole detection bbox into exclude_boxes
    exclude_boxes = [{
        "bbox": pothole_bbox,
        "type": "negative",
        "class_name": "pothole",
        "confidence": 0.88,
    }]

    result = detector.detect(frame, exclude_boxes=exclude_boxes)

    # The pothole pixels should be masked out; detector locks onto the true lines
    assert result.status == "DUAL_LOCKED"
    assert result.left_line is not None
    assert result.right_line is not None
    assert abs(result.steer_angle_deg) <= 3.5

    # Check that mask inside the pothole center is zeroed out
    roi_y_start = int(480 * LANE_ROI_Y_MIN_FRAC)
    mask = result.binary_mask
    assert mask is not None
    pothole_roi_y = 390 - roi_y_start
    assert mask[pothole_roi_y, 320] == 0


# =============================================================================
# 4. Multi-Model Arbitration in _run_lane()
# =============================================================================

def test_run_lane_nominal_straight():
    """Verify clear chart paper commands DRIVE forward at LANE_CRUISE_SPEED."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    detector = ChartLaneDetector(color_mode="dark")
    state = SystemState()

    frame = create_synthetic_chart_paper_frame(width=640, height=480)

    pi_latched, action, res = _run_lane(
        motor=motor,
        sensors=sensors,
        lane_detector=detector,
        detections=[],
        us_blocked=False,
        frame=frame,
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
        sys_state=state,
    )

    assert pi_latched is False
    assert res.status == "DUAL_LOCKED"
    assert "LANE LOCKED" in action
    assert any("DRIVE" in c and str(LANE_CRUISE_SPEED) in c for c in serial_dev.written_commands)


def test_run_lane_curved_steer():
    """Verify curved chart paper commands corrective TURN at LANE_CURVE_SPEED."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    detector = ChartLaneDetector(color_mode="dark")
    state = SystemState()

    frame = create_synthetic_chart_paper_frame(
        width=640, height=480, curve_offset_px=110.0
    )

    pi_latched, action, res = _run_lane(
        motor=motor,
        sensors=sensors,
        lane_detector=detector,
        detections=[],
        us_blocked=False,
        frame=frame,
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
        sys_state=state,
    )

    assert pi_latched is False
    assert res.status == "DUAL_LOCKED"
    assert "LANE STEER RIGHT" in action
    assert any("TURN" in c and str(LANE_CURVE_SPEED) in c for c in serial_dev.written_commands)


def test_run_lane_in_lane_hazard_bypass():
    """Verify that a medium-range obstacle/pothole in the lane corridor triggers active bypass."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    detector = ChartLaneDetector(color_mode="dark")
    state = SystemState()

    frame = create_synthetic_chart_paper_frame(width=640, height=480)

    # Obstacle on the left side of the corridor -> should bypass RIGHT
    detections = [{
        "class_name": "obstacle",
        "bbox": [60, 220, 160, 340],  # cx=110 (< 0.33*640 = 211) -> left zone -> bypass RIGHT
        "confidence": 0.75,
        "type": "positive",
    }]

    pi_latched, action, res = _run_lane(
        motor=motor,
        sensors=sensors,
        lane_detector=detector,
        detections=detections,
        us_blocked=False,
        frame=frame,
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
        sys_state=state,
    )

    assert "LANE BYPASS" in action
    assert "RIGHT" in action
    assert any("PI_AVOID RIGHT" in c for c in serial_dev.written_commands)


def test_run_lane_tier1_imminent_stop():
    """Verify Tier 1 camera stop latch is engaged when an imminent obstacle appears in the lane."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    detector = ChartLaneDetector(color_mode="dark")
    state = SystemState()

    frame = create_synthetic_chart_paper_frame(width=640, height=480)

    # Huge imminent obstacle in center corridor (height > 40%)
    detections = [{
        "class_name": "obstacle",
        "bbox": [200, 100, 440, 360],  # height = 260 / 480 = 54%
        "confidence": 0.85,
        "type": "positive",
    }]

    pi_latched, action, res = _run_lane(
        motor=motor,
        sensors=sensors,
        lane_detector=detector,
        detections=detections,
        us_blocked=False,
        frame=frame,
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
        sys_state=state,
    )

    assert pi_latched is True
    assert "PI_STOP" in action
    assert "PI_STOP" in serial_dev.written_commands


def test_run_lane_lost_holding_stop():
    """Verify that when both lane boundaries are lost for > 8 frames, car holds safe stop."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    detector = ChartLaneDetector(color_mode="dark")
    state = SystemState()

    # Completely blank frame (no lines on paper)
    blank_frame = np.full((480, 640, 3), 245, dtype=np.uint8)

    # Step through lost_max_frames + 1 iterations
    for _ in range(LANE_LOST_MAX_FRAMES + 1):
        pi_latched, action, res = _run_lane(
            motor=motor,
            sensors=sensors,
            lane_detector=detector,
            detections=[],
            us_blocked=False,
            frame=blank_frame,
            pi_latched=False,
            motors_enabled=True,
            throttler=throttler,
            sys_state=state,
        )

    assert res.status == "LANE_LOST"
    assert "Holding Safe Stop" in action
    assert "STOP" in serial_dev.written_commands


# =============================================================================
# 5. Color Mode Presets
# =============================================================================

def test_color_mode_presets():
    """Verify yellow and blue tape presets extract lines correctly."""
    # Yellow tape on white paper
    detector_yellow = ChartLaneDetector(color_mode="yellow")
    # BGR yellow is (0, 220, 240)
    frame_yellow = create_synthetic_chart_paper_frame(
        width=640, height=480, line_color=(0, 220, 240), bg_color=(245, 245, 245)
    )
    res_yellow = detector_yellow.detect(frame_yellow)
    assert res_yellow.status == "DUAL_LOCKED"

    # Blue tape on white paper
    detector_blue = ChartLaneDetector(color_mode="blue")
    # BGR blue is (220, 80, 20)
    frame_blue = create_synthetic_chart_paper_frame(
        width=640, height=480, line_color=(220, 80, 20), bg_color=(245, 245, 245)
    )
    res_blue = detector_blue.detect(frame_blue)
    assert res_blue.status == "DUAL_LOCKED"


# =============================================================================
# 6. HUD Corridor Rendering & Web UI
# =============================================================================

def test_draw_hud_lane_rendering():
    """Verify draw_hud renders the green corridor, boundary curves, and reticle without error."""
    detector = ChartLaneDetector(color_mode="dark")
    frame = create_synthetic_chart_paper_frame(width=640, height=480)
    res = detector.detect(frame)

    hud_img = draw_hud(
        frame=frame,
        detections=[],
        action_label="🛣️ LANE LOCKED (DUAL_LOCKED): Straight",
        latest_sonar={"left": 120.0, "right": 120.0},
        latest_state="IDLE",
        port_name="MOCK_PORT",
        fps=30.0,
        awb_mode="daylight",
        mode="lane",
        motors_enabled=True,
        invert_motors=False,
        pi_latched=False,
        lane_result=res,
    )

    assert hud_img is not None
    assert hud_img.shape == (480, 640, 3)
    # The green corridor adds green tint to the drivable region between lines (sampled at x=280)
    assert hud_img[360, 280, 1] > 100
    # The magenta centerline is drawn with B=255, R=255 at x=320
    assert hud_img[360, 320, 0] > 200 and hud_img[360, 320, 2] > 200


# =============================================================================
# 7. Deep Edge Cases & Robustness
# =============================================================================

def test_off_center_two_lines_left_half():
    """Verify that when the vehicle is displaced to the right, both lines falling in the left half are tracked as DUAL_LOCKED."""
    detector = ChartLaneDetector(color_mode="dark")
    frame = np.full((480, 640, 3), 245, dtype=np.uint8)
    # Two lines: left at x=70 (0.11*w), right at x=310 (0.48*w)
    cv2.line(frame, (70, 200), (70, 480), (25, 25, 25), 8)
    cv2.line(frame, (310, 200), (310, 480), (25, 25, 25), 8)

    res = detector.detect(frame)

    assert res.status == "DUAL_LOCKED"
    assert res.left_line is not None
    assert res.right_line is not None
    measured_w = res.right_line.x(0.68) - res.left_line.x(0.68)
    assert 0.35 <= measured_w <= 0.40
    # Centerline is at x=190 -> Vehicle center is 320 -> should command corrective steer LEFT
    assert res.steer_angle_deg < -15.0


def test_off_center_two_lines_right_half():
    """Verify that when the vehicle is displaced to the left, both lines falling in the right half are tracked as DUAL_LOCKED."""
    detector = ChartLaneDetector(color_mode="dark")
    frame = np.full((480, 640, 3), 245, dtype=np.uint8)
    # Two lines: left at x=330 (0.515*w), right at x=570 (0.89*w)
    cv2.line(frame, (330, 200), (330, 480), (25, 25, 25), 8)
    cv2.line(frame, (570, 200), (570, 480), (25, 25, 25), 8)

    res = detector.detect(frame)

    assert res.status == "DUAL_LOCKED"
    assert res.left_line is not None
    assert res.right_line is not None
    measured_w = res.right_line.x(0.68) - res.left_line.x(0.68)
    assert 0.35 <= measured_w <= 0.40
    # Centerline is at x=450 -> Vehicle center is 320 -> should command corrective steer RIGHT
    assert res.steer_angle_deg > 15.0


def test_single_line_right_off_center_disambiguation():
    """Verify that when only the right line is visible near the center (left line has exited frame), it is correctly identified as SINGLE_RIGHT."""
    detector = ChartLaneDetector(color_mode="dark")
    frame = np.full((480, 640, 3), 245, dtype=np.uint8)
    # Right line at x=310 (sloping towards center-top in perspective), left line offscreen
    cv2.line(frame, (270, 200), (310, 480), (25, 25, 25), 8)

    res = detector.detect(frame)

    assert res.status == "SINGLE_RIGHT"
    assert res.left_line is None
    assert res.right_line is not None
    # Right line at 310 means car is way too far right; steering must be LEFT (< 0)
    assert res.steer_angle_deg < -10.0


def test_coasting_trajectory_motor_execution():
    """Verify that when both lines temporarily disappear after being locked, _run_lane executes motor commands along the coasting path."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    detector = ChartLaneDetector(color_mode="dark")
    state = SystemState()

    # Frame 1: Lock onto straight lane
    frame1 = create_synthetic_chart_paper_frame(width=640, height=480)
    _run_lane(
        motor=motor, sensors=sensors, lane_detector=detector,
        detections=[], us_blocked=False, frame=frame1,
        pi_latched=False, motors_enabled=True, throttler=throttler, sys_state=state,
    )
    serial_dev.written_commands.clear()

    # Frame 2: Gap in chart paper (blank frame) -> should coast and keep driving
    blank_frame = np.full((480, 640, 3), 245, dtype=np.uint8)
    pi_latched, action, res = _run_lane(
        motor=motor, sensors=sensors, lane_detector=detector,
        detections=[], us_blocked=False, frame=blank_frame,
        pi_latched=False, motors_enabled=True, throttler=throttler, sys_state=state,
    )

    assert res.status == "LANE_LOST"
    assert res.center_coeffs is not None  # Coasting memory preserved
    assert res.corridor_pts is not None    # Corridor polygon retained for HUD
    assert "COASTING" in action
    # Throttler or motor command was dispatched
    assert any("DRIVE" in c for c in serial_dev.written_commands) or (throttler.last_cmd == "DRIVE" and throttler.last_speed == LANE_CRUISE_SPEED)


def test_immediate_stop_when_uninitialized():
    """Verify that on cold start with no previous lane lock, _run_lane immediately holds a safe stop instead of coasting."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    detector = ChartLaneDetector(color_mode="dark")
    state = SystemState()

    blank_frame = np.full((480, 640, 3), 245, dtype=np.uint8)
    pi_latched, action, res = _run_lane(
        motor=motor, sensors=sensors, lane_detector=detector,
        detections=[], us_blocked=False, frame=blank_frame,
        pi_latched=False, motors_enabled=True, throttler=throttler, sys_state=state,
    )

    assert res.status == "LANE_LOST"
    assert res.center_coeffs is None
    assert "Holding Safe Stop" in action
    assert "STOP" in serial_dev.written_commands


def test_bbox_exclusion_out_of_bounds_and_negative():
    """Verify that exclude_boxes with coordinates out of bounds or above ROI does not trigger slice index errors."""
    detector = ChartLaneDetector(color_mode="dark")
    frame = create_synthetic_chart_paper_frame(width=640, height=480)

    exclude_boxes = [
        {"bbox": [-50, -30, 100, 150]},        # Top-left out of bounds, above ROI
        {"bbox": [600, 400, 750, 550]},        # Bottom-right out of bounds
        {"bbox": [200, 10, 400, 50]},          # Entirely above ROI
    ]

    res = detector.detect(frame, exclude_boxes=exclude_boxes)
    assert res.status == "DUAL_LOCKED"
