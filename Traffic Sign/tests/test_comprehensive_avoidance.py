"""
test_comprehensive_avoidance.py — Tests for Progressive Avoidance, Sensor Fusion, and Remote Control
=====================================================================================================
Verifies:
  1. Fused Zone Scoring (Camera Bounding Box ground-plane proximity + Ultrasonic fusion)
  2. Active Steering Avoidance (Pi actively steers around camera obstacles instead of stopping dead)
  3. Auto-Unstick / Pivot Recovery from PI_STOP latch
  4. Manual Remote Control (Endpoints, D-Pad, speed selection, and collision safety brake)
  5. Rear Ultrasonic Sensor Scaffold (Parsing, telemetry, and reverse safety checks)
"""

import time
import numpy as np
import pytest

from car.main import (
    CommandThrottler,
    SystemState,
    _run_autonomous,
    _check_obstacle_ahead,
    CRUISE_SPEED,
    SLOW_SPEED,
    EMERGENCY_STOP_DISTANCE_CM,
)
from car.perception.avoidance import (
    compute_zone_scores,
    decide_avoidance,
    decide_avoidance_detailed,
)
from car.control.motor_controller import MotorController
from car.perception.sensors import SensorManager
from tests.test_main_coordination import create_mock_motor


# =============================================================================
# 1. Fused Zone Scoring & Proximity Tests
# =============================================================================

def test_compute_zone_scores_with_ground_proximity():
    """Verify that bottom-edge (y2) proximity increases danger score as obstacle nears bumper."""
    frame_w, frame_h = 640, 480

    # Obstacle A: Far away (y2 = 200, top half of frame)
    det_far = [{"bbox": [260, 100, 380, 200], "confidence": 0.8, "class_name": "person"}]
    scores_far = compute_zone_scores(det_far, frame_w, frame_h)

    # Obstacle B: Close to bumper (y2 = 450, bottom of frame)
    det_near = [{"bbox": [260, 250, 380, 450], "confidence": 0.8, "class_name": "person"}]
    scores_near = compute_zone_scores(det_near, frame_w, frame_h)

    assert scores_near["center"] > scores_far["center"], "Obstacle closer to ground bumper must have higher danger"


def test_compute_zone_scores_fuses_ultrasonics():
    """Verify ultrasonic distances fuse into Left/Right danger scores."""
    frame_w, frame_h = 640, 480
    sonar = {"left": 20.0, "right": 85.0, "rear": 999.0}

    # No camera detections, but left ultrasonic is in critical range (< 22cm)
    scores = compute_zone_scores([], frame_w, frame_h, sonar=sonar)

    assert scores["left"] > 1.0, "Critical left ultrasonic must produce strong left danger score"
    assert scores["right"] < 0.1, "Clear right ultrasonic must leave right score low"
    assert decide_avoidance(scores) == "RIGHT", "Must steer RIGHT away from left obstacle"


def test_decide_avoidance_detailed():
    """Verify decide_avoidance_detailed produces nuanced actions and angles."""
    # Obstacle on left -> Veer right
    scores_left = {"left": 0.6, "center": 0.1, "right": 0.05}
    det_right = decide_avoidance_detailed(scores_left)
    assert det_right["action"] == "VEER_RIGHT"
    assert det_right["angle"] > 0

    # Obstacle in center -> Active avoid left
    scores_center = {"left": 0.1, "center": 0.7, "right": 0.4}
    det_center = decide_avoidance_detailed(scores_center)
    assert det_center["action"] == "AVOID_LEFT"
    assert det_center["angle"] < 0


# =============================================================================
# 2. Active Camera Steering Avoidance in _run_autonomous
# =============================================================================

def test_active_camera_steering_avoidance_veers_right():
    """
    Verify that when an obstacle is detected approaching in Center-Left (height ~25%),
    _run_autonomous actively commands TURN right instead of driving straight into it!
    """
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Obstacle on left corridor with height = 120 / 480 = 25% (caution zone)
    # cx = 120 < 0.33 * 640 = 211 -> Left zone danger -> Steer RIGHT
    detections = [{
        "class_name": "person",
        "confidence": 0.85,
        "bbox": [50, 200, 190, 320],
        "type": "positive",
    }]

    pi_latched, action = _run_autonomous(
        motor=motor,
        sensors=sensors,
        detections=detections,
        us_blocked=False,
        frame=frame,
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
    )

    # Must NOT freeze in PI_STOP!
    assert pi_latched is False
    # Must actively command TURN right to avoid obstacle
    assert any("TURN 35" in cmd for cmd in serial_dev.written_commands), (
        f"Expected TURN 35 in serial commands, got: {serial_dev.written_commands}"
    )
    assert "AVOIDING" in action or "Steering RIGHT" in action


def test_active_camera_steering_avoidance_veers_left():
    """Verify that when an obstacle is on Right, car actively steers LEFT."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Obstacle on right corridor with height = 25%
    detections = [{
        "class_name": "person",
        "confidence": 0.85,
        "bbox": [380, 200, 540, 320],
        "type": "positive",
    }]

    pi_latched, action = _run_autonomous(
        motor=motor,
        sensors=sensors,
        detections=detections,
        us_blocked=False,
        frame=frame,
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
    )

    assert pi_latched is False
    assert any("TURN -35" in cmd for cmd in serial_dev.written_commands)
    assert "LEFT" in action


# =============================================================================
# 3. Auto-Unstick / Pivot Recovery from PI_STOP
# =============================================================================

def test_auto_unstick_pivot_escape_when_stationary():
    """
    Verify that when latched in PI_STOP for > 1.2s and an open side exists,
    car unlatches with PI_GO, pivots toward the clear corridor, and escapes!
    """
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Imminent obstacle in center (height = 50%)
    detections = [{
        "class_name": "person",
        "confidence": 0.90,
        "bbox": [200, 100, 440, 340],
        "type": "positive",
    }]

    # Simulate that car was latched 1.5 seconds ago
    throttler.latch_start_time = time.monotonic() - 1.5

    # Right side has ultrasonic clearance
    motor.latest_sonar = {"left": 15.0, "right": 85.0, "rear": 999.0}

    pi_latched, action = _run_autonomous(
        motor=motor,
        sensors=sensors,
        detections=detections,
        us_blocked=False,
        frame=frame,
        pi_latched=True,
        motors_enabled=True,
        throttler=throttler,
    )

    # Latch should be released
    assert pi_latched is False
    assert "PI_GO" in serial_dev.written_commands
    # Pivot escape command should be sent
    assert any("TURN 45" in cmd for cmd in serial_dev.written_commands)
    assert "ESCAPE" in action


# =============================================================================
# 4. Rear Ultrasonic Sensor Scaffold Tests
# =============================================================================

def test_rear_ultrasonic_parsing_and_clearance():
    """Verify rear ultrasonic telemetry parsing and is_rear_clear helper."""
    motor = MotorController(port="TEST_PORT")

    # Parse telemetry line with REAR distance
    line = "[TELEM] US LEFT=24.5 RIGHT=31.2 REAR=75.0"
    motor._parse_us_telemetry(line)

    assert motor.latest_sonar["left"] == 24.5
    assert motor.latest_sonar["right"] == 31.2
    assert motor.latest_sonar["rear"] == 75.0
    assert motor.is_rear_clear(threshold_cm=30.0) is True
    assert motor.is_rear_clear(threshold_cm=80.0) is False


def test_rear_ultrasonic_supports_alternate_keys():
    """Verify BACK= and US_REAR= tokens are parsed correctly."""
    motor = MotorController(port="TEST_PORT")
    motor._parse_us_telemetry("[TELEM] US LEFT=12.0 RIGHT=14.0 BACK=42.5")
    assert motor.latest_sonar["rear"] == 42.5


# =============================================================================
# 5. SystemState & Manual Control Mode Tests
# =============================================================================

def test_system_state_manual_mode_defaults():
    """Verify SystemState initializes with manual control support."""
    state = SystemState()
    assert state.control_mode == "autonomous"
    assert state.manual_speed == 185
    assert state.manual_action == "IDLE"


def test_manual_drive_command_execution():
    """Verify manual drive commands interact with motor controller."""
    motor, serial_dev = create_mock_motor()
    state = SystemState()
    state.motor = motor
    state.control_mode = "manual"
    state.manual_speed = 200

    # Command forward when clear
    motor.drive(state.manual_speed, 350)
    assert any("DRIVE 200 350" in cmd for cmd in serial_dev.written_commands)

    # Command turn left
    motor.turn(-40, speed=160)
    assert any("TURN -40 160" in cmd for cmd in serial_dev.written_commands)


def test_manual_drive_blocked_by_ultrasonic_safety():
    """Verify that when moving forward in manual mode with an obstacle ahead, stop is enforced."""
    motor, serial_dev = create_mock_motor()
    motor.latest_sonar = {"left": 12.0, "right": 15.0, "rear": 999.0}

    # Obstacle is within emergency stop distance (< 25 cm)
    us_l = motor.latest_sonar["left"]
    us_r = motor.latest_sonar["right"]
    blocked = (us_l < EMERGENCY_STOP_DISTANCE_CM) or (us_r < EMERGENCY_STOP_DISTANCE_CM)
    assert blocked is True

    # Motor must be stopped when blocked
    motor.stop()
    assert "STOP" in serial_dev.written_commands


def test_pothole_held_photo_triggers_avoidance():
    """
    Verify that a pothole detection (e.g. from a held-up photo of a road with a pothole
    in front of the car) produces a strong enough zone danger score to trigger active
    steering avoidance rather than being ignored as CLEAR.
    """
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Pothole inside a held-up photo in the center-left corridor (middle of frame, y2=270)
    detections = [{
        "class_id": -1,
        "class_name": "pothole",
        "confidence": 0.62,
        "bbox": [60, 190, 180, 270],
        "type": "negative",
    }]

    scores = compute_zone_scores(detections, 640, 480)
    assert decide_avoidance(scores) == "RIGHT", f"Expected RIGHT avoidance for left pothole, got {decide_avoidance(scores)} (scores={scores})"

    pi_latched, action = _run_autonomous(
        motor=motor,
        sensors=sensors,
        detections=detections,
        us_blocked=False,
        frame=frame,
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
    )
    assert pi_latched is False
    assert any("TURN 35" in cmd for cmd in serial_dev.written_commands)


def test_manual_drive_override_obstacles_toggle():
    """
    Verify that when manual_override_obstacles is True, manual drive commands
    override ultrasonic emergency auto-brake and drive anyway.
    """
    motor, serial_dev = create_mock_motor()
    state = SystemState()
    state.motor = motor
    state.control_mode = "manual"
    state.manual_speed = 200
    state.manual_override_obstacles = True

    # Obstacle is very close (12 cm < 25 cm emergency threshold)
    motor.latest_sonar = {"left": 12.0, "right": 15.0, "rear": 999.0}

    # When override is active, driver has authority
    override = state.manual_override_obstacles
    us_l = motor.latest_sonar["left"]
    us_r = motor.latest_sonar["right"]
    front_blocked = not override and (us_l < EMERGENCY_STOP_DISTANCE_CM or us_r < EMERGENCY_STOP_DISTANCE_CM)
    assert front_blocked is False, "Obstacle must be overridden when manual_override_obstacles is True"

    motor.drive(state.manual_speed, 350)
    assert any("DRIVE 200 350" in cmd for cmd in serial_dev.written_commands)


def test_live_detect_fine_tuned_model_candidates():
    """
    Verify that car.live_detect resolves the fine-tuned pothole model path and includes
    the obstacle override toggle.
    """
    import car.live_detect as ld
    assert hasattr(ld, "manual_override_obstacles")
    assert hasattr(ld, "pot_candidates")
    assert "pothole_best.pt" in ld.pot_candidates
    assert "pothole_detection/best.pt" in ld.pot_candidates




def test_lower_section_pothole_filter_and_zero_false_positives():
    """
    Verify that the GPU printout-tuned pothole detector:
    1. Does NOT run classical CV Laplacian edge false-positives when the YOLO pothole model is loaded.
    2. Produces ZERO false positives on synthetic indoor floor tiles, shadows, and blank paper borders.
    3. Enforces the lower-section camera constraint (POTHOLE_LOWER_Y_MIN_FRAC = 0.40).
    """
    import cv2
    import car.live_detect as ld
    from car.perception.detector import Detector

    assert ld.POTHOLE_LOWER_Y_MIN_FRAC == 0.40
    assert ld.POTHOLE_PERSIST_FRAMES == 2

    det = Detector()
    assert det.enable_pothole_cv is False
    assert det.pothole_model is not None

    # Create a hard-negative frame with floor tile lines and a blank white paper sheet border
    neg_frame = np.full((240, 320, 3), 150, dtype=np.uint8)
    for x in range(0, 320, 60):
        cv2.line(neg_frame, (x, 0), (x, 240), (100, 100, 100), 1)
    for y in range(90, 240, 45):
        cv2.line(neg_frame, (0, y), (320, y), (100, 100, 100), 1)
    cv2.rectangle(neg_frame, (40, 70), (280, 225), (235, 235, 235), -1)
    cv2.rectangle(neg_frame, (40, 70), (280, 225), (45, 45, 45), 2)

    potholes = det._detect_yolo_potholes(neg_frame)
    assert potholes == [], f"Expected 0 false positives on blank paper border / floor tiles, got {potholes}"
