"""
test_adversarial_coordination.py — Adversarial Verification of Coordination & Streaming Edge Cases
"""

import time
import pytest
import numpy as np

from car.main import (
    CommandThrottler,
    SystemState,
    draw_hud,
    _run_autonomous,
    _run_follow,
    _check_obstacle_ahead,
    is_valid_us,
)
from car.control.motor_controller import MotorController
from car.perception.sensors import SensorManager
from tests.test_main_coordination import create_mock_motor


def test_check_obstacle_ahead_prioritizes_stop_over_slow():
    """Verify that when a SLOW obstacle appears before a STOP obstacle, STOP is returned."""
    frame_w, frame_h = 640, 480
    dets = [
        # Obstacle 1: 20.8% height -> would trigger SLOW (15-40%)
        {"class_name": "person", "confidence": 0.8, "bbox": [200, 100, 440, 200], "type": "positive"},
        # Obstacle 2: 52.0% height -> triggers STOP (>40%)
        {"class_name": "person", "confidence": 0.9, "bbox": [200, 100, 440, 350], "type": "positive"},
    ]
    action, spd = _check_obstacle_ahead(dets, frame_w, frame_h)
    assert action == "STOP", f"Expected STOP, but got {action}"
    assert spd is None


def test_tier1_pi_stop_overrides_mid_avoidance():
    """Verify that Tier 1 camera latch overrides Arduino STATE_AVOIDING."""
    motor, serial_dev = create_mock_motor()
    motor.latest_state = "AVOIDING"
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Imminent obstacle on camera
    dets = [{"class_name": "person", "confidence": 0.9, "bbox": [200, 100, 440, 350], "type": "positive"}]

    pi_latched, action = _run_autonomous(
        motor=motor,
        sensors=sensors,
        detections=dets,
        us_blocked=False,
        frame=frame,
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
    )

    assert pi_latched is True
    assert "PI_STOP" in serial_dev.written_commands
    assert "PI_STOP" in action


def test_tier1_pi_stop_overrides_cliff_reversing():
    """Verify that Tier 1 camera latch overrides Arduino STATE_CLIFF_REVERSING."""
    motor, serial_dev = create_mock_motor()
    motor.latest_state = "CLIFF_REVERSING"
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    dets = [{"class_name": "person", "confidence": 0.9, "bbox": [200, 100, 440, 350], "type": "positive"}]

    pi_latched, action = _run_autonomous(
        motor=motor,
        sensors=sensors,
        detections=dets,
        us_blocked=False,
        frame=frame,
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
    )

    assert pi_latched is True
    assert "PI_STOP" in serial_dev.written_commands


def test_cliff_stopped_does_not_release_if_camera_obstacle_present():
    """Verify that CLIFF_STOPPED does NOT emit PI_GO if camera sees an obstacle."""
    motor, serial_dev = create_mock_motor()
    motor.latest_state = "CLIFF_STOPPED"
    motor.latest_sonar = {"left": 25.0, "right": 25.0}
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Obstacle on camera
    dets = [{"class_name": "person", "confidence": 0.9, "bbox": [200, 100, 440, 350], "type": "positive"}]

    pi_latched, action = _run_autonomous(
        motor=motor,
        sensors=sensors,
        detections=dets,
        us_blocked=False,
        frame=frame,
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
    )

    assert "PI_GO" not in serial_dev.written_commands
    assert pi_latched is True


def test_pi_avoid_hint_not_flooded():
    """Verify that repeated frames with the same avoid decision do not flood PI_AVOID."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Obstacle in left zone (triggers RIGHT avoidance hint)
    dets = [{"class_name": "car", "confidence": 0.8, "bbox": [50, 100, 150, 200], "type": "positive"}]

    for _ in range(10):
        _run_autonomous(
            motor=motor,
            sensors=sensors,
            detections=dets,
            us_blocked=False,
            frame=frame,
            pi_latched=False,
            motors_enabled=True,
            throttler=throttler,
        )

    # PI_AVOID RIGHT should NOT be sent 10 times in 10 consecutive frames!
    avoid_cmds = [cmd for cmd in serial_dev.written_commands if cmd.startswith("PI_AVOID")]
    assert len(avoid_cmds) <= 2, f"PI_AVOID spammed {len(avoid_cmds)} times in 10 frames!"


def test_is_valid_us_handles_none_and_edge_types():
    """Verify is_valid_us does not crash on None, strings, or extreme values."""
    assert is_valid_us(None) is False
    assert is_valid_us("15.2") is False
    assert is_valid_us(0.0) is False
    assert is_valid_us(0.5) is False
    assert is_valid_us(999.0) is False
    assert is_valid_us(-10.0) is False
    assert is_valid_us(30.0) is True


def test_draw_hud_handles_none_fields():
    """Verify draw_hud does not raise AttributeError if optional fields are None."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = draw_hud(
        frame=frame,
        detections=[],
        action_label="TEST",
        latest_sonar=None,
        latest_state=None,
        port_name=None,
        fps=30.0,
        awb_mode="daylight",
        mode="autonomous",
        motors_enabled=True,
        invert_motors=False,
        pi_latched=False,
    )
    assert res.shape == frame.shape


def test_follow_mode_turn_throttling():
    """Verify that rapid steering in follow mode does not spam TURN every iteration."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mock a locked tracker with off-center target requiring turn
    from car.perception.tracker import ObjectTracker
    from car.control.pid import PID
    tracker = ObjectTracker()
    # Initialize tracker with target on far right
    tracker.lock_on(frame, (500, 100, 100, 150))
    pid = PID(P=0.3, I=0.01, D=0.05)
    pid.SetPoint = 320.0

    # 10 rapid iterations (e.g. 60 FPS)
    for _ in range(10):
        _run_follow(
            motor=motor,
            sensors=sensors,
            tracker=tracker,
            pid=pid,
            detections=[],
            us_blocked=False,
            frame=frame,
            follow_state="TRACKING",
            pi_latched=False,
            motors_enabled=True,
            throttler=throttler,
        )

    turn_cmds = [cmd for cmd in serial_dev.written_commands if cmd.startswith("TURN")]
    assert len(turn_cmds) <= 2, f"TURN spammed {len(turn_cmds)} times in 10 rapid frames!"


def test_blocked_telemetry_generalized_parsing():
    """Verify that [ACK] BLOCKED (ANY_STATE) correctly updates latest_state."""
    motor, _ = create_mock_motor(incoming_lines=[
        "[ACK] BLOCKED (AVOIDING)",
        "[ACK] BLOCKED (STATE_CLIFF_STOPPED)",
    ])
    motor.read_telemetry()
    assert motor.latest_state == "AVOIDING"
    motor.read_telemetry()
    assert motor.latest_state == "STATE_CLIFF_STOPPED"


def test_test_drive_blocked_when_latched():
    """Verify that /test_drive does not fire motors if pi_latched is True."""
    from tests.test_main_coordination import simulate_http_get
    sys_state = SystemState()
    motor, serial_dev = create_mock_motor()
    sys_state.motor = motor
    sys_state.motors_enabled = True
    sys_state.pi_latched = True  # Latched!

    status, headers, body = simulate_http_get("/test_drive", sys_state)
    import json
    data = json.loads(body.decode("utf-8"))
    assert data["executed"] is False
    assert len([cmd for cmd in serial_dev.written_commands if cmd.startswith("DRIVE")]) == 0


def test_follow_mode_reacquire_after_lost_target():
    """Verify that when tracker loses target and enters RE_ACQUIRE, candidate persons can be reacquired."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    class MockTracker:
        def __init__(self):
            self.reacquire_called = False

        def update(self, frame):
            return False, None

        def reacquire(self, frame, candidates):
            self.reacquire_called = True
            return True

    tracker = MockTracker()
    from car.control.pid import PID
    pid = PID(P=0.3, I=0.01, D=0.05)

    # Frame 1: Target lost while in TRACKING -> transitions to RE_ACQUIRE with PI_STOP
    follow_state, pi_latched, action = _run_follow(
        motor=motor,
        sensors=sensors,
        tracker=tracker,
        pid=pid,
        detections=[],
        us_blocked=False,
        frame=frame,
        follow_state="TRACKING",
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
    )

    assert follow_state == "RE_ACQUIRE"
    assert pi_latched is True
    assert "PI_STOP" in serial_dev.written_commands

    # Frame 2: Person candidate reappears in front of camera
    candidate_persons = [{"class_name": "person", "confidence": 0.85, "bbox": [200, 100, 300, 300], "type": "positive"}]
    follow_state, pi_latched, action = _run_follow(
        motor=motor,
        sensors=sensors,
        tracker=tracker,
        pid=pid,
        detections=candidate_persons,
        us_blocked=False,
        frame=frame,
        follow_state=follow_state,
        pi_latched=pi_latched,
        motors_enabled=True,
        throttler=throttler,
    )

    assert tracker.reacquire_called is True
    assert follow_state == "TRACKING"
    assert pi_latched is False
    assert "PI_GO" in serial_dev.written_commands
    assert "RE-IDENTIFIED" in action


def test_follow_mode_cliff_release_blocked_by_obstacle():
    """Verify that in follow mode, CLIFF_STOPPED does NOT release latch if non-target obstacle is ahead."""
    motor, serial_dev = create_mock_motor()
    motor.latest_state = "CLIFF_STOPPED"
    motor.latest_sonar = {"left": 25.0, "right": 25.0}
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    class MockTracker:
        def update(self, frame):
            return True, (250, 100, 100, 200)

    tracker = MockTracker()
    from car.control.pid import PID
    pid = PID(P=0.3, I=0.01, D=0.05)

    # Imminent obstacle ahead (e.g. wall/barrier)
    dets = [{"class_name": "car", "confidence": 0.9, "bbox": [200, 100, 440, 350], "type": "positive"}]

    follow_state, pi_latched, action = _run_follow(
        motor=motor,
        sensors=sensors,
        tracker=tracker,
        pid=pid,
        detections=dets,
        us_blocked=False,
        frame=frame,
        follow_state="TRACKING",
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
    )

    assert "PI_GO" not in serial_dev.written_commands
    assert pi_latched is True


def test_follow_mode_steers_when_target_far():
    """Verify that when the target is far away and off-center, follow mode emits TURN instead of DRIVE."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    class MockTracker:
        def update(self, frame):
            # Target is far (height 40px / 480px = 8.3% < 10%) and far to the right (center_x = 550)
            return True, (500, 100, 100, 40)

    tracker = MockTracker()
    from car.control.pid import PID
    pid = PID(P=0.3, I=0.01, D=0.05)
    pid.SetPoint = 320.0

    _run_follow(
        motor=motor,
        sensors=sensors,
        tracker=tracker,
        pid=pid,
        detections=[],
        us_blocked=False,
        frame=frame,
        follow_state="TRACKING",
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
    )

    turn_cmds = [cmd for cmd in serial_dev.written_commands if cmd.startswith("TURN")]
    assert len(turn_cmds) == 1
    assert "TURN" in turn_cmds[0]


def test_motor_controller_thread_safety():
    """Verify concurrent command sending and telemetry reading is thread-safe."""
    import threading
    motor, _ = create_mock_motor(incoming_lines=["[TELEM] US LEFT=20.0 RIGHT=25.0"] * 100)

    errors = []

    def writer_thread():
        try:
            for _ in range(50):
                motor.drive(180, 300)
                motor.stop()
                motor.status()
        except Exception as e:
            errors.append(e)

    def reader_thread():
        try:
            for _ in range(50):
                motor.read_telemetry()
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=writer_thread)
    t2 = threading.Thread(target=reader_thread)
    t1.start()
    t2.start()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    assert len(errors) == 0


def test_auto_connect_cross_platform_port_discovery(monkeypatch):
    """Verify auto_connect discovers system ports using serial.tools.list_ports."""
    import types
    import car.control.motor_controller as mc

    class MockPort:
        def __init__(self, device):
            self.device = device

    mock_serial = types.ModuleType("serial")
    mock_tools = types.ModuleType("tools")
    mock_list_ports = types.ModuleType("list_ports")
    mock_list_ports.comports = lambda: [MockPort("/dev/custom_pi_uart")]
    mock_tools.list_ports = mock_list_ports
    mock_serial.tools = mock_tools

    attempted_ports = []

    def mock_serial_init(port, baud, timeout):
        attempted_ports.append(port)
        raise Exception("Mock fail")

    mock_serial.Serial = mock_serial_init

    monkeypatch.setattr(mc, "serial", mock_serial)

    motor = mc.MotorController(port="auto")
    connected = motor.auto_connect()
    assert connected is False
    assert "/dev/custom_pi_uart" in attempted_ports


def test_check_obstacle_ahead_handles_malformed_detections():
    """Verify _check_obstacle_ahead does not raise on None or invalid detections."""
    malformed = [
        None,
        "not_a_dict",
        {"bbox": None},
        {"bbox": [10, 20]},
        {"bbox": [10, 20, 30]},
        {"bbox": ["a", "b", "c", "d"]},
    ]
    action, spd = _check_obstacle_ahead(malformed, 640, 480)
    assert action == "CLEAR"


def test_telemetry_endpoint_includes_connectivity_and_awb():
    """Verify GET /telemetry includes port, connected, and awb fields."""
    from tests.test_main_coordination import simulate_http_get
    sys_state = SystemState()
    motor, _ = create_mock_motor()
    sys_state.motor = motor
    sys_state.current_awb = "daylight"

    status, headers, body = simulate_http_get("/telemetry", sys_state)
    import json
    data = json.loads(body.decode("utf-8"))
    assert data["port"] == "MOCK_PORT"
    assert data["connected"] is True
    assert data["awb"] == "daylight"


def test_test_turn_endpoint():
    """Verify GET /test_turn triggers turn maneuver."""
    from tests.test_main_coordination import simulate_http_get
    sys_state = SystemState()
    motor, serial_dev = create_mock_motor()
    sys_state.motor = motor
    sys_state.motors_enabled = True
    sys_state.pi_latched = False

    status, headers, body = simulate_http_get("/test_turn?angle=60", sys_state)
    import json
    data = json.loads(body.decode("utf-8"))
    assert data["status"] == "ok"
    assert data["executed"] is True
    assert data["angle"] == 60
    assert any("TURN 60" in cmd for cmd in serial_dev.written_commands)


def test_arduino_pi_stopped_state_sync():
    """Verify that if Arduino is in PI_STOPPED, autonomous mode clears it when path is clear."""
    motor, serial_dev = create_mock_motor()
    motor.latest_state = "PI_STOPPED"
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame with path clear
    pi_latched, action = _run_autonomous(
        motor=motor,
        sensors=sensors,
        detections=[],
        us_blocked=False,
        frame=frame,
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
    )

    assert "PI_GO" in serial_dev.written_commands
