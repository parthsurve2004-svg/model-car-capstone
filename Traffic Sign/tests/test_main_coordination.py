"""
test_main_coordination.py — Verification of car/main.py Pi-Arduino Coordination & Streaming
============================================================================================
Tests state machine synchronization, command throttling, Arduino autonomous state yielding,
bench-top safe mode, live HUD rendering, and HTTP MJPEG streaming on port 8080.
"""

import io
import json
import threading
import time
import urllib.request
import urllib.error
import pytest
import numpy as np

from car.main import (
    CommandThrottler,
    SystemState,
    StreamingOutput,
    StreamingServer,
    DiagnosticHandler,
    draw_hud,
    _run_autonomous,
    _run_follow,
    _check_obstacle_ahead,
    is_valid_us,
    CRUISE_SPEED,
    SLOW_SPEED,
    BBOX_STOP_THRESHOLD,
    BBOX_SLOW_THRESHOLD,
)
from car.control.motor_controller import MotorController
from car.perception.sensors import SensorManager
from car.perception.tracker import ObjectTracker
from car.control.pid import PID


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


# =============================================================================
# 1. State Machine & Protocol Synchronization Tests
# =============================================================================

def test_autonomous_pi_stop_and_latch_preservation():
    """Verify that an imminent obstacle triggers PI_STOP and pi_latched remains True."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Imminent obstacle: bbox height is 50% of frame height (threshold is 40%)
    detections = [{
        "class_name": "person",
        "confidence": 0.85,
        "bbox": [200, 100, 440, 340],  # height = 240 / 480 = 50%
        "type": "positive",
    }]

    # Frame 1: Imminent obstacle triggers PI_STOP
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

    assert pi_latched is True
    assert "PI_STOP" in serial_dev.written_commands
    assert "PI_STOP" in action

    # Frame 2: Skipped inference frame (detections preserved via caching in main loop)
    # pi_latched must NOT oscillate back to False!
    pi_latched, action = _run_autonomous(
        motor=motor,
        sensors=sensors,
        detections=detections,
        us_blocked=False,
        frame=frame,
        pi_latched=pi_latched,
        motors_enabled=True,
        throttler=throttler,
    )

    assert pi_latched is True
    assert "PI_STOP" in action
    # Must NOT send another PI_STOP if already latched
    assert serial_dev.written_commands.count("PI_STOP") == 1


def test_autonomous_pi_go_release():
    """Verify that clearing the path releases the Pi camera latch with PI_GO."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Currently latched with no obstacles ahead
    pi_latched, action = _run_autonomous(
        motor=motor,
        sensors=sensors,
        detections=[],
        us_blocked=False,
        frame=frame,
        pi_latched=True,
        motors_enabled=True,
        throttler=throttler,
    )

    assert pi_latched is False
    assert "PI_GO" in serial_dev.written_commands


def test_autonomous_yields_on_arduino_avoiding():
    """Verify that main.py pauses drive commands when Arduino is in STATE_AVOIDING."""
    motor, serial_dev = create_mock_motor()
    motor.latest_state = "AVOIDING"
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

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

    assert "AVOIDING" in action
    # No DRIVE command must be sent while Arduino is performing autonomous avoidance!
    drive_cmds = [cmd for cmd in serial_dev.written_commands if cmd.startswith("DRIVE")]
    assert len(drive_cmds) == 0


def test_autonomous_yields_on_arduino_cliff_states():
    """Verify that main.py respects CLIFF_REVERSING and CLIFF_STOPPED states."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. CLIFF_REVERSING
    motor.latest_state = "CLIFF_REVERSING"
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
    assert "CLIFF_REVERSING" in action
    assert len([cmd for cmd in serial_dev.written_commands if cmd.startswith("DRIVE")]) == 0

    # 2. CLIFF_STOPPED — when clear, sends PI_GO to release
    motor.latest_state = "CLIFF_STOPPED"
    motor.latest_sonar = {"left": 20.0, "right": 20.0}
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


def test_follow_mode_state_transitions_and_obstacle_stop():
    """Verify follow-me mode handles target locking, non-target obstacle PI_STOP, and latch clearing."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    tracker = ObjectTracker()
    pid = PID(P=0.3, I=0.01, D=0.05)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Non-target obstacle (e.g. car or box) triggers PI_STOP
    non_target_dets = [
        {"class_name": "car", "confidence": 0.85, "bbox": [150, 150, 490, 450], "type": "positive"}
    ]
    follow_state, pi_latched, action = _run_follow(
        motor=motor,
        sensors=sensors,
        tracker=tracker,
        pid=pid,
        detections=non_target_dets,
        us_blocked=False,
        frame=frame,
        follow_state="LOCK_ON",
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
    )

    assert pi_latched is True
    assert "PI_STOP" in serial_dev.written_commands
    assert "PI_STOP" in action

    # 2. Non-target obstacle clears -> PI_GO emitted
    follow_state, pi_latched, action = _run_follow(
        motor=motor,
        sensors=sensors,
        tracker=tracker,
        pid=pid,
        detections=[],
        us_blocked=False,
        frame=frame,
        follow_state=follow_state,
        pi_latched=True,
        motors_enabled=True,
        throttler=throttler,
    )

    assert pi_latched is False
    assert "PI_GO" in serial_dev.written_commands

    # 3. Arduino autonomous state yielding in follow mode
    motor.latest_state = "AVOIDING"
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
    assert "AVOIDING" in action


# =============================================================================
# 2. Command Throttling Verification
# =============================================================================

def test_command_throttling_prevents_serial_flooding():
    """Verify that DRIVE commands are throttled to a controlled cadence."""
    throttler = CommandThrottler(throttle_interval_s=0.10, drive_duration_ms=250)
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Run 10 rapid iterations (e.g. simulated 100 FPS loop with zero delay)
    for _ in range(10):
        _run_autonomous(
            motor=motor,
            sensors=sensors,
            detections=[],
            us_blocked=False,
            frame=frame,
            pi_latched=False,
            motors_enabled=True,
            throttler=throttler,
        )

    # In 10 immediate iterations without sleeping, only 1 DRIVE command should have been emitted
    drive_cmds = [cmd for cmd in serial_dev.written_commands if cmd.startswith("DRIVE")]
    assert len(drive_cmds) == 1

    # Now sleep past throttle interval
    time.sleep(0.12)
    _run_autonomous(
        motor=motor,
        sensors=sensors,
        detections=[],
        us_blocked=False,
        frame=frame,
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
    )

    drive_cmds = [cmd for cmd in serial_dev.written_commands if cmd.startswith("DRIVE")]
    assert len(drive_cmds) == 2


# =============================================================================
# 3. Bench-Test Safe Mode Verification
# =============================================================================

def test_bench_safe_mode_suppresses_motor_actuation():
    """Verify that bench-safe mode suppresses physical motor drive/turn commands."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    pi_latched, action = _run_autonomous(
        motor=motor,
        sensors=sensors,
        detections=[],
        us_blocked=False,
        frame=frame,
        pi_latched=False,
        motors_enabled=False,  # Bench safe!
        throttler=throttler,
    )

    assert "BENCH SAFE" in action
    drive_cmds = [cmd for cmd in serial_dev.written_commands if cmd.startswith("DRIVE")]
    assert len(drive_cmds) == 0


# =============================================================================
# 4. Telemetry Parsing & Auto-Discovery Verification
# =============================================================================

def test_telemetry_draining_and_state_updates():
    """Verify continuous telemetry parsing of US distances and Arduino states."""
    telem_lines = [
        "[TELEM] US LEFT=18.5 RIGHT=42.1",
        "[TELEM] STATE=DRIVING SPEED=180",
        "[TELEM] AVOID RIGHT",
        "[TELEM] DONE AVOID",
    ]
    motor, _ = create_mock_motor(incoming_lines=telem_lines)

    motor.read_telemetry()
    assert motor.latest_sonar["left"] == 18.5
    assert motor.latest_sonar["right"] == 42.1

    motor.read_telemetry()
    assert motor.latest_state == "DRIVING"

    motor.read_telemetry()
    assert motor.latest_state == "AVOIDING"

    motor.read_telemetry()
    assert motor.latest_state == "IDLE"


def test_dry_run_fallback_when_disconnected():
    """Verify graceful dry-run degradation when no microcontroller is present."""
    motor = MotorController(port="/dev/nonexistent_serial_port_xyz")
    connected = motor.connect()
    assert connected is False
    assert motor.port == "SIMULATED"
    # Calling commands in dry-run mode must NOT crash
    assert motor.drive(180, 300) is False
    assert motor.stop() is False


# =============================================================================
# 5. Diagnostic HUD Overlay Verification
# =============================================================================

def test_draw_hud_renders_without_errors():
    """Verify HUD overlay renders active boxes, zones, and diagnostic text."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        {"class_name": "person", "confidence": 0.88, "bbox": [100, 100, 200, 300], "type": "positive"},
        {"class_name": "pothole", "confidence": 0.52, "bbox": [350, 280, 450, 360], "type": "negative"},
    ]
    hud_frame = draw_hud(
        frame=frame,
        detections=detections,
        action_label="CRUISING FORWARD (PWM 180)",
        latest_sonar={"left": 35.0, "right": 55.0},
        latest_state="DRIVING",
        port_name="/dev/ttyUSB0",
        fps=29.4,
        awb_mode="daylight",
        mode="autonomous",
        motors_enabled=True,
        invert_motors=False,
        pi_latched=False,
    )

    assert hud_frame.shape == frame.shape
    # Frame should no longer be completely black
    assert np.any(hud_frame > 0)


# =============================================================================
# 6. Headless HTTP Live Stream & Web Dashboard on Port 8080
# =============================================================================

class MockHttpSocket:
    """Simulates an accepted socket connection for BaseHTTPRequestHandler testing."""
    def __init__(self, request_bytes: bytes):
        self.rfile = io.BytesIO(request_bytes)
        self.wfile = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        if 'r' in mode:
            return self.rfile
        return self.wfile

    def sendall(self, data):
        self.wfile.write(data)


def simulate_http_get(path: str, sys_state: SystemState):
    class MockServer:
        def __init__(self, state):
            self.sys_state = state

    raw_req = f"GET {path} HTTP/1.0\r\nHost: localhost:8080\r\n\r\n".encode("utf-8")
    sock = MockHttpSocket(raw_req)
    server = MockServer(sys_state)
    DiagnosticHandler(sock, ("127.0.0.1", 8080), server)

    resp_data = sock.wfile.getvalue()
    parts = resp_data.split(b"\r\n\r\n", 1)
    header_block = parts[0].decode("utf-8", errors="replace")
    body = parts[1] if len(parts) > 1 else b""
    status_code = int(header_block.splitlines()[0].split()[1])
    return status_code, header_block, body


def test_http_streaming_server_and_telemetry_endpoint():
    """Verify HTTP handler serves web dashboard, telemetry JSON, and controls."""
    sys_state = SystemState()
    motor, serial_dev = create_mock_motor()
    sys_state.motor = motor
    sys_state.streaming_output = StreamingOutput()
    sys_state.motors_enabled = True

    # 1. Test GET / (HTML dashboard)
    status, headers, body = simulate_http_get("/", sys_state)
    assert status == 200
    assert "text/html" in headers
    html = body.decode("utf-8")
    assert "Autonomous RC Car" in html
    assert "stream.mjpg" in html

    # 2. Test GET /telemetry
    status, headers, body = simulate_http_get("/telemetry", sys_state)
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert "sonar" in data
    assert data["motors_enabled"] is True

    # 3. Test GET /toggle_motors
    status, headers, body = simulate_http_get("/toggle_motors", sys_state)
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert data["motors_enabled"] is False
    assert sys_state.motors_enabled is False

    # 4. Test GET /emergency_stop
    status, headers, body = simulate_http_get("/emergency_stop", sys_state)
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert data["pi_latched"] is True
    assert sys_state.pi_latched is True
    assert "PI_STOP" in serial_dev.written_commands

    # 5. Test GET /release_latch
    status, headers, body = simulate_http_get("/release_latch", sys_state)
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert data["pi_latched"] is False
    assert sys_state.pi_latched is False
    assert "PI_GO" in serial_dev.written_commands

    # 6. Test GET /set_awb
    status, headers, body = simulate_http_get("/set_awb?mode=daylight", sys_state)
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert data["mode"] == "daylight"
