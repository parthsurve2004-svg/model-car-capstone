"""
test_follow_me.py — Unit Tests for Follow-Me Person Tracking & Re-ID
====================================================================
Verifies:
  1. 3-Zone Clothing & Figure Profile Discrimination (> 0.85 self, < 0.40 cross)
  2. Multi-Frame Calibration (8 frames -> normalized profile & is_tracking=True)
  3. Anti-Confusion When Bystander Crosses Paths (locks to matching target, yields bystander)
  4. Non-Target Person Treated as Obstacle in _run_follow() (bystander stops car, target does not)
  5. Last-Seen Side Memory & Directional Search in RE_ACQUIRE (left exit -> turn(-28°) up to 4 steps)
"""

import time
import numpy as np
import pytest

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
from car.perception.tracker import (
    ObjectTracker,
    _build_appearance_profile,
    _compare_profiles,
    _box_iou,
)
from car.control.motor_controller import MotorController
from car.perception.sensors import SensorManager
from car.control.pid import PID
from car.main import CommandThrottler, SystemState, _run_follow


class MockSerialDevice:
    """Mock serial transport recording written commands."""

    def __init__(self, incoming_lines=None):
        self.is_open = True
        self.written_commands = []
        self._in_buffer = list(incoming_lines or [])

    def readline(self):
        if self._in_buffer:
            return (self._in_buffer.pop(0) + "\n").encode("utf-8")
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
    motor = MotorController(port="MOCK_PORT", baud=115200)
    mock_serial = MockSerialDevice(incoming_lines)
    motor._serial = mock_serial
    return motor, mock_serial


def make_synthetic_person(
    shirt_bgr,
    pants_bgr,
    bbox=(200, 100, 100, 300),
    frame_shape=(480, 640, 3),
    noise_amplitude=5,
):
    """Generates a synthetic frame containing a two-zone clothed person with subtle texture."""
    f = np.full(frame_shape, 128, dtype=np.uint8)
    x, y, w, h = bbox
    # Upper torso (shirt)
    f[y + int(0.15 * h):y + int(0.52 * h), x:x + w] = shirt_bgr
    # Lower body (pants)
    f[y + int(0.55 * h):y + int(0.92 * h), x:x + w] = pants_bgr
    if noise_amplitude > 0:
        noise = np.random.randint(-noise_amplitude, noise_amplitude + 1, f.shape, dtype=np.int16)
        f = np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return f


# =============================================================================
# 1. 3-Zone Clothing & Figure Profile Discrimination
# =============================================================================

def test_clothing_and_figure_profile_discrimination():
    """Verify distinct clothing and body figure ratios discriminate target vs stranger."""
    # Person A: Blue shirt + Dark Red pants, tall/slender aspect ratio (100x300 -> AR=3.0)
    frame_a = make_synthetic_person([255, 0, 0], [0, 0, 180], bbox=(200, 80, 100, 300))
    prof_a1 = _build_appearance_profile(frame_a, (200, 80, 100, 300))

    # Slightly shifted frame of Person A (different noise sample)
    frame_a_var = make_synthetic_person([255, 0, 0], [0, 0, 180], bbox=(202, 82, 100, 300))
    prof_a2 = _build_appearance_profile(frame_a_var, (202, 82, 100, 300))

    # Person B: Green shirt + Yellow pants, wider aspect ratio (140x210 -> AR=1.5)
    frame_b = make_synthetic_person([0, 255, 0], [0, 255, 255], bbox=(220, 100, 140, 210))
    prof_b = _build_appearance_profile(frame_b, (220, 100, 140, 210))

    sim_self = _compare_profiles(prof_a1, prof_a2)
    sim_cross = _compare_profiles(prof_a1, prof_b)

    assert sim_self > 0.85, f"Self-similarity too low: {sim_self:.3f} <= 0.85"
    assert sim_cross < 0.40, f"Cross-person similarity too high: {sim_cross:.3f} >= 0.40"
    assert sim_self - sim_cross >= FOLLOW_ANTI_CONFUSION_MARGIN


def test_appearance_profile_empty_or_boundary_box():
    """Verify profile extraction safely handles degenerate bounding boxes."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # 0-width or 0-height
    prof_empty = _build_appearance_profile(frame, (100, 100, 0, 0))
    assert prof_empty == {}

    # Outside image frame bounds
    prof_oob = _build_appearance_profile(frame, (700, 500, 50, 50))
    assert prof_oob == {}

    # None comparison
    assert _compare_profiles({}, {}) == 0.0
    assert _compare_profiles(prof_empty, {"aspect_ratio": 2.0}) == 0.0


# =============================================================================
# 2. Multi-Frame Calibration
# =============================================================================

def test_multi_frame_calibration_progression_and_lock():
    """Verify 8-frame calibration gathers samples, averages signature, and transitions to tracking."""
    tracker = ObjectTracker()
    frame = make_synthetic_person([255, 0, 0], [0, 0, 180], bbox=(200, 100, 100, 300))
    init_box = (200, 100, 100, 300)

    # Start calibration
    tracker.start_calibration(frame, init_box, total_frames=FOLLOW_CALIBRATION_FRAMES)
    assert tracker.is_calibrating is True
    assert tracker.is_calibrated is False
    assert tracker.calibration_progress == pytest.approx(1.0 / FOLLOW_CALIBRATION_FRAMES, rel=1e-2)

    dets = [{"class_name": "person", "bbox": [200, 100, 300, 400], "confidence": 0.92}]

    # Step through remaining 7 frames
    for f_idx in range(2, FOLLOW_CALIBRATION_FRAMES):
        done = tracker.step_calibration(frame, detections=dets)
        assert done is False
        assert tracker.is_calibrating is True
        expected_prog = f_idx / FOLLOW_CALIBRATION_FRAMES
        assert tracker.calibration_progress == pytest.approx(expected_prog, rel=1e-2)

    # 8th frame: calibration completes
    done = tracker.step_calibration(frame, detections=dets)
    assert done is True
    assert tracker.is_calibrating is False
    assert tracker.is_calibrated is True
    assert tracker.is_tracking is True
    assert tracker.calibration_progress == 1.0
    assert tracker._owner_profile.get("upper_hist") is not None
    assert tracker._owner_profile.get("aspect_ratio") == pytest.approx(3.0, rel=0.1)


# =============================================================================
# 3. Anti-Confusion When Bystander Crosses Paths
# =============================================================================

def test_anti_confusion_locks_target_and_yields_bystander():
    """Verify that when a bystander overlaps the target, tracker stays locked on target."""
    tracker = ObjectTracker()

    # Target: Person A (Blue shirt, Dark Red pants)
    frame_a = make_synthetic_person([255, 0, 0], [0, 0, 180], bbox=(180, 100, 100, 300))
    tracker.lock_on(frame_a, (180, 100, 100, 300))
    assert tracker.is_calibrated is True

    # Frame with both Target A and Bystander B (Green shirt, Yellow pants) overlapping
    frame_both = frame_a.copy()
    # Bystander B adjacent/overlapping at x=240
    frame_both[140:240, 240:340] = [0, 255, 0]
    frame_both[250:370, 240:340] = [0, 255, 255]

    det_target = {
        "class_name": "person",
        "confidence": 0.90,
        "bbox": [180, 100, 280, 400],
        "type": "positive",
    }
    det_bystander = {
        "class_name": "person",
        "confidence": 0.88,
        "bbox": [240, 100, 340, 400],
        "type": "positive",
    }

    is_tracking, bbox, tgt_det, non_targets, score = tracker.update_with_detections(
        frame_both, [det_target, det_bystander]
    )

    assert is_tracking is True
    assert tgt_det == det_target, "Tracker wrongly locked onto bystander!"
    assert len(non_targets) == 1
    assert non_targets[0] == det_bystander
    assert score >= FOLLOW_SIMILARITY_THRESHOLD


def test_anti_confusion_drops_tracking_when_distractor_occludes():
    """Verify that when target is fully occluded by a distractor, tracker drops to RE_ACQUIRE."""
    tracker = ObjectTracker()

    # Learn Target Profile (Blue shirt)
    frame_target = make_synthetic_person([255, 0, 0], [0, 0, 180], bbox=(200, 100, 100, 300))
    tracker.lock_on(frame_target, (200, 100, 100, 300))

    # Frame now ONLY has Distractor B (Green shirt) at the exact same location
    frame_distractor = make_synthetic_person([0, 255, 0], [0, 255, 255], bbox=(200, 100, 100, 300))
    det_distractor = {
        "class_name": "person",
        "confidence": 0.92,
        "bbox": [200, 100, 300, 400],
        "type": "positive",
    }

    is_tracking, bbox, tgt_det, non_targets, score = tracker.update_with_detections(
        frame_distractor, [det_distractor]
    )

    assert is_tracking is False
    assert bbox is None
    assert tgt_det is None
    assert score < FOLLOW_SIMILARITY_THRESHOLD


# =============================================================================
# 4. Non-Target Person Treated as Obstacle in _run_follow()
# =============================================================================

def test_run_follow_treats_bystander_as_obstacle():
    """Verify non-target bystanders trigger PI_STOP in front corridor while target alone follows."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    tracker = ObjectTracker()
    pid = PID(P=0.3, I=0.01, D=0.05)
    throttler = CommandThrottler()

    frame = make_synthetic_person([255, 0, 0], [0, 0, 180], bbox=(280, 120, 80, 240))
    tracker.lock_on(frame, (280, 120, 80, 240))

    # 1. Target person alone in view -> follow without PI_STOP
    det_target = {
        "class_name": "person",
        "confidence": 0.90,
        "bbox": [280, 120, 360, 360],
        "type": "positive",
    }
    follow_state, pi_latched, action = _run_follow(
        motor=motor,
        sensors=sensors,
        tracker=tracker,
        pid=pid,
        detections=[det_target],
        us_blocked=False,
        frame=frame,
        follow_state="TRACKING",
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
    )
    assert pi_latched is False
    assert "PI_STOP" not in serial_dev.written_commands
    assert "FOLLOWING" in action

    # 2. Target person present PLUS large bystander blocking center corridor
    frame_with_bystander = frame.copy()
    frame_with_bystander[140:240, 180:320] = [0, 255, 0]
    frame_with_bystander[250:370, 180:320] = [0, 255, 255]

    det_bystander = {
        "class_name": "person",
        "confidence": 0.88,
        "bbox": [180, 120, 420, 420],  # Center corridor blocking area
        "type": "positive",
    }

    follow_state, pi_latched, action = _run_follow(
        motor=motor,
        sensors=sensors,
        tracker=tracker,
        pid=pid,
        detections=[det_target, det_bystander],
        us_blocked=False,
        frame=frame_with_bystander,
        follow_state="TRACKING",
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
    )
    assert pi_latched is True
    assert "PI_STOP" in serial_dev.written_commands
    assert "PI_STOP" in action
    assert "Non-Target Obstacle" in action


# =============================================================================
# 5. Last-Seen Side Memory & Directional Search in RE_ACQUIRE
# =============================================================================

def test_last_seen_side_memory_and_stepped_search_turns():
    """Verify target walking off left side records LEFT and triggers stepped turns in RE_ACQUIRE."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    tracker = ObjectTracker()
    pid = PID(P=0.3, I=0.01, D=0.05)
    throttler = CommandThrottler()

    frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    # Frame 1: Center position
    tracker._update_side_memory((270, 100, 100, 300), 640)
    assert tracker.last_seen_side == "CENTER"

    # Frame 2: Moving left
    tracker._update_side_memory((200, 100, 100, 300), 640)
    # Frame 3: Far left edge
    tracker._update_side_memory((50, 100, 100, 300), 640)
    assert tracker.last_seen_side == "LEFT"

    # Enter RE_ACQUIRE with target out of frame
    now = time.monotonic()
    assert tracker.should_step_search(now) is True

    # Run step 1 of search turn
    follow_state, pi_latched, action = _run_follow(
        motor=motor,
        sensors=sensors,
        tracker=tracker,
        pid=pid,
        detections=[],
        us_blocked=False,
        frame=frame,
        follow_state="RE_ACQUIRE",
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
    )
    assert "SEARCHING LEFT (Step 1/4)" in action
    turn_cmds = [cmd for cmd in serial_dev.written_commands if cmd.startswith(f"TURN -{FOLLOW_SEARCH_STEP_DEG}")]
    assert len(turn_cmds) == 1
    assert tracker.search_steps_taken == 1

    # Simulate subsequent search steps until maximum (FOLLOW_SEARCH_MAX_STEPS = 4)
    for step in range(2, FOLLOW_SEARCH_MAX_STEPS + 1):
        # Artificially advance time past FOLLOW_SEARCH_INTERVAL_S
        tracker.last_search_step_time = time.monotonic() - (FOLLOW_SEARCH_INTERVAL_S + 0.1)
        follow_state, pi_latched, action = _run_follow(
            motor=motor,
            sensors=sensors,
            tracker=tracker,
            pid=pid,
            detections=[],
            us_blocked=False,
            frame=frame,
            follow_state="RE_ACQUIRE",
            pi_latched=False,
            motors_enabled=True,
            throttler=throttler,
        )
        assert f"Step {step}/{FOLLOW_SEARCH_MAX_STEPS}" in action

    assert tracker.search_steps_taken == FOLLOW_SEARCH_MAX_STEPS

    # Step 5: Max steps reached -> hold stationary and do not issue further turns
    tracker.last_search_step_time = time.monotonic() - (FOLLOW_SEARCH_INTERVAL_S + 0.1)
    turn_count_before = len([c for c in serial_dev.written_commands if c.startswith("TURN")])
    follow_state, pi_latched, action = _run_follow(
        motor=motor,
        sensors=sensors,
        tracker=tracker,
        pid=pid,
        detections=[],
        us_blocked=False,
        frame=frame,
        follow_state="RE_ACQUIRE",
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
    )
    turn_count_after = len([c for c in serial_dev.written_commands if c.startswith("TURN")])
    assert turn_count_after == turn_count_before
    assert "SEARCH HOLD" in action


# =============================================================================
# 6. Reset Target Functionality
# =============================================================================

def test_reset_target_clears_profile_and_returns_wait_calibration():
    """Verify reset request completely clears target memory and returns to WAIT_CALIBRATION."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    tracker = ObjectTracker()
    pid = PID(P=0.3, I=0.01, D=0.05)
    throttler = CommandThrottler()
    sys_state = SystemState()

    frame = make_synthetic_person([255, 0, 0], [0, 0, 180], bbox=(200, 100, 100, 300))
    tracker.lock_on(frame, (200, 100, 100, 300))
    assert tracker.is_calibrated is True

    sys_state.pending_reset_target = True
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
        sys_state=sys_state,
    )

    assert follow_state == "WAIT_CALIBRATION"
    assert tracker.is_calibrated is False
    assert tracker.is_tracking is False
    assert sys_state.target_bbox is None
    assert sys_state.pending_reset_target is False
    assert "TARGET RESET" in action


# =============================================================================
# 7. Edge Cases: None Ultrasonic Handling, Anti-Confusion Margin & Abort
# =============================================================================

def test_run_follow_handles_none_ultrasonic_in_reacquire():
    """Verify _run_follow does not crash with TypeError when us has None values in RE_ACQUIRE."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    sensors.read_ultrasonic = lambda: {"left": None, "right": 45.0}

    tracker = ObjectTracker()
    tracker.last_seen_side = "LEFT"
    tracker.search_steps_taken = 0
    pid = PID(P=0.3, I=0.01, D=0.05)
    throttler = CommandThrottler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    follow_state, pi_latched, action = _run_follow(
        motor=motor,
        sensors=sensors,
        tracker=tracker,
        pid=pid,
        detections=[],
        us_blocked=False,
        frame=frame,
        follow_state="RE_ACQUIRE",
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
    )
    assert follow_state == "RE_ACQUIRE"
    assert "SEARCHING LEFT" in action


def test_anti_confusion_rejects_overlapping_distractor_within_margin():
    """Verify that when an overlapping distractor is within anti-confusion margin, tracking drops to RE_ACQUIRE."""
    tracker = ObjectTracker()
    frame_target = make_synthetic_person([255, 0, 0], [0, 0, 180], bbox=(200, 100, 100, 300))
    tracker.lock_on(frame_target, (200, 100, 100, 300))
    assert tracker.is_calibrated is True

    # Frame with Target (blue/red) and very similar distractor (blue/slightly darker red) overlapping
    frame_both = frame_target.copy()
    # Distractor overlapping at x=220
    frame_both[100 + int(0.15 * 300):100 + int(0.52 * 300), 220:320] = [255, 10, 10]
    frame_both[100 + int(0.55 * 300):100 + int(0.92 * 300), 220:320] = [0, 0, 170]

    det_target = {"class_name": "person", "bbox": [200, 100, 300, 400], "confidence": 0.90}
    det_distractor = {"class_name": "person", "bbox": [220, 100, 320, 400], "confidence": 0.89}

    # Override profiles to force scores within margin (0.60 vs 0.58 -> diff 0.02 < 0.08)
    import car.perception.tracker as tmod
    orig_compare = tmod._compare_profiles
    try:
        def mock_compare(prof, cand):
            c_ar = cand.get("aspect_ratio", 3.0)
            return 0.60 if c_ar >= 2.9 else 0.58
        tmod._compare_profiles = mock_compare

        is_tracking, bbox, tgt_det, non_targets, score = tracker.update_with_detections(
            frame_both, [det_target, det_distractor]
        )
        # Anti-confusion gate should drop tracking because of ambiguous overlapping distractor
        assert is_tracking is False
        assert bbox is None
    finally:
        tmod._compare_profiles = orig_compare


def test_calibration_aborts_when_target_lost():
    """Verify that if the target person is lost for 5 consecutive frames during calibration, calibration aborts."""
    tracker = ObjectTracker()
    frame = make_synthetic_person([255, 0, 0], [0, 0, 180], bbox=(200, 100, 100, 300))
    tracker.start_calibration(frame, (200, 100, 100, 300), total_frames=8)
    assert tracker.is_calibrating is True

    # Force tracker.update to fail (person walked out of view)
    class FailingTracker:
        def update(self, f):
            return False, None
    tracker.tracker = FailingTracker()

    # Step calibration 5 times with no detections
    for i in range(4):
        done = tracker.step_calibration(frame, detections=[])
        assert done is False
        assert tracker.is_calibrating is True

    # 5th lost frame: should abort
    done = tracker.step_calibration(frame, detections=[])
    assert done is False
    assert tracker.is_calibrating is False
    assert tracker.is_calibrated is False
    assert tracker.is_tracking is False


def test_run_follow_aborts_calibration_and_reports_status():
    """Verify _run_follow catches aborted calibration and transitions to WAIT_CALIBRATION with HUD feedback."""
    motor, serial_dev = create_mock_motor()
    sensors = SensorManager(motor_controller=motor)
    tracker = ObjectTracker()
    pid = PID(P=0.3, I=0.01, D=0.05)
    throttler = CommandThrottler()
    sys_state = SystemState()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker.start_calibration(frame, (200, 100, 100, 300), total_frames=8)

    class FailingTracker:
        def update(self, f):
            return False, None
    tracker.tracker = FailingTracker()

    # Simulate 4 lost frames
    for _ in range(4):
        tracker.step_calibration(frame, detections=[])

    # 5th frame run inside _run_follow
    follow_state, pi_latched, action = _run_follow(
        motor=motor,
        sensors=sensors,
        tracker=tracker,
        pid=pid,
        detections=[],
        us_blocked=False,
        frame=frame,
        follow_state="CALIBRATING",
        pi_latched=False,
        motors_enabled=True,
        throttler=throttler,
        sys_state=sys_state,
    )

    assert follow_state == "WAIT_CALIBRATION"
    assert sys_state.follow_state == "WAIT_CALIBRATION"
    assert "CALIBRATION ABORTED: Person Lost" in action


def test_stale_inference_does_not_reanchor_kcf():
    """Verify that when fresh_inference is False, KCF does not re-anchor to stale YOLO boxes."""
    tracker = ObjectTracker()
    frame = make_synthetic_person([255, 0, 0], [0, 0, 180], bbox=(200, 100, 100, 300))
    tracker.lock_on(frame, (200, 100, 100, 300))

    # KCF tracked box moves to (220, 100, 100, 300)
    tracker.bbox = (220, 100, 100, 300)
    tracker.update = lambda f: (True, (220, 100, 100, 300))

    # Stale YOLO detection still at (200, 100, 300, 400)
    stale_det = {"class_name": "person", "bbox": [200, 100, 300, 400], "confidence": 0.90}

    # When fresh_inference=False, tracker should keep KCF tracked box (220, 100, 100, 300)
    is_tracking, bbox, tgt_det, non_targets, score = tracker.update_with_detections(
        frame, [stale_det], fresh_inference=False
    )
    assert is_tracking is True
    assert bbox == (220, 100, 100, 300)
