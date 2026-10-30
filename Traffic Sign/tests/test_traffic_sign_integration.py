import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from car.main import TrafficSignRuntime, SystemState, DiagnosticHandler, draw_traffic_signs, PAGE
from tests.test_traffic_sign_policy import Clock
from tests.test_main_coordination import MockHttpSocket, simulate_http_get, create_mock_motor

FRAME = np.zeros((480, 640, 3), np.uint8)

def detection(sign, near=True):
    return {"class_name": sign, "type": "traffic_sign", "confidence": .9,
            "bbox": [280, 120, 360, 220] if near else [300, 120, 340, 150]}

class Detector:
    detections = []
    def poll(self, frame):
        return self.detections


def setup_runtime():
    clock = Clock()
    detector = Detector()
    runtime = TrafficSignRuntime(detector, clock)
    motor = Mock(latest_state="IDLE")
    kwargs = dict(camera_valid=True, obstacle_detections=[], obstacle_time=clock(),
                  sonar={"left": 100., "right": 100.}, sonar_time=clock(),
                  pi_latched=False, motors_enabled=True)
    return clock, detector, runtime, motor, kwargs


def step(clock, runtime, motor, kwargs, **extra):
    kwargs = {**kwargs, "obstacle_time": clock(), "sonar_time": clock(), **extra}
    return runtime.step(FRAME, motor, **kwargs)


def confirm(clock, detector, runtime, motor, kwargs, sign, near=True, **extra):
    detector.detections = [detection(sign,near)]
    for _ in range(3):
        step(clock, runtime, motor, kwargs, **extra)
        clock.now += .25


def lane(locked=True):
    return SimpleNamespace(status="DUAL_LOCKED" if locked else "LANE_LOST",
                           steer_angle_deg=0., lateral_offset_pct=0.)


def test_automatic_turn_needs_confirmation_and_current_near_geometry():
    c,d,r,m,k = setup_runtime()
    confirm(c,d,r,m,k,"turn_left",near=False)
    assert "approaching" in r.status
    m.turn.assert_not_called()
    d.detections = [detection("turn_left")]
    assert step(c,r,m,k)
    m.turn.assert_called_once_with(-90, speed=210)
    c.now += .25
    assert step(c,r,m,k)
    m.turn.assert_called_once()
    c.now += 1
    assert "heading unverified" in step(c,r,m,k)
    m.stop.assert_called()
    step(c,r,m,k)
    m.turn.assert_called_once()


@pytest.mark.parametrize("sign", ["stop", "no_entry"])
def test_sign_hold_cannot_fall_through_to_latch_release(sign):
    c,d,r,m,k = setup_runtime()
    confirm(c,d,r,m,k,sign)
    m.latest_state = "PI_STOPPED"
    assert step(c,r,m,k,pi_latched=True)
    m.pi_go.assert_not_called()
    m.drive.assert_not_called()


def test_dwell_counts_only_stopped_intervals_not_reflex():
    c,d,r,m,k = setup_runtime()
    confirm(c,d,r,m,k,"stop")
    remaining = r.policy.stop_remaining
    m.latest_state = "AVOIDING"
    c.now += 2
    step(c,r,m,k)
    assert r.policy.stop_remaining == remaining
    m.latest_state = "IDLE"
    step(c,r,m,k)
    c.now += 3.1
    step(c,r,m,k)
    assert r.policy.hold_reason() is None


def test_dwell_expiration_never_overrides_obstacle():
    c,d,r,m,k = setup_runtime()
    confirm(c,d,r,m,k,"stop")
    c.now += 4
    hazard = [{"bbox": [200, 100, 400, 400], "confidence": .9, "class_name": "person"}]
    assert step(c,r,m,k,obstacle_detections=hazard) is None
    assert "existing obstacle decision" in r.status
    m.turn.assert_not_called()


@pytest.mark.parametrize("extra", [{"camera_valid": False}, {"obstacle_time": -2},
                                    {"sonar_time": -2},
                                    {"sonar": {"left":999., "right":999.}}])
def test_missing_sensing_holds_without_turn(extra):
    c,d,r,m,k = setup_runtime()
    confirm(c,d,r,m,k,"turn_right",near=False)
    d.detections = [detection("turn_right")]
    assert step(c,r,m,k,**extra)
    m.turn.assert_not_called()
    m.stop.assert_called()


def test_pending_instruction_survives_pre_turn_blockage_then_acts_when_clear():
    c,d,r,m,k = setup_runtime()
    confirm(c,d,r,m,k,"turn_left",sonar={"left":20.,"right":20.})
    m.turn.assert_not_called()
    assert r.policy.pending_turn == "turn_left"
    step(c,r,m,k)
    m.turn.assert_called_once_with(-90,speed=210)


def test_inference_exception_and_slow_sign_call_do_not_use_old_clearance():
    c,d,r,m,k = setup_runtime()
    d.poll = Mock(side_effect=RuntimeError("failed"))
    assert step(c,r,m,k)
    m.stop.assert_called()
    d.poll = lambda frame: (setattr(c, "now", c.now + 2) or [])
    assert step(c,r,m,k)
    m.turn.assert_not_called()


def test_safety_reflex_cancels_turn_and_fault_requires_reset():
    c,d,r,m,k = setup_runtime()
    confirm(c,d,r,m,k,"turn_left")
    m.latest_state = "AVOIDING"
    assert step(c,r,m,k)
    assert r.direction is None
    assert r.fault
    m.latest_state = "IDLE"
    step(c,r,m,k)
    m.turn.assert_called_once()
    step(c,r,m,k,reset=True)
    assert r.fault is None


def test_bench_safe_does_not_consume_pending_turn():
    c,d,r,m,k = setup_runtime()
    confirm(c,d,r,m,k,"turn_left",motors_enabled=False)
    m.turn.assert_not_called()
    assert r.policy.pending_turn == "turn_left"


def test_lane_turn_uses_bounded_pulses_then_three_current_reacquired_frames():
    c,d,r,m,k = setup_runtime()
    confirm(c,d,r,m,k,"turn_left",mode="lane",lane_result=lane())
    assert r.direction
    # Observe lane departure during a pulse, then return to fresh centered lane.
    step(c,r,m,k,mode="lane",lane_result=lane(False))
    for _ in range(50):
        c.now += .13
        step(c,r,m,k,mode="lane",lane_result=lane())
        if r.completed_status:
            break
    assert "lane reacquired" in r.completed_status
    assert all(call.args == (-12,) for call in m.turn.call_args_list)
    m.drive.assert_not_called()
    m.stop.assert_called()


def test_unchanged_original_lane_cannot_claim_turn_complete_and_times_out():
    c,d,r,m,k = setup_runtime()
    confirm(c,d,r,m,k,"turn_left",mode="lane",lane_result=lane())
    for _ in range(50):
        c.now += .13
        step(c,r,m,k,mode="lane",lane_result=lane())
    assert r.fault and "timed out" in r.fault
    assert r.completed_status is None
    m.drive.assert_not_called()
    step(c,r,m,k,reset=True,mode="lane",lane_result=lane())
    assert r.fault is None


def post(path, state):
    sock = MockHttpSocket(f"POST {path} HTTP/1.0\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n".encode())
    DiagnosticHandler(sock, ("127.0.0.1", 8080), SimpleNamespace(sys_state=state))
    header, body = sock.wfile.getvalue().split(b"\r\n\r\n", 1)
    return int(header.split()[1]), json.loads(body)


def test_endpoints_queue_reset_only_and_dashboard_has_live_sign_status():
    state = SystemState()
    state.motor = Mock()
    assert post('/traffic-signs/junction', state)[0] == 410
    assert post('/traffic-signs/reset', state)[0] == 409
    state.traffic_signs = object()
    assert post('/traffic-signs/reset', state)[0] == 202
    assert state.traffic_reset_requested
    assert not state.motor.mock_calls
    for path in ('/manual_drive?cmd=forward', '/test_turn', '/test_drive', '/set_mode?mode=manual'):
        assert simulate_http_get(path,state)[0] == 409
    assert not state.motor.mock_calls
    state.motor = None
    state.traffic_sign_status = {"enabled": True, "latest_sign": "turn_left", "bearing_deg": -10,
                                "status": "Left sign detected - taking left"}
    status,_,body=simulate_http_get('/telemetry',state)
    assert status == 200
    assert json.loads(body)["traffic_signs"] == state.traffic_sign_status
    assert 'traffic-sign-card' in PAGE and 'traffic-sign-action' in PAGE
    assert np.any(draw_traffic_signs(FRAME, [detection("turn_left")]) != FRAME)


def test_throttled_results_do_not_count_as_confirmation_or_stay_fresh():
    c,d,r,m,k = setup_runtime()
    d.detections = [detection("stop")]
    step(c,r,m,k)
    d.poll = lambda frame: None
    for _ in range(5):
        c.now += .1
        step(c,r,m,k)
    assert r.policy.hold_reason() is None
    c.now += 1
    assert step(c,r,m,k)
    assert "unavailable" in r.status


def test_stale_obstacle_clearance_never_yields_for_latch_release():
    c,d,r,m,k = setup_runtime()
    m.latest_state = "PI_STOPPED"
    assert step(c,r,m,k,pi_latched=True,obstacle_time=-2)
    m.pi_go.assert_not_called()
    m.turn.assert_not_called()


def test_sonar_receipt_timestamp_requires_new_complete_pair(monkeypatch):
    from car.control import motor_controller
    clock = Clock()
    monkeypatch.setattr(motor_controller.time,"monotonic",clock)
    motor,serial = create_mock_motor(["[TELEM] US LEFT=50 RIGHT=60", "[TELEM] STATE=IDLE", "[TELEM] US LEFT=80"])
    assert motor.latest_sonar_time is None
    motor.read_telemetry()
    assert motor.latest_sonar_time == 0
    clock.now=2
    motor.read_telemetry()
    motor.read_telemetry()
    assert motor.latest_sonar_time == 0


def test_follow_stop_and_no_entry_are_automatic():
    for sign in ("stop", "no_entry"):
        c,d,r,m,k = setup_runtime()
        confirm(c,d,r,m,k,sign,mode="follow")
        assert "stopped" in r.status
        m.stop.assert_called()
        m.turn.assert_not_called()


def test_follow_pause_preserves_owner_and_requires_reacquisition():
    from car.main import _pause_follow_for_sign
    from car.perception.tracker import ObjectTracker
    from car.control.pid import PID
    tracker = ObjectTracker()
    profile = {"upper_hist": np.array([1.0])}
    tracker._owner_profile = profile
    tracker.is_tracking = True
    tracker.bbox = (10, 20, 30, 50)
    state = SystemState()
    state.follow_state = "TRACKING"
    pid = PID()
    pid.SetPoint = 320
    pid.output = 10
    _pause_follow_for_sign(tracker,pid,state)
    assert tracker._owner_profile is profile
    assert not tracker.is_tracking and tracker.bbox is None
    assert state.follow_state == "RE_ACQUIRE"
    assert pid.output == 0
    assert pid.SetPoint == 320
    tracker.reset()
    _pause_follow_for_sign(tracker,pid,state)
    assert state.follow_state == "WAIT_CALIBRATION"


@pytest.mark.parametrize('mode', ['autonomous', 'lane', 'follow'])
def test_actual_pipeline_automatically_arbitrates_sign_and_reuses_current_lane(monkeypatch, mode, tmp_path):
    import sys
    import car.main as main
    import car.perception.traffic_signs as sign_module
    state = SystemState()
    c = Clock()
    monkeypatch.setattr(main.time, 'monotonic', c)
    monkeypatch.setattr(main.time, 'sleep', lambda seconds: setattr(c, 'now', c.now + seconds))
    monkeypatch.setattr(main.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(main, 'SystemState', lambda: state)
    general = SimpleNamespace(detect=Mock(return_value=[]))
    signs = Detector()
    signs.detections = [detection('turn_left')]
    monkeypatch.setattr(main, 'Detector', lambda **kwargs: general)
    monkeypatch.setattr(sign_module, 'TrafficSignDetector', lambda path: signs)
    motor,serial = create_mock_motor()
    motor.connect = lambda: False
    motor.latest_state = 'IDLE'
    motor.latest_sonar = {'left':100., 'right':100., 'rear':999.}
    motor.latest_sonar_time = c()
    monkeypatch.setattr(main, 'MotorController', lambda **kwargs: motor)
    monkeypatch.setitem(sys.modules, 'picamera2', SimpleNamespace(Picamera2=Mock(side_effect=RuntimeError('no CSI'))))
    class Camera:
        frames = 0
        def isOpened(self): return True
        def set(self,*args): pass
        def read(self):
            self.frames += 1
            if self.frames == 9:
                state.running = False
            return True, FRAME.copy()
        def release(self): pass
    camera = Camera()
    monkeypatch.setattr(main.cv2, 'VideoCapture', lambda *args: camera)
    lane_detector = main.ChartLaneDetector()
    original_detect = lane_detector.detect
    lane_detector.detect = Mock(side_effect=original_detect)
    monkeypatch.setattr(main, 'ChartLaneDetector', lambda **kwargs: lane_detector)
    original_lane = main._run_lane
    seen_lane_results = []
    def run_lane(**kwargs):
        seen_lane_results.append(kwargs.get('lane_result'))
        return original_lane(**kwargs)
    monkeypatch.setattr(main, '_run_lane', run_lane)
    model_path = tmp_path / 'injected-test-model.pt'
    model_path.touch()
    main.run_pipeline(mode=mode, enable_stream=False, traffic_sign_model=str(model_path))
    commands=serial.written_commands
    turns=[cmd for cmd in commands if cmd.startswith('TURN')]
    assert turns and turns[0] == ('TURN -12 210' if mode=='lane' else 'TURN -90 210')
    first_turn=next(i for i,cmd in enumerate(commands) if cmd.startswith('TURN'))
    assert not any(cmd.startswith('DRIVE') for cmd in commands[first_turn:])
    if mode=='autonomous':
        assert any(cmd.startswith('DRIVE') for cmd in commands[:first_turn])
    elif mode=='lane':
        assert lane_detector.detect.call_count == camera.frames
        assert any(call.kwargs["exclude_boxes"] for call in lane_detector.detect.call_args_list)
        assert seen_lane_results and all(result is not None for result in seen_lane_results)
    else:
        assert state.follow_state == 'WAIT_CALIBRATION'


def test_sign_enabled_follow_selection_is_allowed_without_direct_motion():
    state = SystemState()
    state.motor = Mock()
    state.traffic_signs = object()
    assert simulate_http_get('/select_target?x=0.5&y=0.5',state)[0] == 200
    assert state.control_mode == 'follow'
    assert state.pending_select_point == (0.5,0.5)
    state.motor.turn.assert_not_called()
    state.motor.drive.assert_not_called()


def test_active_turn_mode_change_requires_reset_instead_of_reusing_deadline():
    c,d,r,m,k = setup_runtime()
    confirm(c,d,r,m,k,"turn_left",mode="lane",lane_result=lane())
    assert step(c,r,m,k,mode="autonomous")
    assert "control mode changed" in r.fault
    assert r.completed_status is None
    m.turn.assert_called_once_with(-12,speed=210)


def test_no_sign_hazard_delegates_to_existing_mode_but_active_turn_is_cancelled():
    c,d,r,m,k = setup_runtime()
    hazard=[{"bbox":[200,100,400,400],"confidence":.9,"class_name":"person"}]
    assert step(c,r,m,k,obstacle_detections=hazard) is None
    m.pi_stop.assert_not_called()
    confirm(c,d,r,m,k,"turn_left")
    assert step(c,r,m,k,obstacle_detections=hazard)
    m.pi_stop.assert_called_once()
    assert r.fault and r.direction is None
