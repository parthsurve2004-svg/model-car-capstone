"""Detector-contract tests use fake predictions, never download/run weights."""

from types import SimpleNamespace

import numpy as np
import pytest

from car.perception.traffic_signs import TrafficSignDetector, sign_geometry, suppress_duplicate_signs


class FakeModel:
    names = {0: "STOP", 1: "No Entry", 2: "compulsory-left", 3: "turn_right"}

    def __init__(self, boxes=()):
        self.boxes = boxes
        self.calls = 0

    def predict(self, frame, **kwargs):
        self.calls += 1
        return [SimpleNamespace(boxes=self.boxes)]


def box(class_id=0, confidence=0.9, coords=(10, 20, 30, 40)):
    return SimpleNamespace(cls=[class_id], conf=[confidence], xyxy=np.array([coords]))


def test_detects_labels_and_distinguishes_skipped_from_empty():
    now = [0.0]
    model = FakeModel([box(2)])
    detector = TrafficSignDetector(None, model=model, clock=lambda: now[0])
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.poll(frame) == [{"class_name": "turn_left", "confidence": 0.9,
                                     "bbox": [10, 20, 30, 40], "type": "traffic_sign"}]
    assert detector.poll(frame) is None
    now[0] = 0.25
    model.boxes = []
    assert detector.poll(frame) == []
    assert model.calls == 2


def test_interval_starts_when_prediction_completes():
    now = [0.0]
    model = FakeModel()
    original = model.predict

    def slow_prediction(*args, **kwargs):
        now[0] += 0.5
        return original(*args, **kwargs)

    model.predict = slow_prediction
    detector = TrafficSignDetector(None, model=model, clock=lambda: now[0])
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.poll(frame) == []
    assert detector.poll(frame) is None
    now[0] = 0.75
    assert detector.poll(frame) == []


def test_rejects_invalid_predictions_and_clips_valid_box():
    model = FakeModel([box(confidence=float("nan")), box(confidence=0.59),
                       box(class_id=0.5), box(class_id=8),
                       box(coords=(1, 1, float("inf"), 4)),
                       box(coords=(4, 4, 2, 2)), box(coords=(-5, 2, 200, 30))])
    detector = TrafficSignDetector(None, model=model)
    assert detector.poll(np.zeros((100, 100, 3), dtype=np.uint8)) == [
        {"class_name": "stop", "confidence": 0.9, "bbox": [0, 2, 100, 30], "type": "traffic_sign"}]


def test_wrong_model_vocabulary_is_rejected():
    model = FakeModel()
    model.names = {0: "person", 1: "no_entry", 2: "turn_left", 3: "turn_right"}
    with pytest.raises(ValueError, match="classes"):
        TrafficSignDetector(None, model=model)


def test_missing_weights_does_not_trigger_download():
    with pytest.raises(FileNotFoundError):
        TrafficSignDetector("/definitely/missing/traffic_signs.pt")


@pytest.mark.parametrize("options", [{"rate_hz": 0}, {"rate_hz": 6},
                                      {"confidence": float("nan")}, {"imgsz": 31}])
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        TrafficSignDetector(None, model=FakeModel(), **options)


def test_prediction_errors_propagate():
    model = FakeModel()

    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")

    model.predict = fail
    detector = TrafficSignDetector(None, model=model)
    with pytest.raises(RuntimeError, match="inference failed"):
        detector.poll(np.zeros((100, 100, 3), dtype=np.uint8))


def prediction(name, confidence, coords=(0, 0, 100, 100)):
    return {"class_name": name, "confidence": confidence, "bbox": list(coords),
            "type": "traffic_sign"}


def test_duplicate_card_keeps_stronger_direction_and_poll_applies_rule():
    left = prediction("turn_left", .998)
    right = prediction("turn_right", .844, (1, 1, 99, 99))
    original = [right, left]
    assert suppress_duplicate_signs(original) == [left]
    assert original == [right, left]
    model = FakeModel([box(2, .998, (0, 0, 100, 100)), box(3, .844, (1, 1, 99, 99))])
    detector = TrafficSignDetector(None, model=model)
    assert [d["class_name"] for d in detector.poll(np.zeros((100, 100, 3), np.uint8))] == ["turn_left"]


def test_close_direction_conflict_is_retained_for_policy_to_withhold_turn():
    left = prediction("turn_left", .91)
    right = prediction("turn_right", .86)
    assert suppress_duplicate_signs([right, left]) == [left, right]


def test_spatially_distinct_direction_signs_are_both_retained():
    left = prediction("turn_left", .99)
    right = prediction("turn_right", .70, (150, 0, 250, 100))
    assert suppress_duplicate_signs([right, left]) == [left, right]


def test_same_class_duplicates_do_not_suppress_an_overlap_chain_neighbor():
    first = prediction("stop", .95)
    suppressed = prediction("stop", .80, (20, 0, 120, 100))
    neighbor = prediction("stop", .70, (40, 0, 140, 100))
    assert suppress_duplicate_signs([neighbor, suppressed, first]) == [first, neighbor]


@pytest.mark.parametrize("options", [{"iou_threshold": 0}, {"iou_threshold": float("nan")},
                                      {"confidence_margin": -1}, {"confidence_margin": float("inf")}])
def test_duplicate_suppression_configuration_is_validated(options):
    with pytest.raises(ValueError):
        suppress_duplicate_signs([], **options)


def sign(coords, **kwargs):
    return {"class_name": "stop", "type": "traffic_sign", "bbox": coords, **kwargs}


def test_sign_geometry_resolution_and_aspect_ratio_invariant():
    first = sign_geometry(sign([128, 96, 256, 192]), (480, 640, 3))
    scaled = sign_geometry(sign([256, 192, 512, 384]), (960, 1280, 3))
    widescreen = sign_geometry(sign([256, 144, 512, 288]), (720, 1280, 3))
    assert scaled == pytest.approx(first)
    assert widescreen == pytest.approx(first)


def test_sign_geometry_pinhole_bearing_and_symmetric_sides():
    # At HFOV90, fx=W/2; quarter-width centers give atan(+/-0.5).
    left = sign_geometry(sign([20, 20, 30, 60]), (100, 100), hfov_deg=90)
    right = sign_geometry(sign([70, 20, 80, 60]), (100, 100), hfov_deg=90)
    assert left["bearing_deg"] == pytest.approx(-26.565051177)
    assert right["bearing_deg"] == pytest.approx(26.565051177)
    assert left["actionable"] is False
    assert right["actionable"] is False
    centered = sign_geometry(sign([40, 20, 60, 60]), (100, 100))
    assert centered["bearing_deg"] == 0
    assert centered["actionable"] is True


def test_sign_geometry_small_sign_is_detectable_but_not_actionable():
    result = sign_geometry(sign([45, 30, 55, 40]), (100, 100))
    assert result["height_frac"] == pytest.approx(0.1)
    assert result["actionable"] is False


def test_sign_geometry_clips_box_before_size_and_bearing_calculation():
    result = sign_geometry(sign([-10, -20, 110, 20]), (100, 100))
    assert result == {"height_frac": 0.2, "bearing_deg": 0.0, "actionable": True}


@pytest.mark.parametrize("coords", [None, [1, 2, 3], [1, 2, 1, 4], [5, 6, 1, 2],
                                    [1, 2, float("nan"), 4], [1, 2, float("inf"), 4],
                                    [200, 200, 300, 300]])
def test_sign_geometry_invalid_box(coords):
    with pytest.raises(ValueError):
        sign_geometry(sign(coords), (100, 100))


@pytest.mark.parametrize("options", [{"hfov_deg": float("nan")}, {"hfov_deg": 0},
                                      {"hfov_deg": 180}, {"min_height_frac": 0},
                                      {"max_bearing_deg": float("nan")},
                                      {"max_bearing_deg": 90}])
def test_sign_geometry_invalid_camera_or_gates(options):
    with pytest.raises(ValueError):
        sign_geometry(sign([10, 10, 30, 30]), (100, 100), **options)


@pytest.mark.parametrize("shape", [None, (), (0, 100), (100, float("nan")), (20.5, 100)])
def test_sign_geometry_invalid_frame(shape):
    with pytest.raises(ValueError):
        sign_geometry(sign([10, 10, 30, 30]), shape)


@pytest.mark.parametrize("changes", [{"type": "positive"}, {"class_name": "person"}])
def test_sign_geometry_rejects_non_sign_detection(changes):
    with pytest.raises(ValueError):
        sign_geometry(sign([10, 10, 30, 30], **changes), (100, 100))
