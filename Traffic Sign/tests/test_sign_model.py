from pathlib import Path
import pytest
from car.perception import sign_model


def make_ncnn(root):
    folder=root/'traffic_signs_ncnn_model'
    folder.mkdir()
    for name in ('model.ncnn.bin','model.ncnn.param','metadata.yaml'):
        (folder/name).write_text('dummy local fixture')
    return folder


def test_bundle_prefers_complete_ncnn_only_with_runtime(tmp_path,monkeypatch):
    pt=tmp_path/'traffic_signs.pt'
    pt.touch()
    ncnn=make_ncnn(tmp_path)
    monkeypatch.setattr(sign_model,'ncnn_available',lambda:True)
    assert sign_model.resolve_sign_model(bundle_root=tmp_path)==ncnn
    monkeypatch.setattr(sign_model,'ncnn_available',lambda:False)
    assert sign_model.resolve_sign_model(bundle_root=tmp_path)==pt


def test_incomplete_ncnn_falls_back_and_missing_bundle_stays_off(tmp_path,monkeypatch):
    monkeypatch.setattr(sign_model,'ncnn_available',lambda:True)
    assert sign_model.resolve_sign_model(bundle_root=tmp_path) is None
    ncnn=make_ncnn(tmp_path)
    (ncnn/'model.ncnn.bin').unlink()
    pt=tmp_path/'traffic_signs.pt'
    pt.touch()
    assert sign_model.resolve_sign_model(bundle_root=tmp_path)==pt


def test_explicit_override_disable_and_missing_paths_are_never_downloads(tmp_path):
    pt=tmp_path/'custom.pt'
    pt.touch()
    assert sign_model.resolve_sign_model(pt)==pt
    assert sign_model.resolve_sign_model(disabled=True,bundle_root=tmp_path) is None
    with pytest.raises(ValueError,match='not both'):
        sign_model.resolve_sign_model(pt,disabled=True)
    with pytest.raises(FileNotFoundError,match='does not exist'):
        sign_model.resolve_sign_model(tmp_path/'yolo11n.pt')
    with pytest.raises(ValueError,match='existing .pt'):
        sign_model.resolve_sign_model(tmp_path)


def test_vehicle_missing_bundle_fails_before_perception_or_hardware(monkeypatch):
    import car.main as main
    from unittest.mock import Mock
    monkeypatch.setattr(main,'resolve_sign_model',lambda *args,**kwargs:None)
    detector = Mock()
    motor = Mock()
    monkeypatch.setattr(main,'Detector',detector)
    monkeypatch.setattr(main,'MotorController',motor)
    with pytest.raises(FileNotFoundError,match='--disable-traffic-signs'):
        main.run_pipeline()
    detector.assert_not_called()
    motor.assert_not_called()


def test_explicit_disable_reaches_ordinary_detector_initialization(monkeypatch):
    import car.main as main
    from unittest.mock import Mock
    monkeypatch.setattr(main,'resolve_sign_model',lambda *args,**kwargs:None)
    # Bound the test at ordinary model initialization; no camera/motor side effects.
    detector = Mock(side_effect=RuntimeError('test boundary before hardware'))
    motor = Mock()
    monkeypatch.setattr(main,'Detector',detector)
    monkeypatch.setattr(main,'MotorController',motor)
    with pytest.raises(SystemExit):
        main.run_pipeline(disable_traffic_signs=True)
    detector.assert_called_once()
    motor.assert_not_called()
