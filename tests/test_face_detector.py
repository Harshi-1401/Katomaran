import numpy as np
import pytest
from pathlib import Path

from app.config import AppConfig, PROJECT_ROOT
from app.detection.face_detector import YOLOFaceDetector, FaceDetection


def test_face_detection_dataclass():
    det = FaceDetection(bbox=(10, 20, 110, 120), confidence=0.85, class_id=0)
    assert det.x1 == 10
    assert det.y1 == 20
    assert det.x2 == 110
    assert det.y2 == 120
    assert det.width == 100
    assert det.height == 100
    assert det.area == 10000
    assert det.confidence == 0.85


def test_detector_initialization_valid():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    detector = YOLOFaceDetector(cfg)
    assert detector.model is not None
    assert detector.device in ["cpu", "cuda"]
    assert detector.confidence_threshold == 0.5


def test_detector_missing_model_error():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    cfg.detection.model_path = "models/non_existent_model.pt"

    with pytest.raises(FileNotFoundError) as exc_info:
        YOLOFaceDetector(cfg)

    assert "Face detection model not found" in str(exc_info.value)


def test_detector_process_frame_and_skipping():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    cfg.detection.detection_skip_frames = 2
    detector = YOLOFaceDetector(cfg)

    frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    # Frame 0: Should run inference
    dets_0, ran_0, latency_0 = detector.process_frame(frame, 0)
    assert ran_0 is True

    # Frame 1: Should skip inference and return cached result
    dets_1, ran_1, latency_1 = detector.process_frame(frame, 1)
    assert ran_1 is False
    assert latency_1 == 0.0

    # Frame 2: Should skip inference
    dets_2, ran_2, latency_2 = detector.process_frame(frame, 2)
    assert ran_2 is False

    # Frame 3 (3 % 3 == 0): Should run inference
    dets_3, ran_3, latency_3 = detector.process_frame(frame, 3)
    assert ran_3 is True


def test_detector_boundary_clipping():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    detector = YOLOFaceDetector(cfg)

    # Empty frame should return empty detections without error
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(empty_frame)
    assert isinstance(detections, list)

    for det in detections:
        assert 0 <= det.x1 < det.x2 <= 640
        assert 0 <= det.y1 < det.y2 <= 480
        assert 0.0 <= det.confidence <= 1.0
