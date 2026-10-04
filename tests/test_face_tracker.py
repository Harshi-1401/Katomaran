import pytest
from app.config import AppConfig, PROJECT_ROOT
from app.detection.face_detector import FaceDetection
from app.tracking.face_tracker import FaceTracker, TrackedFace, compute_iou


def test_compute_iou_identical():
    box1 = (10, 10, 50, 50)
    box2 = (10, 10, 50, 50)
    assert compute_iou(box1, box2) == 1.0


def test_compute_iou_disjoint():
    box1 = (10, 10, 50, 50)
    box2 = (100, 100, 150, 150)
    assert compute_iou(box1, box2) == 0.0


def test_tracker_initialization():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    tracker = FaceTracker(cfg)
    assert tracker.max_disappeared == 30
    assert tracker.total_tracks_created == 0
    assert tracker.total_tracks_removed == 0
    assert len(tracker.get_active_tracks()) == 0


def test_tracker_single_face_continuity():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    tracker = FaceTracker(cfg)

    # Frame 1: First detection
    det1 = FaceDetection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)
    tracks_f1 = tracker.update([det1])

    assert len(tracks_f1) == 1
    t1_id = tracks_f1[0].track_id
    assert t1_id == 1
    assert tracks_f1[0].hits == 1
    assert tracks_f1[0].state == "ACTIVE"

    # Frame 2: Slightly moved detection
    det2 = FaceDetection(bbox=(102, 101, 202, 201), confidence=0.92, class_id=0)
    tracks_f2 = tracker.update([det2])

    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == t1_id
    assert tracks_f2[0].hits == 2
    assert tracks_f2[0].missed_frames == 0


def test_tracker_multiple_face_association():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    tracker = FaceTracker(cfg)

    det_a = FaceDetection(bbox=(50, 50, 150, 150), confidence=0.88, class_id=0)
    det_b = FaceDetection(bbox=(300, 300, 400, 400), confidence=0.91, class_id=0)

    tracks = tracker.update([det_a, det_b])
    assert len(tracks) == 2
    track_ids = {t.track_id for t in tracks}
    assert len(track_ids) == 2

    # Frame 2: Update both
    det_a2 = FaceDetection(bbox=(52, 51, 152, 151), confidence=0.89, class_id=0)
    det_b2 = FaceDetection(bbox=(303, 302, 403, 402), confidence=0.90, class_id=0)

    tracks2 = tracker.update([det_a2, det_b2])
    assert len(tracks2) == 2
    track_ids2 = {t.track_id for t in tracks2}
    assert track_ids2 == track_ids


def test_tracker_missed_frame_and_recovery():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    tracker = FaceTracker(cfg)

    det = FaceDetection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)
    tracks_f1 = tracker.update([det])
    t_id = tracks_f1[0].track_id

    # Frame 2: Missed detection
    tracks_f2 = tracker.update([])
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == t_id
    assert tracks_f2[0].state == "MISSING"
    assert tracks_f2[0].missed_frames == 1

    # Frame 3: Detection recovers
    det_recovered = FaceDetection(bbox=(103, 102, 203, 202), confidence=0.93, class_id=0)
    tracks_f3 = tracker.update([det_recovered])
    assert len(tracks_f3) == 1
    assert tracks_f3[0].track_id == t_id
    assert tracks_f3[0].state == "ACTIVE"
    assert tracks_f3[0].missed_frames == 0


def test_tracker_removal_after_max_disappeared():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    cfg.tracking.max_disappeared_frames = 3
    tracker = FaceTracker(cfg)

    det = FaceDetection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)
    tracker.update([det])

    # Miss 1, 2, 3 frames
    tracker.update([])
    tracker.update([])
    tracker.update([])
    assert len(tracker.get_active_tracks()) == 1

    # Miss 4th frame -> exceeds max_disappeared (3) -> track removed
    active_after = tracker.update([])
    assert len(active_after) == 0
    assert tracker.total_tracks_removed == 1


def test_tracker_reset():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    tracker = FaceTracker(cfg)

    det = FaceDetection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)
    tracker.update([det])
    assert tracker.total_tracks_created == 1

    tracker.reset()
    assert tracker.total_tracks_created == 0
    assert tracker.total_tracks_removed == 0
    assert len(tracker.get_active_tracks()) == 0
