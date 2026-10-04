import numpy as np
import pytest

from app.config import AppConfig, PROJECT_ROOT
from app.recognition.face_matcher import FaceMatcher


def test_face_matcher_registration_and_matching():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    matcher = FaceMatcher(cfg)

    # Unit vector 1
    emb1 = np.array([1.0] + [0.0] * 511, dtype=np.float32)

    # 1. Match on empty index -> returns None
    matched_id, sim = matcher.match_embedding(emb1)
    assert matched_id is None
    assert sim == 0.0

    # 2. Register new identity FACE_0001
    f1_id = matcher.generate_next_face_id()
    assert f1_id == "FACE_0001"
    matcher.register_identity(f1_id, emb1)

    # 3. Match exact embedding -> matches FACE_0001
    matched_id2, sim2 = matcher.match_embedding(emb1)
    assert matched_id2 == "FACE_0001"
    assert pytest.approx(sim2, 1e-5) == 1.0

    # 4. Register second identity FACE_0002 with orthogonal vector
    emb2 = np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32)
    f2_id = matcher.generate_next_face_id()
    assert f2_id == "FACE_0002"
    matcher.register_identity(f2_id, emb2)

    # 5. Dissimilar vector matches FACE_0002
    matched_id3, sim3 = matcher.match_embedding(emb2)
    assert matched_id3 == "FACE_0002"
    assert pytest.approx(sim3, 1e-5) == 1.0


def test_face_matcher_track_binding():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    matcher = FaceMatcher(cfg)

    matcher.bind_track(17, "FACE_0001")
    assert matcher.get_face_for_track(17) == "FACE_0001"

    # Unbinding track leaves persistent identity intact
    matcher.unbind_track(17)
    assert matcher.get_face_for_track(17) is None
