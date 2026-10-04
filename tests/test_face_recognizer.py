import numpy as np
import pytest

from app.config import AppConfig, PROJECT_ROOT
from app.recognition.face_recognizer import FaceRecognizer, FaceEmbedding


def test_face_recognizer_initialization():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    recognizer = FaceRecognizer(cfg)

    assert recognizer.session is not None
    assert recognizer.input_name == "input.1"
    assert recognizer.input_width == 112
    assert recognizer.input_height == 112
    assert recognizer.embedding_dim == 512
    assert recognizer.device in ["cpu", "cuda"]


def test_face_recognizer_preprocessing():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    recognizer = FaceRecognizer(cfg)

    # Synthetic BGR crop
    crop = np.full((80, 80, 3), 128, dtype=np.uint8)
    tensor = recognizer.preprocess(crop)

    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 3, 112, 112)
    assert tensor.dtype == np.float32
    assert -1.0 <= tensor.min() <= 1.0
    assert -1.0 <= tensor.max() <= 1.0


def test_face_recognizer_embedding_generation_and_l2_norm():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    recognizer = FaceRecognizer(cfg)

    # Valid BGR crop (100x100 > min_face_size)
    crop = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    emb_res: FaceEmbedding = recognizer.generate_embedding(crop)

    assert emb_res.success is True
    assert emb_res.embedding is not None
    assert len(emb_res.embedding) == 512
    assert emb_res.dimension == 512
    assert emb_res.reason is None

    # L2 Norm should be exactly 1.0
    norm = np.linalg.norm(emb_res.embedding)
    assert pytest.approx(norm, 1e-5) == 1.0


def test_face_recognizer_too_small_crop():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    cfg.recognition.minimum_face_size = 60
    recognizer = FaceRecognizer(cfg)

    # Small crop (20x20 < 60px)
    small_crop = np.full((20, 20, 3), 128, dtype=np.uint8)
    emb_res = recognizer.generate_embedding(small_crop)

    assert emb_res.success is False
    assert emb_res.embedding is None
    assert "smaller than minimum threshold" in emb_res.reason


def test_face_recognizer_invalid_empty_crop():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    recognizer = FaceRecognizer(cfg)

    empty_crop = np.array([], dtype=np.uint8)
    emb_res = recognizer.generate_embedding(empty_crop)

    assert emb_res.success is False
    assert emb_res.embedding is None
    assert "Empty or invalid face crop" in emb_res.reason


def test_cosine_similarity():
    # Identity test
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sim_identity = FaceRecognizer.cosine_similarity(v1, v2)
    assert pytest.approx(sim_identity, 1e-5) == 1.0

    # Orthogonal test
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    sim_ortho = FaceRecognizer.cosine_similarity(v1, v3)
    assert pytest.approx(sim_ortho, 1e-5) == 0.0

    # Opposite test
    v4 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    sim_opp = FaceRecognizer.cosine_similarity(v1, v4)
    assert pytest.approx(sim_opp, 1e-5) == -1.0

    # Zero vector handling
    v_zero = np.zeros(3, dtype=np.float32)
    sim_zero = FaceRecognizer.cosine_similarity(v1, v_zero)
    assert sim_zero == 0.0
