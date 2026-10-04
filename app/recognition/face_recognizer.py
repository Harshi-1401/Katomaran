import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from app.config import AppConfig
from app.logging.event_logger import get_logger


@dataclass
class FaceEmbedding:
    """Dataclass encapsulating facial feature embedding vector and metadata."""

    embedding: Optional[np.ndarray]  # 512-d L2-normalized float32 array or None
    dimension: int                   # Vector dimension (e.g. 512)
    success: bool                    # True if embedding generated successfully
    reason: Optional[str] = None     # Explanation if embedding generation failed/skipped
    inference_ms: float = 0.0        # Inference execution latency in milliseconds


class FaceRecognizer:
    """ArcFace / InsightFace ONNX Facial Embedding Generator and Matcher."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger()

        self.model_path = config.resolve_path(config.recognition.model_path)
        self.min_face_size = config.recognition.minimum_face_size

        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.output_name: str = ""
        self.input_height: int = 112
        self.input_width: int = 112
        self.embedding_dim: int = 512
        self.device: str = "cpu"

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Loads and verifies the ArcFace ONNX model metadata."""
        if not self.model_path.exists():
            error_msg = (
                f"\n[ERROR] Face recognition model not found at configured path:\n"
                f"        {self.model_path}\n"
                f"Please place the ArcFace ONNX model ('w600k_mbf.onnx') into:\n"
                f"        {self.model_path.parent}\n"
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.logger.info("[RECOGNITION] Loading ArcFace ONNX recognition model...")
        self.logger.info(f"[RECOGNITION] Model Path: {self.model_path}")

        # Provider selection: CUDA if available, else CPU
        available_providers = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
            self.device = "cuda"
        providers.append("CPUExecutionProvider")
        if self.device != "cuda":
            self.device = "cpu"

        try:
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)

            # Inspect input metadata
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            shape = input_meta.shape
            # Typically [batch, channel, height, width] e.g. [1, 3, 112, 112]
            if len(shape) == 4:
                self.input_height = shape[2] if isinstance(shape[2], int) else 112
                self.input_width = shape[3] if isinstance(shape[3], int) else 112

            # Inspect output metadata
            output_meta = self.session.get_outputs()[0]
            self.output_name = output_meta.name
            out_shape = output_meta.shape
            if len(out_shape) == 2 and isinstance(out_shape[1], int):
                self.embedding_dim = out_shape[1]

            self.logger.info("[RECOGNITION] Recognition model loaded successfully.")
            self.logger.info(f"[RECOGNITION] Model Input Node: '{self.input_name}' (Resolution: {self.input_width}x{self.input_height})")
            self.logger.info(f"[RECOGNITION] Model Output Node: '{self.output_name}' (Embedding Dim: {self.embedding_dim})")
            self.logger.info(f"[RECOGNITION] Execution Device: {self.device.upper()}")
            self.logger.info(f"[RECOGNITION] Minimum Face Size: {self.min_face_size} px")

        except Exception as e:
            self.logger.error(f"[RECOGNITION] Failed to initialize ArcFace ONNX model: {e}")
            raise RuntimeError(f"FaceRecognizer initialization failed: {e}") from e

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """Preprocesses face crop for ArcFace ONNX model input.

        Args:
            face_crop (np.ndarray): BGR image crop of detected face.

        Returns:
            np.ndarray: Preprocessed float32 array of shape (1, 3, 112, 112).
        """
        # Convert BGR to RGB
        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Resize to ONNX target input shape (112, 112)
        resized = cv2.resize(rgb_crop, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)

        # Normalize pixel values from [0, 255] to [-1.0, 1.0]
        normalized = (resized.astype(np.float32) - 127.5) / 127.5

        # Transpose from (H, W, C) to (C, H, W)
        chw = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension -> (1, C, H, W)
        batch = np.expand_dims(chw, axis=0)
        return batch

    def generate_embedding(self, face_crop: np.ndarray) -> FaceEmbedding:
        """Generates L2-normalized facial embedding vector from face crop.

        Args:
            face_crop (np.ndarray): BGR face crop image array.

        Returns:
            FaceEmbedding: Result dataclass with embedding vector and metadata.
        """
        if self.session is None:
            return FaceEmbedding(
                embedding=None,
                dimension=self.embedding_dim,
                success=False,
                reason="Recognizer not initialized",
            )

        if face_crop is None or face_crop.size == 0:
            return FaceEmbedding(
                embedding=None,
                dimension=self.embedding_dim,
                success=False,
                reason="Empty or invalid face crop",
            )

        h, w = face_crop.shape[:2]
        if w < self.min_face_size or h < self.min_face_size:
            return FaceEmbedding(
                embedding=None,
                dimension=self.embedding_dim,
                success=False,
                reason=f"Face crop size ({w}x{h}) is smaller than minimum threshold ({self.min_face_size}px)",
            )

        t0 = time.perf_counter()
        try:
            tensor_input = self.preprocess(face_crop)
            outputs = self.session.run([self.output_name], {self.input_name: tensor_input})
            t1 = time.perf_counter()
            inference_ms = (t1 - t0) * 1000.0

            raw_embedding = outputs[0].flatten().astype(np.float32)

            # Validate output dimension
            if len(raw_embedding) != self.embedding_dim:
                return FaceEmbedding(
                    embedding=None,
                    dimension=len(raw_embedding),
                    success=False,
                    reason=f"Unexpected embedding dimension: got {len(raw_embedding)}, expected {self.embedding_dim}",
                    inference_ms=inference_ms,
                )

            # L2 Normalization with zero-vector protection
            norm = np.linalg.norm(raw_embedding)
            if norm == 0 or np.isnan(norm) or np.isinf(norm):
                return FaceEmbedding(
                    embedding=None,
                    dimension=self.embedding_dim,
                    success=False,
                    reason="Invalid or zero-norm embedding vector produced",
                    inference_ms=inference_ms,
                )

            normalized_embedding = raw_embedding / norm

            return FaceEmbedding(
                embedding=normalized_embedding,
                dimension=self.embedding_dim,
                success=True,
                reason=None,
                inference_ms=inference_ms,
            )

        except Exception as e:
            self.logger.error(f"[RECOGNITION] Exception during embedding generation: {e}")
            return FaceEmbedding(
                embedding=None,
                dimension=self.embedding_dim,
                success=False,
                reason=f"Inference error: {e}",
            )

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Computes Cosine Similarity between two embedding vectors.

        Args:
            emb1 (np.ndarray): First embedding vector.
            emb2 (np.ndarray): Second embedding vector.

        Returns:
            float: Cosine similarity score in [-1.0, 1.0].
        """
        if emb1 is None or emb2 is None:
            return 0.0

        v1 = emb1.flatten()
        v2 = emb2.flatten()

        if len(v1) != len(v2) or len(v1) == 0:
            return 0.0

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0 or np.isnan(norm1) or np.isnan(norm2):
            return 0.0

        # Dot product of normalized vectors
        sim = float(np.dot(v1 / norm1, v2 / norm2))

        # Clip numerical precision artifacts
        return float(np.clip(sim, -1.0, 1.0))
