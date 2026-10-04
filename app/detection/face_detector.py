import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.config import AppConfig, PROJECT_ROOT
from app.logging.event_logger import get_logger


@dataclass
class FaceDetection:
    """Structured representation of a single detected face bounding box."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0
    label: str = "face"

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


class YOLOFaceDetector:
    """YOLO-based Face Detection Module supporting CPU/GPU and frame skipping."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger()

        self.model_path = config.resolve_path(config.detection.model_path)
        self.confidence_threshold = config.detection.confidence_threshold
        self.skip_frames = config.detection.detection_skip_frames

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[YOLO] = None

        self._last_detections: List[FaceDetection] = []
        self._last_inference_ms: float = 0.0

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Loads and verifies the YOLO face detection model."""
        if not self.model_path.exists():
            error_msg = (
                f"\n[ERROR] Face detection model not found at configured path:\n"
                f"        {self.model_path}\n"
                f"Please place the required YOLO face model ('yolov8n-face.pt') into:\n"
                f"        {self.model_path.parent}\n"
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.logger.info("[DETECTION] Loading YOLO face detection model...")
        self.logger.info(f"[DETECTION] Model Path: {self.model_path}")
        self.logger.info(f"[DETECTION] Target Device: {self.device.upper()}")

        try:
            self.model = YOLO(str(self.model_path))
            # Send model to device
            self.model.to(self.device)
            self.logger.info("[DETECTION] Model loaded successfully.")
            self.logger.info(f"[DETECTION] YOLO Device: {self.device.upper()}")
            self.logger.info(f"[DETECTION] Confidence Threshold: {self.confidence_threshold}")
            self.logger.info(f"[DETECTION] Detection Skip Frames: {self.skip_frames}")

        except Exception as e:
            self.logger.error(f"[DETECTION] Failed to initialize YOLO face model: {e}")
            raise RuntimeError(f"YOLO Face Detector initialization failed: {e}") from e

    def detect(
        self, frame: np.ndarray, conf_threshold: Optional[float] = None
    ) -> List[FaceDetection]:
        """Runs face detection on an OpenCV BGR image frame.

        Args:
            frame (np.ndarray): Image array (H, W, C).
            conf_threshold (float, optional): Custom confidence threshold override.

        Returns:
            List[FaceDetection]: List of structured face detection objects.
        """
        if self.model is None:
            raise RuntimeError("YOLO Face Detector is not initialized.")

        if frame is None or frame.size == 0:
            return []

        threshold = (
            conf_threshold if conf_threshold is not None else self.confidence_threshold
        )
        img_h, img_w = frame.shape[:2]

        t0 = time.perf_counter()
        results = self.model(frame, conf=threshold, verbose=False, device=self.device)
        t1 = time.perf_counter()
        self._last_inference_ms = (t1 - t0) * 1000.0

        detections: List[FaceDetection] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            xyxy = box.xyxy[0].tolist()

            # Boundary clipping & validation
            x1 = max(0, min(int(round(xyxy[0])), img_w - 1))
            y1 = max(0, min(int(round(xyxy[1])), img_h - 1))
            x2 = max(0, min(int(round(xyxy[2])), img_w))
            y2 = max(0, min(int(round(xyxy[3])), img_h))

            # Reject invalid or zero-area boxes
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    label="face",
                )
            )

        self._last_detections = detections
        return detections

    def process_frame(
        self, frame: np.ndarray, frame_index: int
    ) -> Tuple[List[FaceDetection], bool, float]:
        """Processes frame with optional frame skipping.

        Args:
            frame (np.ndarray): Image frame.
            frame_index (int): Sequential 0-based frame counter.

        Returns:
            Tuple[List[FaceDetection], bool, float]:
                - detections: List of face detections.
                - was_executed: True if YOLO inference ran on this frame.
                - inference_ms: YOLO inference execution latency in milliseconds.
        """
        # If skip_frames > 0, run detection on frame_index % (skip_frames + 1) == 0
        interval = self.skip_frames + 1 if self.skip_frames > 0 else 1
        should_detect = (frame_index % interval == 0)

        if should_detect:
            detections = self.detect(frame)
            return detections, True, self._last_inference_ms
        else:
            # On skipped frames, return cached previous detections for visual demo
            return self._last_detections, False, 0.0

    @property
    def last_inference_ms(self) -> float:
        """Returns the latency of the most recent YOLO inference call in milliseconds."""
        return self._last_inference_ms
