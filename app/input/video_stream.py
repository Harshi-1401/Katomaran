from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from app.input.base_stream import BaseStream
from app.logging.event_logger import get_logger


class VideoStream(BaseStream):
    """File-based video stream reader using OpenCV."""

    def __init__(self, video_path: Path | str):
        self.video_path = Path(video_path).resolve()
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._width: int = 0
        self._height: int = 0
        self._total_frames: int = 0
        self._logger = get_logger()

    def open(self) -> bool:
        """Opens the video file and validates stream parameters."""
        if not self.video_path.exists():
            self._logger.error(f"[STREAM] Video file does not exist: {self.video_path}")
            return False

        self._logger.info(f"[STREAM] Opening video: {self.video_path}")
        self._cap = cv2.VideoCapture(str(self.video_path))

        if not self._cap.isOpened():
            self._logger.error(f"[STREAM] Failed to open video file with OpenCV: {self.video_path}")
            return False

        # Read video metadata
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or width <= 0 or height <= 0:
            self._logger.error(
                f"[STREAM] Corrupted video metadata (FPS={fps}, Resolution={width}x{height})"
            )
            self.release()
            return False

        self._fps = fps
        self._width = width
        self._height = height
        self._total_frames = total_frames

        self._logger.info(f"[STREAM] Video opened successfully")
        self._logger.info(f"[STREAM] Video FPS: {self._fps:.2f}")
        self._logger.info(f"[STREAM] Video resolution: {self._width}x{self._height}")
        self._logger.info(f"[STREAM] Total frame count: {self._total_frames}")

        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads the next sequential frame from the video file."""
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._logger.info("[STREAM] End of video reached")
            return False, None

        return True, frame

    def is_opened(self) -> bool:
        """Checks if the video stream is currently open."""
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        """Releases the OpenCV video capture object."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._logger.info("[STREAM] Video released")

    def get_fps(self) -> float:
        return self._fps

    def get_frame_size(self) -> Tuple[int, int]:
        return self._width, self._height

    def get_frame_count(self) -> int:
        return self._total_frames
