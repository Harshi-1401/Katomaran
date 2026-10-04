import time
import threading
from typing import Optional, Tuple
import cv2
import numpy as np
from app.input.base_stream import BaseStream
from app.logging.event_logger import get_logger


class RTSPStream(BaseStream):
    """Threaded RTSP camera stream reader with auto-reconnect and frame-dropping."""

    def __init__(
        self,
        rtsp_url: str,
        reconnect_delay_seconds: float = 5.0,
        buffer_size: int = 1,
    ):
        self.rtsp_url = rtsp_url
        self.reconnect_delay_seconds = max(0.1, reconnect_delay_seconds)
        self.buffer_size = max(1, buffer_size)

        self._logger = get_logger()
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0  # Default stream FPS estimate
        self._width: int = 0
        self._height: int = 0

        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._is_connected: bool = False
        self._reconnect_count: int = 0

    def open(self) -> bool:
        """Starts the background worker thread to read RTSP frames."""
        if not self.rtsp_url:
            self._logger.error("[RTSP] RTSP URL is empty.")
            return False

        self._logger.info(f"[RTSP] Connecting to RTSP stream: {self.rtsp_url}")
        self._stop_event.clear()

        # Attempt initial sync connection
        if not self._connect():
            self._logger.warning(
                f"[RTSP] Initial connection failed. Background worker will attempt reconnect every {self.reconnect_delay_seconds}s."
            )

        # Launch background daemon reader thread
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _connect(self) -> bool:
        """Internal helper to initialize OpenCV VideoCapture connection."""
        if self._cap is not None:
            self._cap.release()

        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            return False

        # Query stream parameters
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._cap = cap
        self._fps = fps if fps > 0 else 30.0
        self._width = width
        self._height = height
        self._is_connected = True

        self._logger.info("[RTSP] Connected successfully")
        self._logger.info(f"[RTSP] Resolution: {self._width}x{self._height}, Estimated FPS: {self._fps:.1f}")
        return True

    def _capture_loop(self) -> None:
        """Background thread loop that continuously reads frames and handles auto-reconnect."""
        last_reconnect_log_time = 0.0

        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened() or not self._is_connected:
                now = time.time()
                if now - last_reconnect_log_time >= self.reconnect_delay_seconds:
                    self._reconnect_count += 1
                    self._logger.warning(
                        f"[RTSP] Connection lost. Reconnecting... (Attempt #{self._reconnect_count})"
                    )
                    last_reconnect_log_time = now

                if self._connect():
                    self._logger.info(f"[RTSP] Reconnected successfully after {self._reconnect_count} attempts.")
                    self._reconnect_count = 0
                else:
                    time.sleep(self.reconnect_delay_seconds)
                    continue

            # Grab and retrieve latest frame (drops internal OpenCV buffer queue)
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame.copy()
            else:
                self._is_connected = False
                self._logger.warning("[RTSP] Frame grab failed. Stream may have disconnected.")
                time.sleep(0.1)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the most recent frame read by the background thread."""
        with self._lock:
            if self._latest_frame is None:
                return False, None
            # Return copy of latest frame
            frame = self._latest_frame.copy()
            return True, frame

    def is_opened(self) -> bool:
        """Returns True if the stream is currently active and thread is running."""
        return not self._stop_event.is_set() and (self._thread is not None and self._thread.is_alive())

    def release(self) -> None:
        """Signals the background thread to stop and cleans up resources."""
        self._logger.info("[RTSP] Releasing RTSP stream...")
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._is_connected = False
        with self._lock:
            self._latest_frame = None

        self._logger.info("[RTSP] Stream released cleanly.")

    def get_fps(self) -> float:
        return self._fps

    def get_frame_size(self) -> Tuple[int, int]:
        return self._width, self._height

    def get_frame_count(self) -> int:
        return -1  # Live RTSP streams do not have a fixed frame count
