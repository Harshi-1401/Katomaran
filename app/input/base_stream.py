from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class BaseStream(ABC):
    """Abstract Base Class for video and RTSP stream inputs."""

    @abstractmethod
    def open(self) -> bool:
        """Opens the video source. Returns True if successful, False otherwise."""
        pass

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads the next frame.

        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - success (bool): True if frame was read successfully.
                - frame (np.ndarray | None): BGR image frame array or None.
        """
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Checks if the video stream is currently open and accessible."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Releases underlying stream resources and handles thread cleanup."""
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """Returns the frame rate of the video stream."""
        pass

    @abstractmethod
    def get_frame_size(self) -> Tuple[int, int]:
        """Returns frame dimensions as (width, height)."""
        pass

    @abstractmethod
    def get_frame_count(self) -> int:
        """Returns total frame count for file videos, or -1 for live streams."""
        pass
