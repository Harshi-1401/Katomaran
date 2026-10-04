from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class FaceRecord:
    """Database entity representing a persistent registered face identity."""

    face_id: str
    registered_at: str
    last_seen_at: str
    embedding: np.ndarray  # 512-float32 numpy array
    first_crop_path: str
    id: Optional[int] = None


@dataclass
class EventRecord:
    """Database entity representing an Entry, Exit, or Registration log event."""

    face_id: str
    event_type: str        # 'ENTRY', 'EXIT', 'REGISTRATION', 'REIDENTIFICATION'
    timestamp: str
    image_path: str
    tracking_id: int
    confidence: float = 0.0
    similarity: float = 0.0
    id: Optional[int] = None
