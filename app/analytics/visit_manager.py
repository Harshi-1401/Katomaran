import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import AppConfig
from app.logging.event_logger import get_logger


@dataclass
class VisitEvent:
    """Structured representation of an Entry or Exit visit event."""

    face_id: str
    event_type: str           # 'ENTRY', 'EXIT', 'REGISTRATION', 'REIDENTIFICATION'
    timestamp: str            # ISO format string YYYY-MM-DD HH:MM:SS
    image_path: str           # Path to saved face crop image
    track_id: int             # Temporary track ID
    confidence: float = 0.0   # Face detection confidence
    similarity: float = 0.0   # Embedding cosine similarity score


class ActiveVisit:
    """Tracks state of an active physical visit in progress."""

    def __init__(self, face_id: str, track_id: int, start_time: datetime):
        self.face_id = face_id
        self.track_id = track_id
        self.start_time = start_time
        self.last_seen_time = start_time
        self.has_logged_entry: bool = False
        self.has_logged_exit: bool = False
        self.last_known_crop: Optional[np.ndarray] = None
        self.last_confidence: float = 0.0


class VisitManager:
    """Manages visit lifecycles ensuring EXACTLY ONE ENTRY and ONE EXIT per visit."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger()

        self.entries_dir = config.resolve_path("logs/entries")
        self.exits_dir = config.resolve_path("logs/exits")

        self.entries_dir.mkdir(parents=True, exist_ok=True)
        self.exits_dir.mkdir(parents=True, exist_ok=True)

        # Map face_id -> ActiveVisit
        self._active_visits: Dict[str, ActiveVisit] = {}

        # Logged events queue
        self._event_history: List[VisitEvent] = []

    def _get_date_subfolder(self, base_dir: Path, dt: datetime) -> Path:
        """Returns YYYY-MM-DD subfolder path inside base_dir."""
        date_str = dt.strftime("%Y-%m-%d")
        subfolder = base_dir / date_str
        subfolder.mkdir(parents=True, exist_ok=True)
        return subfolder

    def _save_crop_image(
        self, crop: np.ndarray, base_dir: Path, face_id: str, event_type: str, dt: datetime, track_id: int = 0
    ) -> str:
        """Saves timestamped face crop image to structured directory."""
        if crop is None or crop.size == 0:
            return ""

        date_dir = self._get_date_subfolder(base_dir, dt)
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{face_id}_{event_type}_{timestamp_str}_tr{track_id}.jpg"
        full_path = date_dir / filename

        try:
            cv2.imwrite(str(full_path), crop)
            return str(full_path)
        except Exception as e:
            self.logger.error(f"[VISIT] Failed to save crop image to {full_path}: {e}")
            return ""

    def process_presence(
        self,
        face_id: str,
        track_id: int,
        face_crop: np.ndarray,
        confidence: float,
        similarity: float = 1.0,
        dt: Optional[datetime] = None,
    ) -> Optional[VisitEvent]:
        """Processes visible presence of a face. Triggers ENTRY on first appearance of a visit."""
        if dt is None:
            dt = datetime.now()

        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")

        if face_id not in self._active_visits:
            # Create new active visit
            visit = ActiveVisit(face_id=face_id, track_id=track_id, start_time=dt)
            visit.last_known_crop = face_crop.copy() if face_crop is not None else None
            visit.last_confidence = confidence

            # Save ENTRY crop image
            img_path = self._save_crop_image(face_crop, self.entries_dir, face_id, "ENTRY", dt, track_id)

            # Generate ENTRY event
            entry_event = VisitEvent(
                face_id=face_id,
                event_type="ENTRY",
                timestamp=timestamp_str,
                image_path=img_path,
                track_id=track_id,
                confidence=confidence,
                similarity=similarity,
            )

            visit.has_logged_entry = True
            self._active_visits[face_id] = visit
            self._event_history.append(entry_event)

            self.logger.info(f"[ENTRY] {face_id} entered (track_id={track_id}, confidence={confidence:.2f})")
            return entry_event
        else:
            # Update existing visit timestamp and last known crop
            visit = self._active_visits[face_id]
            visit.last_seen_time = dt
            visit.track_id = track_id
            visit.last_confidence = confidence
            if face_crop is not None and face_crop.size > 0:
                visit.last_known_crop = face_crop.copy()
            return None

    def process_absence(
        self, face_id: str, track_id: int, dt: Optional[datetime] = None
    ) -> Optional[VisitEvent]:
        """Triggers EXIT event when a face is confirmed absent beyond max_disappeared timeout."""
        if dt is None:
            dt = datetime.now()

        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")

        if face_id in self._active_visits:
            visit = self._active_visits[face_id]

            if visit.has_logged_entry and not visit.has_logged_exit:
                # Save EXIT crop image
                img_path = self._save_crop_image(
                    visit.last_known_crop, self.exits_dir, face_id, "EXIT", dt, track_id
                )

                exit_event = VisitEvent(
                    face_id=face_id,
                    event_type="EXIT",
                    timestamp=timestamp_str,
                    image_path=img_path,
                    track_id=track_id,
                    confidence=visit.last_confidence,
                    similarity=1.0,
                )

                visit.has_logged_exit = True
                del self._active_visits[face_id]
                self._event_history.append(exit_event)

                self.logger.info(f"[EXIT] {face_id} exited (track_id={track_id})")
                return exit_event

        return None

    def get_active_visitor_ids(self) -> List[str]:
        """Returns list of currently present face IDs."""
        return list(self._active_visits.keys())

    def get_event_history(self) -> List[VisitEvent]:
        """Returns list of all logged VisitEvents."""
        return self._event_history
