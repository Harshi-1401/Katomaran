import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.config import AppConfig
from app.detection.face_detector import FaceDetection
from app.logging.event_logger import get_logger


@dataclass
class TrackedFace:
    """Dataclass representing a single temporally tracked face object."""

    track_id: int                            # Temporary integer track identity (1, 2, 3...)
    bbox: Tuple[int, int, int, int]          # Bounding box (x1, y1, x2, y2)
    confidence: float                        # Detection confidence score
    class_id: int = 0                        # Class identifier (0 for face)
    age: int = 1                             # Total frames track has existed
    hits: int = 1                            # Total frames matched with a detection
    missed_frames: int = 0                  # Consecutive frames missed
    state: str = "ACTIVE"                    # State: "NEW", "ACTIVE", "MISSING", "REMOVED"
    velocity: Tuple[float, float] = (0.0, 0.0) # Estimated centroid velocity (dx, dy)

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
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


class FaceTracker:
    """Custom IoU-based multi-object tracker using Hungarian assignment and motion estimation."""

    def __init__(self, config: AppConfig, iou_threshold: float = 0.25):
        self.config = config
        self.logger = get_logger()

        self.max_disappeared = config.tracking.max_disappeared_frames
        self.iou_threshold = iou_threshold

        self._next_track_id: int = 1
        self._tracks: List[TrackedFace] = []
        self._last_update_ms: float = 0.0

        # Diagnostics metrics
        self.total_tracks_created: int = 0
        self.total_tracks_removed: int = 0

        self.logger.info("[TRACKING] FaceTracker initialized.")
        self.logger.info(f"[TRACKING] Max Disappeared Frames: {self.max_disappeared}")
        self.logger.info(f"[TRACKING] IoU Threshold: {self.iou_threshold}")

    def update(
        self, detections: List[FaceDetection], is_skipped_frame: bool = False
    ) -> List[TrackedFace]:
        """Updates active tracks with new detections from the current frame.

        Args:
            detections (List[FaceDetection]): Current frame face detections.
            is_skipped_frame (bool): True if YOLO inference was skipped for this frame.

        Returns:
            List[TrackedFace]: Currently active and visible tracked faces.
        """
        t0 = time.perf_counter()

        # Handle skipped frames (extrapolate track positions via motion velocity)
        if is_skipped_frame:
            for track in self._tracks:
                if track.state != "REMOVED":
                    dx, dy = track.velocity
                    x1 = int(round(track.x1 + dx))
                    y1 = int(round(track.y1 + dy))
                    x2 = int(round(track.x2 + dx))
                    y2 = int(round(track.y2 + dy))
                    track.bbox = (x1, y1, x2, y2)
                    track.age += 1
            t1 = time.perf_counter()
            self._last_update_ms = (t1 - t0) * 1000.0
            return self.get_active_tracks()

        active_tracks = [t for t in self._tracks if t.state != "REMOVED"]

        # Case 1: No detections in current frame
        if len(detections) == 0:
            for track in active_tracks:
                track.missed_frames += 1
                track.age += 1
                if track.state == "ACTIVE" and track.missed_frames == 1:
                    track.state = "MISSING"
                    self.logger.info(f"[TRACKING] Track #{track.track_id} temporarily lost")
                elif track.missed_frames > self.max_disappeared:
                    track.state = "REMOVED"
                    self.total_tracks_removed += 1
                    self.logger.info(
                        f"[TRACKING] Track #{track.track_id} removed (missed {track.missed_frames} frames)"
                    )
            t1 = time.perf_counter()
            self._last_update_ms = (t1 - t0) * 1000.0
            return self.get_active_tracks()

        # Case 2: No existing tracks -> Spawn new tracks for all detections
        if len(active_tracks) == 0:
            for det in detections:
                new_track = TrackedFace(
                    track_id=self._next_track_id,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    age=1,
                    hits=1,
                    missed_frames=0,
                    state="ACTIVE",
                )
                self._tracks.append(new_track)
                self.logger.info(f"[TRACKING] Track #{new_track.track_id} created")
                self._next_track_id += 1
                self.total_tracks_created += 1

            t1 = time.perf_counter()
            self._last_update_ms = (t1 - t0) * 1000.0
            return self.get_active_tracks()

        # Case 3: ByteTrack IoU Hungarian Assignment Matrix
        cost_matrix = np.zeros((len(active_tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(active_tracks):
            for j, det in enumerate(detections):
                iou = compute_iou(track.bbox, det.bbox)
                cost_matrix[i, j] = 1.0 - iou

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_detections = set()

        for r, c in zip(row_ind, col_ind):
            iou = 1.0 - cost_matrix[r, c]
            if iou >= self.iou_threshold:
                track = active_tracks[r]
                det = detections[c]

                # Update velocity (moving average)
                old_cx, old_cy = track.center
                new_cx, new_cy = ((det.x1 + det.x2) / 2.0, (det.y1 + det.y2) / 2.0)
                dx = 0.6 * (new_cx - old_cx) + 0.4 * track.velocity[0]
                dy = 0.6 * (new_cy - old_cy) + 0.4 * track.velocity[1]

                if track.state == "MISSING":
                    self.logger.info(f"[TRACKING] Track #{track.track_id} recovered")

                track.bbox = det.bbox
                track.confidence = det.confidence
                track.hits += 1
                track.age += 1
                track.missed_frames = 0
                track.state = "ACTIVE"
                track.velocity = (dx, dy)

                matched_tracks.add(r)
                matched_detections.add(c)

        # Handle unmatched active tracks
        for i, track in enumerate(active_tracks):
            if i not in matched_tracks:
                track.missed_frames += 1
                track.age += 1
                if track.state == "ACTIVE" and track.missed_frames == 1:
                    track.state = "MISSING"
                    self.logger.info(f"[TRACKING] Track #{track.track_id} temporarily lost")
                elif track.missed_frames > self.max_disappeared:
                    track.state = "REMOVED"
                    self.total_tracks_removed += 1
                    self.logger.info(
                        f"[TRACKING] Track #{track.track_id} removed (missed {track.missed_frames} frames)"
                    )

        # Handle unmatched detections -> Spawn new tracks
        for j, det in enumerate(detections):
            if j not in matched_detections:
                new_track = TrackedFace(
                    track_id=self._next_track_id,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    age=1,
                    hits=1,
                    missed_frames=0,
                    state="ACTIVE",
                )
                self._tracks.append(new_track)
                self.logger.info(f"[TRACKING] Track #{new_track.track_id} created")
                self._next_track_id += 1
                self.total_tracks_created += 1

        t1 = time.perf_counter()
        self._last_update_ms = (t1 - t0) * 1000.0
        return self.get_active_tracks()

    def get_active_tracks(self) -> List[TrackedFace]:
        """Returns list of currently active or temporarily missing tracks."""
        return [t for t in self._tracks if t.state in ["ACTIVE", "MISSING"]]

    def reset(self) -> None:
        """Resets all internal tracks and counter states."""
        self._tracks.clear()
        self._next_track_id = 1
        self.total_tracks_created = 0
        self.total_tracks_removed = 0
        self.logger.info("[TRACKING] Tracker reset.")

    @property
    def last_update_ms(self) -> float:
        """Returns latency of the most recent tracker update call in milliseconds."""
        return self._last_update_ms
