import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import AppConfig, PROJECT_ROOT
from app.input.base_stream import BaseStream
from app.input.stream_factory import StreamFactory
from app.detection.face_detector import YOLOFaceDetector, FaceDetection
from app.tracking.face_tracker import FaceTracker, TrackedFace
from app.recognition.face_recognizer import FaceRecognizer, FaceEmbedding
from app.recognition.face_matcher import FaceMatcher
from app.analytics.visit_manager import VisitManager, VisitEvent
from app.analytics.visitor_counter import VisitorCounter
from app.database.database import Database
from app.logging.event_logger import setup_logger, get_logger


class FrameProcessor:
    """End-to-End Integrated Pipeline Orchestrator."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = setup_logger(config.resolve_path(config.logging.log_file))

        self.logger.info("[PIPELINE] Initializing Integrated Processing Pipeline...")

        self.stream: BaseStream = StreamFactory.create_stream(config)
        self.detector = YOLOFaceDetector(config)
        self.tracker = FaceTracker(config)
        self.recognizer = FaceRecognizer(config)
        self.matcher = FaceMatcher(config)
        self.database = Database(config)
        self.visit_manager = VisitManager(config)
        self.visitor_counter = VisitorCounter(self.database, self.visit_manager)

        self.show_window = config.output.show_window
        self.window_name = f"Katomaran Intelligent Face Tracker [{config.input.source_type.upper()}]"

        # Load existing DB identities into memory matcher so persistent IDs survive restart
        self._load_persistent_identities()

    def _load_persistent_identities(self) -> None:
        """Loads registered faces from SQLite database into memory matcher."""
        records = self.database.get_all_faces()
        if records:
            self.logger.info(f"[PIPELINE] Loading {len(records)} persistent face identities from database...")
            for rec in records:
                self.matcher.register_identity(rec.face_id, rec.embedding)
            self.logger.info("[PIPELINE] Persistent identities loaded into memory index.")

    def run(self, max_frames: int = 0) -> Dict[str, float]:
        """Runs the complete end-to-end processing pipeline loop.

        Args:
            max_frames (int): Maximum frames to process (0 = process until end of stream).

        Returns:
            Dict[str, float]: Execution metrics summary dictionary.
        """
        self.logger.info("[PIPELINE] Starting Processing Loop...")

        if not self.stream.open():
            self.logger.error("[PIPELINE] Failed to open stream input.")
            return {}

        width, height = self.stream.get_frame_size()
        source_type = self.config.input.source_type.upper()

        frame_index = 0
        start_time = time.time()

        last_yolo_ms = 0.0
        last_rec_ms = 0.0
        last_tracker_ms = 0.0

        try:
            while self.stream.is_opened():
                ret, frame = self.stream.read_frame()
                if not ret or frame is None:
                    self.logger.info("[PIPELINE] End of input stream reached.")
                    break

                now_dt = datetime.now()

                # 1. YOLO Face Detection
                detections, ran_detection, yolo_ms = self.detector.process_frame(frame, frame_index)
                if ran_detection:
                    last_yolo_ms = yolo_ms

                # 2. ByteTrack Temporal Tracking
                active_tracks: List[TrackedFace] = self.tracker.update(
                    detections, is_skipped_frame=not ran_detection
                )
                last_tracker_ms = self.tracker.last_update_ms

                # 3. Recognition & Identity Binding Policy
                for track in active_tracks:
                    track_id = track.track_id
                    x1, y1, x2, y2 = track.bbox

                    # Safe face crop
                    crop = frame[y1:y2, x1:x2]

                    face_id = self.matcher.get_face_for_track(track_id)
                    sim_score = 1.0

                    if face_id is None and crop is not None and crop.size > 0:
                        # Generate ArcFace embedding for unassigned track
                        emb_res: FaceEmbedding = self.recognizer.generate_embedding(crop)
                        if emb_res.success and emb_res.embedding is not None:
                            last_rec_ms = emb_res.inference_ms

                            matched_id, max_sim = self.matcher.match_embedding(emb_res.embedding)
                            if matched_id is not None:
                                # Re-identify existing face
                                face_id = matched_id
                                sim_score = max_sim
                                self.matcher.bind_track(track_id, face_id)
                                self.database.update_face_last_seen(face_id, now_dt.strftime("%Y-%m-%d %H:%M:%S"))
                                self.logger.info(
                                    f"[REIDENTIFICATION] Track #{track_id} matched existing identity {face_id} (sim={max_sim:.4f})"
                                )
                            else:
                                # Register new persistent identity
                                face_id = self.matcher.generate_next_face_id()
                                sim_score = 1.0
                                self.matcher.register_identity(face_id, emb_res.embedding)
                                self.matcher.bind_track(track_id, face_id)
                                self.logger.info(
                                    f"[REGISTER] Registered new persistent identity {face_id} for Track #{track_id}"
                                )

                    # 4. Visit State Engine (Entry Triggering)
                    if face_id is not None and crop is not None and crop.size > 0:
                        entry_event = self.visit_manager.process_presence(
                            face_id=face_id,
                            track_id=track_id,
                            face_crop=crop,
                            confidence=track.confidence,
                            similarity=sim_score,
                            dt=now_dt,
                        )
                        if entry_event is not None:
                            # Save to persistent SQLite database faces table if first time registered
                            if not self.database.get_face(face_id):
                                rec_face = self.matcher.get_embedding(face_id)
                                emb_bytes = rec_face if rec_face is not None else np.zeros(512, dtype=np.float32)
                                dt_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
                                self.database.add_face(
                                    face_id, emb_bytes, entry_event.image_path, dt_str, dt_str
                                )

                            # Save ENTRY event to database
                            self.database.add_event(
                                face_id=entry_event.face_id,
                                event_type="ENTRY",
                                timestamp=entry_event.timestamp,
                                image_path=entry_event.image_path,
                                tracking_id=entry_event.track_id,
                                confidence=entry_event.confidence,
                                similarity=entry_event.similarity,
                            )

                # 5. Check missing/removed tracks for EXIT events
                for track in self.tracker._tracks:
                    if track.state == "REMOVED":
                        face_id = self.matcher.get_face_for_track(track.track_id)
                        if face_id is not None:
                            exit_event = self.visit_manager.process_absence(face_id, track.track_id, now_dt)
                            if exit_event is not None:
                                self.database.add_event(
                                    face_id=exit_event.face_id,
                                    event_type="EXIT",
                                    timestamp=exit_event.timestamp,
                                    image_path=exit_event.image_path,
                                    tracking_id=exit_event.track_id,
                                    confidence=exit_event.confidence,
                                    similarity=exit_event.similarity,
                                )
                            self.matcher.unbind_track(track.track_id)

                # 6. Render HUD and Bounding Boxes
                for track in active_tracks:
                    face_id = self.matcher.get_face_for_track(track.track_id)
                    x1, y1, x2, y2 = track.bbox

                    box_color = (0, 255, 0) if face_id else (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    display_id = face_id if face_id else f"Track:{track.track_id}"
                    label = f"{display_id} | Track:{track.track_id} | {track.confidence:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        box_color,
                        2,
                    )

                elapsed = time.time() - start_time
                current_fps = (frame_index + 1) / elapsed if elapsed > 0 else 0.0

                unique_count = self.visitor_counter.get_unique_visitor_count()
                active_count = self.visitor_counter.get_active_visitor_count()

                hud_fps = f"FPS: {current_fps:.1f}"
                hud_timing = f"YOLO: {last_yolo_ms:.1f}ms | ArcFace: {last_rec_ms:.1f}ms | Track: {last_tracker_ms:.1f}ms"
                hud_visitors = f"Active Visitors: {active_count} | Unique Visitors: {unique_count}"
                hud_source = f"Source: {source_type} ({width}x{height})"

                cv2.putText(frame, hud_fps, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(frame, hud_timing, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(frame, hud_visitors, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, hud_source, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                cv2.putText(frame, "Press 'q' to exit demo", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

                if self.show_window:
                    try:
                        cv2.imshow(self.window_name, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            self.logger.info("[PIPELINE] User requested exit by pressing 'q'.")
                            break
                    except cv2.error:
                        self.show_window = False
                        self.logger.warning("[PIPELINE] OpenCV GUI unavailable. Running in console mode.")

                frame_index += 1

                if frame_index % 30 == 0:
                    self.logger.info(
                        f"[PIPELINE] Frame #{frame_index} | FPS: {current_fps:.1f} | Active: {active_count} | Unique: {unique_count}"
                    )

                if max_frames > 0 and frame_index >= max_frames:
                    self.logger.info(f"[PIPELINE] Reached maximum requested frames ({max_frames}). Exiting.")
                    break

        finally:
            self.stream.release()
            if self.show_window:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

        elapsed = time.time() - start_time
        avg_fps = frame_index / elapsed if elapsed > 0 else 0

        summary = {
            "total_frames": float(frame_index),
            "elapsed_seconds": elapsed,
            "avg_fps": avg_fps,
            "unique_visitors": float(self.visitor_counter.get_unique_visitor_count()),
            "active_visitors": float(self.visitor_counter.get_active_visitor_count()),
        }

        self.logger.info("=" * 68)
        self.logger.info("[PIPELINE] Pipeline Execution Summary:")
        self.logger.info(f"  Total Frames Processed:    {frame_index}")
        self.logger.info(f"  Elapsed Processing Time:   {elapsed:.2f}s")
        self.logger.info(f"  Average Processing FPS:    {avg_fps:.1f}")
        self.logger.info(f"  Unique Visitors Count:     {int(summary['unique_visitors'])}")
        entries, exits = self.visitor_counter.get_entry_exit_counts()
        self.logger.info(f"  Total Entry Events:        {entries}")
        self.logger.info(f"  Total Exit Events:         {exits}")
        self.logger.info("=" * 68)

        return summary
