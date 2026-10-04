import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path

from app.config import AppConfig, PROJECT_ROOT
from app.input.stream_factory import StreamFactory
from app.detection.face_detector import YOLOFaceDetector
from app.recognition.face_recognizer import FaceRecognizer, FaceEmbedding
from app.tracking.face_tracker import FaceTracker, TrackedFace
from app.pipeline.processor import FrameProcessor
from app.logging.event_logger import setup_logger, get_logger


def print_config_summary(cfg: AppConfig) -> None:
    print("=" * 68)
    print("       KATOMARAN INTELLIGENT FACE TRACKER - CONFIGURATION CHECK")
    print("=" * 68)
    print(f"  Project Root Directory:  {PROJECT_ROOT}")
    print("-" * 68)
    print("  INPUT CONFIGURATION:")
    print(f"    - Source Type:         {cfg.input.source_type.upper()}")
    if cfg.input.source_type == "video":
        print(f"    - Video Path:          {cfg.resolve_path(cfg.input.video_path)}")
    else:
        print(f"    - RTSP Stream URL:     {cfg.input.rtsp_url}")
        print(f"    - Reconnect Delay:     {cfg.input.rtsp_reconnect_delay_seconds}s")
        print(f"    - Buffer Size:         {cfg.input.rtsp_buffer_size}")

    print("-" * 68)
    print("  FACE DETECTION CONFIGURATION:")
    print(f"    - Model Path:          {cfg.resolve_path(cfg.detection.model_path)}")
    print(f"    - Confidence Threshold:{cfg.detection.confidence_threshold}")
    print(f"    - Skip Frames:         {cfg.detection.detection_skip_frames}")

    print("-" * 68)
    print("  FACE RECOGNITION CONFIGURATION:")
    print(f"    - Model Path:          {cfg.resolve_path(cfg.recognition.model_path)}")
    print(f"    - Similarity Threshold:{cfg.recognition.similarity_threshold}")
    print(f"    - Min Face Size:       {cfg.recognition.minimum_face_size} px")

    print("-" * 68)
    print("  TEMPORAL TRACKING CONFIGURATION:")
    print(f"    - Max Disappeared:     {cfg.tracking.max_disappeared_frames} frames")

    print("-" * 68)
    print("  DATABASE & LOGGING CONFIGURATION:")
    print(f"    - SQLite DB Path:      {cfg.resolve_path(cfg.database.path)}")
    print(f"    - Events Log File:     {cfg.resolve_path(cfg.logging.log_file)}")
    print(f"    - Entries Image Dir:   {cfg.resolve_path('logs/entries')}")
    print(f"    - Exits Image Dir:     {cfg.resolve_path('logs/exits')}")

    print("-" * 68)
    print("  OUTPUT & DISPLAY CONFIGURATION:")
    print(f"    - Show OpenCV Window:  {cfg.output.show_window}")
    print(f"    - Save Output Video:   {cfg.output.save_video}")
    print("=" * 68)


def check_config(config_path: str = "config.json") -> None:
    logger = setup_logger()
    logger.info("Initializing configuration validation...")

    try:
        cfg = AppConfig.load_from_json(config_path)
        logger.info("config.json loaded and validated successfully.")
    except Exception as e:
        logger.error(f"Configuration validation failed:\n{e}")
        sys.exit(1)

    created_dirs = cfg.ensure_directories()
    if created_dirs:
        logger.info("Created missing system directories:")
        for d in created_dirs:
            logger.info(f"   + {d}")
    else:
        logger.info("All required system directories exist.")

    print_config_summary(cfg)
    logger.info("Phase 2 Configuration Validation PASSED successfully!")


def test_stream(config_path: str = "config.json", max_test_frames: int = 0) -> None:
    cfg = AppConfig.load_from_json(config_path)
    logger = setup_logger(cfg.resolve_path(cfg.logging.log_file))

    logger.info("Starting Phase 3 Stream Ingestion Test...")
    stream = StreamFactory.create_stream(cfg)

    if not stream.open():
        logger.error(f"Failed to open input stream ({cfg.input.source_type}).")
        sys.exit(1)

    width, height = stream.get_frame_size()
    fps = stream.get_fps()
    total_frames = stream.get_frame_count()
    source_type = cfg.input.source_type.upper()

    logger.info(f"Stream Source: {source_type}")
    logger.info(f"Resolution:    {width}x{height}")
    logger.info(f"FPS:           {fps:.2f}")

    frame_num = 0
    start_time = time.time()
    show_window = cfg.output.show_window
    window_name = f"Katomaran Stream Test [{source_type}]"

    try:
        while stream.is_opened():
            ret, frame = stream.read_frame()
            if not ret or frame is None:
                logger.info("End of stream or no frame available.")
                break

            frame_num += 1

            hud_text_source = f"Source: {source_type}"
            hud_text_res = f"Res: {width}x{height} | FPS: {fps:.1f}"
            hud_text_frame = f"Frame: {frame_num}" + (f" / {total_frames}" if total_frames > 0 else "")

            cv2.putText(frame, hud_text_source, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, hud_text_res, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, hud_text_frame, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if show_window:
                try:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                except cv2.error:
                    show_window = False

            if max_test_frames > 0 and frame_num >= max_test_frames:
                break

    finally:
        stream.release()
        if show_window:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    elapsed = time.time() - start_time
    avg_fps = frame_num / elapsed if elapsed > 0 else 0
    logger.info(f"Stream test finished. Total frames: {frame_num}, Avg FPS: {avg_fps:.1f}")


def test_detection(config_path: str = "config.json", max_test_frames: int = 0) -> None:
    cfg = AppConfig.load_from_json(config_path)
    logger = setup_logger(cfg.resolve_path(cfg.logging.log_file))

    logger.info("[DETECTION] Detection test started")
    stream = StreamFactory.create_stream(cfg)

    if not stream.open():
        sys.exit(1)

    detector = YOLOFaceDetector(cfg)
    frame_num = 0
    start_time = time.time()

    try:
        while stream.is_opened():
            ret, frame = stream.read_frame()
            if not ret or frame is None:
                break

            detections, ran_inference, yolo_ms = detector.process_frame(frame, frame_num)
            frame_num += 1

            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if cfg.output.show_window:
                try:
                    cv2.imshow("Detection Test", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    pass

            if max_test_frames > 0 and frame_num >= max_test_frames:
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()


def test_recognition(config_path: str = "config.json", max_test_frames: int = 0) -> None:
    cfg = AppConfig.load_from_json(config_path)
    logger = setup_logger(cfg.resolve_path(cfg.logging.log_file))

    logger.info("[RECOGNITION] Recognition test started")
    stream = StreamFactory.create_stream(cfg)

    if not stream.open():
        sys.exit(1)

    detector = YOLOFaceDetector(cfg)
    recognizer = FaceRecognizer(cfg)
    frame_num = 0

    try:
        while stream.is_opened():
            ret, frame = stream.read_frame()
            if not ret or frame is None:
                break

            detections, ran_inference, yolo_ms = detector.process_frame(frame, frame_num)
            frame_num += 1

            for det in detections:
                x1, y1, x2, y2 = det.bbox
                crop = frame[y1:y2, x1:x2]
                emb_res = recognizer.generate_embedding(crop)
                if emb_res.success:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)

            if cfg.output.show_window:
                try:
                    cv2.imshow("Recognition Test", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    pass

            if max_test_frames > 0 and frame_num >= max_test_frames:
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()


def test_tracking(config_path: str = "config.json", max_test_frames: int = 0) -> None:
    cfg = AppConfig.load_from_json(config_path)
    logger = setup_logger(cfg.resolve_path(cfg.logging.log_file))

    logger.info("[TRACKING] Face tracking test started")
    stream = StreamFactory.create_stream(cfg)

    if not stream.open():
        sys.exit(1)

    detector = YOLOFaceDetector(cfg)
    tracker = FaceTracker(cfg)
    frame_num = 0

    try:
        while stream.is_opened():
            ret, frame = stream.read_frame()
            if not ret or frame is None:
                break

            detections, ran_inference, yolo_ms = detector.process_frame(frame, frame_num)
            frame_num += 1

            tracks = tracker.update(detections, is_skipped_frame=not ran_inference)
            for t in tracks:
                x1, y1, x2, y2 = t.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Track:{t.track_id}", (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if cfg.output.show_window:
                try:
                    cv2.imshow("Tracking Test", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    pass

            if max_test_frames > 0 and frame_num >= max_test_frames:
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()


def run_pipeline(config_path: str = "config.json", max_frames: int = 0) -> None:
    """Runs the complete end-to-end processing pipeline."""
    cfg = AppConfig.load_from_json(config_path)
    processor = FrameProcessor(cfg)
    processor.run(max_frames=max_frames)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intelligent Face Tracker with Auto-Registration and Visitor Counting"
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration and verify system directory structure",
    )
    parser.add_argument(
        "--test-stream",
        action="store_true",
        help="Run Phase 3 frame ingestion stream test",
    )
    parser.add_argument(
        "--test-detection",
        action="store_true",
        help="Run Phase 4 YOLO face detection test",
    )
    parser.add_argument(
        "--test-recognition",
        action="store_true",
        help="Run Phase 5 ArcFace face recognition test",
    )
    parser.add_argument(
        "--test-tracking",
        action="store_true",
        help="Run Phase 6 ByteTrack face tracking test",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run full end-to-end integrated face tracking & visitor counter pipeline",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit number of frames for testing (0 = process full video/stream)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config.json file (default: config.json)",
    )

    args = parser.parse_args()

    if args.check_config:
        check_config(args.config)
        sys.exit(0)
    elif args.test_stream:
        test_stream(args.config, max_test_frames=args.max_frames)
        sys.exit(0)
    elif args.test_detection:
        test_detection(args.config, max_test_frames=args.max_frames)
        sys.exit(0)
    elif args.test_recognition:
        test_recognition(args.config, max_test_frames=args.max_frames)
        sys.exit(0)
    elif args.test_tracking:
        test_tracking(args.config, max_test_frames=args.max_frames)
        sys.exit(0)
    elif args.run:
        run_pipeline(args.config, max_frames=args.max_frames)
        sys.exit(0)
    else:
        # Default behavior when no flag is provided: Run full pipeline!
        run_pipeline(args.config, max_frames=args.max_frames)
        sys.exit(0)


if __name__ == "__main__":
    main()
