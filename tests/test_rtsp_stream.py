import pytest
from app.input.rtsp_stream import RTSPStream
from app.input.stream_factory import StreamFactory
from app.config import AppConfig, InputConfig, PROJECT_ROOT


def test_rtsp_stream_init():
    stream = RTSPStream("rtsp://127.0.0.1:8554/live", reconnect_delay_seconds=2.0, buffer_size=1)
    assert stream.rtsp_url == "rtsp://127.0.0.1:8554/live"
    assert stream.reconnect_delay_seconds == 2.0
    assert stream.buffer_size == 1
    assert stream.get_frame_count() == -1


def test_rtsp_stream_empty_url():
    stream = RTSPStream("")
    success = stream.open()
    assert success is False
    stream.release()


def test_stream_factory_video():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    stream = StreamFactory.create_stream(cfg)
    assert stream.__class__.__name__ == "VideoStream"


def test_stream_factory_rtsp():
    cfg_data = {
        "input": {
            "source_type": "rtsp",
            "video_path": "",
            "rtsp_url": "rtsp://192.168.1.100:554/stream1",
            "rtsp_reconnect_delay_seconds": 3,
            "rtsp_buffer_size": 1,
        },
        "detection": {"model_path": "models/yolov8n-face.pt", "confidence_threshold": 0.5, "detection_skip_frames": 5},
        "recognition": {"similarity_threshold": 0.5, "minimum_face_size": 60, "model_path": "models/insightface/w600k_mbf.onnx"},
        "tracking": {"max_disappeared_frames": 30},
        "database": {"path": "data/faces.db"},
        "logging": {"log_file": "logs/events.log"},
        "output": {"show_window": False, "save_video": False},
    }
    cfg = AppConfig(**cfg_data)
    stream = StreamFactory.create_stream(cfg)
    assert stream.__class__.__name__ == "RTSPStream"
    assert stream.rtsp_url == "rtsp://192.168.1.100:554/stream1"
