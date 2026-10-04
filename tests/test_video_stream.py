from pathlib import Path
import pytest
from app.input.video_stream import VideoStream
from app.config import PROJECT_ROOT


def test_video_stream_open_valid():
    video_path = PROJECT_ROOT / "data" / "sample.mp4"
    assert video_path.exists(), "Sample video file must exist for testing"

    stream = VideoStream(video_path)
    success = stream.open()
    assert success is True
    assert stream.is_opened() is True

    width, height = stream.get_frame_size()
    assert width > 0
    assert height > 0
    assert stream.get_fps() > 0
    assert stream.get_frame_count() > 0

    stream.release()
    assert stream.is_opened() is False


def test_video_stream_invalid_path():
    invalid_path = PROJECT_ROOT / "data" / "non_existent_video.mp4"
    stream = VideoStream(invalid_path)
    success = stream.open()
    assert success is False
    assert stream.is_opened() is False


def test_video_stream_read_and_release():
    video_path = PROJECT_ROOT / "data" / "sample.mp4"
    stream = VideoStream(video_path)
    assert stream.open() is True

    width, height = stream.get_frame_size()
    ret, frame = stream.read_frame()
    assert ret is True
    assert frame is not None
    assert frame.shape == (height, width, 3)

    stream.release()
    ret_after, frame_after = stream.read_frame()
    assert ret_after is False
    assert frame_after is None
