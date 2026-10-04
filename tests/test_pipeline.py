import pytest
from app.config import AppConfig, PROJECT_ROOT
from app.pipeline.processor import FrameProcessor


def test_frame_processor_pipeline_integration():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    # Disable GUI window during automated test
    cfg.output.show_window = False

    processor = FrameProcessor(cfg)
    summary = processor.run(max_frames=30)

    assert isinstance(summary, dict)
    assert summary["total_frames"] == 30.0
    assert summary["elapsed_seconds"] > 0.0
    assert summary["avg_fps"] > 0.0
    assert summary["unique_visitors"] >= 0.0
