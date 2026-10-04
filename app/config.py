import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

# Project root directory (d:\katomaran)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class InputConfig(BaseModel):
    source_type: Literal["video", "rtsp"] = Field(
        ..., description="Input source type: 'video' or 'rtsp'"
    )
    video_path: str = Field(
        default="data/sample.mp4", description="Path to video file"
    )
    rtsp_url: str = Field(
        default="", description="RTSP camera stream URL"
    )
    rtsp_reconnect_delay_seconds: float = Field(
        default=5.0, description="Seconds to wait between RTSP reconnection attempts"
    )
    rtsp_buffer_size: int = Field(
        default=1, description="Buffer size for RTSP frame queue (1 for latest frame)"
    )

    @field_validator("rtsp_reconnect_delay_seconds")
    @classmethod
    def validate_reconnect_delay(cls, v: float) -> float:
        if v < 0.1:
            raise ValueError(f"rtsp_reconnect_delay_seconds must be >= 0.1, got {v}")
        return v

    @field_validator("rtsp_buffer_size")
    @classmethod
    def validate_buffer_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"rtsp_buffer_size must be >= 1, got {v}")
        return v

    @model_validator(mode="after")
    def validate_source_inputs(self) -> "InputConfig":
        if self.source_type == "video" and not self.video_path.strip():
            raise ValueError("video_path cannot be empty when source_type is 'video'")
        if self.source_type == "rtsp" and not self.rtsp_url.strip():
            raise ValueError("rtsp_url cannot be empty when source_type is 'rtsp'")
        return self


class DetectionConfig(BaseModel):
    model_path: str = Field(
        default="models/yolov8n-face.pt", description="Path to YOLO face detector model"
    )
    confidence_threshold: float = Field(
        default=0.5, description="Detection confidence threshold (0.0 to 1.0)"
    )
    detection_skip_frames: int = Field(
        default=5, description="Number of frames to skip between detections"
    )

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence_threshold must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("detection_skip_frames")
    @classmethod
    def validate_skip_frames(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"detection_skip_frames must be >= 0, got {v}")
        return v


class RecognitionConfig(BaseModel):
    similarity_threshold: float = Field(
        default=0.5, description="Cosine similarity threshold for matching (0.0 to 1.0)"
    )
    minimum_face_size: int = Field(
        default=60, description="Minimum face size in pixels"
    )
    model_path: str = Field(
        default="models/insightface/w600k_mbf.onnx", description="Path to ArcFace embedding model"
    )

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"similarity_threshold must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("minimum_face_size")
    @classmethod
    def validate_min_face_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"minimum_face_size must be > 0, got {v}")
        return v


class TrackingConfig(BaseModel):
    max_disappeared_frames: int = Field(
        default=30, description="Max consecutive missing frames before triggering EXIT"
    )

    @field_validator("max_disappeared_frames")
    @classmethod
    def validate_max_disappeared(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"max_disappeared_frames must be > 0, got {v}")
        return v


class DatabaseConfig(BaseModel):
    path: str = Field(
        default="data/faces.db", description="SQLite database file path"
    )


class LoggingConfig(BaseModel):
    log_file: str = Field(
        default="logs/events.log", description="Events log file path"
    )


class OutputConfig(BaseModel):
    show_window: bool = Field(
        default=True, description="Whether to display OpenCV window"
    )
    save_video: bool = Field(
        default=False, description="Whether to save output video"
    )


class AppConfig(BaseModel):
    input: InputConfig
    detection: DetectionConfig
    recognition: RecognitionConfig
    tracking: TrackingConfig
    database: DatabaseConfig
    logging: LoggingConfig
    output: OutputConfig

    @classmethod
    def load_from_json(cls, config_path: Path | str = PROJECT_ROOT / "config.json") -> "AppConfig":
        path = Path(config_path)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(**data)

    def resolve_path(self, relative_or_abs: str) -> Path:
        """Resolves a file/directory path relative to project root safely."""
        p = Path(relative_or_abs)
        if p.is_absolute():
            return p
        return (PROJECT_ROOT / p).resolve()

    def ensure_directories(self) -> list[Path]:
        """Creates missing directories required for data, models, logs, entries, and exits."""
        created_dirs: list[Path] = []
        target_dirs = [
            self.resolve_path(self.database.path).parent,
            self.resolve_path(self.logging.log_file).parent,
            self.resolve_path("logs/entries"),
            self.resolve_path("logs/exits"),
            self.resolve_path(self.detection.model_path).parent,
            self.resolve_path(self.recognition.model_path).parent,
        ]
        for d in target_dirs:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                created_dirs.append(d)
        return created_dirs
