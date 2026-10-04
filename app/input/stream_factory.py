from app.config import AppConfig
from app.input.base_stream import BaseStream
from app.input.video_stream import VideoStream
from app.input.rtsp_stream import RTSPStream
from app.logging.event_logger import get_logger


class StreamFactory:
    """Factory for creating stream instances based on application configuration."""

    @staticmethod
    def create_stream(config: AppConfig) -> BaseStream:
        """Instantiates and returns appropriate BaseStream subclass.

        Args:
            config (AppConfig): System configuration.

        Returns:
            BaseStream: Configured VideoStream or RTSPStream instance.
        """
        logger = get_logger()
        source_type = config.input.source_type.lower()

        if source_type == "video":
            video_file = config.resolve_path(config.input.video_path)
            logger.info(f"[FACTORY] Creating VideoStream for source: {video_file}")
            return VideoStream(video_file)

        elif source_type == "rtsp":
            rtsp_url = config.input.rtsp_url
            logger.info(f"[FACTORY] Creating RTSPStream for source: {rtsp_url}")
            return RTSPStream(
                rtsp_url=rtsp_url,
                reconnect_delay_seconds=config.input.rtsp_reconnect_delay_seconds,
                buffer_size=config.input.rtsp_buffer_size,
            )

        else:
            raise ValueError(f"Unsupported stream source_type: '{source_type}'. Must be 'video' or 'rtsp'.")
