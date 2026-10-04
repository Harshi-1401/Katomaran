import logging
import sys
from pathlib import Path
from typing import Optional

_LOGGER_INSTANCE: Optional[logging.Logger] = None


def setup_logger(log_file_path: Optional[Path | str] = None, log_level: int = logging.INFO) -> logging.Logger:
    """Configures and returns the central application logger."""
    global _LOGGER_INSTANCE

    if _LOGGER_INSTANCE is not None:
        return _LOGGER_INSTANCE

    logger = logging.getLogger("KatomaranTracker")
    logger.setLevel(log_level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file_path:
        path = Path(log_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGER_INSTANCE = logger
    return logger


def get_logger() -> logging.Logger:
    """Returns existing logger instance or initializes a default console logger."""
    global _LOGGER_INSTANCE
    if _LOGGER_INSTANCE is None:
        return setup_logger()
    return _LOGGER_INSTANCE
