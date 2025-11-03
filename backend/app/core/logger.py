import logging
import sys
from typing import Optional

from app.core.config import settings


def setup_logging() -> None:
    """Configure application logging."""
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    root_logger.addHandler(console_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    # Application logger
    app_logger = logging.getLogger("sailor")
    app_logger.setLevel(getattr(logging, settings.log_level.upper()))


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance."""
    if name is None:
        name = "sailor"
    return logging.getLogger(name)


# Initialize logging on import
setup_logging()