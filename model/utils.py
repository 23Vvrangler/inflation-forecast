"""
Utility functions for the ML pipeline.
Handles configuration loading, directory creation, and logging setup.
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "model/config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary with configuration parameters.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging with consistent formatting.

    Args:
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories if they do not exist.

    Args:
        config: Configuration dictionary containing paths.
    """
    paths = config.get("paths", {})
    directories = [
        paths.get("artifacts_dir", "model/artifacts"),
        paths.get("raw_data", "data/raw"),
        paths.get("processed_data", "data/processed"),
        "data/splits",
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path object pointing to the project root.
    """
    return Path(__file__).parent.parent
