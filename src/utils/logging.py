import logging
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


def setup_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    experiment_name: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_dir: Directory for log files
        level: Logging level
        experiment_name: Name of experiment (used in log filenames)

    Returns:
        Logger instance
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    else:
        log_file = log_dir / f"training_{timestamp}.log"

    # Setup logger
    logger = logging.getLogger("TimingNet")
    logger.setLevel(level)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging to {log_file}")

    return logger


def log_metrics(metrics: dict, output_file: str = "metrics.json"):
    """Save metrics to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
