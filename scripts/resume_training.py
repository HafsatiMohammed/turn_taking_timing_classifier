#!/usr/bin/env python3
"""
Resume training from checkpoint.

Usage:
    python scripts/resume_training.py --config configs/config.yaml --checkpoint checkpoints/checkpoint_latest.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import main as train_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume training from checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Call train with resume flag
    train_main(args)
