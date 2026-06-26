#!/usr/bin/env python3
"""
Main training script for turn-taking timing classifier.

Usage:
    python scripts/train.py --config configs/config.yaml [--resume checkpoint.pt]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging

from src.utils import load_config, setup_logging
from src.data import get_dataloaders
from src.models import TimingActionNet
from src.training import Trainer
from src.evaluation import Evaluator


def main(args):
    # Load config
    config = load_config(args.config)

    # Setup logging
    logger = setup_logging(
        log_dir=config.get("logging", {}).get("log_dir", "logs"),
        level=config.get("logging", {}).get("level", "INFO"),
        experiment_name=config.get("logging", {}).get("experiment_name", "timing_net"),
    )

    logger.info("=" * 80)
    logger.info("Turn-Taking Timing Classifier - Training")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")

    # Device
    device = config.get("training", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    logger.info(f"Device: {device}")

    # Load data
    logger.info("Loading datasets...")
    dataset_path = config.get("dataset", {}).get("path", "data/processed/hf_export")
    dataloaders = get_dataloaders(
        dataset_path=dataset_path,
        batch_size=config.get("training", {}).get("batch_size", 32),
        val_batch_size=config.get("training", {}).get("val_batch_size"),
        test_batch_size=config.get("training", {}).get("test_batch_size"),
        num_workers=config.get("dataset", {}).get("num_workers", 4),
        seed=config.get("training", {}).get("seed", 42),
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    train_dataset = dataloaders["train_dataset"]

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Label distribution: {train_dataset.get_label_distribution()}")

    # Create model
    logger.info("Creating model...")
    model = TimingActionNet(config)
    model = model.to(device)
    logger.info(f"Model: {model.__class__.__name__}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {num_params:,}")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Train
    logger.info("Starting training...")
    resume_checkpoint = args.resume
    if resume_checkpoint is None and trainer.get_latest_checkpoint():
        logger.info("Found latest checkpoint, resuming...")
        resume_checkpoint = trainer.get_latest_checkpoint()

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    logger.info("=" * 80)
    logger.info("Training Complete")
    logger.info("=" * 80)
    logger.info(f"Best {trainer.early_stopping_metric}: {train_result['best_metric']:.4f}")
    logger.info(f"Best model: {train_result['best_model_path']}")

    return train_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train turn-taking timing classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()
    main(args)
