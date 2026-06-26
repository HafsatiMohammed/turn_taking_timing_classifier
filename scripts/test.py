#!/usr/bin/env python3
"""
Evaluation script for trained model.

Usage:
    python scripts/test.py --config configs/config.yaml --checkpoint checkpoints/checkpoint_best.pt
"""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging

from src.utils import load_config, setup_logging
from src.data import get_dataloaders
from src.models import TimingActionNet
from src.evaluation import Evaluator


def main(args):
    # Load config
    config = load_config(args.config)

    # Setup logging
    logger = setup_logging(
        log_dir=config.get("logging", {}).get("log_dir", "logs"),
        level=config.get("logging", {}).get("level", "INFO"),
        experiment_name="test",
    )

    logger.info("=" * 80)
    logger.info("Turn-Taking Timing Classifier - Evaluation")
    logger.info("=" * 80)

    # Device
    device = config.get("training", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    # Load data
    logger.info("Loading test dataset...")
    dataset_path = config.get("dataset", {}).get("path", "data/processed/hf_export")
    dataloaders = get_dataloaders(
        dataset_path=dataset_path,
        batch_size=config.get("training", {}).get("batch_size", 32),
        val_batch_size=config.get("training", {}).get("val_batch_size"),
        test_batch_size=config.get("training", {}).get("test_batch_size"),
        num_workers=config.get("dataset", {}).get("num_workers", 4),
        seed=config.get("training", {}).get("seed", 42),
    )

    test_loader = dataloaders["test"]
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    logger.info("Creating model...")
    model = TimingActionNet(config)
    model = model.to(device)

    # Load checkpoint
    if not args.checkpoint:
        raise ValueError("Must provide --checkpoint path")

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Evaluate
    logger.info("Running evaluation...")
    evaluator = Evaluator(model, device=device)

    # Test set evaluation
    test_metrics, test_predictions = evaluator.evaluate(
        test_loader,
        return_predictions=True,
    )

    evaluator.log_metrics_summary(test_metrics, prefix="Test")

    # Save results
    results = {
        "metrics": test_metrics,
        "predictions": {
            "sample_ids": test_predictions["sample_ids"],
            "labels": test_predictions["labels"].tolist(),
            "predictions": test_predictions["predictions"].tolist(),
            "probabilities": test_predictions["probabilities"].tolist(),
        },
        "model_config": config,
        "checkpoint": args.checkpoint,
    }

    output_path = Path(args.output) if args.output else Path("results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Primary Metrics Summary:")
    logger.info("=" * 80)
    for metric in ["macro_f1", "backchannel_f1", "start_speaking_f1", "false_entry_rate", "missed_entry_rate"]:
        if metric in test_metrics:
            logger.info(f"{metric:30s}: {test_metrics[metric]:.4f}")

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate turn-taking timing classifier")
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
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Path to save results JSON",
    )

    args = parser.parse_args()
    main(args)
