import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
import json
from tqdm import tqdm
from pathlib import Path
import logging

from ..evaluation import Evaluator


logger = logging.getLogger(__name__)


class Trainer:
    """Train model with checkpointing and resumption support."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = "cuda",
    ):
        """
        Args:
            model: PyTorch model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Configuration dict
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.train_cfg = config.get("training", {})
        self.checkpoint_dir = Path(self.train_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training parameters
        self.num_epochs = self.train_cfg.get("num_epochs", 100)
        self.batch_size = self.train_cfg.get("batch_size", 32)
        self.learning_rate = self.train_cfg.get("learning_rate", 0.001)
        self.gradient_clip = self.train_cfg.get("gradient_clip", 1.0)

        # Loss function
        self._setup_loss()

        # Optimizer
        self._setup_optimizer()

        # Learning rate scheduler
        self._setup_scheduler()

        # Evaluator
        self.evaluator = Evaluator(model, device=device)

        # Training state
        self.current_epoch = 0
        self.best_val_metric = -np.inf
        self.best_model_path = None
        self.early_stopping_counter = 0
        self.early_stopping_patience = self.train_cfg.get("early_stopping_patience", 15)
        self.early_stopping_metric = self.train_cfg.get("early_stopping_metric", "val_macro_f1")

        # History
        self.train_history = []
        self.val_history = []

        # Random seed
        seed = self.train_cfg.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _setup_loss(self):
        """Setup loss function with class weighting if specified."""
        loss_cfg = self.train_cfg.get("loss", {})
        loss_type = loss_cfg.get("type", "weighted_ce")

        # Get class weights
        class_weights = loss_cfg.get("class_weights")
        if class_weights is None:
            # Auto-compute from training data
            labels = self.train_loader.dataset.labels
            counts = np.bincount(labels, minlength=3)
            class_weights = 1.0 / (counts + 1e-8)
            class_weights = class_weights / class_weights.sum() * 3

        class_weights = torch.FloatTensor(class_weights).to(self.device)

        if loss_type == "weighted_ce":
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_type == "ce":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        logger.info(f"Loss: {loss_type}, Class weights: {class_weights.cpu().numpy()}")

    def _setup_optimizer(self):
        """Setup optimizer."""
        optimizer_type = self.train_cfg.get("optimizer", "adam").lower()
        optimizer_params = self.train_cfg.get("optimizer_params", {})

        if optimizer_type == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, **optimizer_params)
        elif optimizer_type == "adamw":
            self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, **optimizer_params)
        elif optimizer_type == "sgd":
            self.optimizer = SGD(self.model.parameters(), lr=self.learning_rate, **optimizer_params)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        logger.info(f"Optimizer: {optimizer_type}, LR: {self.learning_rate}")

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_type = self.train_cfg.get("lr_scheduler", "cosine").lower()
        scheduler_params = self.train_cfg.get("lr_scheduler_params", {})

        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_params.get("T_max", self.num_epochs),
                eta_min=scheduler_params.get("eta_min", 1e-5),
            )
        elif scheduler_type == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_params.get("step_size", 10),
                gamma=scheduler_params.get("gamma", 0.1),
            )
        elif scheduler_type == "exponential":
            self.scheduler = ExponentialLR(
                self.optimizer,
                gamma=scheduler_params.get("gamma", 0.95),
            )
        elif scheduler_type == "none":
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

    def train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            frame = batch["frame"].to(self.device)
            scalar = batch["scalar"].to(self.device)
            labels = batch["label"].to(self.device)
            weights = batch["weight"].to(self.device)

            # Forward
            logits = self.model(frame, scalar)

            # Weighted loss
            loss = self.criterion(logits, labels)
            if weights is not None:
                loss = (loss * weights).mean()

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                pbar.set_postfix({"loss": loss.item():.4f})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> Dict:
        """Validate model. Returns metrics dict."""
        metrics, _ = self.evaluator.evaluate(self.val_loader, return_predictions=False)
        return metrics

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict:
        """
        Train model with checkpointing.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training history dict
        """
        # Resume if checkpoint provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
            logger.info(f"Resumed from checkpoint at epoch {self.current_epoch}")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            self.train_history.append({"epoch": epoch, "loss": train_loss})

            # Validate
            val_metrics = self.validate()
            val_metrics["epoch"] = epoch
            self.val_history.append(val_metrics)

            # Logging
            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
            )

            # Learning rate step
            if self.scheduler:
                self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.train_cfg.get("save_every_epoch", 5) == 0:
                self.save_checkpoint(tag=f"epoch_{epoch + 1}")

            # Early stopping
            metric_value = val_metrics.get(self.early_stopping_metric)
            if metric_value is not None:
                if metric_value > self.best_val_metric:
                    self.best_val_metric = metric_value
                    self.best_model_path = self.save_checkpoint(tag="best")
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1

                if self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        logger.info(f"Training complete. Best {self.early_stopping_metric}: {self.best_val_metric:.4f}")

        return {
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_model_path": self.best_model_path,
            "best_metric": self.best_val_metric,
        }

    def save_checkpoint(self, tag: str = "latest") -> str:
        """
        Save checkpoint.

        Args:
            tag: Tag for checkpoint filename

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{tag}.pt"

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_metric": self.best_val_metric,
            "config": self.config,
            "train_history": self.train_history,
            "val_history": self.val_history,
        }

        if self.scheduler:
            checkpoint["scheduler_state"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if self.scheduler and "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.best_val_metric = checkpoint.get("best_val_metric", -np.inf)
        self.train_history = checkpoint.get("train_history", [])
        self.val_history = checkpoint.get("val_history", [])

        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}, best metric: {self.best_val_metric:.4f}")

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint if it exists."""
        latest = self.checkpoint_dir / "checkpoint_latest.pt"
        if latest.exists():
            return str(latest)
        return None
