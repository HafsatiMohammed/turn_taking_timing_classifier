import torch
import numpy as np
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader
import time
from .metrics import MetricsCalculator


class Evaluator:
    """Evaluate model on dataset with comprehensive metrics."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        compute_ece: bool = True,
    ):
        """
        Args:
            model: PyTorch model to evaluate
            device: Device to run on
            compute_ece: Whether to compute ECE
        """
        self.model = model
        self.device = device
        self.metrics_calc = MetricsCalculator()
        self.compute_ece = compute_ece

    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False,
    ) -> Tuple[Dict, Optional[Dict]]:
        """
        Evaluate model on dataloader.

        Args:
            dataloader: DataLoader with batches
            return_predictions: If True, also return predictions dict

        Returns:
            (metrics_dict, predictions_dict) or (metrics_dict, None)
        """
        self.model.eval()

        all_logits = []
        all_labels = []
        all_weights = []
        all_sample_ids = []
        inference_times = []

        with torch.no_grad():
            for batch in dataloader:
                frame = batch["frame"].to(self.device)
                scalar = batch["scalar"].to(self.device)
                labels = batch["label"].to(self.device)
                weights = batch["weight"]
                sample_ids = batch["sample_id"]

                # Forward pass with timing
                start_time = time.perf_counter()
                logits = self.model(frame, scalar)
                elapsed = time.perf_counter() - start_time
                inference_times.append(elapsed)

                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_weights.append(weights.numpy())
                all_sample_ids.extend(sample_ids)

        # Concatenate
        logits = np.concatenate(all_logits, axis=0)  # [N, 3]
        labels = np.concatenate(all_labels, axis=0)  # [N]
        weights = np.concatenate(all_weights, axis=0)  # [N]

        # Probabilities and predictions
        probs = self._softmax(logits)
        preds = np.argmax(logits, axis=1)

        # Compute metrics
        metrics = self.metrics_calc.compute_all_metrics(labels, preds, probs, weights)

        # Add inference metrics
        total_inference_time = sum(inference_times)
        total_samples = len(labels)
        metrics["inference_latency_ms"] = (total_inference_time / total_samples) * 1000

        # Real-time factor (assuming 100ms window, 6s context)
        metrics["real_time_factor"] = metrics["inference_latency_ms"] / 100

        # Label distribution
        metrics["label_distribution"] = self.metrics_calc.get_label_distribution(labels, preds)

        # Per-class metrics
        metrics["per_class_metrics"] = self.metrics_calc.get_per_class_metrics(labels, preds, probs)

        if return_predictions:
            predictions = {
                "sample_ids": all_sample_ids,
                "labels": labels,
                "predictions": preds,
                "probabilities": probs,
                "weights": weights,
            }
            return metrics, predictions

        return metrics, None

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Compute softmax in a numerically stable way."""
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def evaluate_with_filtering(
        self,
        dataloader: DataLoader,
        filter_func=None,
    ) -> Dict:
        """
        Evaluate with optional sample filtering.

        Args:
            dataloader: DataLoader
            filter_func: Optional function to filter samples

        Returns:
            Metrics dict
        """
        metrics, predictions = self.evaluate(dataloader, return_predictions=True)

        if filter_func is not None:
            mask = np.array([filter_func(pred) for pred in predictions["sample_ids"]])
            for key in ["labels", "predictions", "probabilities", "weights"]:
                predictions[key] = predictions[key][mask]

        # Recompute metrics on filtered set
        metrics = self.metrics_calc.compute_all_metrics(
            predictions["labels"],
            predictions["predictions"],
            predictions["probabilities"],
            predictions["weights"],
        )

        metrics["num_samples"] = len(predictions["labels"])

        return metrics

    def get_model_size_mb(self) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0

        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

    def log_metrics_summary(self, metrics: Dict, prefix: str = ""):
        """Print summary of key metrics."""
        print(f"\n{prefix} Evaluation Results:")
        print("=" * 60)

        # Primary metrics
        print("Primary Metrics:")
        for key in ["macro_f1", "backchannel_f1", "start_speaking_f1", "false_entry_rate", "missed_entry_rate"]:
            if key in metrics:
                print(f"  {key:30s}: {metrics[key]:.4f}")

        # Secondary metrics
        print("\nSecondary Metrics:")
        for key in ["wait_f1", "balanced_accuracy", "bc_as_turn_error", "turn_as_bc_error", "ece"]:
            if key in metrics:
                print(f"  {key:30s}: {metrics[key]:.4f}")

        # Real-time metrics
        if "inference_latency_ms" in metrics:
            print("\nReal-time Performance:")
            print(f"  inference_latency_ms      : {metrics['inference_latency_ms']:.2f}")
            print(f"  real_time_factor          : {metrics['real_time_factor']:.4f}")

        print("=" * 60)
