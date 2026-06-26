import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
from typing import Dict, Optional, Tuple, List
import time


class MetricsCalculator:
    """Calculate all KPIs for turn-taking timing model."""

    LABEL_NAMES = ["WAIT", "BACKCHANNEL", "START_SPEAKING"]
    LABEL_TO_IDX = {name: i for i, name in enumerate(LABEL_NAMES)}
    IDX_TO_LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}

    def __init__(self, start_speaking_threshold: float = 0.70, backchannel_threshold: float = 0.55):
        """
        Args:
            start_speaking_threshold: Threshold for START_SPEAKING policy
            backchannel_threshold: Threshold for BACKCHANNEL policy
        """
        self.start_speaking_threshold = start_speaking_threshold
        self.backchannel_threshold = backchannel_threshold

    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute all metrics.

        Args:
            y_true: True labels [N]
            y_pred: Predicted labels [N]
            y_probs: Predicted probabilities [N, 3]
            sample_weights: Optional sample weights [N]

        Returns:
            Dict of metrics
        """
        metrics = {}

        # Primary metrics
        metrics.update(self._primary_metrics(y_true, y_pred, y_probs, sample_weights))

        # Secondary metrics
        metrics.update(self._secondary_metrics(y_true, y_pred, y_probs, sample_weights))

        # Error analysis
        metrics.update(self._error_analysis(y_true, y_probs))

        return metrics

    def _primary_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict:
        """Primary metrics: Macro-F1, BC F1, START F1, entry/miss rates."""
        metrics = {}

        # Macro F1
        metrics["macro_f1"] = f1_score(
            y_true, y_pred, average="macro", zero_division=0, sample_weight=sample_weights
        )

        # Per-class F1
        f1_scores = f1_score(
            y_true, y_pred, average=None, zero_division=0, sample_weight=sample_weights
        )
        metrics["wait_f1"] = f1_scores[0]
        metrics["backchannel_f1"] = f1_scores[1]
        metrics["start_speaking_f1"] = f1_scores[2]

        # False entry and missed entry rates
        # False entry: predicted entry (BC or START) but true is WAIT
        # Missed entry: true entry (BC or START) but predicted WAIT
        is_entry_true = y_true != 0  # 0 = WAIT
        is_entry_pred = y_pred != 0

        false_entries = np.sum((is_entry_pred) & (~is_entry_true))
        total_wait = np.sum(~is_entry_true)
        metrics["false_entry_rate"] = false_entries / (total_wait + 1e-8)

        missed_entries = np.sum((~is_entry_pred) & (is_entry_true))
        total_entries = np.sum(is_entry_true)
        metrics["missed_entry_rate"] = missed_entries / (total_entries + 1e-8)

        return metrics

    def _secondary_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict:
        """Secondary metrics: balanced accuracy, confusion-based errors, ECE."""
        metrics = {}

        # Balanced accuracy
        metrics["balanced_accuracy"] = balanced_accuracy_score(
            y_true, y_pred, sample_weight=sample_weights
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        # BC-as-turn error: predicted START_SPEAKING but true is BACKCHANNEL
        bc_as_turn = cm[1, 2] if len(cm) > 2 else 0
        total_bc = np.sum(y_true == 1)
        metrics["bc_as_turn_error"] = bc_as_turn / (total_bc + 1e-8)

        # Turn-as-BC error: predicted BACKCHANNEL but true is START_SPEAKING
        turn_as_bc = cm[2, 1] if len(cm) > 2 else 0
        total_turn = np.sum(y_true == 2)
        metrics["turn_as_bc_error"] = turn_as_bc / (total_turn + 1e-8)

        # Floor violation: aggressive false positive START predictions
        metrics["floor_violation_rate"] = metrics["false_entry_rate"]

        # Aggressiveness: tendency to predict entry vs WAIT
        metrics["aggressiveness_rate"] = np.sum(y_pred != 0) / len(y_pred)

        # Expected Calibration Error (ECE)
        metrics["ece"] = self._compute_ece(y_true, y_probs, n_bins=10)

        return metrics

    def _error_analysis(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
    ) -> Dict:
        """Error patterns and policy analysis."""
        metrics = {}

        # Policy compliance: how often thresholds are met
        p_wait = y_probs[:, 0]
        p_bc = y_probs[:, 1]
        p_start = y_probs[:, 2]

        # Fraction where START threshold is met
        metrics["policy_start_eligible"] = np.mean(p_start > self.start_speaking_threshold)

        # Fraction where BC threshold is met (given human active, not tracked here)
        metrics["policy_bc_eligible"] = np.mean(
            (p_bc > self.backchannel_threshold) & (p_start <= self.start_speaking_threshold)
        )

        return metrics

    def _compute_ece(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error.

        Measures how well predicted probabilities match actual accuracy.
        """
        confidences = np.max(y_probs, axis=1)
        predictions = np.argmax(y_probs, axis=1)
        correct = predictions == y_true

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        total_samples = len(y_true)

        for lower, upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > lower) & (confidences <= upper)
            if np.sum(in_bin) == 0:
                continue

            avg_confidence = np.mean(confidences[in_bin])
            accuracy = np.mean(correct[in_bin])
            bin_size = np.sum(in_bin)

            ece += np.abs(avg_confidence - accuracy) * (bin_size / total_samples)

        return ece

    def get_label_distribution(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Get label distribution in data and predictions."""
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)

        dist = {}
        for idx in range(3):
            true_count = counts_true[np.where(unique_true == idx)[0][0]] if idx in unique_true else 0
            pred_count = counts_pred[np.where(unique_pred == idx)[0][0]] if idx in unique_pred else 0

            label_name = self.IDX_TO_LABEL[idx]
            dist[label_name] = {
                "true_percent": true_count / len(y_true) * 100,
                "pred_percent": pred_count / len(y_pred) * 100,
            }

        return dist

    def get_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
    ) -> Dict:
        """Get precision, recall, F1 for each class."""
        precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
        recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)

        metrics = {}
        for idx, label_name in enumerate(self.LABEL_NAMES):
            metrics[label_name] = {
                "precision": precisions[idx],
                "recall": recalls[idx],
                "f1": f1_scores[idx],
            }

        return metrics

    def format_metrics_for_logging(self, metrics: Dict) -> Dict:
        """Format metrics dict for clean logging."""
        formatted = {}
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted[key] = round(value, 4)
            else:
                formatted[key] = value
        return formatted
