"""
Voice Activity Threshold baselines for turn-taking prediction.

Implements:
1. VA-Silence: Predicts START_SPEAKING based on silence duration
2. VA-Threshold: Predicts using voice activity patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PredictionResult:
    """Container for baseline predictions."""
    sample_id: str
    split: str
    true_label: str
    pred_va_silence: str
    pred_va_threshold: str
    theta_start: float
    theta_bc_min_speech: float
    human_active_at_t: bool
    num_humans_active_at_t: int
    overlap_active_at_t: bool
    silence_duration_before_t: float
    current_human_speech_duration: float


class VASilenceBaseline:
    """
    VA-Silence baseline: predicts based on silence duration only.

    Rule:
    - if silence_duration_before_t >= theta_start:
        predict START_SPEAKING
    - else:
        predict WAIT

    Never predicts BACKCHANNEL.
    """

    def __init__(self, theta_start: float = 0.6):
        """
        Args:
            theta_start: Silence duration threshold (seconds)
        """
        self.theta_start = theta_start

    def predict(self, silence_duration_before_t: float) -> str:
        """
        Predict action based on silence duration.

        Args:
            silence_duration_before_t: Duration of silence before current time (seconds)

        Returns:
            Prediction: "START_SPEAKING" or "WAIT"
        """
        if silence_duration_before_t >= self.theta_start:
            return "START_SPEAKING"
        return "WAIT"


class VAThresholdBaseline:
    """
    VA-Threshold baseline: predicts using multiple voice activity features.

    Rule:
    - if silence_duration_before_t >= theta_start:
        return START_SPEAKING
    - elif (human_active_at_t and num_humans_active_at_t == 1 and
            not overlap_active_at_t and
            current_human_speech_duration >= theta_bc_min_speech):
        return BACKCHANNEL
    - else:
        return WAIT
    """

    def __init__(self, theta_start: float = 0.6, theta_bc_min_speech: float = 1.0):
        """
        Args:
            theta_start: Silence duration threshold (seconds)
            theta_bc_min_speech: Minimum speech duration for backchannel (seconds)
        """
        self.theta_start = theta_start
        self.theta_bc_min_speech = theta_bc_min_speech

    def predict(
        self,
        human_active_at_t: bool,
        num_humans_active_at_t: int,
        overlap_active_at_t: bool,
        silence_duration_before_t: float,
        current_human_speech_duration: float,
    ) -> str:
        """
        Predict action based on voice activity features.

        Args:
            human_active_at_t: Is any human speaking at time t?
            num_humans_active_at_t: Number of humans active at time t
            overlap_active_at_t: Are multiple humans overlapping at time t?
            silence_duration_before_t: Duration of silence before time t (seconds)
            current_human_speech_duration: Current continuous human speech duration (seconds)

        Returns:
            Prediction: "START_SPEAKING", "BACKCHANNEL", or "WAIT"
        """
        # Check for turn start (silence threshold)
        if silence_duration_before_t >= self.theta_start:
            return "START_SPEAKING"

        # Check for backchannel (single speaker, not too long)
        if (
            human_active_at_t
            and num_humans_active_at_t == 1
            and not overlap_active_at_t
            and current_human_speech_duration >= self.theta_bc_min_speech
        ):
            return "BACKCHANNEL"

        # Default: wait
        return "WAIT"


class BaselineEvaluator:
    """Evaluate baselines and select optimal thresholds."""

    LABEL_MAP = {"WAIT": 0, "BACKCHANNEL": 1, "START_SPEAKING": 2}
    ENTRY_LABELS = {"BACKCHANNEL", "START_SPEAKING"}

    def __init__(self):
        self.baselines_silence = {}
        self.baselines_threshold = {}

    def create_baselines(
        self,
        theta_start_candidates: List[float],
        theta_bc_min_speech_candidates: List[float],
    ) -> Dict[str, object]:
        """Create baseline instances for all threshold combinations."""
        baselines = {
            "silence": {theta: VASilenceBaseline(theta) for theta in theta_start_candidates},
            "threshold": {
                (ts, tbc): VAThresholdBaseline(ts, tbc)
                for ts in theta_start_candidates
                for tbc in theta_bc_min_speech_candidates
            },
        }
        return baselines

    def predict_batch(
        self,
        df: pd.DataFrame,
        baselines: Dict[str, object],
    ) -> pd.DataFrame:
        """
        Generate predictions for all samples using all baselines.

        Args:
            df: DataFrame with features and true labels
            baselines: Dict of baseline instances

        Returns:
            DataFrame with predictions
        """
        results = []

        for idx, row in df.iterrows():
            # Extract features
            human_active = row.get("human_active_at_t", False)
            num_humans_active = row.get("num_humans_active_at_t", 0)
            overlap_active = row.get("overlap_active_at_t", False)
            silence_dur = row.get("silence_duration_before_t", 0.0)
            speech_dur = row.get("current_human_speech_duration", 0.0)

            # Get true label
            true_label = row.get("final_label") or row.get("label")

            # Silence baseline predictions (all use same thresholds)
            for theta_start, baseline_silence in baselines["silence"].items():
                pred_silence = baseline_silence.predict(silence_dur)

                # VA-Threshold predictions
                for (ts, tbc), baseline_threshold in baselines["threshold"].items():
                    pred_threshold = baseline_threshold.predict(
                        human_active, num_humans_active, overlap_active, silence_dur, speech_dur
                    )

                    result = PredictionResult(
                        sample_id=row.get("sample_id", f"sample_{idx}"),
                        split=row.get("split", "unknown"),
                        true_label=true_label,
                        pred_va_silence=pred_silence,
                        pred_va_threshold=pred_threshold,
                        theta_start=ts,
                        theta_bc_min_speech=tbc,
                        human_active_at_t=human_active,
                        num_humans_active_at_t=num_humans_active,
                        overlap_active_at_t=overlap_active,
                        silence_duration_before_t=silence_dur,
                        current_human_speech_duration=speech_dur,
                    )
                    results.append(result)

        # Convert to DataFrame
        results_df = pd.DataFrame([vars(r) for r in results])
        return results_df

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """
        Compute metrics for predictions.

        Args:
            y_true: True labels (encoded as 0, 1, 2)
            y_pred: Predicted labels (encoded as 0, 1, 2)

        Returns:
            Dict of metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            confusion_matrix,
        )

        metrics = {}

        # Overall metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Per-class metrics
        for label_idx, label_name in enumerate(["WAIT", "BACKCHANNEL", "START_SPEAKING"]):
            y_true_binary = (y_true == label_idx).astype(int)
            y_pred_binary = (y_pred == label_idx).astype(int)

            metrics[f"{label_name}_precision"] = float(
                precision_score(y_true_binary, y_pred_binary, zero_division=0)
            )
            metrics[f"{label_name}_recall"] = float(
                recall_score(y_true_binary, y_pred_binary, zero_division=0)
            )
            metrics[f"{label_name}_f1"] = float(
                f1_score(y_true_binary, y_pred_binary, zero_division=0)
            )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        metrics["confusion_matrix"] = cm.tolist()

        # Entry rate errors
        entry_mask_true = np.isin(y_true, [1, 2])  # BACKCHANNEL or START_SPEAKING
        entry_mask_pred = np.isin(y_pred, [1, 2])

        # False entry rate: predicted entry but true WAIT
        false_entries = np.sum((entry_mask_pred) & (y_true == 0))
        num_false_entries = np.sum(entry_mask_pred)
        metrics["false_entry_rate"] = (
            float(false_entries / num_false_entries) if num_false_entries > 0 else None
        )

        # Missed entry rate: true entry but predicted WAIT
        missed_entries = np.sum((y_pred == 0) & (entry_mask_true))
        num_true_entries = np.sum(entry_mask_true)
        metrics["missed_entry_rate"] = (
            float(missed_entries / num_true_entries) if num_true_entries > 0 else None
        )

        # Backchannel-as-turn error: true BC predicted as START
        bc_as_turn = np.sum((y_true == 1) & (y_pred == 2))
        num_true_bc = np.sum(y_true == 1)
        metrics["backchannel_as_turn_error"] = (
            float(bc_as_turn / num_true_bc) if num_true_bc > 0 else None
        )

        # Turn-as-backchannel error: true START predicted as BC
        turn_as_bc = np.sum((y_true == 2) & (y_pred == 1))
        num_true_start = np.sum(y_true == 2)
        metrics["turn_as_backchannel_error"] = (
            float(turn_as_bc / num_true_start) if num_true_start > 0 else None
        )

        return metrics

    def select_best_thresholds(
        self,
        predictions_df: pd.DataFrame,
        val_split: str = "validation",
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Select best thresholds based on validation Macro-F1.

        Args:
            predictions_df: DataFrame with all predictions
            val_split: Split name for validation

        Returns:
            (best_theta_start_silence, (best_theta_start_threshold, best_theta_bc_min_speech))
        """
        val_df = predictions_df[predictions_df["split"] == val_split].copy()

        if len(val_df) == 0:
            raise ValueError(f"No validation samples found with split='{val_split}'")

        # Encode labels
        val_df["true_label_encoded"] = val_df["true_label"].map(self.LABEL_MAP)

        # Find best VA-Silence threshold
        best_theta_silence = None
        best_f1_silence = -1

        for theta in val_df["theta_start"].unique():
            subset = val_df[val_df["theta_start"] == theta]
            pred_encoded = subset["pred_va_silence"].map(self.LABEL_MAP).values
            true_encoded = subset["true_label_encoded"].values

            f1 = float(f1_score(true_encoded, pred_encoded, average="macro", zero_division=0))
            if f1 > best_f1_silence:
                best_f1_silence = f1
                best_theta_silence = theta

        # Find best VA-Threshold thresholds
        best_thresholds_threshold = None
        best_f1_threshold = -1

        for ts in val_df["theta_start"].unique():
            for tbc in val_df["theta_bc_min_speech"].unique():
                subset = val_df[
                    (val_df["theta_start"] == ts) & (val_df["theta_bc_min_speech"] == tbc)
                ]
                pred_encoded = subset["pred_va_threshold"].map(self.LABEL_MAP).values
                true_encoded = subset["true_label_encoded"].values

                f1 = float(f1_score(true_encoded, pred_encoded, average="macro", zero_division=0))
                if f1 > best_f1_threshold:
                    best_f1_threshold = f1
                    best_thresholds_threshold = (ts, tbc)

        return best_theta_silence, best_thresholds_threshold
