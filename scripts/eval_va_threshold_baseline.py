#!/usr/bin/env python3
"""
Evaluate VA-Threshold baselines on turn-taking dataset.

Usage:
    python scripts/eval_va_threshold_baseline.py \
        --manifest data/processed/final_manifest.parquet \
        --output-dir reports/va_threshold_baseline \
        --use-precomputed-if-available true \
        --max-samples 100

Outputs:
    - va_baseline_predictions.parquet
    - va_baseline_metrics.json
    - va_baseline_metrics.md
    - va_baseline_confusion_matrix.csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines import VAThresholdBaseline, VASilenceBaseline, BaselineEvaluator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VAThresholdEvaluationPipeline:
    """Pipeline for evaluating VA-Threshold baselines."""

    def __init__(
        self,
        manifest_path: str,
        output_dir: str,
        use_precomputed: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            manifest_path: Path to manifest parquet file
            output_dir: Output directory for results
            use_precomputed: Use precomputed features if available
            max_samples: Limit to N samples (for testing)
        """
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir)
        self.use_precomputed = use_precomputed
        self.max_samples = max_samples

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

    def load_manifest(self) -> pd.DataFrame:
        """Load and validate manifest."""
        logger.info(f"Loading manifest from {self.manifest_path}")

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        # Try parquet first, then CSV
        if str(self.manifest_path).endswith(".parquet"):
            df = pd.read_parquet(self.manifest_path)
        else:
            df = pd.read_csv(self.manifest_path)

        logger.info(f"Loaded {len(df)} samples")

        # Limit samples if requested
        if self.max_samples:
            df = df.head(self.max_samples)
            logger.info(f"Limited to {len(df)} samples (max_samples={self.max_samples})")

        # Validate required columns
        required_cols = ["sample_id", "split"]
        label_col = "final_label" if "final_label" in df.columns else "label"
        required_cols.append(label_col)

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Rename label column if needed
        if label_col != "final_label":
            df["final_label"] = df[label_col]

        return df

    def check_feature_availability(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Check if precomputed features are available.

        Returns:
            (all_available, missing_columns)
        """
        required_features = [
            "human_active_at_t",
            "num_humans_active_at_t",
            "overlap_active_at_t",
            "silence_duration_before_t",
            "current_human_speech_duration",
        ]

        missing = [col for col in required_features if col not in df.columns]
        all_available = len(missing) == 0

        return all_available, missing

    def validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean feature columns.

        Returns:
            DataFrame with validated features
        """
        features = [
            "human_active_at_t",
            "num_humans_active_at_t",
            "overlap_active_at_t",
            "silence_duration_before_t",
            "current_human_speech_duration",
        ]

        for feat in features:
            if feat not in df.columns:
                logger.warning(f"Missing feature: {feat}, filling with defaults")
                if "active" in feat or "overlap" in feat:
                    df[feat] = False
                else:
                    df[feat] = 0.0

            # Convert to expected types
            if "active" in feat or "overlap" in feat:
                df[feat] = df[feat].astype(bool)
            else:
                df[feat] = df[feat].astype(float)

        return df

    def run(self):
        """Execute full evaluation pipeline."""
        logger.info("=" * 80)
        logger.info("VA-Threshold Baseline Evaluation Pipeline")
        logger.info("=" * 80)

        # Load manifest
        df = self.load_manifest()
        logger.info(f"Splits: {df['split'].unique().tolist()}")
        logger.info(f"Label distribution:\n{df['final_label'].value_counts()}")

        # Check feature availability
        features_available, missing = self.check_feature_availability(df)
        if not features_available:
            logger.warning(f"Missing features: {missing}")
            if self.use_precomputed:
                logger.warning("Precomputed features not available, filling with defaults")
        else:
            logger.info("All precomputed features available ✓")

        # Validate and fill features
        df = self.validate_features(df)

        # Create baselines
        evaluator = BaselineEvaluator()
        theta_start_candidates = [0.3, 0.5, 0.6, 0.7, 1.0]
        theta_bc_min_speech_candidates = [0.5, 1.0, 1.5, 2.0]

        baselines = evaluator.create_baselines(
            theta_start_candidates,
            theta_bc_min_speech_candidates,
        )

        logger.info(f"Created baselines:")
        logger.info(f"  - VA-Silence: {len(baselines['silence'])} thresholds")
        logger.info(f"  - VA-Threshold: {len(baselines['threshold'])} threshold combinations")

        # Generate predictions
        logger.info("Generating predictions...")
        predictions_df = evaluator.predict_batch(df, baselines)
        logger.info(f"Generated {len(predictions_df)} predictions")

        # Select best thresholds
        logger.info("Selecting best thresholds based on validation Macro-F1...")
        try:
            best_theta_silence, best_thresholds_threshold = evaluator.select_best_thresholds(
                predictions_df
            )
            logger.info(f"Best VA-Silence theta_start: {best_theta_silence:.3f}")
            logger.info(f"Best VA-Threshold thresholds: theta_start={best_thresholds_threshold[0]:.3f}, theta_bc_min_speech={best_thresholds_threshold[1]:.3f}")
        except ValueError as e:
            logger.warning(f"Could not select thresholds: {e}")
            best_theta_silence = 0.6
            best_thresholds_threshold = (0.6, 1.0)

        # Filter to best thresholds for test evaluation
        test_df = df[df["split"] == "test"].copy()
        if len(test_df) == 0:
            logger.warning("No test split found, using all data")
            test_df = df

        logger.info(f"Evaluating on {len(test_df)} test samples")

        # Get predictions with best thresholds
        baseline_silence = VASilenceBaseline(best_theta_silence)
        baseline_threshold = VAThresholdBaseline(
            best_thresholds_threshold[0],
            best_thresholds_threshold[1],
        )

        test_df["pred_va_silence"] = test_df["silence_duration_before_t"].apply(
            baseline_silence.predict
        )

        test_df["pred_va_threshold"] = test_df.apply(
            lambda row: baseline_threshold.predict(
                row["human_active_at_t"],
                row["num_humans_active_at_t"],
                row["overlap_active_at_t"],
                row["silence_duration_before_t"],
                row["current_human_speech_duration"],
            ),
            axis=1,
        )

        # Encode labels
        label_map = {"WAIT": 0, "BACKCHANNEL": 1, "START_SPEAKING": 2}
        test_df["true_label_encoded"] = test_df["final_label"].map(label_map)
        test_df["pred_va_silence_encoded"] = test_df["pred_va_silence"].map(label_map)
        test_df["pred_va_threshold_encoded"] = test_df["pred_va_threshold"].map(label_map)

        # Compute metrics
        logger.info("Computing metrics...")
        metrics_silence = evaluator.compute_metrics(
            test_df["true_label_encoded"].values,
            test_df["pred_va_silence_encoded"].values,
        )
        metrics_threshold = evaluator.compute_metrics(
            test_df["true_label_encoded"].values,
            test_df["pred_va_threshold_encoded"].values,
        )

        # Save predictions
        output_predictions = self.output_dir / "va_baseline_predictions.parquet"
        predictions_to_save = test_df[
            [
                "sample_id",
                "split",
                "final_label",
                "pred_va_silence",
                "pred_va_threshold",
                "human_active_at_t",
                "num_humans_active_at_t",
                "overlap_active_at_t",
                "silence_duration_before_t",
                "current_human_speech_duration",
            ]
        ].copy()
        predictions_to_save["theta_start"] = best_theta_silence
        predictions_to_save["theta_bc_min_speech"] = best_thresholds_threshold[1]

        predictions_to_save.to_parquet(output_predictions, index=False)
        logger.info(f"Saved predictions: {output_predictions}")

        # Save metrics
        metrics_all = {
            "va_silence": metrics_silence,
            "va_threshold": metrics_threshold,
            "best_thresholds": {
                "theta_start_silence": float(best_theta_silence),
                "theta_start_threshold": float(best_thresholds_threshold[0]),
                "theta_bc_min_speech": float(best_thresholds_threshold[1]),
            },
            "num_samples": len(test_df),
        }

        output_metrics_json = self.output_dir / "va_baseline_metrics.json"
        with open(output_metrics_json, "w") as f:
            json.dump(metrics_all, f, indent=2)
        logger.info(f"Saved metrics JSON: {output_metrics_json}")

        # Save metrics markdown
        output_metrics_md = self.output_dir / "va_baseline_metrics.md"
        self._save_metrics_markdown(output_metrics_md, metrics_all)
        logger.info(f"Saved metrics markdown: {output_metrics_md}")

        # Save confusion matrices
        output_cm_silence = self.output_dir / "va_baseline_confusion_matrix_silence.csv"
        output_cm_threshold = self.output_dir / "va_baseline_confusion_matrix_threshold.csv"

        cm_silence = np.array(metrics_silence["confusion_matrix"])
        cm_threshold = np.array(metrics_threshold["confusion_matrix"])

        np.savetxt(output_cm_silence, cm_silence, delimiter=",", fmt="%d")
        np.savetxt(output_cm_threshold, cm_threshold, delimiter=",", fmt="%d")
        logger.info(f"Saved confusion matrices: {output_cm_silence}, {output_cm_threshold}")

        logger.info("=" * 80)
        logger.info("✅ Evaluation complete!")
        logger.info("=" * 80)

        # Print summary
        print("\n" + "=" * 80)
        print("VA-Threshold Baseline Results")
        print("=" * 80)
        print(f"\nTest samples: {len(test_df)}")
        print(f"Label distribution:")
        print(test_df["final_label"].value_counts().to_string())
        print(f"\n--- VA-Silence (theta_start={best_theta_silence:.3f}) ---")
        print(f"Accuracy: {metrics_silence['accuracy']:.4f}")
        print(f"Macro-F1: {metrics_silence['macro_f1']:.4f}")
        print(f"False Entry Rate: {metrics_silence['false_entry_rate']}")
        print(f"Missed Entry Rate: {metrics_silence['missed_entry_rate']}")

        print(f"\n--- VA-Threshold (theta_start={best_thresholds_threshold[0]:.3f}, theta_bc_min_speech={best_thresholds_threshold[1]:.3f}) ---")
        print(f"Accuracy: {metrics_threshold['accuracy']:.4f}")
        print(f"Macro-F1: {metrics_threshold['macro_f1']:.4f}")
        print(f"False Entry Rate: {metrics_threshold['false_entry_rate']}")
        print(f"Missed Entry Rate: {metrics_threshold['missed_entry_rate']}")
        print(f"BACKCHANNEL-as-Turn Error: {metrics_threshold['backchannel_as_turn_error']}")
        print(f"Turn-as-BACKCHANNEL Error: {metrics_threshold['turn_as_backchannel_error']}")

        print(f"\n📁 Results saved to: {self.output_dir}")
        print("=" * 80)

    def _save_metrics_markdown(self, output_path: Path, metrics: dict):
        """Save metrics in markdown format."""
        md_lines = [
            "# VA-Threshold Baseline Evaluation Results\n",
            f"## Test Set Size: {metrics['num_samples']}\n",
            "\n### Best Thresholds Selected\n",
            f"- VA-Silence theta_start: {metrics['best_thresholds']['theta_start_silence']:.3f}\n",
            f"- VA-Threshold theta_start: {metrics['best_thresholds']['theta_start_threshold']:.3f}\n",
            f"- VA-Threshold theta_bc_min_speech: {metrics['best_thresholds']['theta_bc_min_speech']:.3f}\n",
            "\n### VA-Silence Results\n",
            self._metrics_to_markdown(metrics["va_silence"]),
            "\n### VA-Threshold Results\n",
            self._metrics_to_markdown(metrics["va_threshold"]),
        ]

        with open(output_path, "w") as f:
            f.writelines(md_lines)

    @staticmethod
    def _metrics_to_markdown(metrics: dict) -> str:
        """Convert metrics dict to markdown table."""
        lines = ["| Metric | Value |\n", "|--------|-------|\n"]

        for key, value in metrics.items():
            if key == "confusion_matrix":
                continue
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.4f} |\n")
            elif value is not None:
                lines.append(f"| {key} | {value} |\n")

        return "".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VA-Threshold baselines on turn-taking dataset"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest parquet/CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/va_threshold_baseline",
        help="Output directory for results",
    )
    parser.add_argument(
        "--use-precomputed-if-available",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Use precomputed features if available",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit to N samples (for testing)",
    )

    args = parser.parse_args()

    use_precomputed = args.use_precomputed_if_available.lower() == "true"

    pipeline = VAThresholdEvaluationPipeline(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        use_precomputed=use_precomputed,
        max_samples=args.max_samples,
    )

    try:
        pipeline.run()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
