#!/usr/bin/env python3
"""
Analyze and compare ablation study results.

Usage:
    python scripts/analyze_ablations.py --results results_ablation_*.json --output ablation_comparison.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np


class AblationAnalyzer:
    """Analyze ablation study results."""

    PRIMARY_METRICS = [
        "macro_f1",
        "backchannel_f1",
        "start_speaking_f1",
        "false_entry_rate",
        "missed_entry_rate",
    ]

    SECONDARY_METRICS = [
        "wait_f1",
        "balanced_accuracy",
        "bc_as_turn_error",
        "turn_as_bc_error",
        "floor_violation_rate",
        "aggressiveness_rate",
        "ece",
    ]

    REALTIME_METRICS = [
        "inference_latency_ms",
        "real_time_factor",
    ]

    def __init__(self):
        self.results = {}

    def load_results(self, result_files: List[str]):
        """Load results from JSON files."""
        for result_file in result_files:
            path = Path(result_file)
            if not path.exists():
                print(f"Warning: {result_file} not found")
                continue

            with open(path) as f:
                data = json.load(f)

            # Extract ablation name from filename
            ablation_name = path.stem.replace("results_", "").replace("_", " ").title()
            self.results[ablation_name] = data.get("metrics", data)

    def compute_improvements(self) -> Dict:
        """Compute improvements relative to baseline (Ablation 4: full model)."""
        baseline_name = None
        baseline_metrics = None

        # Find baseline (usually the full model)
        for name, metrics in self.results.items():
            if "full" in name.lower() or "baseline" in name.lower():
                baseline_name = name
                baseline_metrics = metrics
                break

        if baseline_metrics is None:
            # Use best overall F1 as baseline
            best_f1 = -1
            for name, metrics in self.results.items():
                f1 = metrics.get("macro_f1", 0)
                if f1 > best_f1:
                    best_f1 = f1
                    baseline_name = name
                    baseline_metrics = metrics

        improvements = {}
        for name, metrics in self.results.items():
            if name == baseline_name:
                improvements[name] = {"is_baseline": True}
                continue

            ablation_improvement = {}

            # F1 metrics (higher is better)
            for metric in self.PRIMARY_METRICS + self.SECONDARY_METRICS:
                if metric in baseline_metrics and metric in metrics:
                    baseline_val = baseline_metrics[metric]
                    ablation_val = metrics[metric]

                    if "rate" in metric.lower() or "error" in metric.lower():
                        # Lower is better
                        improvement = (baseline_val - ablation_val) / (abs(baseline_val) + 1e-8)
                    else:
                        # Higher is better
                        improvement = (ablation_val - baseline_val) / (baseline_val + 1e-8)

                    ablation_improvement[metric] = {
                        "baseline": baseline_val,
                        "ablation": ablation_val,
                        "change_pct": improvement * 100,
                    }

            improvements[name] = ablation_improvement

        return improvements

    def generate_report(self) -> str:
        """Generate comparison report."""
        report = []
        report.append("=" * 80)
        report.append("ABLATION STUDY COMPARISON")
        report.append("=" * 80)
        report.append("")

        # Summary table
        report.append("Ablation Results Summary:")
        report.append("-" * 80)
        report.append(f"{'Ablation':<30} {'Macro-F1':>12} {'BC-F1':>12} {'Turn-F1':>12}")
        report.append("-" * 80)

        for name, metrics in sorted(self.results.items()):
            macro_f1 = metrics.get("macro_f1", 0)
            bc_f1 = metrics.get("backchannel_f1", 0)
            turn_f1 = metrics.get("start_speaking_f1", 0)
            report.append(f"{name:<30} {macro_f1:>12.4f} {bc_f1:>12.4f} {turn_f1:>12.4f}")

        report.append("")

        # Error rates
        report.append("Error Rates:")
        report.append("-" * 80)
        report.append(f"{'Ablation':<30} {'False Entry':>15} {'Missed Entry':>15}")
        report.append("-" * 80)

        for name, metrics in sorted(self.results.items()):
            false_entry = metrics.get("false_entry_rate", 0)
            missed_entry = metrics.get("missed_entry_rate", 0)
            report.append(f"{name:<30} {false_entry:>15.4f} {missed_entry:>15.4f}")

        report.append("")

        # Real-time performance
        report.append("Real-time Performance:")
        report.append("-" * 80)
        report.append(f"{'Ablation':<30} {'Latency (ms)':>20} {'RT Factor':>15}")
        report.append("-" * 80)

        for name, metrics in sorted(self.results.items()):
            latency = metrics.get("inference_latency_ms", 0)
            rt_factor = metrics.get("real_time_factor", 0)
            report.append(f"{name:<30} {latency:>20.2f} {rt_factor:>15.4f}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def save_comparison(self, output_file: str):
        """Save comparison to JSON file."""
        improvements = self.compute_improvements()

        comparison = {
            "results": self.results,
            "improvements_vs_baseline": improvements,
        }

        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2, default=str)

        print(f"Comparison saved to {output_file}")


def main(args):
    analyzer = AblationAnalyzer()
    analyzer.load_results(args.results)

    # Print report
    report = analyzer.generate_report()
    print(report)

    # Save comparison
    analyzer.save_comparison(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Result JSON files from ablations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ablation_comparison.json",
        help="Output comparison JSON",
    )

    args = parser.parse_args()
    main(args)
