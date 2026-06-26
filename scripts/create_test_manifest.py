#!/usr/bin/env python3
"""
Create a test manifest for baseline evaluation.

Usage:
    python scripts/create_test_manifest.py --output data/processed/final_manifest_test.parquet
"""

import argparse
import json
from pathlib import Path

import numpy as np


def create_test_manifest(
    output_path: str,
    num_validation: int = 30,
    num_test: int = 30,
    seed: int = 42,
):
    """
    Create a synthetic test manifest for baseline evaluation.

    Args:
        output_path: Path to save manifest
        num_validation: Number of validation samples
        num_test: Number of test samples
        seed: Random seed
    """
    np.random.seed(seed)

    data = []

    # Create validation samples
    for i in range(num_validation):
        sample = {
            "sample_id": f"val_sample_{i:03d}",
            "split": "validation",
            "final_label": np.random.choice(["WAIT", "BACKCHANNEL", "START_SPEAKING"]),
            "human_active_at_t": bool(np.random.choice([True, False])),
            "num_humans_active_at_t": int(np.random.choice([0, 1, 2, 3])),
            "overlap_active_at_t": bool(np.random.choice([True, False])),
            "silence_duration_before_t": float(np.random.uniform(0, 2.5)),
            "current_human_speech_duration": float(np.random.uniform(0, 3.5)),
        }
        data.append(sample)

    # Create test samples
    for i in range(num_test):
        sample = {
            "sample_id": f"test_sample_{i:03d}",
            "split": "test",
            "final_label": np.random.choice(["WAIT", "BACKCHANNEL", "START_SPEAKING"]),
            "human_active_at_t": bool(np.random.choice([True, False])),
            "num_humans_active_at_t": int(np.random.choice([0, 1, 2, 3])),
            "overlap_active_at_t": bool(np.random.choice([True, False])),
            "silence_duration_before_t": float(np.random.uniform(0, 2.5)),
            "current_human_speech_duration": float(np.random.uniform(0, 3.5)),
        }
        data.append(sample)

    # Save as JSON for inspection, then convert to parquet if pandas available
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to save as parquet with pandas
    try:
        import pandas as pd

        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)
        print(f"✓ Saved parquet to {output_path}")

    except ImportError:
        # Fallback: save as JSON lines
        json_path = output_path.with_suffix(".jsonl")
        with open(json_path, "w") as f:
            for sample in data:
                f.write(json.dumps(sample) + "\n")

        print(f"⚠ pandas not available, saved JSONL to {json_path}")
        print(f"  To convert to parquet: pd.read_json('{json_path}', lines=True).to_parquet(...)")

    # Print summary
    import json as json_module

    print(f"\nManifest Summary:")
    print(f"  Total samples: {len(data)}")
    print(f"  Validation: {num_validation}")
    print(f"  Test: {num_test}")
    print(f"\n  Sample structure:")
    print(f"  {json_module.dumps(data[0], indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create test manifest for baseline evaluation")
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/final_manifest_test.parquet",
        help="Output path for manifest",
    )
    parser.add_argument(
        "--num-validation",
        type=int,
        default=30,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=30,
        help="Number of test samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    create_test_manifest(
        output_path=args.output,
        num_validation=args.num_validation,
        num_test=args.num_test,
        seed=args.seed,
    )
