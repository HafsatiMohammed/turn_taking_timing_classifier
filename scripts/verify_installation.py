#!/usr/bin/env python3
"""
Verify installation and setup of turn-taking timing classifier.

Usage:
    python scripts/verify_installation.py [--check-data]
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np

print("=" * 80)
print("Turn-Taking Timing Classifier - Installation Verification")
print("=" * 80)
print()

# Check 1: Python version
print("✓ Python version:", sys.version.split()[0])
assert sys.version_info >= (3, 7), "Python 3.7+ required"

# Check 2: PyTorch
print("✓ PyTorch:", torch.__version__)
print("  CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU:", torch.cuda.get_device_name(0))
    print("  CUDA version:", torch.version.cuda)

# Check 3: Dependencies
print()
print("Checking dependencies...")
try:
    import pandas as pd
    print("✓ pandas:", pd.__version__)
except ImportError:
    print("✗ pandas not installed")
    sys.exit(1)

try:
    import pyarrow
    print("✓ pyarrow:", pyarrow.__version__)
except ImportError:
    print("✗ pyarrow not installed")
    sys.exit(1)

try:
    import yaml
    print("✓ pyyaml:", yaml.__version__)
except ImportError:
    print("✗ pyyaml not installed")
    sys.exit(1)

try:
    import sklearn
    print("✓ scikit-learn:", sklearn.__version__)
except ImportError:
    print("✗ scikit-learn not installed")
    sys.exit(1)

# Check 4: Source modules
print()
print("Checking source modules...")
try:
    from src.utils import load_config
    print("✓ src.utils")
except ImportError as e:
    print("✗ src.utils:", e)
    sys.exit(1)

try:
    from src.data import TimingDataset, get_dataloaders
    print("✓ src.data")
except ImportError as e:
    print("✗ src.data:", e)
    sys.exit(1)

try:
    from src.models import TimingActionNet
    print("✓ src.models")
except ImportError as e:
    print("✗ src.models:", e)
    sys.exit(1)

try:
    from src.training import Trainer
    print("✓ src.training")
except ImportError as e:
    print("✗ src.training:", e)
    sys.exit(1)

try:
    from src.evaluation import Evaluator, MetricsCalculator
    print("✓ src.evaluation")
except ImportError as e:
    print("✗ src.evaluation:", e)
    sys.exit(1)

# Check 5: Config file
print()
print("Checking configuration...")
config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
if config_path.exists():
    print(f"✓ Found config at {config_path}")
    try:
        config = load_config(str(config_path))
        print(f"  - Dataset path: {config['dataset']['path']}")
        print(f"  - Model type: {config['model']['frame_branch']['type']}")
        print(f"  - Training epochs: {config['training']['num_epochs']}")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        sys.exit(1)
else:
    print(f"✗ Config not found at {config_path}")
    sys.exit(1)

# Check 6: Model instantiation
print()
print("Checking model...")
try:
    model = TimingActionNet(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created successfully")
    print(f"  - Parameters: {num_params:,}")
    print(f"  - Device: {device}")

    # Test forward pass
    frame = torch.randn(2, 120, 7).to(device)
    scalar = torch.randn(2, 6).to(device)
    logits = model(frame, scalar)
    assert logits.shape == (2, 3), f"Expected shape (2, 3), got {logits.shape}"
    print(f"  - Forward pass OK: {logits.shape}")

except Exception as e:
    print(f"✗ Model error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 7: Data setup
print()
print("Checking data setup...")
dataset_path = config['dataset']['path']
if Path(dataset_path).exists():
    print(f"✓ Dataset path exists: {dataset_path}")

    # List parquet files
    parquet_files = list(Path(dataset_path).glob("*.parquet"))
    print(f"  - Found {len(parquet_files)} parquet files:")
    for f in parquet_files:
        print(f"    - {f.name}")

        if args.check_data:
            try:
                df = pd.read_parquet(f)
                print(f"      Rows: {len(df)}, Columns: {len(df.columns)}")

                # Check required columns
                required = ["final_label", "training_weight", "X_frame", "X_scalar"]
                missing = [col for col in required if col not in df.columns]
                if missing:
                    print(f"      Missing columns: {missing}")
                else:
                    print(f"      ✓ Has all required columns")

            except Exception as e:
                print(f"      ✗ Error reading: {e}")
else:
    print(f"⚠ Dataset path not found (expected on first run): {dataset_path}")
    print("  Once you download the data, the checks above will pass")

# Check 8: Directories
print()
print("Checking directories...")
dirs = ["checkpoints", "logs", "configs", "scripts", "src"]
for d in dirs:
    p = Path(__file__).parent.parent / d
    if p.exists():
        print(f"✓ {d}/")
    else:
        print(f"✗ {d}/ not found")

# Check 9: Scripts
print()
print("Checking scripts...")
scripts = ["train.py", "test.py", "resume_training.py", "analyze_ablations.py"]
script_dir = Path(__file__).parent
for s in scripts:
    p = script_dir / s
    if p.exists():
        print(f"✓ {s}")
    else:
        print(f"✗ {s} not found")

# Final summary
print()
print("=" * 80)
print("✓ Installation verification complete!")
print("=" * 80)
print()
print("Next steps:")
print("1. Place parquet files in:", dataset_path)
print("2. Run: python scripts/train.py --config configs/config.yaml")
print("3. After training: python scripts/test.py --config configs/config.yaml --checkpoint checkpoints/checkpoint_best.pt")
print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify installation")
    parser.add_argument("--check-data", action="store_true", help="Check data files")
    args = parser.parse_args()
