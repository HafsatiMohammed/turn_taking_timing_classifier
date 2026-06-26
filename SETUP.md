# Complete Setup & Usage Guide

This guide walks you through everything from setup to running your first ablation study.

## 🚀 Quick Start (15 minutes)

### 1. Install Dependencies (2 min)
```bash
pip install -r requirements.txt
python scripts/verify_installation.py
```

### 2. Prepare Data (10 min)
You have two paths:

**Path A: Create metadata from scratch**
```python
# Save this as create_metadata.py
import json
from pathlib import Path

samples = []
# Add your samples here based on your dataset curation
# Each sample needs: meeting_id, pseudo_robot, time, context_start, context_end, 
#                    split, final_label, training_weight, exclude_from_training

output = Path("data/processed/final_manifest.jsonl")
output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

print(f"Created {len(samples)} samples")
```

**Path B: I already have metadata.jsonl**
```bash
# Just copy it to the right location
cp your_metadata.jsonl data/processed/final_manifest.jsonl
```

### 3. Extract Features from AMI (3 min)
```bash
python scripts/prepare_dataset.py \
    --metadata data/processed/final_manifest.jsonl \
    --output data/processed/hf_export
```

**Output:** 
```
data/processed/hf_export/
├── train.parquet       (X_frame, X_scalar extracted)
├── validation.parquet
└── test.parquet
```

### 4. Train Model (varies)
```bash
python scripts/train.py --config configs/config.yaml
```

Checkpoints auto-saved to `checkpoints/checkpoint_best.pt`

### 5. Evaluate (2 min)
```bash
python scripts/test.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/checkpoint_best.pt
```

Results saved to `results.json`

---

## 📋 Complete Workflow

### Phase 1: Setup
```bash
# Clone/navigate to project
cd turn_taking_timing_classifier

# Install all dependencies
pip install -r requirements.txt

# Verify everything works
python scripts/verify_installation.py
```

### Phase 2: Data Preparation

See [DATA_PREPARATION.md](DATA_PREPARATION.md) for detailed guide.

**TL;DR:**
1. Create `data/processed/final_manifest.jsonl` with your samples
2. Run `python scripts/prepare_dataset.py`
3. Verify: `ls -lh data/processed/hf_export/*.parquet`

### Phase 3: Training

#### Single Model (Baseline)
```bash
# Default config (full model)
python scripts/train.py --config configs/config.yaml

# View logs
tail -f logs/timing_net_*.log
```

#### Ablation Study (6 experiments)
See [ABLATION_STUDIES.md](ABLATION_STUDIES.md) for complete guide.

**Quick ablation workflow:**
```bash
# 1. Create config files for each ablation
for i in 1 2 3 4 5 6; do
  cp configs/config.yaml configs/ablation_${i}.yaml
done

# 2. Edit each config with different architectures
# (See ABLATION_STUDIES.md for which parameters to change)

# 3. Run training for each
for i in 1 2 3 4 5 6; do
  echo "Running Ablation $i..."
  python scripts/train.py --config configs/ablation_${i}.yaml
  sleep 5  # Let GPU cool down between runs
done
```

#### Resume Training
```bash
# Auto-resumes from latest checkpoint
python scripts/train.py --config configs/config.yaml

# Or explicit resume
python scripts/resume_training.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/checkpoint_epoch_50.pt
```

### Phase 4: Evaluation

#### Evaluate Single Model
```bash
python scripts/test.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/checkpoint_best.pt \
    --output results.json
```

#### Compare Ablations
```bash
python scripts/analyze_ablations.py \
    --results results_ablation_*.json \
    --output ablation_comparison.json
```

---

## 📁 Project Layout

```
turn_taking_timing_classifier/
├── README.md                       # Full documentation
├── QUICKSTART.md                   # 5-min quick start
├── SETUP.md                        # This file
├── DATA_PREPARATION.md             # Data extraction guide
├── ABLATION_STUDIES.md             # Ablation setup guide
├── PROJECT_SUMMARY.md              # Architecture overview
│
├── configs/
│   ├── config.yaml                 # Main config
│   └── ablation_studies.yaml       # Ablation reference
│
├── src/
│   ├── data/                       # Data loading
│   │   ├── dataset.py
│   │   └── loaders.py
│   ├── models/                     # Model architecture
│   │   ├── timing_net.py
│   │   └── branches.py
│   ├── training/                   # Training loop
│   │   └── trainer.py
│   ├── evaluation/                 # Metrics & evaluation
│   │   ├── metrics.py
│   │   └── evaluator.py
│   └── utils/                      # Config, logging
│       ├── config.py
│       └── logging.py
│
├── scripts/
│   ├── prepare_dataset.py          # Feature extraction
│   ├── train.py                    # Train model
│   ├── test.py                     # Evaluate model
│   ├── resume_training.py          # Resume from checkpoint
│   ├── analyze_ablations.py        # Compare ablations
│   ├── verify_installation.py      # Verify setup
│   └── run_experiment.sh           # Full workflow
│
├── data/
│   └── processed/
│       ├── final_manifest.jsonl    # Sample metadata
│       └── hf_export/
│           ├── train.parquet       # Training data
│           ├── validation.parquet  # Val data
│           └── test.parquet        # Test data
│
├── checkpoints/                    # Model checkpoints
│   └── checkpoint_best.pt          # Best model
│
├── logs/                           # Training logs
│   └── timing_net_*.log            # Log files
│
└── requirements.txt                # Python dependencies
```

---

## 🔧 Configuration

All settings in `configs/config.yaml`:

```yaml
# Dataset
dataset:
  path: "data/processed/hf_export"
  batch_size: 32

# Model
model:
  frame_branch:
    type: "tcn"                # TCN or GRU
    output_dim: 128
  scalar_branch:
    output_dim: 64
  fusion:
    hidden_dims: [256, 128]
    output_dim: 3

# Training
training:
  num_epochs: 100
  learning_rate: 0.001
  lr_scheduler: "cosine"
  early_stopping_patience: 15

# Evaluation
evaluation:
  thresholds:
    start_speaking: 0.70
    backchannel: 0.55
```

---

## 📊 Training Flow

```
Start Training
    ↓
Load Data (normalize features)
    ↓
Create Model & Optimizer
    ↓
[For each epoch:]
    ├─ Train on batches
    │   ├─ Forward pass
    │   ├─ Compute loss (weighted by sample weights)
    │   ├─ Backward pass
    │   └─ Update weights
    ├─ Validate on val set
    │   └─ Compute metrics (15+ KPIs)
    ├─ Save checkpoint (every N epochs)
    ├─ Check early stopping
    │   └─ If no improvement → stop
    └─ Log metrics
    ↓
Save Best Model
    ↓
Complete Training
```

## 📈 Evaluation

Each evaluation computes:

**Primary KPIs:**
- Macro-F1
- BACKCHANNEL F1
- START_SPEAKING F1
- False entry rate
- Missed entry rate

**Secondary:**
- WAIT F1
- Balanced accuracy
- BC-as-turn error
- Turn-as-BC error
- Floor violation rate
- Aggressiveness rate
- ECE

**Real-time:**
- Inference latency (ms)
- Real-time factor
- Model size (MB)

---

## 🧪 Ablation Studies

### 6 Configurations to Compare

| # | Name | Features | Purpose |
|---|------|----------|---------|
| 1 | Previous Activity Only | Robot past activity | Baseline |
| 2 | Activity + Context | + Scalar features | Does context help? |
| 3 | Frame History | All temporal | Is history important? |
| 4 | Full Model | All features | Best performance |
| 5 | Current State Only | Scalar only | Need history? |
| 6 | Human History | Human activity | Robot awareness? |

### Expected Pattern
```
Ablation 1:  ~0.45 Macro-F1  [Baseline]
Ablation 2:  ~0.58 Macro-F1  [+13 points]
Ablation 3:  ~0.63 Macro-F1  [+18 points]
Ablation 4:  ~0.72 Macro-F1  [+27 points] ← Best
Ablation 5:  ~0.42 Macro-F1  [-3 points]
Ablation 6:  ~0.60 Macro-F1  [+15 points]
```

(Your results will vary - use this as reference)

---

## 🐛 Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: datasets` | `pip install datasets` |
| CUDA out of memory | Reduce `batch_size` in config |
| `FileNotFoundError: *.parquet` | Run `python scripts/prepare_dataset.py` first |
| NaN loss during training | Lower `learning_rate` to 0.0005 |
| Very slow data loading | Increase `num_workers` in dataset config |
| Model not improving | Check label distribution, verify data |
| Feature extraction is slow | Normal on first run (HF download cache) |

---

## ✅ Verification Checklist

- [ ] Dependencies installed: `pip list | grep -E "torch|pandas|scikit"`
- [ ] Metadata file exists: `ls data/processed/final_manifest.jsonl`
- [ ] Features extracted: `ls data/processed/hf_export/*.parquet`
- [ ] Data loads: `python scripts/verify_installation.py --check-data`
- [ ] Model trains: First epoch completes without error
- [ ] Model saves: `ls checkpoints/checkpoint_latest.pt`
- [ ] Evaluation works: `results.json` created with metrics

---

## 📚 Next Steps

1. **Start with basics**: Run full model on your data
   ```bash
   python scripts/train.py --config configs/config.yaml
   ```

2. **Set up ablations**: Modify configs for 6 variants
   ```bash
   # See ABLATION_STUDIES.md for detailed setup
   ```

3. **Run experiments**: Train each ablation
   ```bash
   ./scripts/run_experiment.sh ablation_1
   ./scripts/run_experiment.sh ablation_2
   # ... etc
   ```

4. **Analyze results**: Compare performance
   ```bash
   python scripts/analyze_ablations.py --results results_ablation_*.json
   ```

5. **Write paper**: Use metrics for paper submission

---

## 📖 Full Documentation

- **README.md** - Complete architecture & API reference
- **QUICKSTART.md** - 5-minute setup
- **DATA_PREPARATION.md** - Feature extraction details
- **ABLATION_STUDIES.md** - Ablation study guide
- **PROJECT_SUMMARY.md** - Technical overview
- **configs/config.yaml** - All hyperparameters documented

---

## 🎯 Your First End-to-End Run

```bash
# 1. Verify install (2 min)
python scripts/verify_installation.py

# 2. Create sample metadata (1 min)
python -c "
import json
from pathlib import Path

sample = {
    'sample_id': 'test_001',
    'meeting_id': 'ES2002',
    'pseudo_robot': 'A',
    'time': 100.0,
    'context_start': 94.0,
    'context_end': 100.0,
    'split': 'train',
    'final_label': 'WAIT',
    'training_weight': 0.95,
    'exclude_from_training': False,
}

output = Path('data/processed/final_manifest.jsonl')
output.parent.mkdir(parents=True, exist_ok=True)
with open(output, 'w') as f:
    f.write(json.dumps(sample) + '\n')
print('Created sample metadata')
"

# 3. Extract features (5 min on first run)
python scripts/prepare_dataset.py

# 4. Check data loaded
python -c "
from src.data import get_dataloaders
loaders = get_dataloaders('data/processed/hf_export', batch_size=1)
batch = next(iter(loaders['train']))
print(f'✓ Data loaded: frame {batch[\"frame\"].shape}, scalar {batch[\"scalar\"].shape}')
"

# 5. Train for 3 epochs (test run)
python scripts/train.py --config configs/config.yaml 2>&1 | head -50

# 6. Evaluate
python scripts/test.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/checkpoint_best.pt

# 7. Check results
python -c "import json; print(json.dumps(json.load(open('results.json'))['metrics'], indent=2))"
```

You're done! Now scale up with your full dataset and ablation studies.

---

## 💾 Saving Your Progress

```bash
# Git track everything except data/logs/checkpoints
git add -A
git commit -m "turn-taking timing classifier setup complete"

# Checkpoints are auto-saved during training
# Training resumes automatically from latest checkpoint
```

---

**Ready to go?** Start with:
```bash
python scripts/train.py --config configs/config.yaml
```

Questions? Check the relevant guide:
- Data issues → [DATA_PREPARATION.md](DATA_PREPARATION.md)
- Training issues → [QUICKSTART.md](QUICKSTART.md)
- Architecture questions → [README.md](README.md)
- Ablations → [ABLATION_STUDIES.md](ABLATION_STUDIES.md)
