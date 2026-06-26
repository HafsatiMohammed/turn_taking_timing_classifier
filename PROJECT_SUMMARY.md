# Project Summary: Turn-Taking Timing Classifier

## What's Been Built

A complete machine learning pipeline for training and evaluating a deep neural network that predicts conversational turn-taking timing in multi-speaker meetings. This includes:

✅ **Data Pipeline**: Parquet loading, preprocessing, normalization  
✅ **Model Architecture**: Two-branch network (TCN/GRU + MLP)  
✅ **Training System**: Full training loop with checkpointing and resumption  
✅ **Evaluation**: Comprehensive KPI calculation (15+ metrics)  
✅ **Ablation Framework**: Complete setup for ablation studies  
✅ **Documentation**: Guides, examples, and configurations  

## Project Structure

```
turn_taking_timing_classifier/
├── README.md                        # Main documentation
├── QUICKSTART.md                    # Get started in 5 minutes
├── ABLATION_STUDIES.md              # Ablation study guide
├── PROJECT_SUMMARY.md               # This file
│
├── configs/
│   ├── config.yaml                  # Main configuration (all hyperparams)
│   └── ablation_studies.yaml        # Ablation configurations reference
│
├── src/
│   ├── data/
│   │   ├── dataset.py               # Parquet loading, preprocessing
│   │   └── loaders.py               # PyTorch DataLoaders
│   │
│   ├── models/
│   │   ├── timing_net.py            # Complete model (TimingActionNet)
│   │   └── branches.py              # TCN, GRU, MLP, Fusion modules
│   │
│   ├── training/
│   │   └── trainer.py               # Training loop with checkpointing
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # 15+ KPI calculations
│   │   └── evaluator.py             # Evaluation pipeline
│   │
│   └── utils/
│       ├── config.py                # YAML config loading
│       └── logging.py               # Logging setup
│
├── scripts/
│   ├── train.py                     # Main training script
│   ├── test.py                      # Evaluation script
│   ├── resume_training.py           # Resume from checkpoint
│   ├── analyze_ablations.py         # Ablation analysis
│   └── run_experiment.sh            # Full experiment workflow
│
├── checkpoints/                     # Saved model checkpoints
├── logs/                            # Training logs
└── requirements.txt                 # Python dependencies
```

## Key Features

### 1. **Training with Checkpointing**
- Saves model every N epochs
- Tracks best model by validation metric
- Preserves optimizer and scheduler state
- Full resumption capability
- Early stopping support

```bash
# Train from scratch
python scripts/train.py --config configs/config.yaml

# Resume automatically
python scripts/train.py --config configs/config.yaml

# Or explicit resume
python scripts/resume_training.py --config configs/config.yaml --checkpoint checkpoints/checkpoint_epoch_50.pt
```

### 2. **Comprehensive Evaluation**
15+ metrics computed automatically:
- Macro-F1, per-class F1
- False entry and missed entry rates
- Floor violation rates
- Calibration error (ECE)
- Inference latency
- Real-time performance metrics

```bash
# Evaluate on test set
python scripts/test.py --config configs/config.yaml --checkpoint checkpoints/checkpoint_best.pt
```

### 3. **Flexible Model Architecture**
Two-branch network:
- **Frame branch**: TCN or GRU on temporal context (120 frames, 6 seconds)
- **Scalar branch**: MLP on current state (6 features)
- **Fusion**: Concatenate and predict 3 classes

Configure in `config.yaml`:
```yaml
model:
  frame_branch:
    type: "tcn"                    # or "gru"
    out_channels: [32, 64, 128]    # TCN channel progression
    output_dim: 128
  scalar_branch:
    hidden_dims: [64, 64]
    output_dim: 64
```

### 4. **Ablation Study Framework**
Built-in support for 6 different feature combinations:
1. Previous activity only (baseline)
2. + Current scalar context
3. All frame features
4. Full model
5. Current state only (no history)
6. Human activity history

See [ABLATION_STUDIES.md](ABLATION_STUDIES.md) for complete guide.

### 5. **Data Handling**
- Reads parquet files (train/val/test split)
- Automatic normalization (per-feature statistics)
- Class-weighted sampling for imbalanced data
- Configurable batch sizes and data loading

## Quick Usage

### Setup (5 minutes)
```bash
pip install -r requirements.txt
```

### Train (from scratch)
```bash
python scripts/train.py --config configs/config.yaml
```

### Evaluate (after training)
```bash
python scripts/test.py --config configs/config.yaml --checkpoint checkpoints/checkpoint_best.pt
```

### Run Ablation Studies
```bash
# See ABLATION_STUDIES.md for complete setup

# Run single ablation
python scripts/train.py --config configs/ablation_1.yaml
python scripts/test.py --config configs/ablation_1.yaml --checkpoint checkpoints/checkpoint_best.pt

# Analyze results
python scripts/analyze_ablations.py --results results_ablation_*.json
```

## Configuration

All hyperparameters in `configs/config.yaml`:

```yaml
dataset:
  path: "data/processed/hf_export"
  batch_size: 32
  num_workers: 4

training:
  num_epochs: 100
  learning_rate: 0.001
  lr_scheduler: "cosine"
  early_stopping_patience: 15
  checkpoint_dir: "checkpoints"

model:
  frame_branch:
    type: "tcn"
    output_dim: 128
  scalar_branch:
    output_dim: 64
  fusion:
    hidden_dims: [256, 128]

evaluation:
  batch_size: 64
  thresholds:
    start_speaking: 0.70
    backchannel: 0.55
```

See `configs/config.yaml` for all 100+ parameters with descriptions.

## Model Input/Output

### Input
```python
batch = {
    "frame": torch.Tensor[B, 120, 7],   # 120 frames, 7 features
    "scalar": torch.Tensor[B, 6],       # 6 scalar features
    "label": torch.Tensor[B],           # 0=WAIT, 1=BACKCHANNEL, 2=START_SPEAKING
    "weight": torch.Tensor[B],          # Sample weights (confidence)
}
```

### Output
```python
logits = model(frame, scalar)  # [B, 3]
probs = torch.softmax(logits, dim=1)  # [B, 3]
predictions = torch.argmax(logits, dim=1)  # [B]
```

### Inference Policy
```python
p_wait, p_bc, p_start = probs

if p_start > 0.70:
    action = "START_SPEAKING"
elif p_bc > 0.55 and human_active:
    action = "BACKCHANNEL"
else:
    action = "WAIT"
```

## Training Workflow

1. **Load Data**
   - Read parquet files (train/val/test)
   - Compute normalization statistics
   - Create DataLoaders with class weighting

2. **Create Model**
   - Build two-branch network
   - Initialize weights
   - Count parameters

3. **Train Loop** (per epoch)
   - Forward pass on training batch
   - Compute weighted loss
   - Backward pass with gradient clipping
   - Update weights
   - Save checkpoint

4. **Validation Loop** (per epoch)
   - Evaluate on validation set
   - Compute all metrics
   - Track best metric
   - Early stopping check

5. **Complete**
   - Save best model
   - Print results
   - Log metrics to tensorboard (optional)

## Checkpointing Details

Each checkpoint contains:
```python
{
    "epoch": int,                    # Current epoch
    "model_state": dict,             # Model weights
    "optimizer_state": dict,         # Optimizer state
    "scheduler_state": dict,         # LR scheduler state
    "best_val_metric": float,        # Best validation metric
    "train_history": list,           # Loss per epoch
    "val_history": list,             # Metrics per epoch
    "config": dict,                  # Configuration
}
```

Resume automatically:
- Restores epoch number
- Restores optimizer momentum/variance
- Restores LR schedule
- Continues training from exact point

## KPIs Calculated

### Primary (for paper)
- Macro-F1
- BACKCHANNEL F1
- START_SPEAKING F1
- False entry rate (predicted entry but true WAIT)
- Missed entry rate (true entry but predicted WAIT)

### Secondary
- WAIT F1
- Balanced accuracy
- BC-as-turn error rate
- Turn-as-BC error rate
- Floor violation rate
- Aggressiveness rate (fraction of entry predictions)
- ECE (Expected Calibration Error)

### Real-time
- Inference latency (ms)
- Real-time factor (latency / 100ms window)
- Model size (MB)

All metrics computed on weighted samples to match training distribution.

## Paper-Ready Output

After training and evaluation:

```
results.json                          # Predictions + metrics
├── metrics
│   ├── macro_f1: 0.724
│   ├── backchannel_f1: 0.681
│   ├── start_speaking_f1: 0.758
│   ├── false_entry_rate: 0.094
│   ├── missed_entry_rate: 0.087
│   └── ... (15+ metrics)
├── predictions
│   ├── sample_ids
│   ├── labels
│   ├── predictions
│   └── probabilities
└── model_config

logs/timing_net_*.log                 # Training logs
ablation_comparison.json              # Ablation results (if run)
```

## Extending the Project

### Add Custom Loss Functions
Edit `src/training/trainer.py:_setup_loss()`:
```python
elif loss_type == "focal":
    self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

### Add New Metrics
Edit `src/evaluation/metrics.py:compute_all_metrics()`:
```python
metrics["custom_metric"] = compute_custom_metric(y_true, y_pred, y_probs)
```

### Change Model Architecture
Edit `configs/config.yaml`:
```yaml
model:
  frame_branch:
    type: "lstm"           # Change to LSTM
    num_layers: 3
    hidden_dim: 256
```

### Add Logging/Monitoring
Enable wandb in config.yaml:
```yaml
logging:
  wandb: true
  wandb_project: "turn-taking-timing"
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| `FileNotFoundError: train.parquet` | Check dataset path in config.yaml |
| CUDA out of memory | Reduce batch_size in config |
| NaN loss during training | Lower learning_rate or use loss_type: "ce" |
| Model not improving | Check label distribution, verify data loading |
| Very slow data loading | Increase num_workers in dataset config |

## References

- **Dataset**: Parquet format with frame and scalar features
- **Model**: Two-branch architecture (TCN + MLP)
- **Training**: PyTorch with gradient clipping and learning rate scheduling
- **Evaluation**: Scikit-learn metrics + custom KPIs
- **Configuration**: YAML with full hyperparameter control

## Citation

If you use this code for your paper:

```bibtex
@software{timing_classifier_2024,
  title={Turn-Taking Timing Classifier for Multi-Speaker Conversations},
  author={...},
  year={2024},
  url={https://github.com/your-repo/turn_taking_timing_classifier}
}
```

## Next Steps

1. **Verify Setup**: See [QUICKSTART.md](QUICKSTART.md)
2. **Run First Experiment**: `python scripts/train.py --config configs/config.yaml`
3. **Set Up Ablations**: See [ABLATION_STUDIES.md](ABLATION_STUDIES.md)
4. **Analyze Results**: Use `scripts/analyze_ablations.py`
5. **Write Paper**: Use results for your ablation study

---

Built with ❤️ for conversational AI research.
