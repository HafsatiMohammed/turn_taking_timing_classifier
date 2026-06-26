# Turn-Taking Timing Classifier

A PyTorch-based deep learning model for predicting conversational turn-taking timing in multi-speaker settings. This project implements a two-branch neural network architecture combining temporal convolutional networks (TCN) and multi-layer perceptrons (MLP) for turn-taking action prediction.

## Overview

### Problem
Predict three actions for a pseudo-robot in multi-speaker meetings:
- **WAIT**: Do not speak
- **BACKCHANNEL**: Provide minimal feedback (e.g., "mm-hmm")
- **START_SPEAKING**: Take a full turn

### Model Architecture
```
Frame Branch (TCN/GRU)         Scalar Branch (MLP)
    ↓                               ↓
[B, 120, 7]                     [B, 6]
    ↓                               ↓
[32→64→128]                     [64→64]
    ↓                               ↓
[B, 128] ──→ Concat ←─── [B, 64]
               ↓
          Fusion MLP
               ↓
          [B, 3] logits
```

### Inputs

**Frame-level context** (last 6 seconds at 0.05s frame shift):
1. any_human_active
2. num_humans_active_norm
3. overlap_active
4. silence_active
5. human_onset
6. human_offset
7. pseudo_robot_past_active

**Scalar features** (current time):
1. silence_duration_before_t
2. current_human_speech_duration
3. human_speech_ratio_last_1s
4. human_speech_ratio_last_6s
5. overlap_ratio_last_6s
6. time_since_pseudo_robot_last_spoke

## Setup

### Installation

```bash
pip install -r requirements.txt
```

### Dataset
Place parquet files in `data/processed/hf_export/`:
```
data/processed/
├── hf_export/
│   ├── train.parquet
│   ├── validation.parquet
│   └── test.parquet
└── final_manifest.jsonl
```

## Usage

### Training

Start training from scratch:
```bash
python scripts/train.py --config configs/config.yaml
```

The trainer will automatically:
- Load and normalize data
- Create weighted sampler for class balancing
- Save checkpoints every N epochs
- Track best model by validation macro-F1
- Implement early stopping

### Resume Training

Continue from latest checkpoint:
```bash
python scripts/train.py --config configs/config.yaml
```

Or specify a checkpoint:
```bash
python scripts/resume_training.py --config configs/config.yaml --checkpoint checkpoints/checkpoint_epoch_50.pt
```

The system maintains:
- Current epoch
- Optimizer state
- Learning rate scheduler state
- Training/validation history
- Best metric tracking

### Evaluation

Test the trained model:
```bash
python scripts/test.py --config configs/config.yaml --checkpoint checkpoints/checkpoint_best.pt
```

Output:
- `results.json`: Predictions and metrics
- Console logs with KPI summary

## Configuration

Edit `configs/config.yaml` to customize:

### Dataset
- `path`: Path to parquet files
- `batch_size`: Training batch size

### Model
- `frame_branch.type`: "tcn" or "gru"
- `frame_branch.out_channels`: TCN channel progression
- `scalar_branch.hidden_dims`: MLP hidden layers

### Training
- `num_epochs`: Max training epochs
- `learning_rate`: Initial learning rate
- `lr_scheduler`: "cosine", "step", "exponential", or "none"
- `early_stopping_patience`: Patience for stopping

### Loss
- `loss.type`: "weighted_ce" (recommended) or "ce"
- `loss.class_weights`: Auto-computed if null

## Evaluation Metrics

### Primary KPIs
- **Macro-F1**: Average F1 across all classes
- **BACKCHANNEL F1**: F1 for backchannel prediction
- **START_SPEAKING F1**: F1 for turn-start prediction
- **False Entry Rate**: Predicted entry but true WAIT
- **Missed Entry Rate**: True entry but predicted WAIT

### Secondary KPIs
- **WAIT F1**: F1 for wait prediction
- **Balanced Accuracy**: Macro-average of per-class recall
- **BC-as-Turn Error**: BACKCHANNEL predicted as START_SPEAKING
- **Turn-as-BC Error**: START_SPEAKING predicted as BACKCHANNEL
- **Floor Violation Rate**: Aggressive false positives
- **Aggressiveness Rate**: Fraction of entry predictions
- **ECE**: Expected Calibration Error

### Real-time Metrics
- **Inference Latency**: Milliseconds per sample
- **Real-time Factor**: Latency / 100ms window
- **Model Size**: MB

## Policy Integration

At inference (every 100ms or 200ms):

```python
p_wait, p_backchannel, p_start_speaking = model(frame, scalar)

if p_start_speaking > 0.70:
    action = START_SPEAKING
elif p_backchannel > 0.55 and human_active_at_t:
    action = BACKCHANNEL
else:
    action = WAIT
```

## Project Structure

```
turn_taking_timing_classifier/
├── configs/
│   └── config.yaml              # Main configuration
├── src/
│   ├── data/
│   │   ├── dataset.py           # Parquet loading, preprocessing
│   │   └── loaders.py           # PyTorch DataLoaders
│   ├── models/
│   │   ├── timing_net.py        # Full model architecture
│   │   └── branches.py          # TCN, GRU, MLP modules
│   ├── training/
│   │   └── trainer.py           # Training loop with checkpointing
│   ├── evaluation/
│   │   ├── metrics.py           # All KPI calculations
│   │   └── evaluator.py         # Evaluation pipeline
│   ├── baselines/
│   │   ├── va_threshold.py      # VA-Threshold baseline implementation
│   │   └── __init__.py
│   └── utils/
│       ├── config.py            # Config loading
│       └── logging.py           # Logging setup
├── scripts/
│   ├── train.py                 # Main training script
│   ├── test.py                  # Evaluation script
│   ├── resume_training.py       # Resume from checkpoint
│   └── eval_va_threshold_baseline.py  # VA-Threshold baseline evaluation
├── checkpoints/                 # Saved models
├── logs/                        # Training logs
└── README.md
```

## Voice Activity Threshold Baselines

Two simple voice activity baselines use your precomputed AMI features to establish baseline performance:

### Quick Run

```bash
python scripts/eval_va_threshold_baseline.py \
    --manifest data/processed/final_manifest.parquet \
    --output-dir reports/va_threshold_baseline
```

### VA-Silence
Predicts **START_SPEAKING** only when silence duration exceeds threshold.

```bash
python scripts/eval_va_threshold_baseline.py \
    --manifest data/processed/final_manifest.parquet \
    --output-dir reports/va_threshold_baseline
```

### VA-Threshold
Predicts based on multiple voice activity patterns:
- **START_SPEAKING**: When silence duration ≥ θ_start
- **BACKCHANNEL**: When single human active for ≥ θ_bc_min_speech seconds
- **WAIT**: Default

#### Inputs
- `human_active_at_t`: Is any human speaking at current time?
- `num_humans_active_at_t`: Number of active humans
- `overlap_active_at_t`: Multiple humans overlapping?
- `silence_duration_before_t`: Seconds of silence before current time
- `current_human_speech_duration`: Current continuous human speech duration

#### Threshold Tuning
The script automatically:
1. Tries all threshold combinations on **validation** split
2. Selects best thresholds based on Macro-F1
3. Evaluates on **test** split with selected thresholds

#### Output Files
```
reports/va_threshold_baseline/
├── va_baseline_predictions.parquet  # Sample predictions
├── va_baseline_metrics.json         # Detailed metrics
├── va_baseline_metrics.md           # Human-readable metrics
├── va_baseline_confusion_matrix_silence.csv
└── va_baseline_confusion_matrix_threshold.csv
```

## Ablation Studies

To run ablation studies, modify `configs/config.yaml`:

1. **Ablation 1: Previous Activity Only** (current)
   - Uses only `pseudo_robot_past_active` from frame features
   - Baseline comparison

2. **Ablation 2: Add Current Context**
   - Include current scalar features

3. **Ablation 3: Full Features**
   - Use all available features

```bash
# Run ablations with different configs
python scripts/train.py --config configs/ablation_1.yaml
python scripts/train.py --config configs/ablation_2.yaml
python scripts/train.py --config configs/ablation_3.yaml
```

## Checkpointing Details

Checkpoints save:
- Model weights
- Optimizer state
- Learning rate scheduler state
- Epoch number
- Best metric value
- Complete training/validation history
- Configuration

Resume automatically continues from last saved state with reproducible RNG.

## Troubleshooting

**Out of memory**: Reduce `batch_size` in config
**Slow data loading**: Increase `num_workers` in dataset config
**Poor metrics**: Check dataset path and label distribution
**NaN loss**: Try reducing `learning_rate` or using `loss.type: "ce"` without class weights

