# Ablation Study Guide

## Overview

This document explains how to set up and run ablation studies for the turn-taking timing classifier. Ablation studies systematically remove or modify components to understand their contribution to the model's performance.

## Study Design

We investigate 6 different feature combinations:

| # | Name | Frame Features | Scalar Features | Purpose |
|---|------|----------------|-----------------|---------|
| 1 | Previous Activity Only | Feature 6 only | None | **Baseline**: What can be predicted from just robot's own past activity? |
| 2 | Activity + Context | Feature 6 | All | Does current context help beyond historical activity? |
| 3 | Frame History | All 1-7 | None | Is temporal context sufficient without current state? |
| 4 | Full Model | All 1-7 | All | Full model with all available features |
| 5 | Current State Only | None | All | Can decision be made from just instant state? |
| 6 | Human History | Features 1-5 | None | Impact of human activity awareness vs. robot state? |

## Features Explained

### Frame Features (120 timesteps, 6 seconds of history)
```
[0] any_human_active         - Is any human speaking?
[1] num_humans_active_norm   - How many humans active (normalized)
[2] overlap_active           - Are multiple humans overlapping?
[3] silence_active           - Is there silence (no one speaking)?
[4] human_onset              - Did a human just start speaking?
[5] human_offset             - Did a human just stop speaking?
[6] pseudo_robot_past_active - Was robot speaking in this frame?
```

### Scalar Features (single value at prediction time)
```
[0] silence_duration_before_t         - Seconds of silence before now
[1] current_human_speech_duration     - How long humans have been speaking
[2] human_speech_ratio_last_1s        - % of last 1s with human speech
[3] human_speech_ratio_last_6s        - % of last 6s with human speech
[4] overlap_ratio_last_6s             - % of last 6s with overlapping speech
[5] time_since_pseudo_robot_last_spoke - Seconds since robot's last utterance
```

## Setup

### Step 1: Create Config Files

For each ablation, create a dedicated config file:

```bash
# Create ablation configs by copying main config
for i in 1 2 3 4 5 6; do
  cp configs/config.yaml configs/ablation_${i}.yaml
done
```

### Step 2: Modify Each Config

Edit each `configs/ablation_N.yaml` with different model architectures:

**ablation_1.yaml** (Previous Activity Only):
```yaml
model:
  frame_branch:
    in_channels: 1              # Only 1 feature (robot past activity)
    out_channels: [16, 32, 64]
    output_dim: 64
  scalar_branch:
    hidden_dims: []             # No scalar branch
    output_dim: 0
  fusion:
    hidden_dims: [128, 64]
    output_dim: 3
```

**ablation_2.yaml** (Activity + Scalar Context):
```yaml
model:
  frame_branch:
    in_channels: 1
    out_channels: [16, 32, 64]
    output_dim: 64
  scalar_branch:
    in_dim: 6
    hidden_dims: [32, 32]
    output_dim: 32
  fusion:
    hidden_dims: [128, 64]
    output_dim: 3
```

**ablation_3.yaml** (All Frame Features Only):
```yaml
model:
  frame_branch:
    in_channels: 7              # All frame features
    out_channels: [32, 64, 128]
    output_dim: 128
  scalar_branch:
    hidden_dims: []             # No scalar branch
    output_dim: 0
  fusion:
    hidden_dims: [256, 128]
    output_dim: 3
```

**ablation_4.yaml** (Full Model - copy of main config.yaml):
```yaml
# Keep as is - this is the baseline
```

**ablation_5.yaml** (Current State Only):
```yaml
model:
  frame_branch:
    in_channels: 0              # No frame features
    out_channels: []
    output_dim: 0
  scalar_branch:
    in_dim: 6
    hidden_dims: [64, 64]
    output_dim: 64
  fusion:
    hidden_dims: [128, 64]
    output_dim: 3
```

**ablation_6.yaml** (Human History Only):
```yaml
model:
  frame_branch:
    in_channels: 6              # Features 0-5 (human-related)
    out_channels: [32, 64, 128]
    output_dim: 128
  scalar_branch:
    hidden_dims: []             # No scalar branch
    output_dim: 0
  fusion:
    hidden_dims: [256, 128]
    output_dim: 3
```

### Step 3: Adjust Feature Masking (Optional)

Optionally, modify `src/data/dataset.py` to mask features:

```python
# In TimingDataset.__getitem__()
def _mask_features(self, frame, enabled_features):
    """Zero out disabled features."""
    if enabled_features is not None:
        mask = np.zeros(self.frame_dim, dtype=bool)
        for idx in enabled_features:
            mask[idx] = True
        frame[:, ~mask] = 0
    return frame
```

## Running Ablations

### Option 1: Individual Training (Simple)

Train each ablation separately:

```bash
# Train ablation 1
python scripts/train.py --config configs/ablation_1.yaml

# Train ablation 2
python scripts/train.py --config configs/ablation_2.yaml

# ... and so on
```

### Option 2: Batch Training (Recommended)

Use the provided script to run all ablations:

```bash
# Create a script to run all
cat > run_all_ablations.sh << 'EOF'
#!/bin/bash
for i in 1 2 3 4 5 6; do
  echo "Running Ablation $i..."
  python scripts/train.py --config configs/ablation_${i}.yaml
  
  # Wait for training to complete, then evaluate
  BEST_CKPT=$(ls -t checkpoints/checkpoint_best.pt | head -1)
  python scripts/test.py \
    --config configs/ablation_${i}.yaml \
    --checkpoint $BEST_CKPT \
    --output results_ablation_${i}.json
  
  echo "Ablation $i complete!"
  echo ""
done
EOF
chmod +x run_all_ablations.sh
./run_all_ablations.sh
```

### Option 3: Using the Experiment Script

```bash
# Run single ablation
./scripts/run_experiment.sh ablation_1

# Run all
for i in 1 2 3 4 5 6; do
  ./scripts/run_experiment.sh ablation_${i}
done
```

## Analysis

### Step 1: Collect Results

After all ablations complete, you should have:
```
results_ablation_1.json
results_ablation_2.json
results_ablation_3.json
results_ablation_4.json
results_ablation_5.json
results_ablation_6.json
```

### Step 2: Generate Comparison Report

```bash
python scripts/analyze_ablations.py \
  --results results_ablation_*.json \
  --output ablation_comparison.json
```

This generates a comparison table showing:
- Macro-F1 for each ablation
- Per-class F1 (BACKCHANNEL, START_SPEAKING)
- Error rates (false entry, missed entry)
- Real-time metrics (latency, RTF)
- Improvements vs. baseline

### Step 3: Visualize Results

```python
import json
import matplotlib.pyplot as plt
import pandas as pd

# Load comparison
with open('ablation_comparison.json') as f:
    comparison = json.load(f)

results = comparison['results']
ablations = list(results.keys())
f1_scores = [results[a]['macro_f1'] for a in ablations]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(ablations, f1_scores)
plt.ylabel('Macro-F1')
plt.title('Ablation Study: Macro-F1 Scores')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('ablation_comparison.png')
```

## Expected Results

Typical pattern (your results may vary):

```
Ablation 1 (Prev Activity Only):         0.45 Macro-F1  [Baseline]
Ablation 2 (+ Scalar Context):           0.58 Macro-F1  [+13 points]
Ablation 3 (All Frame):                  0.63 Macro-F1  [+18 points]
Ablation 4 (Full Model):                 0.72 Macro-F1  [+27 points] ← Best
Ablation 5 (Current State Only):         0.42 Macro-F1  [-3 points]
Ablation 6 (Human History Only):         0.60 Macro-F1  [+15 points]
```

### Interpretation

1. **Ablation 1 vs 2**: Scalar context adds ~13 points
   - → Immediate context helps

2. **Ablation 1 vs 3**: Temporal variation matters
   - → Seeing how conversation evolves is important

3. **Ablation 4 is best**: Full information is helpful
   - → No redundancy between frame and scalar features

4. **Ablation 5 is poor**: Current instant state insufficient
   - → Temporal history is crucial

5. **Ablation 6 matters**: Human activity is important
   - → But robot's own state (feature 6) also critical

## Statistical Significance

For paper submission, compute confidence intervals:

```python
# Bootstrap confidence intervals
from sklearn.metrics import f1_score
import numpy as np

def bootstrap_f1(y_true, y_pred, n_bootstrap=1000):
    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        score = f1_score(y_true[idx], y_pred[idx], average='macro')
        scores.append(score)
    
    return np.mean(scores), np.std(scores), np.percentile(scores, [2.5, 97.5])
```

## Hyperparameter Considerations

When comparing ablations, keep constant:
- Learning rate
- Batch size
- Number of epochs
- Random seed
- Data preprocessing

Only vary:
- Input features (frame/scalar selection)
- Model architecture (layer sizes to accommodate)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not training (all ablations) | Check dataset path in config |
| Ablation 5 very poor | Normal - instant state is insufficient |
| Similar results across ablations | Check feature masking is working |
| Memory issues | Reduce batch_size or model layer sizes |
| Training too slow | Reduce num_epochs, use learning rate warmup |

## Paper Write-up

### Suggested Results Section

```
We conducted ablation studies to understand the contribution of each input 
modality. Table 3 shows results across six configurations:

- Ablation 1 (baseline): Only robot's past activity → 45% Macro-F1
- Ablation 2: + scalar context → 58% Macro-F1 (+13pp)
- Ablation 3: + human activity history → 63% Macro-F1 (+18pp)
- Ablation 4: Full model → 72% Macro-F1 (+27pp)
- Ablation 5: Scalar only (no history) → 42% Macro-F1 (-3pp)
- Ablation 6: Human history only → 60% Macro-F1 (+15pp)

These results demonstrate that both temporal context and current state are 
important. The temporal dimension (6 seconds of history) contributes more 
than instant state alone, suggesting that understanding conversation 
dynamics is crucial for timing decisions.
```

## References

- See [configs/ablation_studies.yaml](configs/ablation_studies.yaml) for complete feature definitions
- See [QUICKSTART.md](QUICKSTART.md) for training instructions
- See [README.md](README.md) for model architecture details
