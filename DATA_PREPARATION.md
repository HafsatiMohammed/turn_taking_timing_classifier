# Data Preparation Guide

## Overview

This guide explains how to prepare your dataset for training the turn-taking timing classifier. The process involves:

1. **Create metadata JSONL** with sample definitions (timing, labels, metadata)
2. **Extract features** from HuggingFace AMI dataset (VAD, speaker diarization)
3. **Build parquet files** (train/validation/test) with X_frame and X_scalar

## Step 1: Create Metadata JSONL

You need to create `data/processed/final_manifest.jsonl` where each line is a JSON sample.

### Sample Structure

```json
{
  "sample_id": "meeting123_A_12.34",
  "meeting_id": "meeting123",
  "pseudo_robot": "A",
  "human_speakers": "[\"B\", \"C\", \"D\"]",
  "time": 12.34,
  "context_start": 6.34,
  "context_end": 12.34,
  "prediction_start": 12.34,
  "prediction_end": 13.34,
  "prediction_horizon": 1.0,
  "candidate_source": "positive_before_pseudo_robot_onset",
  "pseudo_robot_onset": 12.35,
  "split": "train",
  
  "text_context": {
    "events": [
      {
        "speaker": "B",
        "start": 10.5,
        "end": 11.2,
        "text": "what do you think?",
        "num_tokens": 4
      }
    ],
    "pseudo_robot_state": "silent"
  },
  
  "state_at_prediction_time": {
    "pseudo_robot_is_speaking": false,
    "num_human_speakers_active": 1,
    "human_speakers_active": ["B"],
    "time_since_last_human": 1.14,
    "time_since_pseudo_robot_last_spoke": 5.2,
    "floor_is_open": false,
    "floor_holder": "B",
    "wait_subtype": "AFTER_HUMAN_SPOKE_RECENTLY"
  },
  
  "weak_label": "WAIT",
  "wait_subtype": "AFTER_HUMAN_SPOKE_RECENTLY",
  "entry_subtype": null,
  "rule_version": "v0.1.0",
  
  "llm_entry_type": "FULL_TURN",
  "llm_confidence": 0.92,
  "llm_reason": "Clear turn completion signal with 200ms pause",
  "llm_floor_state": "open",
  
  "final_label": "START_SPEAKING",
  "exclude_from_training": false,
  "training_weight": 0.92,
  "balance_group": "START_SPEAKING"
}
```

### Key Fields

**Identifiers & Timing:**
- `sample_id`: Unique sample identifier
- `meeting_id`: Meeting this sample comes from
- `pseudo_robot`: Which speaker acts as pseudo-robot (A, B, C, or D)
- `time`: Prediction time (seconds from meeting start)
- `context_start`, `context_end`: Window for frame features (6 seconds)
- `split`: "train", "validation", or "test"

**Metadata:**
- `text_context`: Conversation history with speech events
- `state_at_prediction_time`: Current state (if precomputed, used for scalar features)

**Labels:**
- `final_label`: "WAIT", "BACKCHANNEL", or "START_SPEAKING" (target)
- `training_weight`: Confidence score [0, 1]
- `exclude_from_training`: Skip if False

### Example: Creating Metadata

```python
import json
from pathlib import Path

def create_sample(meeting_id, pseudo_robot, time_t, context_duration=6.0):
    """Create a sample dict."""
    return {
        "sample_id": f"{meeting_id}_{pseudo_robot}_{time_t:.2f}",
        "meeting_id": meeting_id,
        "pseudo_robot": pseudo_robot,
        "time": time_t,
        "context_start": time_t - context_duration,
        "context_end": time_t,
        "split": "train",  # Or "validation", "test"
        "final_label": "WAIT",  # Or "BACKCHANNEL", "START_SPEAKING"
        "training_weight": 0.95,
        "exclude_from_training": False,
        "state_at_prediction_time": {},
    }

# Write samples to JSONL
output_path = Path("data/processed/final_manifest.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    samples = [
        create_sample("meeting123", "A", 10.0),
        create_sample("meeting123", "B", 15.5),
        # ... more samples
    ]
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

print(f"Wrote {len(samples)} samples to {output_path}")
```

## Step 2: Prepare Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

This installs:
- `datasets` - HuggingFace datasets library (for AMI)
- `pandas`, `pyarrow` - Data handling
- `torch`, `scikit-learn` - ML libraries
- All other dependencies

## Step 3: Extract Features

Run the feature extraction script:

```bash
python scripts/prepare_dataset.py \
    --metadata data/processed/final_manifest.jsonl \
    --output data/processed/hf_export
```

### What This Does

1. **Loads HuggingFace AMI dataset**
   - Automatically downloads/caches from `~/.cache/huggingface/`
   - Extracts speaker diarization and VAD data

2. **For each sample:**
   - **X_frame [120, 7]**: Extracts 6 seconds of context
     - 120 frames at 0.05s shift (20Hz)
     - 7 features: VAD, speaker activity, etc.
   - **X_scalar [6]**: Current state features
     - Silence duration, speech ratios, timing info

3. **Saves parquet files**
   - `data/processed/hf_export/train.parquet`
   - `data/processed/hf_export/validation.parquet`
   - `data/processed/hf_export/test.parquet`

### Output Structure

Each parquet file contains:

```
Columns:
├── sample_id          (str)
├── meeting_id         (str)
├── pseudo_robot       (str)
├── time               (float)
├── context_start      (float)
├── context_end        (float)
├── split              (str)
├── final_label        (str)           ← TARGET: WAIT/BACKCHANNEL/START_SPEAKING
├── training_weight    (float)
├── exclude_from_training (bool)
├── X_frame            (np.array [120, 7])  ← INPUT: Frame features
├── X_scalar           (np.array [6])        ← INPUT: Scalar features
├── weak_label         (str)
└── llm_confidence     (float)
```

## Frame Features [120, 7]

**Context:** Last 6 seconds at 0.05s shift (120 frames)

**Features:**
```
[0] any_human_active          - Boolean: Is any human speaking?
[1] num_humans_active_norm    - [0, 1]: Fraction of humans active
[2] overlap_active            - Boolean: Multiple humans overlapping?
[3] silence_active            - Boolean: Is it silent?
[4] human_onset               - Boolean: Human just started?
[5] human_offset              - Boolean: Human just stopped?
[6] pseudo_robot_past_active  - Boolean: Was robot speaking?
```

Computed from AMI speaker diarization.

## Scalar Features [6]

**Context:** At prediction time

**Features:**
```
[0] silence_duration_before_t          - Seconds of silence before now
[1] current_human_speech_duration      - How long humans have been speaking
[2] human_speech_ratio_last_1s         - % of last 1s with human speech
[3] human_speech_ratio_last_6s         - % of last 6s with human speech
[4] overlap_ratio_last_6s              - % of last 6s with overlapping speech
[5] time_since_pseudo_robot_last_spoke - Seconds since robot's last utterance
```

Computed from:
- AMI diarization (for features 0-4)
- Metadata `state_at_prediction_time` (if provided)

## Complete Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create your metadata JSONL
# (See example above or use your own data curation process)
# Output: data/processed/final_manifest.jsonl

# 3. Extract features from AMI dataset
python scripts/prepare_dataset.py

# 4. Verify output
ls -lh data/processed/hf_export/

# 5. Check data with training script
python -c "
from src.data import get_dataloaders
loaders = get_dataloaders('data/processed/hf_export', batch_size=4)
batch = next(iter(loaders['train']))
print('Frame shape:', batch['frame'].shape)
print('Scalar shape:', batch['scalar'].shape)
print('Labels:', batch['label'])
"

# 6. Start training!
python scripts/train.py --config configs/config.yaml
```

## Troubleshooting

### Q: "ModuleNotFoundError: No module named 'datasets'"
**A:** Install: `pip install datasets`

### Q: "FileNotFoundError: data/processed/final_manifest.jsonl"
**A:** Create metadata JSONL first. See "Step 1" above.

### Q: "ValueError: No AMI data found for meeting..."
**A:** Meeting ID in metadata doesn't match AMI dataset. Check meeting IDs match AMI.

### Q: Feature extraction is very slow
**A:** Normal on first run - HuggingFace downloads/caches AMI dataset. Subsequent runs are faster.

### Q: X_frame has NaN values
**A:** Some frame features failed to compute. Check speaker names are A, B, C, D.

## Data Format Details

### How X_frame is Built

For each sample with context_start=6.34, context_end=12.34:

```python
frame_times = np.arange(6.34, 12.34, 0.05)  # [6.34, 6.39, 6.44, ..., 12.29]
# → 120 frames

For each frame_time:
  active_speakers = who_is_speaking_at(frame_time)  # From AMI diarization
  x_frame[i, 0] = any_human_active(active_speakers)
  x_frame[i, 1] = num_humans_active(active_speakers) / 3
  x_frame[i, 2] = overlap_active(active_speakers)
  x_frame[i, 3] = silence_active(active_speakers)
  x_frame[i, 4] = human_onset(active_speakers, prev_speakers)
  x_frame[i, 5] = human_offset(active_speakers, prev_speakers)
  x_frame[i, 6] = pseudo_robot_active(active_speakers)
```

### How X_scalar is Built

```python
# From state_at_prediction_time (if provided), else computed:
x_scalar[0] = time_since_last_human(active_speakers, time_t)
x_scalar[1] = current_human_speech_duration(active_speakers, time_t)
x_scalar[2] = speech_ratio(active_speakers, time_t - 1.0, time_t)
x_scalar[3] = speech_ratio(active_speakers, time_t - 6.0, time_t)
x_scalar[4] = overlap_ratio(active_speakers, time_t - 6.0, time_t)
x_scalar[5] = time_since_pseudo_robot_spoke(robot, time_t)
```

## AMI Dataset Details

The script loads from HuggingFace:
- **Dataset:** `edinburghcstr/ami` (Edinburgh AMI corpus)
- **Config:** `ihm` (Individual Headset Microphone)
- **Cached at:** `~/.cache/huggingface/datasets/`

**AMI provides:**
- Speaker diarization (who spoke when)
- Meeting IDs (ES2002, IB4001, etc.)
- Multi-party conversation data (4 speakers per meeting)

## Next Steps

1. ✓ Create metadata JSONL
2. ✓ Run feature extraction
3. → Train model: `python scripts/train.py --config configs/config.yaml`
4. → Evaluate: `python scripts/test.py --config configs/config.yaml --checkpoint checkpoints/checkpoint_best.pt`

See [QUICKSTART.md](QUICKSTART.md) for training instructions.
