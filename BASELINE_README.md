# VA-Threshold Baseline Evaluation Guide

## Overview

Two simple voice activity (VA) threshold baselines are provided to establish baseline performance on your turn-taking dataset.

### Baseline 1: VA-Silence
**Input:** `silence_duration_before_t`  
**Rule:**
```
if silence_duration_before_t >= theta_start:
    predict START_SPEAKING
else:
    predict WAIT
```
**Limitation:** Never predicts BACKCHANNEL

### Baseline 2: VA-Threshold  
**Inputs:**
- `human_active_at_t`: Is any human speaking at current time?
- `num_humans_active_at_t`: Number of currently active humans
- `overlap_active_at_t`: Are multiple humans overlapping?
- `silence_duration_before_t`: Duration of silence before current time (seconds)
- `current_human_speech_duration`: Current continuous human speech duration (seconds)

**Rule:**
```python
if silence_duration_before_t >= theta_start:
    return START_SPEAKING
elif (human_active_at_t and 
      num_humans_active_at_t == 1 and 
      not overlap_active_at_t and 
      current_human_speech_duration >= theta_bc_min_speech):
    return BACKCHANNEL
else:
    return WAIT
```

## Dataset Requirements

Your manifest should be a parquet or CSV file with at least these columns:

```
Required:
- sample_id           (str)
- split               (str: "train", "validation", "test")
- final_label         (str: "WAIT", "BACKCHANNEL", "START_SPEAKING")

Feature columns (Mode A: precomputed):
- human_active_at_t                (bool)
- num_humans_active_at_t           (int)
- overlap_active_at_t              (bool)
- silence_duration_before_t        (float) - seconds
- current_human_speech_duration    (float) - seconds
```

If these features are not precomputed, see "Feature Extraction" below.

## Running the Baseline

### Mode A: Precomputed Features

If your manifest already has the voice activity features:

```bash
python scripts/eval_va_threshold_baseline.py \
    --manifest data/processed/final_manifest.parquet \
    --output-dir reports/va_threshold_baseline \
    --use-precomputed-if-available true
```

### Mode B: Auto-Fill Missing Features

If features are missing, the script will fill them with defaults:

```bash
python scripts/eval_va_threshold_baseline.py \
    --manifest data/processed/final_manifest.parquet \
    --output-dir reports/va_threshold_baseline \
    --use-precomputed-if-available true
```

The script logs warnings when features are missing and uses:
- `human_active_at_t = False`
- `num_humans_active_at_t = 0`
- `overlap_active_at_t = False`
- `silence_duration_before_t = 0.0`
- `current_human_speech_duration = 0.0`

### Dry-Run Mode (Test First)

Test on a small sample before running full evaluation:

```bash
python scripts/eval_va_threshold_baseline.py \
    --manifest data/processed/final_manifest.parquet \
    --output-dir reports/va_threshold_baseline \
    --max-samples 100
```

This processes only the first 100 samples.

## Threshold Tuning

The script automatically:

1. **On Validation Split:**
   - Tests all combinations of:
     - `theta_start` ∈ {0.3, 0.5, 0.6, 0.7, 1.0}
     - `theta_bc_min_speech` ∈ {0.5, 1.0, 1.5, 2.0}
   - Selects best thresholds based on **Macro-F1**

2. **On Test Split:**
   - Uses best thresholds found on validation
   - Computes final metrics

3. **Output:**
   - Selected thresholds logged and saved
   - Full metrics reported for test set

## Output Files

### Predictions
**File:** `va_baseline_predictions.parquet`

Columns:
```
- sample_id                        (str)
- split                            (str)
- final_label                      (str) - ground truth
- pred_va_silence                  (str) - VA-Silence prediction
- pred_va_threshold                (str) - VA-Threshold prediction
- theta_start                      (float) - selected threshold
- theta_bc_min_speech              (float) - selected threshold
- human_active_at_t                (bool)
- num_humans_active_at_t           (int)
- overlap_active_at_t              (bool)
- silence_duration_before_t        (float)
- current_human_speech_duration    (float)
```

### Metrics (JSON)
**File:** `va_baseline_metrics.json`

```json
{
  "va_silence": {
    "accuracy": 0.65,
    "macro_f1": 0.62,
    "weighted_f1": 0.64,
    "WAIT_precision": 0.68,
    "WAIT_recall": 0.71,
    "WAIT_f1": 0.69,
    "BACKCHANNEL_precision": 0.0,
    "BACKCHANNEL_recall": 0.0,
    "BACKCHANNEL_f1": 0.0,
    "START_SPEAKING_precision": 0.60,
    "START_SPEAKING_recall": 0.55,
    "START_SPEAKING_f1": 0.57,
    "confusion_matrix": [[12, 0, 5], [0, 0, 2], [3, 0, 3]],
    "false_entry_rate": 1.0,
    "missed_entry_rate": 0.45,
    "backchannel_as_turn_error": null,
    "turn_as_backchannel_error": null
  },
  "va_threshold": {
    "accuracy": 0.68,
    "macro_f1": 0.65,
    "weighted_f1": 0.68,
    "WAIT_precision": 0.70,
    "WAIT_recall": 0.75,
    "WAIT_f1": 0.72,
    "BACKCHANNEL_precision": 0.33,
    "BACKCHANNEL_recall": 0.40,
    "BACKCHANNEL_f1": 0.36,
    "START_SPEAKING_precision": 0.64,
    "START_SPEAKING_recall": 0.60,
    "START_SPEAKING_f1": 0.62,
    "confusion_matrix": [[13, 1, 3], [0, 2, 0], [2, 0, 4]],
    "false_entry_rate": 0.75,
    "missed_entry_rate": 0.30,
    "backchannel_as_turn_error": 0.0,
    "turn_as_backchannel_error": 0.33
  },
  "best_thresholds": {
    "theta_start_silence": 0.6,
    "theta_start_threshold": 0.5,
    "theta_bc_min_speech": 1.0
  },
  "num_samples": 25
}
```

### Metrics (Markdown)
**File:** `va_baseline_metrics.md`

Human-readable format for reports.

### Confusion Matrices
**Files:**
- `va_baseline_confusion_matrix_silence.csv`
- `va_baseline_confusion_matrix_threshold.csv`

Format: 3×3 CSV (rows=true labels, cols=predictions)

```
12,0,5
0,0,2
3,0,3
```

## Example Workflow

```bash
# 1. Create manifest with your samples
# Ensure it has all required columns
python create_manifest.py

# 2. Test with small sample
python scripts/eval_va_threshold_baseline.py \
    --manifest data/processed/final_manifest.parquet \
    --output-dir reports/va_threshold_baseline \
    --max-samples 100

# 3. Review output
cat reports/va_threshold_baseline/va_baseline_metrics.md

# 4. Run full evaluation
python scripts/eval_va_threshold_baseline.py \
    --manifest data/processed/final_manifest.parquet \
    --output-dir reports/va_threshold_baseline

# 5. View results
python -c "
import json
with open('reports/va_threshold_baseline/va_baseline_metrics.json') as f:
    metrics = json.load(f)
    print('VA-Silence Macro-F1:', metrics['va_silence']['macro_f1'])
    print('VA-Threshold Macro-F1:', metrics['va_threshold']['macro_f1'])
"
```

## Features Already Precomputed

Your curated AMI dataset already includes the voice activity features as columns:
- `human_active_at_t` (bool)
- `num_humans_active_at_t` (int)
- `overlap_active_at_t` (bool)
- `silence_duration_before_t` (float)
- `current_human_speech_duration` (float)

The baseline script reads these directly - no feature extraction needed!

## Metrics Explained

### Primary
- **Accuracy:** Fraction of correct predictions
- **Macro-F1:** Unweighted average F1 across classes
- **Weighted F1:** Weighted by class frequency

### Per-Class
- **Precision:** TP / (TP + FP) - when model predicts this class, how often is it correct?
- **Recall:** TP / (TP + FN) - of all true instances of this class, how many did model find?
- **F1:** Harmonic mean of precision and recall

### Turn-Taking Specific
- **False Entry Rate:** Predictions of entry (BACKCHANNEL or START) when true is WAIT
  - Lower is better
- **Missed Entry Rate:** Predicted WAIT when true is entry
  - Lower is better
- **BC-as-Turn Error:** Predicted START_SPEAKING when true is BACKCHANNEL
- **Turn-as-BC Error:** Predicted BACKCHANNEL when true is START_SPEAKING

## Troubleshooting

### "Missing required columns"
Ensure your manifest has: `sample_id`, `split`, `final_label`

### "No validation split found"
Add a `split` column with values "validation" and "test"

### "No test split found"
The script will use all data for evaluation. Add `split == "test"` rows for proper test evaluation.

### Features all zeros/defaults
Check that your feature columns have the correct names and types.

### Poor baseline performance
- Check label distribution (very imbalanced datasets are harder)
- Verify ground truth labels are correct
- Visualize the feature distributions

## Next Steps

After establishing baseline:
1. Compare neural network models against these baselines
2. Try different feature combinations
3. Tune thresholds for your specific use case
4. Use as floor for model performance

## References

- Silero VAD: https://github.com/snakers4/silero-vad
- Voice Activity Detection: [Your citation]
- Turn-taking literature: [Your citations]
