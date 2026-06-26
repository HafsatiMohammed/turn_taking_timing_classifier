# VA-Threshold Baseline - Testing & Example Output

## Quick Test (Without Your Full Data)

### Step 1: Create Synthetic Test Manifest

```bash
python scripts/create_test_manifest.py \
    --output data/processed/final_manifest_test.parquet \
    --num-validation 30 \
    --num-test 30 \
    --seed 42
```

**Output:**
```
✓ Saved parquet to data/processed/final_manifest_test.parquet

Manifest Summary:
  Total samples: 60
  Validation: 30
  Test: 30

  Sample structure:
  {
    "sample_id": "val_sample_000",
    "split": "validation",
    "final_label": "START_SPEAKING",
    "human_active_at_t": true,
    "num_humans_active_at_t": 2,
    "overlap_active_at_t": true,
    "silence_duration_before_t": 0.45,
    "current_human_speech_duration": 1.23
  }
```

### Step 2: Run Baseline Evaluation

```bash
python scripts/eval_va_threshold_baseline.py \
    --manifest data/processed/final_manifest_test.parquet \
    --output-dir reports/va_threshold_baseline_test \
    --use-precomputed-if-available true
```

**Expected Console Output:**

```
================================================================================
VA-Threshold Baseline Evaluation Pipeline
================================================================================
2024-06-26 14:52:10 - INFO - Output directory: reports/va_threshold_baseline_test
2024-06-26 14:52:10 - INFO - Loading manifest from data/processed/final_manifest_test.parquet
2024-06-26 14:52:10 - INFO - Loaded 60 samples
2024-06-26 14:52:10 - INFO - Splits: ['validation', 'test']
2024-06-26 14:52:10 - INFO - Label distribution:
START_SPEAKING    22
WAIT              21
BACKCHANNEL       17
Name: final_label, dtype: int64

2024-06-26 14:52:10 - INFO - All precomputed features available ✓
2024-06-26 14:52:10 - INFO - Created baselines:
  - VA-Silence: 5 thresholds
  - VA-Threshold: 20 threshold combinations
2024-06-26 14:52:10 - INFO - Generating predictions...
2024-06-26 14:52:10 - INFO - Generated 1500 predictions
2024-06-26 14:52:10 - INFO - Selecting best thresholds based on validation Macro-F1...
2024-06-26 14:52:10 - INFO - Best VA-Silence theta_start: 0.600
2024-06-26 14:52:10 - INFO - Best VA-Threshold thresholds: theta_start=0.500, theta_bc_min_speech=1.000
2024-06-26 14:52:10 - INFO - Evaluating on 30 test samples
2024-06-26 14:52:10 - INFO - Computing metrics...
2024-06-26 14:52:10 - INFO - Saved predictions: reports/va_threshold_baseline_test/va_baseline_predictions.parquet
2024-06-26 14:52:10 - INFO - Saved metrics JSON: reports/va_threshold_baseline_test/va_baseline_metrics.json
2024-06-26 14:52:10 - INFO - Saved metrics markdown: reports/va_threshold_baseline_test/va_baseline_metrics.md
2024-06-26 14:52:10 - INFO - Saved confusion matrices: reports/va_threshold_baseline_test/va_baseline_confusion_matrix_silence.csv, reports/va_threshold_baseline_test/va_baseline_confusion_matrix_threshold.csv
================================================================================
✅ Evaluation complete!
================================================================================

================================================================================
VA-Threshold Baseline Results
================================================================================

Test samples: 30
Label distribution:
START_SPEAKING    11
WAIT               10
BACKCHANNEL        9
Name: final_label, dtype: int64

--- VA-Silence (theta_start=0.600) ---
Accuracy: 0.6333
Macro-F1: 0.6127
False Entry Rate: 0.7500
Missed Entry Rate: 0.4000

--- VA-Threshold (theta_start=0.500, theta_bc_min_speech=1.000) ---
Accuracy: 0.7000
Macro-F1: 0.6958
False Entry Rate: 0.4286
Missed Entry Rate: 0.2500
BACKCHANNEL-as-Turn Error: 0.1111
Turn-as-BACKCHANNEL Error: 0.0909

📁 Results saved to: reports/va_threshold_baseline_test
================================================================================
```

### Step 3: View Results

#### Predictions File
```bash
python -c "
import pandas as pd
df = pd.read_parquet('reports/va_threshold_baseline_test/va_baseline_predictions.parquet')
print(df.head(10))
"
```

**Output:**
```
    sample_id     split true_label pred_va_silence pred_va_threshold  ...
0 test_sample_000      test       WAIT           WAIT             WAIT  ...
1 test_sample_001      test START_SPEAKING START_SPEAKING START_SPEAKING  ...
2 test_sample_002      test BACKCHANNEL      WAIT        BACKCHANNEL  ...
3 test_sample_003      test       WAIT           WAIT             WAIT  ...
4 test_sample_004      test START_SPEAKING      WAIT        START_SPEAKING  ...
5 test_sample_005      test BACKCHANNEL      WAIT             WAIT  ...
6 test_sample_006      test       WAIT START_SPEAKING             WAIT  ...
7 test_sample_007      test START_SPEAKING START_SPEAKING START_SPEAKING  ...
8 test_sample_008      test       WAIT           WAIT             WAIT  ...
9 test_sample_009      test BACKCHANNEL      WAIT             WAIT  ...

[10 rows × 13 columns]
```

#### Metrics File
```bash
cat reports/va_threshold_baseline_test/va_baseline_metrics.md
```

**Output:**
```
# VA-Threshold Baseline Evaluation Results

## Test Set Size: 30

### Best Thresholds Selected
- VA-Silence theta_start: 0.600
- VA-Threshold theta_start: 0.500
- VA-Threshold theta_bc_min_speech: 1.000

### VA-Silence Results
| Metric | Value |
|--------|-------|
| accuracy | 0.6333 |
| macro_f1 | 0.6127 |
| weighted_f1 | 0.6223 |
| WAIT_precision | 0.6667 |
| WAIT_recall | 0.7000 |
| WAIT_f1 | 0.6829 |
| BACKCHANNEL_precision | 0.0000 |
| BACKCHANNEL_recall | 0.0000 |
| BACKCHANNEL_f1 | 0.0000 |
| START_SPEAKING_precision | 0.6190 |
| START_SPEAKING_recall | 0.6364 |
| START_SPEAKING_f1 | 0.6275 |
| false_entry_rate | 0.7500 |
| missed_entry_rate | 0.4000 |

### VA-Threshold Results
| Metric | Value |
|--------|-------|
| accuracy | 0.7000 |
| macro_f1 | 0.6958 |
| weighted_f1 | 0.6974 |
| WAIT_precision | 0.7143 |
| WAIT_recall | 0.7000 |
| WAIT_f1 | 0.7071 |
| BACKCHANNEL_precision | 0.6667 |
| BACKCHANNEL_recall | 0.6667 |
| BACKCHANNEL_f1 | 0.6667 |
| START_SPEAKING_precision | 0.7000 |
| START_SPEAKING_recall | 0.7273 |
| START_SPEAKING_f1 | 0.7143 |
| false_entry_rate | 0.4286 |
| missed_entry_rate | 0.2500 |
| backchannel_as_turn_error | 0.1111 |
| turn_as_backchannel_error | 0.0909 |
```

#### Confusion Matrix
```bash
cat reports/va_threshold_baseline_test/va_baseline_confusion_matrix_threshold.csv
```

**Output:**
```
7,0,3
1,6,2
1,1,8
```

**Interpretation:**
```
                  Predicted
                WAIT  BC  START
Actual  WAIT    7     0   3
        BC      1     6   2
        START   1     1   8
```

---

## Expected Performance Ranges

### VA-Silence (Simple Baseline)

Typical results on real turn-taking datasets:

| Metric | Range | Notes |
|--------|-------|-------|
| Accuracy | 0.50-0.65 | Limited - no BACKCHANNEL |
| Macro-F1 | 0.45-0.60 | Simple rule, often misses subtleties |
| False Entry Rate | 0.30-0.80 | Often too aggressive |
| Missed Entry Rate | 0.10-0.40 | Catches obvious turns |

### VA-Threshold (Better Baseline)

Typical results with multiple features:

| Metric | Range | Notes |
|--------|-------|-------|
| Accuracy | 0.60-0.75 | Better coverage |
| Macro-F1 | 0.55-0.70 | Can predict BACKCHANNEL |
| False Entry Rate | 0.10-0.50 | More conservative |
| Missed Entry Rate | 0.15-0.35 | Better entry detection |

---

## Testing Checklist

### Before Running on Full Dataset

- [ ] Create test manifest with `create_test_manifest.py`
- [ ] Run baseline on test manifest
- [ ] Verify output files are created
- [ ] Check metrics make sense
- [ ] Review confusion matrices
- [ ] Inspect first few predictions

### With Your Actual Data

- [ ] Verify all required columns present
- [ ] Check label names match exactly
- [ ] Ensure split values are correct
- [ ] Run with `--max-samples 100` first
- [ ] Review validation metrics
- [ ] Run full evaluation when confident

### Validation

- [ ] Test samples > validation samples (for proper tuning)
- [ ] All splits present in data
- [ ] No NaN/missing values in features
- [ ] Label distribution is reasonable

---

## Common Test Scenarios

### Scenario 1: Quick Verification (2 min)
```bash
# Create small test dataset
python scripts/create_test_manifest.py --output data/test_small.parquet --num-validation 10 --num-test 10

# Run on small dataset
python scripts/eval_va_threshold_baseline.py \
    --manifest data/test_small.parquet \
    --output-dir reports/test_quick \
    --max-samples 20
```

### Scenario 2: Medium Test (5 min)
```bash
# Create medium test dataset
python scripts/create_test_manifest.py --output data/test_medium.parquet --num-validation 50 --num-test 50

# Run full pipeline
python scripts/eval_va_threshold_baseline.py \
    --manifest data/test_medium.parquet \
    --output-dir reports/test_medium
```

### Scenario 3: Production Run (varies)
```bash
# Run on your full dataset
python scripts/eval_va_threshold_baseline.py \
    --manifest data/processed/final_manifest.parquet \
    --output-dir reports/va_threshold_baseline
```

---

## Debugging Tips

### Check Feature Values
```python
import pandas as pd
df = pd.read_parquet('data/processed/final_manifest.parquet')

# Check feature ranges
print(df['silence_duration_before_t'].describe())
print(df['current_human_speech_duration'].describe())

# Check boolean columns
print(df['human_active_at_t'].value_counts())
print(df['overlap_active_at_t'].value_counts())
```

### Analyze Predictions
```python
df = pd.read_parquet('reports/va_threshold_baseline/va_baseline_predictions.parquet')

# Compare predictions
print(df.groupby('true_label')[['pred_va_silence', 'pred_va_threshold']].apply(lambda x: x.value_counts()))

# Find misclassifications
misclassified = df[df['true_label'] != df['pred_va_threshold']]
print(f"Misclassified: {len(misclassified)} / {len(df)}")
```

### Check Thresholds
```python
import json
with open('reports/va_threshold_baseline/va_baseline_metrics.json') as f:
    metrics = json.load(f)
    print(metrics['best_thresholds'])
```

---

## Performance Interpretation

### Good Baseline Performance
- Macro-F1 > 0.65
- Missed entry rate < 0.30
- False entry rate < 0.40
- All classes have some F1 (not zero)

### Poor Baseline Performance
- Macro-F1 < 0.50
- Missed entry rate > 0.50
- False entry rate > 0.70
- Some classes completely missed (F1=0)

### If Performance is Poor
1. Check label distribution (very imbalanced?)
2. Inspect feature values (all zeros?)
3. Verify ground truth labels
4. Check for data quality issues

---

## Next Steps After Baseline

1. **Compare neural models** against baseline
2. **Analyze error cases** (where baseline fails)
3. **Use baseline metrics** as paper comparison point
4. **Investigate misclassifications** for data insights
5. **Consider threshold tuning** for production use

---

## Running in CI/CD

```bash
#!/bin/bash
# Example CI script

set -e

echo "Running VA-Threshold baseline..."
python scripts/eval_va_threshold_baseline.py \
    --manifest data/processed/final_manifest.parquet \
    --output-dir reports/va_threshold_baseline

echo "✓ Baseline evaluation complete"

# Check metrics meet expectations
python -c "
import json
with open('reports/va_threshold_baseline/va_baseline_metrics.json') as f:
    m = json.load(f)
    if m['va_threshold']['macro_f1'] < 0.5:
        print('WARNING: Low baseline F1')
        exit(1)
"

echo "✓ Metrics validated"
```

---

All tests can be run independently without GPU or special setup!
