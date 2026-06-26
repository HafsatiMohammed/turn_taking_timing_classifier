# VA-Threshold Baseline - Quick Guide

## Your Setup ✅

You have:
- ✅ Curated AMI dataset with samples
- ✅ Precomputed voice activity features in manifest
- ✅ Train/validation/test splits
- ✅ Ground truth labels

**No feature extraction needed!**

---

## One Command to Run Baselines

```bash
python scripts/eval_va_threshold_baseline.py \
    --manifest data/processed/final_manifest.parquet \
    --output-dir reports/va_threshold_baseline
```

That's it! ✅

---

## What You Need in Your Manifest

**Required columns:**
```
- sample_id (str)
- split (str: "validation", "test")
- final_label (str: "WAIT", "BACKCHANNEL", "START_SPEAKING")
```

**Precomputed voice activity features:**
```
- human_active_at_t (bool)
- num_humans_active_at_t (int)
- overlap_active_at_t (bool)
- silence_duration_before_t (float)
- current_human_speech_duration (float)
```

**That's all!** The script reads these directly.

---

## What You Get

```
reports/va_threshold_baseline/
├── va_baseline_predictions.parquet  ← All predictions + features
├── va_baseline_metrics.json         ← All metrics (JSON)
├── va_baseline_metrics.md           ← Human-readable metrics
├── va_baseline_confusion_matrix_silence.csv
└── va_baseline_confusion_matrix_threshold.csv
```

---

## Two Baselines Compared

### VA-Silence
- Input: `silence_duration_before_t` only
- Output: START_SPEAKING or WAIT
- Simple threshold-based

### VA-Threshold  
- Input: 5 voice activity features
- Output: START_SPEAKING, BACKCHANNEL, or WAIT
- More sophisticated

**Both automatically tuned on validation set** → evaluated on test set

---

## Example Workflow

### 1. Prepare your manifest
```python
# Ensure you have:
df = pd.read_parquet('data/processed/final_manifest.parquet')
# Columns: sample_id, split, final_label, 
#          human_active_at_t, num_humans_active_at_t, 
#          overlap_active_at_t, silence_duration_before_t,
#          current_human_speech_duration
```

### 2. Run baseline
```bash
python scripts/eval_va_threshold_baseline.py \
    --manifest data/processed/final_manifest.parquet \
    --output-dir reports/va_threshold_baseline
```

### 3. View results
```bash
cat reports/va_threshold_baseline/va_baseline_metrics.md
```

### 4. Compare with your neural models
```bash
# Your neural model Macro-F1: ?
# VA-Silence Macro-F1: from metrics
# VA-Threshold Macro-F1: from metrics
```

---

## Key Metrics

**Automatically computed:**
- Accuracy
- Macro-F1 (main metric)
- Weighted-F1
- Per-class precision/recall/F1
- False entry rate
- Missed entry rate
- Confusion matrices

---

## No Extra Dependencies

Already in `requirements.txt`:
- pandas
- numpy
- scikit-learn
- pyarrow

**Nothing to install!**

---

## That's All You Need to Know

1. Make sure your manifest has the 5 voice activity features
2. Run the one command above
3. Review the results

Done! ✅

For more details, see:
- **BASELINE_README.md** - Complete guide
- **VA_BASELINE_SUMMARY.md** - Implementation details
- **BASELINE_TEST_GUIDE.md** - Testing & examples
