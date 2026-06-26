# VA-Threshold Baseline Implementation Summary

## ✅ What Was Built

A complete voice activity threshold baseline system for turn-taking timing prediction.

### Components

#### 1. **Baseline Module** (`src/baselines/va_threshold.py` - 350 lines)
- `VASilenceBaseline`: Predicts based on silence duration only
- `VAThresholdBaseline`: Predicts using multiple voice activity features
- `BaselineEvaluator`: Threshold tuning and metrics computation
- `PredictionResult`: Data class for predictions

#### 2. **Evaluation Script** (`scripts/eval_va_threshold_baseline.py` - 421 lines)
- `VAThresholdEvaluationPipeline`: End-to-end evaluation workflow
- Automatic threshold tuning on validation set
- Comprehensive metrics computation
- Multiple output formats (parquet, JSON, markdown, CSV)

#### 3. **Test Data Generator** (`scripts/create_test_manifest.py`)
- Create synthetic test manifest for baseline validation
- Generates both validation and test samples

#### 4. **Documentation**
- `BASELINE_README.md`: Complete usage guide
- `VA_BASELINE_SUMMARY.md`: This file

### Files Created

```
src/baselines/
├── __init__.py              (121 bytes)
└── va_threshold.py          (13 KB) - Baseline implementations

scripts/
├── eval_va_threshold_baseline.py  (421 lines, 13 KB)
└── create_test_manifest.py        (80 lines, 2.5 KB)

Documentation:
├── BASELINE_README.md       (Complete usage guide)
└── VA_BASELINE_SUMMARY.md   (This summary)
```

---

## 🎯 Baseline Definitions

### VA-Silence
**Simplest baseline**: Predicts based on silence duration alone

```python
def predict(silence_duration_before_t):
    if silence_duration_before_t >= theta_start:
        return "START_SPEAKING"
    return "WAIT"
```

- **Never predicts:** BACKCHANNEL
- **Single parameter:** `theta_start` (silence threshold in seconds)
- **Candidates tested:** [0.3, 0.5, 0.6, 0.7, 1.0]

### VA-Threshold
**More sophisticated**: Uses multiple voice activity features

```python
def predict(human_active_at_t,
            num_humans_active_at_t,
            overlap_active_at_t,
            silence_duration_before_t,
            current_human_speech_duration,
            theta_start=0.6,
            theta_bc_min_speech=1.0):
    
    if silence_duration_before_t >= theta_start:
        return "START_SPEAKING"
    
    if (human_active_at_t and 
        num_humans_active_at_t == 1 and 
        not overlap_active_at_t and 
        current_human_speech_duration >= theta_bc_min_speech):
        return "BACKCHANNEL"
    
    return "WAIT"
```

- **Parameters:** 2 thresholds
  - `theta_start`: Silence threshold (seconds)
  - `theta_bc_min_speech`: Min speech duration for backchannel (seconds)
- **Candidates tested:** 5 × 4 = 20 combinations

---

## 📊 Features Required

Your dataset manifest should include:

```
Required columns:
- sample_id              (str)
- split                  (str: "validation", "test")
- final_label            (str: "WAIT", "BACKCHANNEL", "START_SPEAKING")

Voice activity features:
- human_active_at_t                (bool)
- num_humans_active_at_t           (int)
- overlap_active_at_t              (bool)
- silence_duration_before_t        (float) - seconds
- current_human_speech_duration    (float) - seconds
```

**If features missing:** Script auto-fills with defaults (0, False)

---

## 🚀 Quick Start

### 1. Create Test Data (Demo)
```bash
python scripts/create_test_manifest.py \
    --output data/processed/final_manifest_test.parquet \
    --num-validation 30 \
    --num-test 30
```

### 2. Run Baseline on Test Data (Dry-Run)
```bash
python scripts/eval_va_threshold_baseline.py \
    --manifest data/processed/final_manifest_test.parquet \
    --output-dir reports/va_threshold_baseline \
    --max-samples 100
```

### 3. Run Full Evaluation
```bash
python scripts/eval_va_threshold_baseline.py \
    --manifest data/processed/final_manifest.parquet \
    --output-dir reports/va_threshold_baseline
```

### 4. View Results
```bash
cat reports/va_threshold_baseline/va_baseline_metrics.md
```

---

## 📈 Output Files

### 1. Predictions (Parquet)
**File:** `va_baseline_predictions.parquet`

Contains per-sample predictions with:
- True label
- VA-Silence prediction
- VA-Threshold prediction
- All voice activity features
- Selected thresholds

### 2. Metrics (JSON)
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
    "false_entry_rate": 1.0,
    "missed_entry_rate": 0.45,
    "confusion_matrix": [[12, 0, 5], [0, 0, 2], [3, 0, 3]]
  },
  "va_threshold": {
    "accuracy": 0.68,
    "macro_f1": 0.65,
    ...
  },
  "best_thresholds": {
    "theta_start_silence": 0.6,
    "theta_start_threshold": 0.5,
    "theta_bc_min_speech": 1.0
  }
}
```

### 3. Metrics (Markdown)
**File:** `va_baseline_metrics.md`

Human-readable format for reports and papers.

### 4. Confusion Matrices (CSV)
**Files:**
- `va_baseline_confusion_matrix_silence.csv`
- `va_baseline_confusion_matrix_threshold.csv`

3×3 matrices for confusion analysis.

---

## 🔍 Workflow

```
1. Load Manifest
   ↓
2. Validate Features
   (Fill missing with defaults)
   ↓
3. Create Baseline Instances
   (All threshold combinations)
   ↓
4. Generate Predictions
   (On entire dataset)
   ↓
5. Select Best Thresholds
   (Maximize Macro-F1 on validation)
   ↓
6. Evaluate on Test Set
   (Compute all metrics)
   ↓
7. Save Results
   (Parquet, JSON, Markdown, CSV)
```

---

## 📋 Threshold Tuning

### Search Space

**VA-Silence:**
- `theta_start`: [0.3, 0.5, 0.6, 0.7, 1.0] → 5 configurations

**VA-Threshold:**
- `theta_start`: [0.3, 0.5, 0.6, 0.7, 1.0] (5 values)
- `theta_bc_min_speech`: [0.5, 1.0, 1.5, 2.0] (4 values)
- **Total:** 5 × 4 = 20 configurations

### Selection Criteria

For each configuration:
1. Evaluate on **validation** split only
2. Compute **Macro-F1** (unweighted average across classes)
3. Select configuration with **highest Macro-F1**
4. Use selected thresholds for **test** set evaluation

---

## 📊 Metrics Computed

### Primary Metrics
- **Accuracy:** Fraction correct
- **Macro-F1:** Unweighted average F1 across classes
- **Weighted-F1:** Class-weighted F1

### Per-Class Metrics
For each label (WAIT, BACKCHANNEL, START_SPEAKING):
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1:** Harmonic mean

### Turn-Taking Specific
- **False Entry Rate:** `pred in {BC, START} AND true == WAIT`
  - Fraction of predicted entries that are false positives
  - **Lower is better** (don't speak when you shouldn't)
  
- **Missed Entry Rate:** `pred == WAIT AND true in {BC, START}`
  - Fraction of true entries that are missed
  - **Lower is better** (don't miss opportunities)
  
- **BC-as-Turn Error:** `true == BC AND pred == START`
  - Confusion between backchannel and turn-start
  
- **Turn-as-BC Error:** `true == START AND pred == BC`
  - Confusion between turn-start and backchannel

### Confusion Matrix
Standard 3×3 matrix: rows=true, columns=predictions

---

## 🔧 Implementation Details

### Key Features

✅ **Automatic threshold tuning** on validation set  
✅ **Flexible feature handling** (auto-fill missing)  
✅ **Comprehensive metrics** (15+ metrics)  
✅ **Multiple output formats** (parquet, JSON, markdown, CSV)  
✅ **Dry-run mode** for testing (`--max-samples`)  
✅ **Detailed logging** with progress  
✅ **Error handling** with helpful messages  
✅ **Production-ready** code quality  

### Dependencies

```
pandas >= 2.0.0      (Data handling)
numpy >= 1.24.0      (Numerics)
scikit-learn >= 1.3.0 (Metrics)
pyarrow >= 12.0.0    (Parquet I/O)
```

**All already in `requirements.txt` - nothing extra to install!**

---

## 💡 Usage Examples

### Example 1: Full Pipeline
```bash
# Create test data
python scripts/create_test_manifest.py

# Run evaluation
python scripts/eval_va_threshold_baseline.py \
    --manifest data/processed/final_manifest_test.parquet \
    --output-dir reports/baseline_test

# View results
cat reports/baseline_test/va_baseline_metrics.md
```

### Example 2: With Your Data
```bash
# Assuming you have: data/processed/final_manifest.parquet
python scripts/eval_va_threshold_baseline.py \
    --manifest data/processed/final_manifest.parquet \
    --output-dir reports/va_threshold_baseline
```

### Example 3: Dry-Run (100 samples)
```bash
python scripts/eval_va_threshold_baseline.py \
    --manifest data/processed/final_manifest.parquet \
    --output-dir reports/va_threshold_baseline \
    --max-samples 100
```

---

## 🎯 Next Steps

1. **Prepare your dataset** with required features
2. **Run baseline** to establish floor performance
3. **Compare against neural models** (from main pipeline)
4. **Analyze errors** using confusion matrices
5. **Use baselines** as paper baseline comparisons

---

## 📚 Integration with Main Pipeline

The baseline system integrates with your main neural network pipeline:

```
Your Dataset
    ↓
├─→ VA-Threshold Baselines (this module)
│   ├─ VA-Silence
│   └─ VA-Threshold
│
└─→ Neural Network Models (main pipeline)
    ├─ TimingActionNet
    └─ Ablations

Comparison:
  Baseline Performance << Neural Network Performance
```

Use baselines to establish expectations and debug data quality.

---

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Missing required columns" | Add `sample_id`, `split`, `final_label` |
| "No validation split" | Add `split` column with "validation" values |
| "Features all zeros" | Check feature column names match expected |
| Metrics look strange | Verify label names are exactly "WAIT", "BACKCHANNEL", "START_SPEAKING" |
| Script is slow | Use `--max-samples 100` to test |

---

## 📖 Documentation

- **BASELINE_README.md** - Comprehensive usage guide
- **README.md** - Updated with baseline section
- **va_threshold.py** - Inline code documentation
- **eval_va_threshold_baseline.py** - Detailed script documentation

---

## ✨ Quality Checklist

- ✅ Modular, readable code
- ✅ Comprehensive error handling
- ✅ Multiple output formats
- ✅ Detailed logging
- ✅ Production-ready quality
- ✅ Tested with synthetic data
- ✅ Full documentation
- ✅ Integrates with main pipeline
- ✅ Deterministic (fixed seed)
- ✅ Efficient (parallelizable if needed)

---

## 🎉 Ready to Use!

The baseline system is complete and ready for your data. Simply:

1. Prepare your manifest (add voice activity features)
2. Run the evaluation script
3. Compare against your neural models

See **BASELINE_README.md** for detailed usage instructions.
