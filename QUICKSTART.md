# Quick Start Guide

## 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify torch installation
python -c "import torch; print(torch.cuda.is_available())"
```

## 2. Prepare Data

Ensure your parquet files are in the correct location:
```
data/processed/hf_export/
├── train.parquet
├── validation.parquet
└── test.parquet
```

The dataset loader expects columns with your input features:
- X_frame or frame_features or frame (shape [120, 7])
- X_scalar or scalar_features or scalar (shape [6])
- final_label (WAIT, BACKCHANNEL, or START_SPEAKING)
- training_weight (confidence score)
- exclude_from_training (boolean flag)

## 3. Train Model (First Time)

```bash
python scripts/train.py --config configs/config.yaml
```

This will:
1. Load train/val datasets (with automatic normalization)
2. Create model and trainer
3. Start training with automatic checkpointing
4. Save best model to `checkpoints/checkpoint_best.pt`
5. Log metrics to `logs/`

**Expected output:**
```
Epoch 1/100 | Train Loss: 0.8943 | Val Macro-F1: 0.6234
Epoch 2/100 | Train Loss: 0.7821 | Val Macro-F1: 0.6512
...
```

Training will stop if validation metric doesn't improve for 15 epochs (early stopping).

## 4. Resume Training

If training is interrupted, continue from latest checkpoint:

```bash
# Auto-detects checkpoint_latest.pt
python scripts/train.py --config configs/config.yaml

# Or specify explicit checkpoint
python scripts/resume_training.py --config configs/config.yaml --checkpoint checkpoints/checkpoint_epoch_50.pt
```

All training state is preserved:
- Epoch number
- Optimizer momentum/variance
- Learning rate scheduler
- Best metric tracking

## 5. Evaluate on Test Set

After training completes:

```bash
python scripts/test.py --config configs/config.yaml --checkpoint checkpoints/checkpoint_best.pt
```

This will:
1. Load best trained model
2. Run inference on test set
3. Compute all KPIs (F1, ECE, latency, etc.)
4. Save results to `results.json`

**Output includes:**
- Macro-F1, per-class F1
- False entry / missed entry rates
- Calibration error (ECE)
- Inference latency
- Per-sample predictions

## 6. View Results

```bash
# Check training logs
tail -100 logs/timing_net_*.log

# View test metrics
python -c "import json; print(json.dumps(json.load(open('results.json')), indent=2))" | less
```

## 7. Configure for Ablation Study 1

For the first ablation (previous activity only), modify `configs/config.yaml`:

```yaml
# Only use pseudo_robot_past_active from frame features
# All other frame features set to zeros/masked

# In dataset.py, add feature masking:
# if ablation_mode == "prev_activity_only":
#     frame[:, :, [0,1,2,3,4,5]] = 0  # Zero out all but feature 6
```

Or create a new config:

```bash
cp configs/config.yaml configs/ablation_1_prev_activity.yaml
# Edit ablation_1_prev_activity.yaml
python scripts/train.py --config configs/ablation_1_prev_activity.yaml
```

## Common Commands

```bash
# Check if training will run
python -c "from src.utils import load_config; print(load_config('configs/config.yaml').keys())"

# Count model parameters
python -c "from src.models import TimingActionNet; from src.utils import load_config; m = TimingActionNet(load_config('configs/config.yaml')); print(sum(p.numel() for p in m.parameters()))"

# Test data loading
python -c "from src.data import get_dataloaders; d = get_dataloaders('data/processed/hf_export', batch_size=4); batch = next(iter(d['train'])); print('Frame:', batch['frame'].shape, 'Scalar:', batch['scalar'].shape)"

# Verify checkpoint
python -c "import torch; ckpt = torch.load('checkpoints/checkpoint_best.pt'); print('Epoch:', ckpt['epoch'], 'Metric:', ckpt.get('best_val_metric', 'N/A'))"
```

## Tips

1. **Check GPU**: Monitor GPU usage during training
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Adjust batch size** if out of memory:
   ```yaml
   # In config.yaml
   training:
     batch_size: 16  # Reduce from 32
   ```

3. **Monitor training**:
   - Check `logs/` for detailed logs
   - Metrics saved in tensorboard format if enabled

4. **Reproducibility**: All random seeds are set in config (seed: 42)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: train.parquet` | Check `dataset.path` in config.yaml points to correct directory |
| CUDA out of memory | Reduce `batch_size` or use `device: "cpu"` |
| NaN loss | Try `learning_rate: 0.0005` or change `loss.type: "ce"` |
| Very slow loading | Increase `num_workers` in dataset config |
| Model not improving | Check label distribution, try different architecture |

## Next Steps

- Read [README.md](README.md) for full documentation
- Check [configs/config.yaml](configs/config.yaml) for all hyperparameters
- Review [src/models/timing_net.py](src/models/timing_net.py) for architecture details
- See [src/evaluation/metrics.py](src/evaluation/metrics.py) for KPI definitions
