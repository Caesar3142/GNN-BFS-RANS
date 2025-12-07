# Training & Pressure Fixes - Implementation Summary

## ✅ Implemented Fixes

### 1. Loss Function Rebalancing (CRITICAL) ✅

**Problem**: Pressure error was under-trained compared to velocity, causing drift and instability.

**Solution**: Increased pressure weight from 1.0 to **3.0** in the loss function.

**Implementation**:
- Updated `WeightedMSELoss` default weights:
  - `w_U = 1.0` (velocity)
  - `w_p = 3.0` ← **CRITICAL: prevents pressure drift**
  - `w_k = 0.5` (turbulence)
  - `w_epsilon = 0.5`
  - `w_nut = 0.5`

**Location**: `normalization.py` - `WeightedMSELoss.__init__()`

---

### 2. Reduced Initial Learning Rate ✅

**Problem**: Initial LR = 1e-3 was too high for CFD-GNNs, causing early instability.

**Solution**: Reduced default learning rate to **3e-4**.

**Implementation**:
- Changed default `--lr` from `0.0005` to `3e-4`
- Added note in help text about CFD-GNN recommendations

**Location**: `train.py` - argument parser

**Usage**:
```bash
python train.py --lr 3e-4  # or 1e-4 for even more stability
```

---

### 3. Pressure Reference Constraint ✅

**Problem**: Pressure is defined up to a constant → global drift is common.

**Solution**: Added soft constraint to anchor absolute pressure level.

**Implementation**:
- Added `pressure_ref_weight` parameter (default: 0.1)
- Constraint: `L_pref = (mean(p_pred) - mean(p_ref))²`
- Applied in addition to standard MSE loss for pressure

**Location**: 
- `normalization.py` - `WeightedMSELoss.forward()`
- `train.py` - `--pressure_ref_weight` argument

**Usage**:
```bash
python train.py --pressure_ref_weight 0.1  # default
python train.py --pressure_ref_weight 0.2  # stronger constraint
```

---

### 4. Two-Stage (Curriculum) Training for Pressure ✅

**Problem**: Pressure learns last and is most fragile. Early training can cause oscillations.

**Solution**: Two-phase training strategy.

**Implementation**:
- **Phase 1** (epochs 1 to `curriculum_epochs`):
  - Train U + turbulence fields
  - Freeze pressure output (zero gradients for pressure)
- **Phase 2** (epochs `curriculum_epochs+1` to end):
  - Unfreeze pressure
  - Reduce learning rate by 50%

**Location**: `train.py` - training loop

**Usage**:
```bash
# Enable curriculum training for first 25 epochs
python train.py --curriculum_epochs 25
```

**How it works**:
- In Phase 1, pressure output gradients are zeroed during backward pass
- In Phase 2, pressure is unfrozen and LR is reduced automatically

---

## 📊 Recommended Training Configuration

### Basic (Recommended Starting Point)
```bash
python train.py \
    --case_path OpenFOAM-data \
    --time_dirs 0 100 200 282 \
    --output_dir checkpoints \
    --hidden_dim 256 \
    --num_layers 6 \
    --layer_type GCN \
    --epochs 200 \
    --lr 3e-4 \
    --batch_size 1 \
    --pressure_ref_weight 0.1
```

### With Curriculum Training (For Better Pressure Stability)
```bash
python train.py \
    --case_path OpenFOAM-data \
    --time_dirs 0 100 200 282 \
    --output_dir checkpoints \
    --hidden_dim 256 \
    --num_layers 6 \
    --layer_type GCN \
    --epochs 200 \
    --lr 3e-4 \
    --batch_size 1 \
    --pressure_ref_weight 0.1 \
    --curriculum_epochs 25
```

### Conservative (If Still Having Issues)
```bash
python train.py \
    --case_path OpenFOAM-data \
    --time_dirs 0 100 200 282 \
    --output_dir checkpoints \
    --hidden_dim 256 \
    --num_layers 6 \
    --layer_type GCN \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 1 \
    --pressure_ref_weight 0.2 \
    --curriculum_epochs 30
```

---

## 🔍 What Changed in the Code

### `normalization.py`
1. **Default field weights updated**: `p: 1.0 → 3.0`
2. **Added `pressure_ref_weight` parameter**: Controls pressure reference constraint
3. **Pressure reference constraint**: `L_pref = (mean(p_pred) - mean(p_ref))²`

### `train.py`
1. **Default learning rate**: `0.0005 → 3e-4`
2. **New argument**: `--pressure_ref_weight` (default: 0.1)
3. **New argument**: `--curriculum_epochs` (default: 0, disabled)
4. **Curriculum training logic**: Freezes pressure in Phase 1, unfreezes in Phase 2
5. **Updated loss calls**: All loss computations now pass `pressure_ref_weight`

---

## 📈 Expected Improvements

1. **Stable validation loss**: Pressure weight prevents drift
2. **Better pressure predictions**: Reference constraint anchors absolute level
3. **Smoother training**: Lower LR reduces early oscillations
4. **Controlled pressure learning**: Curriculum training prevents early pressure corruption

---

## 🎯 Monitoring Training

Watch for these indicators:

**Good signs**:
- Validation loss decreases smoothly
- Pressure error decreases over time
- Field errors are balanced (not dominated by one field)
- No sudden jumps in loss

**Warning signs**:
- Validation loss plateaus early → try lower LR or more epochs
- Pressure error stays high → increase `pressure_ref_weight` or `curriculum_epochs`
- One field dominates → adjust field weights
- Loss jumps → LR too high, reduce further

**Use training curves**:
```bash
python plot_training.py --history checkpoints/training_history.json --detailed
```

---

## 🔧 Troubleshooting

### If pressure errors are still high:
1. Increase `--pressure_ref_weight` to 0.2 or 0.3
2. Increase `--curriculum_epochs` to 30-40
3. Reduce `--lr` to 1e-4
4. Increase pressure weight in code (modify `train.py` line ~340: `'p': 3.0` → `'p': 5.0`)

### If training is too slow:
1. Increase `--lr` to 5e-4 (but watch for instability)
2. Reduce `--curriculum_epochs` or disable it
3. Reduce `--pressure_ref_weight`

### If validation loss doesn't decrease:
1. Check data quality (use `test_data_loading.py`)
2. Verify normalization is working
3. Try different GNN layer type (`--layer_type GAT` or `Transformer`)
4. Increase model capacity (`--hidden_dim 512`, `--num_layers 8`)

---

## 📝 Notes

- **Model architecture**: ✅ No changes needed
- **Graph construction**: ✅ No changes needed  
- **Data quality**: ✅ No changes needed
- **Visualization**: ✅ No changes needed

**This is a TRAINING STRATEGY fix, not a physics or coding bug.**

The fixes address:
- Loss function imbalance (pressure under-weighted)
- Learning rate too high (early instability)
- Pressure drift (reference constraint)
- Pressure fragility (curriculum training)
