# Physics-Based Fixes for GNN Training

## Issues Identified and Fixed

### 1. **Velocity Normalization Issue** ✅ FIXED
**Problem**: Velocity components (Ux, Uy, Uz) were normalized together as a flattened array, not separately. This can cause issues if components have different scales.

**Fix**: Modified `FieldNormalizer` to normalize each velocity component separately:
- `U[:, 0]` (Ux), `U[:, 1]` (Uy), `U[:, 2]` (Uz) are now normalized independently
- Each component gets its own mean and std
- Transform and inverse_transform now handle per-component normalization

### 2. **Loss Function Issue** ✅ FIXED
**Problem**: The weighted MSE loss applied element-wise weights, but after independent field normalization, fields might still have different scales, making the loss dominated by one field.

**Fix**: Added field-wise loss computation:
- Computes MSE loss separately for each field (U, p, k, epsilon, nut)
- Applies field weights to each field's loss, then sums
- This ensures balanced training across all fields regardless of their normalized scales
- Enabled by default with `use_fieldwise=True`

### 3. **Field Ordering** ✅ VERIFIED
**Confirmed**: Field ordering is consistent across all code:
- Training: `[U(3), p(1), k(1), epsilon(1), nut(1)]` = 7 dimensions
- Model output: Same ordering
- Inference: Same ordering

## Next Steps

### 1. **Retrain the Model**
You need to retrain the model with the fixed normalization:
```bash
python train.py \
    --case_path OpenFOAM-data \
    --time_dirs 0 100 200 282 \
    --output_dir checkpoints \
    --hidden_dim 256 \
    --num_layers 6 \
    --layer_type GCN \
    --epochs 100 \
    --lr 0.0005
```

**Important**: The old checkpoint won't work correctly because:
- The normalizer format has changed (now stores per-component stats for U)
- The model needs to be retrained with the new normalization scheme

### 2. **Training Recommendations**

If errors are still high after retraining, consider:

**a) Increase model capacity:**
```bash
--hidden_dim 512 \
--num_layers 8
```

**b) Adjust learning rate:**
```bash
--lr 0.001  # Try higher learning rate
# or
--lr 0.0001  # Try lower learning rate
```

**c) Train for more epochs:**
```bash
--epochs 200
```

**d) Use different GNN layer:**
```bash
--layer_type GAT  # or Transformer
```

**e) Adjust field weights in loss:**
Modify `train.py` to give more weight to problematic fields:
```python
criterion = WeightedMSELoss(
    field_weights={
        'U': 2.0,      # Increase if velocity errors are high
        'p': 2.0,      # Increase if pressure errors are high
        'k': 1.0,      # Increase turbulence field weights
        'epsilon': 1.0,
        'nut': 1.0
    },
    use_fieldwise=True
)
```

### 3. **Verify Training Progress**

Monitor the per-field errors during training:
- They should decrease over epochs
- If one field's error is much higher, increase its weight
- If errors plateau early, try higher learning rate or more capacity

### 4. **Check Data Quality**

Verify that:
- All time directories have consistent field data
- Field values are physically reasonable
- Mesh connectivity is correct (use `test_data_loading.py`)

## Technical Details

### Normalization Changes

**Before:**
```python
# U normalized as flattened array
U_flat = U.flatten()
mean = np.mean(U_flat)
std = np.std(U_flat)
U_normalized = (U - mean) / std  # Same mean/std for all components
```

**After:**
```python
# U normalized per component
mean = np.mean(U, axis=0)  # [mean_Ux, mean_Uy, mean_Uz]
std = np.std(U, axis=0)     # [std_Ux, std_Uy, std_Uz]
U_normalized = (U - mean) / std  # Component-wise normalization
```

### Loss Function Changes

**Before:**
```python
# Element-wise weighting
squared_error = (pred - target) ** 2
weighted_error = squared_error * weights  # [U(3), p(1), k(1), ...]
loss = weighted_error.mean()
```

**After:**
```python
# Field-wise computation
u_loss = mean((U_pred - U_target)^2)
p_loss = mean((p_pred - p_target)^2)
# ... for each field
loss = w_U * u_loss + w_p * p_loss + w_k * k_loss + ...
```

## Expected Improvements

After retraining with these fixes:
1. **Better velocity prediction**: Per-component normalization should improve Ux, Uy, Uz predictions
2. **Balanced field training**: Field-wise loss ensures all fields are learned equally
3. **More stable training**: Better normalization should lead to more stable gradients
4. **Lower overall errors**: Combined improvements should reduce prediction errors

## If Errors Are Still High

1. **Check normalization statistics**: Print normalizer stats to verify they're reasonable
2. **Visualize training loss**: Check if loss is decreasing or plateauing
3. **Compare normalized vs raw errors**: Verify denormalization is working correctly
4. **Check model capacity**: Try larger models or different architectures
5. **Data augmentation**: Consider using more training time steps or different geometries
