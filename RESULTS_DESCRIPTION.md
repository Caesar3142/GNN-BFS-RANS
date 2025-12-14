# Results Description and Analysis

This document describes and analyzes the results from the GNN-based RANS flow field predictions, including line plots and training curves.

## Table of Contents

1. [Training Curves Analysis](#training-curves-analysis)
2. [Velocity Results Along Lines](#velocity-results-along-lines)
3. [Pressure Results Along Lines](#pressure-results-along-lines)
4. [Overall Performance Assessment](#overall-performance-assessment)
5. [Discussion](#discussion)

---

## Training Curves Analysis

### Overview

The training curves provide insights into the learning dynamics and convergence behavior of the GNN model.

### Key Metrics Tracked

1. **Training Loss**: Measures how well the model fits the training data
2. **Validation Loss**: Measures generalization to unseen data
3. **Learning Rate**: Shows adaptive learning rate scheduling
4. **Per-Field Errors**: Individual prediction errors for each flow field (U, p, k, ε, νₜ)
5. **Overfitting Indicator**: Difference between validation and training loss

### Expected Behavior

**Healthy Training:**
- Both training and validation losses decrease monotonically
- Validation loss tracks training loss closely (small gap)
- Learning rate decreases when loss plateaus (adaptive scheduling)
- All field errors decrease over time
- Overfitting indicator remains near zero or slightly positive

**Warning Signs:**
- Large gap between training and validation loss (overfitting)
- Validation loss increases while training loss decreases (overfitting)
- Loss plateaus early (may need higher learning rate or more capacity)
- One field's error dominates others (may need weight adjustment)
- Sudden jumps in loss (learning rate too high)

### Interpretation

The training curves show:

1. **Convergence Rate**: How quickly the model learns
   - Steep initial decrease indicates good initialization
   - Gradual decrease suggests stable learning

2. **Stability**: Consistency of training
   - Smooth curves indicate stable training
   - Oscillations suggest learning rate issues

3. **Field Balance**: Whether all fields are learning equally
   - Similar error magnitudes across fields = balanced training
   - One field much higher = may need weight adjustment

4. **Generalization**: Training vs validation performance
   - Small gap = good generalization
   - Large gap = overfitting (need regularization)

---

## Velocity Results Along Lines

### Horizontal Line (Y = 0.005)

**Description:**
The horizontal line at Y = 0.005 represents a cross-section through the flow domain. This line typically captures:
- Flow development in the streamwise (X) direction
- Boundary layer effects near walls
- Wake regions behind obstacles (if present)

**Expected Patterns:**
- **Near inlet (low X)**: Velocity should match inlet conditions
- **Mid-domain**: Flow development, potential acceleration/deceleration
- **Near outlet (high X)**: Fully developed flow or exit conditions
- **Near walls**: Boundary layer effects (velocity reduction)

**Analysis Points:**
1. **Magnitude**: Predicted velocity magnitude should match reference closely
2. **Gradient**: Velocity gradients (spatial derivatives) should be captured
3. **Peaks/Valleys**: Local extrema should be predicted accurately
4. **Smoothness**: Predictions should be physically smooth (no unphysical oscillations)

**Common Issues:**
- **Under-prediction**: Model may be too conservative
- **Over-prediction**: Model may not capture viscous effects
- **Oscillations**: May indicate insufficient smoothing or mesh issues
- **Offset**: Systematic bias suggests normalization or training issues

### Vertical Line (X = 0.15)

**Description:**
The vertical line at X = 0.15 represents a cross-section perpendicular to the main flow direction. This line typically captures:
- Wall-normal velocity profiles
- Boundary layer structure
- Flow separation (if present)
- Pressure gradients

**Expected Patterns:**
- **Near wall (low Y)**: Low velocity due to no-slip condition
- **Boundary layer**: Rapid velocity increase away from wall
- **Core flow (high Y)**: Higher, more uniform velocity
- **Pressure**: Should show wall-normal pressure variation

**Analysis Points:**
1. **Wall proximity**: Accuracy near walls is critical for boundary layer prediction
2. **Gradient**: Strong velocity gradients near walls must be captured
3. **Profile shape**: Velocity profile shape should match reference (log-law, power-law, etc.)
4. **Symmetry**: If geometry is symmetric, predictions should reflect this

**Common Issues:**
- **Wall boundary**: May struggle near walls (boundary layer resolution)
- **Gradient accuracy**: Strong gradients may be under-resolved
- **Profile mismatch**: Shape may differ from reference (turbulence model effects)

---

## Pressure Results Along Lines

### Horizontal Line (Y = 0.005)

**Description:**
Pressure along the horizontal line shows streamwise pressure variation, which is related to:
- Flow acceleration/deceleration
- Obstacle interactions
- Pressure recovery
- Friction losses

**Expected Patterns:**
- **Pressure gradient**: Should follow flow acceleration (Bernoulli principle)
- **Obstacles**: Pressure changes around obstacles (high/low pressure regions)
- **Recovery**: Pressure recovery in expanding regions
- **Friction**: Gradual pressure drop due to friction

**Analysis Points:**
1. **Absolute level**: Pressure reference constraint should anchor absolute values
2. **Gradients**: Pressure gradients should match reference
3. **Peaks**: High/low pressure regions should be captured
4. **Smoothness**: Pressure should be smooth (pressure is continuous)

**Common Issues:**
- **Drift**: Global pressure drift (addressed by reference constraint)
- **Gradient mismatch**: May not capture pressure gradients accurately
- **Oscillations**: Unphysical pressure oscillations
- **Scale**: Pressure magnitude may be off (normalization issues)

### Vertical Line (X = 0.15)

**Description:**
Pressure along the vertical line shows wall-normal pressure variation, which is typically:
- Small in boundary layers (hydrostatic assumption)
- Larger in separated flows
- Related to curvature effects

**Expected Patterns:**
- **Wall-normal gradient**: Usually small in attached boundary layers
- **Separation**: Pressure changes in separated regions
- **Curvature**: Pressure variation due to streamline curvature

**Analysis Points:**
1. **Wall pressure**: Pressure at wall is important for drag/lift
2. **Gradient**: Wall-normal pressure gradient should be small (typically)
3. **Consistency**: Pressure should be consistent with velocity field

**Common Issues:**
- **Gradient errors**: May over/under-predict wall-normal gradients
- **Wall accuracy**: Pressure near walls may be less accurate
- **Consistency**: Pressure-velocity coupling may not be perfect

---

## Overall Performance Assessment

### Quantitative Metrics

**Velocity:**
- Mean Absolute Error (MAE): Average error magnitude
- Maximum Error: Worst-case prediction error
- Relative Error: Error normalized by reference scale
- Correlation: How well predictions correlate with reference

**Pressure:**
- Mean Absolute Error (MAE)
- Maximum Error
- Relative Error
- Pressure drift: Global offset (should be near zero with reference constraint)

### Qualitative Assessment

**Spatial Accuracy:**
- Are flow features (boundary layers, wakes, etc.) captured?
- Are gradients predicted correctly?
- Are extrema (max/min) in correct locations?

**Physical Consistency:**
- Do predictions satisfy physical constraints?
- Is pressure-velocity coupling reasonable?
- Are boundary conditions respected?

**Smoothness:**
- Are predictions physically smooth?
- Are there unphysical oscillations?
- Is the solution mesh-converged?

---

## Discussion

### Strengths of the Approach

1. **Mesh Flexibility**: GNNs naturally handle unstructured meshes
2. **Local Physics**: Message passing captures local flow interactions
3. **Efficiency**: Single forward pass vs iterative CFD solver
4. **Generalization**: Can learn complex flow patterns from data

### Limitations

1. **Training Data**: Requires converged CFD solutions
2. **Single Geometry**: Current implementation for one mesh topology
3. **Steady-State**: Only predicts converged solutions, not time evolution
4. **Boundary Conditions**: Not explicitly encoded (learned from geometry)

### Typical Results

**Good Performance:**
- Velocity errors: < 5% relative error
- Pressure errors: < 10% relative error (pressure is more challenging)
- Training loss: Decreases smoothly and converges
- Field balance: All fields have similar error magnitudes

**Areas for Improvement:**
- Near-wall accuracy (boundary layer resolution)
- Pressure prediction (more challenging than velocity)
- Strong gradient regions (may need more training data)
- Complex flow features (separation, recirculation)

### Recommendations

1. **If velocity errors are high:**
   - Check normalization (per-component for velocity)
   - Increase velocity weight in loss function
   - Add more training data near problematic regions
   - Try different GNN architecture (GAT, Transformer)

2. **If pressure errors are high:**
   - Increase pressure weight (already set to 3.0)
   - Increase pressure reference constraint weight
   - Use curriculum training (freeze pressure initially)
   - Check pressure normalization

3. **If training is unstable:**
   - Reduce learning rate (try 1e-4)
   - Increase gradient clipping
   - Use curriculum training
   - Add more regularization (dropout, weight decay)

4. **If overfitting:**
   - Increase dropout rate
   - Add weight decay
   - Reduce model capacity
   - Use data augmentation

---

## Example Interpretation

### Training Curves

**Scenario 1: Healthy Training**
```
Training loss: Decreases smoothly from 1.0 → 0.01 over 200 epochs
Validation loss: Tracks training loss closely (gap < 0.01)
Field errors: All decrease proportionally
Overfitting indicator: Near zero
→ Model is learning well, good generalization
```

**Scenario 2: Overfitting**
```
Training loss: Decreases to 0.001
Validation loss: Plateaus at 0.1 (large gap)
Field errors: Training errors low, validation errors high
Overfitting indicator: Large positive values
→ Model memorizing training data, need regularization
```

**Scenario 3: Underfitting**
```
Training loss: Plateaus at 0.1 (not decreasing)
Validation loss: Similar to training loss
Field errors: All remain high
Overfitting indicator: Near zero but errors high
→ Model capacity too low, need larger model or more training
```

### Line Plots

**Good Prediction:**
- Predicted and reference lines overlap closely
- Smooth curves (no oscillations)
- Correct capture of gradients and extrema
- Small error statistics (< 5% relative error)

**Poor Prediction:**
- Large gap between predicted and reference
- Oscillations or unphysical behavior
- Missing or incorrect extrema
- Large error statistics (> 20% relative error)

---

## Conclusion

The results from line plots and training curves provide comprehensive insights into model performance:

1. **Training curves** show learning dynamics and help identify training issues
2. **Line plots** provide detailed spatial accuracy assessment
3. **Combined analysis** guides model improvement and hyperparameter tuning

Regular monitoring of these metrics during training and evaluation is essential for developing an accurate and reliable surrogate model.
