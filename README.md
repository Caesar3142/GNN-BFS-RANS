# GNN-Based RANS Flow Field Surrogate Simulator

This repository contains a Graph Neural Network (GNN) framework for training a surrogate simulator based on OpenFOAM steady RANS simulation data. The framework learns to predict flow fields (velocity, pressure, and turbulence quantities) directly from mesh geometry.

## Overview

The framework consists of:
- **OpenFOAM Data Loader**: Parses OpenFOAM mesh and field files
- **Graph Constructor**: Builds graph structure from mesh connectivity
- **GNN Model**: Graph Neural Network for flow field prediction
- **Training Script**: End-to-end training pipeline with history tracking
- **Inference Script**: Prediction and evaluation tools
- **Visualization Tools**: Field comparison plots, line plots, and training curve visualization

## Features

- Supports multiple GNN architectures (GCN, GAT, GIN, Transformer)
- Predicts velocity (U), pressure (p), and turbulence fields (k, epsilon, nut)
- Handles unstructured meshes via cell-to-cell connectivity
- Robust graph construction with automatic validation and error handling
- Automatic handling of isolated nodes and edge connectivity issues
- **Improved normalization**: Per-component velocity normalization for better accuracy
- **Field-wise loss**: Balanced training across all flow fields with optimized pressure weight (3.0)
- **Training visualization**: Plot training curves and monitor progress
- **Line plots**: Extract and visualize velocity/pressure along specific lines (horizontal/vertical)
- Can save predictions in both NumPy and OpenFOAM formats
- Includes evaluation metrics for field-wise error analysis
- Triangulation-based visualization for accurate mesh representation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd GNN-BFS-RANS
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: On macOS, if you get an "externally-managed-environment" error, use a virtual environment as shown above. After activation, you can use `pip` directly.

## Data Structure

The framework expects OpenFOAM case data in the following structure:
```
OpenFOAM-data/
├── constant/
│   ├── polyMesh/
│   │   ├── points
│   │   ├── faces
│   │   ├── owner
│   │   ├── neighbour
│   │   └── boundary
│   ├── transportProperties
│   └── turbulenceProperties
├── 0/          # Initial conditions
│   ├── U
│   ├── p
│   ├── k
│   ├── epsilon
│   └── nut
├── 100/        # Time snapshot 1
├── 200/        # Time snapshot 2
└── 282/        # Final/converged solution
```

## Usage

**Important**: Always activate the virtual environment before running scripts:
```bash
source venv/bin/activate
```

### Testing Data Loading

Before training, verify that your OpenFOAM data can be loaded correctly:

```bash
python test_data_loading.py --case_path OpenFOAM-data
```

This script checks:
- Mesh data loading (points, faces, owner, neighbour)
- Field data loading (U, p, k, epsilon, nut)
- Graph construction and connectivity
- Node and edge feature computation

### Visualization

Visualize predicted flow fields and compare with reference data:

```bash
python visualize.py \
    --checkpoint checkpoints/best_model.pt \
    --case_path OpenFOAM-data \
    --reference_time 282 \
    --output_dir visualizations
```

**Basic Usage:**
```bash
# Minimum required: just specify the checkpoint
python visualize.py --checkpoint checkpoints/best_model.pt

# Full example with all options
python visualize.py \
    --checkpoint checkpoints/best_model.pt \
    --case_path OpenFOAM-data \
    --reference_time 282 \
    --output_dir visualizations \
    --device cuda
```

**Arguments:**
- `--checkpoint` (required): Path to trained model checkpoint (e.g., `checkpoints/best_model.pt`)
- `--case_path` (optional): Path to OpenFOAM case directory (default: `OpenFOAM-data`)
- `--reference_time` (optional): Time directory for reference comparison (default: `282`)
- `--output_dir` (optional): Directory to save visualization plots (default: `visualizations`)
- `--device` (optional): Device to use - `cuda` or `cpu` (default: auto-detect)

**What it does:**
1. Loads the trained model from the checkpoint
2. Loads the normalizer (if saved with the checkpoint)
3. Constructs the graph from the mesh
4. Runs inference to predict flow fields
5. Loads reference fields from the specified time directory
6. Creates comparison plots for each field

**Output:**
- Creates comparison plots for each field: **U** (velocity), **p** (pressure), **k**, **epsilon**, **nut**
- Each plot contains 3 subplots stacked vertically:
  1. **Top**: Predicted field
  2. **Middle**: Reference field (from OpenFOAM)
  3. **Bottom**: Normalized error (|Predicted - Reference| / Range(Reference) × 100%, capped at 10%)
- All plots saved as PNG images in the output directory:
  - `U_comparison.png`
  - `p_comparison.png`
  - `k_comparison.png`
  - `epsilon_comparison.png`
  - `nut_comparison.png`

**Example Output:**
```
Loading model...
Normalizer found - will denormalize predictions
Loading mesh and constructing graph...
Running prediction...
Loading reference fields...
Filtering to z >= 0 (positive z side): 12345 cells (from 24690 total)
Creating visualization plots...
  U Error Stats:
    Mean absolute error: 1.234567e-03
    Max absolute error: 5.678901e-03
    Reference scale (range): 1.000000e+00 (min: 0.000000e+00, max: 1.000000e+00)
    Mean normalized error: 0.12%
    Max normalized error: 0.57%
  Saved comparison plot: visualizations/U_comparison.png
  ...
Visualization complete! Plots saved to visualizations
```

**Tips:**
- Make sure you have a trained model checkpoint before visualizing
- The reference time should match a time directory in your OpenFOAM case
- The visualization automatically filters to 2D (z >= 0) for 2D plots
- Error statistics are printed to console for each field

### Line Plots

Plot velocity and pressure along specific lines (horizontal or vertical):

```bash
# Plot along Y = 0.005 (horizontal) and X = 0.15 (vertical)
python plot_lines.py \
    --checkpoint checkpoints/best_model.pt \
    --x_line 0.15 \
    --y_line 0.005
```

**Arguments:**
- `--checkpoint` (required): Path to trained model checkpoint
- `--case_path` (optional): Path to OpenFOAM case directory (default: `OpenFOAM-data`)
- `--reference_time` (optional): Time directory for reference (default: `282`)
- `--x_line` (optional): X coordinate for vertical line (default: `0.15`)
- `--y_line` (optional): Y coordinate for horizontal line (default: `0.005`)
- `--output_dir` (optional): Directory to save plots (default: `visualizations`)
- `--tol` (optional): Tolerance for line extraction (default: `1e-4`)

**Output:**
- `line_Y_0.005.png`: Velocity and pressure along horizontal line Y = 0.005
- `line_X_0.15.png`: Velocity and pressure along vertical line X = 0.15
- Each plot shows predicted vs reference for both velocity and pressure
- Statistics printed to console (ranges and mean absolute errors)

### Training

Train a GNN model on your OpenFOAM data:

```bash
python train.py \
    --case_path OpenFOAM-data \
    --time_dirs 0 100 200 282 \
    --output_dir checkpoints \
    --hidden_dim 128 \
    --num_layers 4 \
    --layer_type GCN \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 1
```

**Arguments:**
- `--case_path`: Path to OpenFOAM case directory
- `--time_dirs`: List of time directories to use for training
- `--output_dir`: Directory to save model checkpoints
- `--hidden_dim`: Hidden dimension for GNN layers (default: 128)
- `--num_layers`: Number of GNN layers (default: 4)
- `--layer_type`: GNN layer type - GCN, GAT, GIN, or Transformer (default: GCN)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 3e-4, optimized for CFD-GNNs)
- `--batch_size`: Batch size (default: 1, typically for single mesh)
- `--pressure_ref_weight`: Weight for pressure reference constraint to prevent drift (default: 0.1)
- `--curriculum_epochs`: Number of epochs for curriculum training (0 = disabled). Phase 1: train U+turbulence, Phase 2: train all (default: 0)

**Recommended Training Configuration:**
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
    --pressure_ref_weight 0.1 \
    --curriculum_epochs 25
```

**Training Strategy:**
- **Pressure weight = 3.0**: Higher weight prevents pressure under-training and drift
- **Lower learning rate (3e-4)**: More stable for CFD-GNNs, reduces early oscillations
- **Pressure reference constraint**: Anchors absolute pressure level to prevent global drift
- **Curriculum training**: Optional two-phase training (freeze pressure in Phase 1, then unfreeze in Phase 2)

**For detailed information on training fixes and troubleshooting, see `TRAINING_FIXES.md`**

**Training Output:**
- Model checkpoints saved in `output_dir/`:
  - `best_model.pt`: Best model based on validation loss
  - `checkpoint_epoch_N.pt`: Periodic checkpoints
  - `config.json`: Training configuration
  - `training_history.json`: Training history (losses, field errors, learning rate)

### Plot Training Curves

Visualize training progress and loss curves:

```bash
# Basic usage (plots from checkpoints/training_history.json)
python plot_training.py

# Specify custom history file
python plot_training.py --history checkpoints/training_history.json

# Also create detailed field errors plot
python plot_training.py --detailed
```

**Output:**
- `training_curves.png`: Main plot with 4 subplots:
  1. Training and validation loss (log scale)
  2. Learning rate schedule
  3. Per-field errors (U, p, k, epsilon, nut)
  4. Overfitting indicator (val_loss - train_loss)
- `field_errors_detailed.png`: Detailed per-field error plots (if `--detailed` flag used)

**What to look for:**
- **Loss curves**: Should decrease over time; validation loss should track training loss
- **Learning rate**: Shows how the scheduler adjusts the learning rate
- **Field errors**: Individual field prediction errors (computed every 10 epochs)
- **Overfitting indicator**: Positive values suggest overfitting; negative values suggest underfitting

### Inference

Run inference with a trained model:

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --case_path OpenFOAM-data \
    --output_dir predictions \
    --reference_time 282 \
    --save_format both
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint
- `--case_path`: Path to OpenFOAM case directory
- `--output_dir`: Directory to save predictions
- `--reference_time`: Optional time directory for comparison (e.g., "282")
- `--save_format`: Output format - numpy, openfoam, or both

## Theory and Methods

For detailed theoretical background and methodological approaches, see **[THEORY_AND_METHODS.md](THEORY_AND_METHODS.md)**.

This document covers:
- RANS equations and turbulence modeling
- Graph Neural Networks theory
- Graph construction from unstructured meshes
- Message passing mechanisms
- GNN architectures (GCN, GAT, GIN, Transformer)
- Training methodology and loss functions
- Normalization strategies
- Surrogate modeling approach

## Results Analysis

For guidance on interpreting training curves and line plot results, see **[RESULTS_DESCRIPTION.md](RESULTS_DESCRIPTION.md)**.

This document covers:
- Training curves analysis and interpretation
- Velocity results along horizontal and vertical lines
- Pressure results along horizontal and vertical lines
- Performance assessment metrics
- Common issues and troubleshooting
- Recommendations for improvement

## Model Architecture

The GNN model (`FlowGNN`) consists of:
1. **Input Projection**: Maps cell center coordinates to hidden dimension
2. **GNN Layers**: Stack of graph convolution layers with residual connections
3. **Output Projection**: Maps hidden features to flow field predictions

**Message Passing Robustness:**
- Automatic validation of edge connectivity before message passing
- Handles edge cases (empty graphs, invalid indices) gracefully
- Informative error messages for debugging connectivity issues
- Automatic self-loop addition for isolated nodes to ensure message propagation

The model predicts:
- Velocity vector U: [n_cells, 3]
- Pressure p: [n_cells, 1]
- Turbulent kinetic energy k: [n_cells, 1]
- Dissipation rate epsilon: [n_cells, 1]
- Turbulent viscosity nut: [n_cells, 1]

## Graph Construction

The graph is constructed from OpenFOAM mesh connectivity:
- **Nodes**: Mesh cells (represented by cell centers)
- **Edges**: Cell-to-cell connections via shared faces (bidirectional)
- **Node Features**: Cell center coordinates (x, y, z)
- **Edge Features**: Direction vector and distance between cell centers

**Robustness Features:**
- Automatic validation of edge indices to ensure all connections are within valid node range
- Automatic handling of isolated nodes by adding self-loops to ensure graph connectivity
- Proper handling of internal cell filtering when working with boundary conditions
- Edge attribute validation and error handling for message passing

## Training Details

- **Loss Function**: Weighted MSE with field-wise computation for balanced training across all fields
  - Computes loss separately for each field (U, p, k, epsilon, nut)
  - Applies field-specific weights, then combines
  - **Default weights**: U=1.0, **p=3.0** (critical for pressure stability), k=0.5, epsilon=0.5, nut=0.5
  - Ensures balanced training regardless of normalized field scales
  - **Pressure reference constraint**: `L_pref = (mean(p_pred) - mean(p_ref))²` prevents global drift
- **Normalization**: Per-component normalization for velocity (Ux, Uy, Uz normalized separately)
  - Each velocity component gets its own mean and std
  - Scalar fields normalized independently
  - Improves training stability and prediction accuracy
- **Learning Rate**: Default 3e-4 (optimized for CFD-GNNs, reduces early instability)
- **Optimizer**: Adam with learning rate scheduling (ReduceLROnPlateau)
- **Curriculum Training**: Optional two-phase training strategy
  - Phase 1: Train U + turbulence fields, freeze pressure output
  - Phase 2: Unfreeze pressure, reduce learning rate by 50%
  - Prevents early pressure oscillations and improves stability
- **Regularization**: Gradient clipping, dropout, batch normalization
- **Evaluation**: Per-field error metrics (MAE, RMSE, relative error) computed every 10 epochs
- **Training History**: Automatically saved to `training_history.json` for plotting and analysis

## Output

### Training Output
- Model checkpoints saved in `checkpoints/`:
  - `best_model.pt`: Best model based on validation loss
  - `checkpoint_epoch_N.pt`: Periodic checkpoints
  - `config.json`: Training configuration
  - `training_history.json`: Training history (losses, field errors, learning rate)

### Inference Output
- Predictions saved as:
  - `predictions.npz`: NumPy format (if `save_format` includes 'numpy')
  - `predictions/predicted/`: OpenFOAM format field files (if `save_format` includes 'openfoam')
- Field comparison metrics printed to console (if reference provided)

### Visualization Output
- **Field comparison plots**: Contour plots comparing predicted vs reference fields
  - Saved as PNG images: `U_comparison.png`, `p_comparison.png`, etc.
  - Each plot shows predicted, reference, and normalized error
- **Line plots**: Velocity and pressure along specific lines
  - `line_Y_0.005.png`: Horizontal line at Y = 0.005
  - `line_X_0.15.png`: Vertical line at X = 0.15
  - Each plot shows predicted vs reference for velocity and pressure
- **Training curves**: Training and validation loss plots
  - `training_curves.png`: Main training progress plot
  - `field_errors_detailed.png`: Detailed per-field error plots (optional)

## Example Workflow

1. **Prepare Data**: Ensure your OpenFOAM case data is in the correct structure
2. **Test Data Loading** (optional but recommended):
   ```bash
   python test_data_loading.py --case_path OpenFOAM-data
   ```
3. **Train Model**:
   ```bash
   python train.py \
       --case_path OpenFOAM-data \
       --time_dirs 0 100 200 282 \
       --epochs 200 \
       --lr 3e-4 \
       --pressure_ref_weight 0.1 \
       --curriculum_epochs 25
   ```
   This will:
   - Load and normalize the data (per-component velocity normalization)
   - Train the GNN model with optimized settings
   - Use curriculum training (freeze pressure in Phase 1, then unfreeze)
   - Save checkpoints and training history
4. **Plot Training Curves**:
   ```bash
   python plot_training.py --history checkpoints/training_history.json --detailed
   ```
   This creates plots showing:
   - Training and validation loss curves
   - Learning rate schedule
   - Per-field errors over time
   - Overfitting indicator
5. **Evaluate Model**:
   ```bash
   python inference.py --checkpoint checkpoints/best_model.pt --reference_time 282
   ```
6. **Visualize Results**:
   ```bash
   python visualize.py --checkpoint checkpoints/best_model.pt --reference_time 282 --output_dir visualizations
   ```
   This creates contour plots comparing predicted vs reference fields for:
   - Velocity magnitude
   - Pressure
   - Turbulent kinetic energy (k)
   - Dissipation rate (epsilon)
   - Turbulent viscosity (nut)
7. **Plot Line Comparisons** (optional):
   ```bash
   python plot_lines.py --checkpoint checkpoints/best_model.pt --x_line 0.15 --y_line 0.005
   ```
   This creates line plots showing velocity and pressure along:
   - Horizontal line at Y = 0.005
   - Vertical line at X = 0.15
8. **Use Predictions**: Load predictions from `predictions/` directory for further analysis

## Notes

- The current implementation uses cell center coordinates as input features
- Boundary conditions are not explicitly encoded (can be added as future enhancement)
- The model learns steady-state flow fields from geometry
- For best results, use converged solution data (e.g., final time step) for training
- Graph connectivity is automatically validated and fixed during construction
- Isolated nodes are handled automatically with self-loops to ensure message passing works correctly
- **Normalization**: Velocity components are normalized separately; each field is normalized independently
- **Training**: The loss function uses field-wise computation to ensure balanced learning across all fields
- **Visualization**: Plots use triangulation-based rendering for accurate representation of unstructured meshes
- **Training fixes**: Pressure weight (3.0), lower LR (3e-4), pressure reference constraint, and optional curriculum training improve stability
- **Line plots**: Use `plot_lines.py` to extract and visualize field values along specific lines for detailed analysis

## Troubleshooting

### Graph Connectivity Issues
If you encounter errors related to graph connectivity or message passing:
- The framework automatically validates and fixes edge indices
- Isolated nodes are automatically handled with self-loops
- Check that your mesh data is properly loaded (use `test_data_loading.py`)

### Common Issues
- **"edge_index must have shape [2, num_edges]"**: This is automatically validated and fixed
- **Isolated nodes**: Automatically handled by adding self-loops
- **Invalid edge indices**: Automatically filtered out during graph construction

## Recent Improvements

### Graph Connectivity & Message Passing
- ✅ Fixed graph connectivity issues with proper edge index remapping
- ✅ Added automatic validation of edge indices and node connectivity
- ✅ Implemented robust message passing with error handling
- ✅ Automatic handling of isolated nodes with self-loops
- ✅ Enhanced error messages for debugging connectivity issues

### Normalization & Training
- ✅ **Per-component velocity normalization**: Ux, Uy, Uz normalized separately for better accuracy
- ✅ **Field-wise loss computation**: Improved loss function that computes loss per field, ensuring balanced training
- ✅ **Pressure weight = 3.0**: Critical fix to prevent pressure under-training and drift
- ✅ **Reduced learning rate (3e-4)**: Optimized for CFD-GNNs, reduces early instability
- ✅ **Pressure reference constraint**: Anchors absolute pressure level to prevent global drift
- ✅ **Curriculum training**: Optional two-phase training (freeze pressure in Phase 1)
- ✅ **Training history tracking**: Automatic saving of training metrics for analysis
- ✅ **Training curve plotting**: New script to visualize training progress and loss curves

### Visualization
- ✅ **Triangulation-based plotting**: Direct use of mesh structure for more accurate visualizations
- ✅ **2D collapse function**: Automatic averaging of 3D data to 2D for cleaner plots
- ✅ **Improved error metrics**: Normalized error calculation using field range instead of individual values
- ✅ **Diagnostic output**: Detailed error statistics printed during visualization
- ✅ **Line plots**: New script to plot velocity and pressure along specific lines (horizontal/vertical)

## Future Enhancements

- [ ] Support for boundary condition encoding
- [ ] Multi-mesh training with data augmentation
- [ ] Uncertainty quantification
- [ ] Transfer learning for different geometries
- [ ] Integration with OpenFOAM for online prediction

## License

[Add your license here]

## Citation

If you use this code, please cite:
```
[Add citation]
```

