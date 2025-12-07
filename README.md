# GNN-Based RANS Flow Field Surrogate Simulator

This repository contains a Graph Neural Network (GNN) framework for training a surrogate simulator based on OpenFOAM steady RANS simulation data. The framework learns to predict flow fields (velocity, pressure, and turbulence quantities) directly from mesh geometry.

## Overview

The framework consists of:
- **OpenFOAM Data Loader**: Parses OpenFOAM mesh and field files
- **Graph Constructor**: Builds graph structure from mesh connectivity
- **GNN Model**: Graph Neural Network for flow field prediction
- **Training Script**: End-to-end training pipeline
- **Inference Script**: Prediction and evaluation tools

## Features

- Supports multiple GNN architectures (GCN, GAT, GIN, Transformer)
- Predicts velocity (U), pressure (p), and turbulence fields (k, epsilon, nut)
- Handles unstructured meshes via cell-to-cell connectivity
- Can save predictions in both NumPy and OpenFOAM formats
- Includes evaluation metrics for field-wise error analysis

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

### Visualization

Visualize predicted flow fields and compare with reference data:

```bash
python visualize.py \
    --checkpoint checkpoints/best_model.pt \
    --case_path OpenFOAM-data \
    --reference_time 282 \
    --output_dir visualizations
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint
- `--case_path`: Path to OpenFOAM case directory
- `--reference_time`: Time directory for reference comparison (e.g., "282")
- `--output_dir`: Directory to save visualization plots
- `--device`: Device to use (default: 'cuda' if available, else 'cpu')

**Output:**
- Creates contour plots for each field (U, p, k, epsilon, nut)
- Each plot shows: Predicted | Reference | Absolute Error
- Saved as PNG images in the output directory

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
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 1, typically for single mesh)

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

## Model Architecture

The GNN model (`FlowGNN`) consists of:
1. **Input Projection**: Maps cell center coordinates to hidden dimension
2. **GNN Layers**: Stack of graph convolution layers with residual connections
3. **Output Projection**: Maps hidden features to flow field predictions

The model predicts:
- Velocity vector U: [n_cells, 3]
- Pressure p: [n_cells, 1]
- Turbulent kinetic energy k: [n_cells, 1]
- Dissipation rate epsilon: [n_cells, 1]
- Turbulent viscosity nut: [n_cells, 1]

## Graph Construction

The graph is constructed from OpenFOAM mesh connectivity:
- **Nodes**: Mesh cells (represented by cell centers)
- **Edges**: Cell-to-cell connections via shared faces
- **Node Features**: Cell center coordinates (x, y, z)
- **Edge Features**: Direction vector and distance between cell centers

## Training Details

- **Loss Function**: Mean Squared Error (MSE) over all predicted fields
- **Optimizer**: Adam with learning rate scheduling
- **Regularization**: Gradient clipping, dropout, batch normalization
- **Evaluation**: Per-field error metrics (MAE, RMSE, relative error)

## Output

### Training Output
- Model checkpoints saved in `checkpoints/`:
  - `best_model.pt`: Best model based on validation loss
  - `checkpoint_epoch_N.pt`: Periodic checkpoints
  - `config.json`: Training configuration

### Inference Output
- Predictions saved as:
  - `predictions.npz`: NumPy format (if `save_format` includes 'numpy')
  - `predictions/predicted/`: OpenFOAM format field files (if `save_format` includes 'openfoam')
- Field comparison metrics printed to console (if reference provided)

### Visualization
- Contour plots comparing predicted vs reference fields
- Error maps showing absolute differences
- Saved as PNG images in the visualization directory

## Example Workflow

1. **Prepare Data**: Ensure your OpenFOAM case data is in the correct structure
2. **Train Model**:
   ```bash
   python train.py --case_path OpenFOAM-data --time_dirs 0 100 200 282 --epochs 100
   ```
3. **Evaluate Model**:
   ```bash
   python inference.py --checkpoint checkpoints/best_model.pt --reference_time 282
   ```
4. **Visualize Results**:
   ```bash
   python visualize.py --checkpoint checkpoints/best_model.pt --reference_time 282 --output_dir visualizations
   ```
   This creates contour plots comparing predicted vs reference fields for:
   - Velocity magnitude
   - Pressure
   - Turbulent kinetic energy (k)
   - Dissipation rate (epsilon)
   - Turbulent viscosity (nut)
5. **Use Predictions**: Load predictions from `predictions/` directory for further analysis

## Notes

- The current implementation uses cell center coordinates as input features
- Boundary conditions are not explicitly encoded (can be added as future enhancement)
- The model learns steady-state flow fields from geometry
- For best results, use converged solution data (e.g., final time step) for training

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

