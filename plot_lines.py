"""
Plot velocity and pressure along specific lines (Y = constant, X = constant).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from openfoam_loader import OpenFOAMLoader
from graph_constructor import GraphConstructor
from gnn_model import FlowGNN
from normalization import FieldNormalizer
import pickle


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {
            'hidden_dim': 256,
            'num_layers': 6,
            'layer_type': 'GCN'
        }
    
    model = FlowGNN(
        input_dim=3,
        hidden_dim=config.get('hidden_dim', 256),
        output_dim=7,
        num_layers=config.get('num_layers', 6),
        layer_type=config.get('layer_type', 'GCN'),
        use_edge_attr=True,
        dropout=0.0,
        use_batch_norm=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    # Load normalizer if available
    normalizer = None
    if 'normalizer' in checkpoint and checkpoint['normalizer'] is not None:
        normalizer = FieldNormalizer()
        normalizer.field_stats = checkpoint['normalizer']['field_stats']
        normalizer.scalers = checkpoint['normalizer']['scalers']
    
    return model, config, normalizer


def predict_fields(model, graph_data, device='cpu', normalizer=None):
    """Predict flow fields from graph data."""
    model.eval()
    
    with torch.no_grad():
        x = graph_data.x.to(device)
        edge_index = graph_data.edge_index.to(device)
        edge_attr = graph_data.edge_attr.to(device)
        
        output = model(x, edge_index, edge_attr)
        fields = model.predict_fields(output)
        
        # Convert to numpy
        fields_numpy = {}
        for key in fields:
            fields_numpy[key] = fields[key].cpu().numpy()
        
        # Denormalize if normalizer available
        if normalizer is not None:
            fields_numpy = normalizer.inverse_transform(fields_numpy)
        
        return fields_numpy


def extract_line_data(cell_centers, field_values, x_line=None, y_line=None, tol=1e-4):
    """
    Extract field values along a line.
    
    Args:
        cell_centers: [n_cells, 3] array of cell center coordinates
        field_values: [n_cells] or [n_cells, 3] array of field values
        x_line: X coordinate for vertical line (if None, use y_line for horizontal)
        y_line: Y coordinate for horizontal line (if None, use x_line for vertical)
        tol: Tolerance for line extraction
    
    Returns:
        positions: Array of positions along the line
        values: Field values along the line
    """
    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    z = cell_centers[:, 2]
    
    if x_line is not None:
        # Vertical line at X = x_line
        mask = np.abs(x - x_line) < tol
        if np.sum(mask) == 0:
            # If no exact match, find closest points
            distances = np.abs(x - x_line)
            mask = distances < (np.min(distances) + tol)
        
        positions = y[mask]
        values = field_values[mask] if len(field_values.shape) == 1 else field_values[mask]
        
        # Sort by Y
        sort_idx = np.argsort(positions)
        positions = positions[sort_idx]
        values = values[sort_idx] if len(field_values.shape) == 1 else values[sort_idx]
        
        return positions, values
    
    elif y_line is not None:
        # Horizontal line at Y = y_line
        mask = np.abs(y - y_line) < tol
        if np.sum(mask) == 0:
            # If no exact match, find closest points
            distances = np.abs(y - y_line)
            mask = distances < (np.min(distances) + tol)
        
        positions = x[mask]
        values = field_values[mask] if len(field_values.shape) == 1 else field_values[mask]
        
        # Sort by X
        sort_idx = np.argsort(positions)
        positions = positions[sort_idx]
        values = values[sort_idx] if len(field_values.shape) == 1 else values[sort_idx]
        
        return positions, values
    
    else:
        raise ValueError("Either x_line or y_line must be specified")


def plot_line_comparison(predicted_fields, reference_fields, cell_centers, 
                        x_line=None, y_line=None, output_path=None, tol=1e-4):
    """
    Plot velocity and pressure along specified lines.
    
    Args:
        predicted_fields: Dictionary of predicted fields
        reference_fields: Dictionary of reference fields
        cell_centers: [n_cells, 3] array of cell centers
        x_line: X coordinate for vertical line
        y_line: Y coordinate for horizontal line
        output_path: Path to save plot
        tol: Tolerance for line extraction
    """
    # Extract velocity magnitude
    pred_U = predicted_fields['U']
    ref_U = reference_fields['U']
    
    if len(pred_U.shape) > 1:
        pred_U_mag = np.linalg.norm(pred_U, axis=1)
    else:
        pred_U_mag = pred_U
    
    if len(ref_U.shape) > 1:
        ref_U_mag = np.linalg.norm(ref_U, axis=1)
    else:
        ref_U_mag = ref_U
    
    # Extract pressure
    pred_p = predicted_fields['p'].flatten() if len(predicted_fields['p'].shape) > 1 else predicted_fields['p']
    ref_p = reference_fields['p'].flatten() if len(reference_fields['p'].shape) > 1 else reference_fields['p']
    
    # Determine line type and label
    if x_line is not None:
        line_type = 'vertical'
        line_label = f'X = {x_line:.3f}'
        position_label = 'Y [m]'
    else:
        line_type = 'horizontal'
        line_label = f'Y = {y_line:.3f}'
        position_label = 'X [m]'
    
    # Extract data along line
    pred_pos_U, pred_U_line = extract_line_data(cell_centers, pred_U_mag, x_line=x_line, y_line=y_line, tol=tol)
    ref_pos_U, ref_U_line = extract_line_data(cell_centers, ref_U_mag, x_line=x_line, y_line=y_line, tol=tol)
    
    pred_pos_p, pred_p_line = extract_line_data(cell_centers, pred_p, x_line=x_line, y_line=y_line, tol=tol)
    ref_pos_p, ref_p_line = extract_line_data(cell_centers, ref_p, x_line=x_line, y_line=y_line, tol=tol)
    
    # Create figure with two subplots - larger figure size
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot velocity
    ax1 = axes[0]
    ax1.plot(pred_pos_U, pred_U_line, 'b-', label='Predicted', linewidth=2.5, marker='o', markersize=5)
    ax1.plot(ref_pos_U, ref_U_line, 'r--', label='Reference', linewidth=2.5, marker='s', markersize=5)
    ax1.set_xlabel(position_label, fontsize=14)
    ax1.set_ylabel('Velocity Magnitude [m/s]', fontsize=14)
    ax1.set_title(f'Velocity along {line_label}', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)
    
    # Plot pressure
    ax2 = axes[1]
    ax2.plot(pred_pos_p, pred_p_line, 'b-', label='Predicted', linewidth=2.5, marker='o', markersize=5)
    ax2.plot(ref_pos_p, ref_p_line, 'r--', label='Reference', linewidth=2.5, marker='s', markersize=5)
    ax2.set_xlabel(position_label, fontsize=14)
    ax2.set_ylabel('Pressure [m²/s²]', fontsize=14)
    ax2.set_title(f'Pressure along {line_label}', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved line plot to {output_path}")
    
    plt.close()
    
    # Print statistics
    print(f"\n{line_label} Statistics:")
    print(f"  Velocity - Predicted range: [{pred_U_line.min():.6e}, {pred_U_line.max():.6e}]")
    print(f"  Velocity - Reference range: [{ref_U_line.min():.6e}, {ref_U_line.max():.6e}]")
    print(f"  Velocity - Mean absolute error: {np.mean(np.abs(pred_U_line - ref_U_line)):.6e}")
    print(f"  Pressure - Predicted range: [{pred_p_line.min():.6e}, {pred_p_line.max():.6e}]")
    print(f"  Pressure - Reference range: [{ref_p_line.min():.6e}, {ref_p_line.max():.6e}]")
    print(f"  Pressure - Mean absolute error: {np.mean(np.abs(pred_p_line - ref_p_line)):.6e}")


def main():
    parser = argparse.ArgumentParser(description='Plot velocity and pressure along specific lines')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--case_path', type=str, default='OpenFOAM-data',
                       help='Path to OpenFOAM case directory')
    parser.add_argument('--reference_time', type=str, default='282',
                       help='Time directory for reference comparison')
    parser.add_argument('--x_line', type=float, default=0.15,
                       help='X coordinate for vertical line (default: 0.15)')
    parser.add_argument('--y_line', type=float, default=0.005,
                       help='Y coordinate for horizontal line (default: 0.005)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save plots')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--tol', type=float, default=1e-4,
                       help='Tolerance for line extraction (default: 1e-4)')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, config, normalizer = load_model(args.checkpoint, args.device)
    if normalizer is not None:
        print("Normalizer found - will denormalize predictions")
    
    # Load mesh and construct graph
    print("Loading mesh and constructing graph...")
    loader = OpenFOAMLoader(args.case_path)
    mesh_data = loader.load_mesh()
    graph_constructor = GraphConstructor(mesh_data)
    
    # Get number of internal cells from reference
    ref_fields = loader.load_fields(args.reference_time)
    n_internal = len(ref_fields['p'])
    
    # Build graph
    graph = graph_constructor.build_graph(
        node_features=mesh_data['cell_centers'],
        filter_internal=True,
        n_internal_cells=n_internal
    )
    
    # Run prediction
    print("Running prediction...")
    predicted_fields = predict_fields(model, graph, args.device, normalizer)
    
    # Load reference fields
    print("Loading reference fields...")
    reference_fields = loader.load_fields(args.reference_time)
    
    # Convert reference to same format
    ref_dict = {
        'U': reference_fields['U'],
        'p': reference_fields['p'].reshape(-1, 1) if len(reference_fields['p'].shape) == 1 else reference_fields['p'],
    }
    
    # Get cell centers for internal cells
    cell_centers_all = mesh_data['cell_centers'][:n_internal]
    
    # Filter to z >= 0 (positive z side)
    z_mask = cell_centers_all[:, 2] >= 0
    n_filtered = np.sum(z_mask)
    
    print(f"Filtering to z >= 0: {n_filtered} cells (from {n_internal} total)")
    
    if n_filtered == 0:
        print("Warning: No cells found with z >= 0, using all cells")
        cell_centers = cell_centers_all
    else:
        cell_centers = cell_centers_all[z_mask]
        for field_name in predicted_fields:
            if len(predicted_fields[field_name].shape) > 1:
                predicted_fields[field_name] = predicted_fields[field_name][z_mask]
            else:
                predicted_fields[field_name] = predicted_fields[field_name][z_mask]
        
        for field_name in ref_dict:
            if len(ref_dict[field_name].shape) > 1:
                ref_dict[field_name] = ref_dict[field_name][z_mask]
            else:
                ref_dict[field_name] = ref_dict[field_name][z_mask]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot horizontal line (Y = constant)
    print(f"\nPlotting along horizontal line Y = {args.y_line}...")
    output_path_h = output_dir / f'line_Y_{args.y_line:.3f}.png'
    plot_line_comparison(
        predicted_fields, ref_dict, cell_centers,
        y_line=args.y_line, output_path=output_path_h, tol=args.tol
    )
    
    # Plot vertical line (X = constant)
    print(f"\nPlotting along vertical line X = {args.x_line}...")
    output_path_v = output_dir / f'line_X_{args.x_line:.3f}.png'
    plot_line_comparison(
        predicted_fields, ref_dict, cell_centers,
        x_line=args.x_line, output_path=output_path_v, tol=args.tol
    )
    
    print(f"\nLine plots saved to {output_dir}")


if __name__ == '__main__':
    main()
