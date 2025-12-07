"""
Visualization script for GNN predictions and OpenFOAM results.
Creates contour plots for velocity, pressure, and turbulence fields.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import argparse
import json
from collections import defaultdict

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
        normalizer.scalers = checkpoint['normalizer']['scalers']  # Already in correct format
    
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


def create_2d_contour_plot(cell_centers, field_values, field_name, title, 
                           levels=20, cmap='viridis', output_path=None):
    """
    Create 2D contour plot from cell-centered data.
    
    Args:
        cell_centers: [n_cells, 3] array of cell center coordinates
        field_values: [n_cells] or [n_cells, 3] array of field values
        field_name: Name of the field
        title: Plot title
        levels: Number of contour levels
        cmap: Colormap
        output_path: Path to save figure
    """
    # Extract 2D coordinates (assuming z is constant for 2D case)
    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    
    # Handle vector fields (take magnitude)
    if len(field_values.shape) > 1 and field_values.shape[1] > 1:
        if field_values.shape[1] == 3:
            # Velocity magnitude
            field_values = np.linalg.norm(field_values, axis=1)
        else:
            field_values = field_values[:, 0]
    
    # Create triangulation for contour plot
    from matplotlib.tri import Triangulation
    
    # For unstructured mesh, use triangulation
    # Simple approach: create grid-based interpolation
    # Get bounding box
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Create regular grid
    nx, ny = 200, 200
    xi = np.linspace(x_min, x_max, nx)
    yi = np.linspace(y_min, y_max, ny)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate to grid using nearest neighbor or linear interpolation
    from scipy.interpolate import griddata
    Zi = griddata((x, y), field_values, (Xi, Yi), method='linear', fill_value=np.nan)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Contour plot
    if field_name in ['p']:
        # Use symmetric colormap for pressure
        vmin, vmax = field_values.min(), field_values.max()
        vcenter = (vmin + vmax) / 2
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        contour = ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap, norm=norm)
    else:
        contour = ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(field_name, fontsize=12)
    
    # Scatter plot of actual cell centers for reference
    ax.scatter(x, y, c='k', s=0.1, alpha=0.3)
    
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    return fig, ax


def collapse_to_2d(cell_centers, field, tol=1e-6):
    """
    Collapse extruded 3D cell-centered data into true 2D data
    by clustering points in (x,y) with tolerance.
    """
    bins = {}
    
    for (x, y, z), val in zip(cell_centers, field):
        key = (round(x / tol), round(y / tol))
        if key not in bins:
            bins[key] = {'x': [], 'y': [], 'val': []}
        bins[key]['x'].append(x)
        bins[key]['y'].append(y)
        bins[key]['val'].append(val)
    
    x2d = np.array([np.mean(v['x']) for v in bins.values()])
    y2d = np.array([np.mean(v['y']) for v in bins.values()])
    v2d = np.array([np.mean(v['val']) for v in bins.values()])
    
    return x2d, y2d, v2d


def compare_fields(predicted_fields, reference_fields, cell_centers, output_dir):
    """Create comparison plots for predicted vs reference fields."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Field names and their properties - all using diverging blue-white-red colormap
    field_configs = {
        'U': {'name': 'Velocity Magnitude', 'cmap': 'RdBu_r', 'unit': 'm/s'},
        'p': {'name': 'Pressure', 'cmap': 'RdBu_r', 'unit': 'm²/s²'},
        'k': {'name': 'Turbulent Kinetic Energy', 'cmap': 'RdBu_r', 'unit': 'm²/s²'},
        'epsilon': {'name': 'Dissipation Rate', 'cmap': 'RdBu_r', 'unit': 'm²/s³'},
        'nut': {'name': 'Turbulent Viscosity', 'cmap': 'RdBu_r', 'unit': 'm²/s'},
    }
    
    for field_name in ['U', 'p', 'k', 'epsilon', 'nut']:
        if field_name not in predicted_fields or field_name not in reference_fields:
            continue
        
        pred = predicted_fields[field_name]
        ref = reference_fields[field_name]
        
        # Handle vector fields
        if field_name == 'U':
            pred_mag = np.linalg.norm(pred, axis=1) if len(pred.shape) > 1 else pred
            ref_mag = np.linalg.norm(ref, axis=1) if len(ref.shape) > 1 else ref
        else:
            pred_mag = pred.flatten() if len(pred.shape) > 1 else pred
            ref_mag = ref.flatten() if len(ref.shape) > 1 else ref
        
        config = field_configs[field_name]
        
        # Create comparison figure - stacked vertically (larger figure)
        fig, axes = plt.subplots(3, 1, figsize=(12, 20))
        
        # Collapse to 2D by averaging values at same (x, y) coordinates
        x, y, pred_mag_2d = collapse_to_2d(cell_centers, pred_mag)
        _, _, ref_mag_2d = collapse_to_2d(cell_centers, ref_mag)
        
        from matplotlib.tri import Triangulation
        
        # Create Delaunay triangulation from collapsed 2D cell centers
        try:
            tri = Triangulation(x, y)
        except:
            # Fallback: create triangulation manually if automatic fails
            from scipy.spatial import Delaunay
            points_2d = np.column_stack([x, y])
            delaunay_tri = Delaunay(points_2d)
            tri = Triangulation(x, y, delaunay_tri.simplices)
        
        # Calculate normalized error: |pred - ref| / range(ref) * 100
        # This normalizes by the range (max - min) of the reference field
        # This provides a consistent scale and avoids high percentage errors for small individual values
        ref_max = np.nanmax(ref_mag_2d)
        ref_min = np.nanmin(ref_mag_2d)
        ref_range = ref_max - ref_min
        
        # If range is very small, use max absolute value as fallback
        if ref_range < 1e-10:
            ref_scale = max(np.abs(ref_max), np.abs(ref_min))
        else:
            ref_scale = ref_range
        
        # Add small epsilon to avoid division by zero
        eps = max(ref_scale * 1e-6, 1e-10)
        
        if ref_scale > eps:
            # Normalize by the scale of the reference field
            error_normalized = (np.abs(pred_mag_2d - ref_mag_2d) / (ref_scale + eps)) * 100
        else:
            # Fallback: use absolute error if reference is essentially zero everywhere
            error_normalized = np.abs(pred_mag_2d - ref_mag_2d) * 100
        
        # Clip error to maximum 10% for better visualization
        error_normalized = np.clip(error_normalized, 0, 10.0)
        
        # Print diagnostic information
        abs_error = np.abs(pred_mag_2d - ref_mag_2d)
        mean_abs_error = np.mean(abs_error)
        max_abs_error = np.max(abs_error)
        mean_error_pct = np.mean(error_normalized)
        max_error_pct = np.max(error_normalized)
        print(f"  {field_name} Error Stats:")
        print(f"    Mean absolute error: {mean_abs_error:.6e}")
        print(f"    Max absolute error: {max_abs_error:.6e}")
        print(f"    Reference scale (range): {ref_scale:.6e} (min: {ref_min:.6e}, max: {ref_max:.6e})")
        print(f"    Mean normalized error: {mean_error_pct:.2f}%")
        print(f"    Max normalized error: {max_error_pct:.2f}%")
        
        # Common scale for predicted and reference
        vmin = min(np.nanmin(pred_mag_2d), np.nanmin(ref_mag_2d))
        vmax = max(np.nanmax(pred_mag_2d), np.nanmax(ref_mag_2d))
        levels = np.linspace(vmin, vmax, 50)  # More contour levels for smoother appearance
        
        # Predicted field - top (using triangulation)
        im1 = axes[0].tricontourf(tri, pred_mag_2d, levels=levels, vmin=vmin, vmax=vmax, 
                                  cmap=config['cmap'], extend='neither')
        axes[0].set_title(f'Predicted {config["name"]}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X [m]', fontsize=12)
        axes[0].set_ylabel('Y [m]', fontsize=12)
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
        # Colorbar without extend arrows (smaller size)
        cbar1 = plt.colorbar(im1, ax=axes[0], label=config['unit'], fraction=0.035, pad=0.02)
        cbar1.ax.tick_params(labelsize=9)
        cbar1.ax.yaxis.label.set_size(10)
        
        # Reference - middle (using triangulation)
        im2 = axes[1].tricontourf(tri, ref_mag_2d, levels=levels, vmin=vmin, vmax=vmax, 
                                  cmap=config['cmap'], extend='neither')
        axes[1].set_title(f'Reference {config["name"]}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X [m]', fontsize=12)
        axes[1].set_ylabel('Y [m]', fontsize=12)
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        # Colorbar without extend arrows (smaller size)
        cbar2 = plt.colorbar(im2, ax=axes[1], label=config['unit'], fraction=0.035, pad=0.02)
        cbar2.ax.tick_params(labelsize=9)
        cbar2.ax.yaxis.label.set_size(10)
        
        # Error - bottom (normalized error, capped at 10%, using triangulation)
        error_levels = np.linspace(0, 10.0, 50)  # Fixed range 0-10%
        im3 = axes[2].tricontourf(tri, error_normalized, levels=error_levels, vmin=0, vmax=10.0, 
                                  cmap='RdBu_r', extend='neither')  # No extend arrows
        axes[2].set_title(f'Normalized Error: |Predicted - Reference| / Range(Reference) × 100% (capped at 10%)', 
                         fontsize=14, fontweight='bold')
        axes[2].set_xlabel('X [m]', fontsize=12)
        axes[2].set_ylabel('Y [m]', fontsize=12)
        axes[2].set_aspect('equal')
        axes[2].grid(True, alpha=0.3)
        # Colorbar with percentage label, fixed range 0-10% (smaller size)
        cbar3 = plt.colorbar(im3, ax=axes[2], label='Error [%]', fraction=0.035, pad=0.02)
        cbar3.ax.tick_params(labelsize=9)
        cbar3.ax.yaxis.label.set_size(10)
        cbar3.set_ticks(np.linspace(0, 10, 11))  # Show ticks at 0, 1, 2, ..., 10%
        
        plt.tight_layout()
        output_path = output_dir / f'{field_name}_comparison.png'
        plt.savefig(output_path, dpi=400, bbox_inches='tight')
        print(f"Saved comparison plot: {output_path}")
        plt.close()


def visualize_predictions(checkpoint_path, case_path, reference_time, output_dir, device='cpu'):
    """Main visualization function."""
    print("Loading model...")
    model, config, normalizer = load_model(checkpoint_path, device)
    if normalizer is not None:
        print("Normalizer found - will denormalize predictions")
    
    print("Loading mesh and constructing graph...")
    loader = OpenFOAMLoader(case_path)
    mesh_data = loader.load_mesh()
    graph_constructor = GraphConstructor(mesh_data)
    
    # Get number of internal cells from reference
    ref_fields = loader.load_fields(reference_time)
    n_internal = len(ref_fields['p'])
    
    # Build graph
    graph = graph_constructor.build_graph(
        node_features=mesh_data['cell_centers'],
        filter_internal=True,
        n_internal_cells=n_internal
    )
    
    print("Running prediction...")
    predicted_fields = predict_fields(model, graph, device, normalizer)
    
    print("Loading reference fields...")
    reference_fields = loader.load_fields(reference_time)
    
    # Convert reference to same format
    ref_dict = {
        'U': reference_fields['U'],
        'p': reference_fields['p'].reshape(-1, 1) if len(reference_fields['p'].shape) == 1 else reference_fields['p'],
        'k': reference_fields['k'].reshape(-1, 1) if len(reference_fields['k'].shape) == 1 else reference_fields['k'],
        'epsilon': reference_fields['epsilon'].reshape(-1, 1) if len(reference_fields['epsilon'].shape) == 1 else reference_fields['epsilon'],
        'nut': reference_fields['nut'].reshape(-1, 1) if len(reference_fields['nut'].shape) == 1 else reference_fields['nut'],
    }
    
    # Get cell centers for internal cells
    cell_centers_all = mesh_data['cell_centers'][:n_internal]
    
    # Filter to z >= 0 (positive z side, corresponding to z = 0.0005 in polyMesh points)
    # Since cell centers are averaged from points at z = -0.0005 and z = 0.0005,
    # cells with z >= 0 are closer to the z = 0.0005 plane
    z_mask = cell_centers_all[:, 2] >= 0
    n_filtered = np.sum(z_mask)
    
    print(f"Filtering to z >= 0 (positive z side, corresponding to z = 0.0005 in points): {n_filtered} cells (from {n_internal} total)")
    
    if n_filtered == 0:
        print("Warning: No cells found with z > 0, using all cells")
        cell_centers = cell_centers_all
    else:
        # Filter cell centers and fields
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
    
    print("Creating visualization plots...")
    compare_fields(predicted_fields, ref_dict, cell_centers, output_dir)
    
    print(f"\nVisualization complete! Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize GNN predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--case_path', type=str, default='OpenFOAM-data',
                       help='Path to OpenFOAM case directory')
    parser.add_argument('--reference_time', type=str, default='282',
                       help='Time directory for reference comparison')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save plots')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    visualize_predictions(
        args.checkpoint,
        args.case_path,
        args.reference_time,
        args.output_dir,
        args.device
    )


if __name__ == '__main__':
    main()

