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


def compare_fields(predicted_fields, reference_fields, cell_centers, output_dir):
    """Create comparison plots for predicted vs reference fields."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Field names and their properties
    field_configs = {
        'U': {'name': 'Velocity Magnitude', 'cmap': 'plasma', 'unit': 'm/s'},
        'p': {'name': 'Pressure', 'cmap': 'RdBu_r', 'unit': 'm²/s²'},
        'k': {'name': 'Turbulent Kinetic Energy', 'cmap': 'viridis', 'unit': 'm²/s²'},
        'epsilon': {'name': 'Dissipation Rate', 'cmap': 'hot', 'unit': 'm²/s³'},
        'nut': {'name': 'Turbulent Viscosity', 'cmap': 'coolwarm', 'unit': 'm²/s'},
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
        
        # Create comparison figure - stacked vertically
        fig, axes = plt.subplots(3, 1, figsize=(10, 18))
        
        # Use griddata with Delaunay domain masking (from sample project)
        x = cell_centers[:, 0]
        y = cell_centers[:, 1]
        
        # Create regular grid - higher resolution
        nx, ny = 500, 500
        xi = np.linspace(x.min(), x.max(), nx)
        yi = np.linspace(y.min(), y.max(), ny)
        Xi, Yi = np.meshgrid(xi, yi)
        
        from scipy.interpolate import griddata
        from scipy.spatial import Delaunay
        
        # Interpolate to grid
        Zi_pred = griddata((x, y), pred_mag, (Xi, Yi), method='linear', fill_value=np.nan)
        Zi_ref = griddata((x, y), ref_mag, (Xi, Yi), method='linear', fill_value=np.nan)
        
        # Calculate percentage error: (|pred - ref| / |ref|) * 100
        # Avoid division by zero
        ref_magnitude = np.abs(ref_mag)
        threshold = np.max(ref_magnitude) * 1e-6  # Small threshold to avoid division by zero
        error_percent = np.where(
            ref_magnitude > threshold,
            (np.abs(pred_mag - ref_mag) / ref_magnitude) * 100,
            np.zeros_like(ref_mag)
        )
        # Clip error to maximum 5% for better visualization
        error_percent = np.clip(error_percent, 0, 5.0)
        Zi_err = griddata((x, y), error_percent, (Xi, Yi), method='linear', fill_value=np.nan)
        
        # Domain masking with Delaunay triangulation (key fix from sample project)
        try:
            points_2d = np.column_stack([x, y])
            tri = Delaunay(points_2d)
            grid_points = np.column_stack([Xi.ravel(), Yi.ravel()])
            mask = tri.find_simplex(grid_points) >= 0
            mask = mask.reshape(Xi.shape)
            
            # Mask points outside domain
            Zi_pred[~mask] = np.nan
            Zi_ref[~mask] = np.nan
            Zi_err[~mask] = np.nan
        except:
            pass  # If Delaunay fails, continue without masking
        
        # Common scale for predicted and reference
        vmin = min(np.nanmin(Zi_pred), np.nanmin(Zi_ref))
        vmax = max(np.nanmax(Zi_pred), np.nanmax(Zi_ref))
        levels = np.linspace(vmin, vmax, 50)  # More contour levels for smoother appearance
        
        # Predicted field - top
        im1 = axes[0].contourf(Xi, Yi, Zi_pred, levels=levels, vmin=vmin, vmax=vmax, 
                               cmap=config['cmap'], extend='neither')
        axes[0].set_title(f'Predicted {config["name"]}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X [m]', fontsize=12)
        axes[0].set_ylabel('Y [m]', fontsize=12)
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
        # Colorbar without extend arrows (no pointy legends)
        cbar1 = plt.colorbar(im1, ax=axes[0], label=config['unit'], fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=10)
        
        # Reference - middle
        im2 = axes[1].contourf(Xi, Yi, Zi_ref, levels=levels, vmin=vmin, vmax=vmax, 
                               cmap=config['cmap'], extend='neither')
        axes[1].set_title(f'Reference {config["name"]}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X [m]', fontsize=12)
        axes[1].set_ylabel('Y [m]', fontsize=12)
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        # Colorbar without extend arrows
        cbar2 = plt.colorbar(im2, ax=axes[1], label=config['unit'], fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=10)
        
        # Error - bottom (percentage error, capped at 5%)
        error_levels = np.linspace(0, 5.0, 50)  # Fixed range 0-5%
        im3 = axes[2].contourf(Xi, Yi, Zi_err, levels=error_levels, vmin=0, vmax=5.0, 
                               cmap='hot', extend='max')  # Show arrow for values > 5%
        axes[2].set_title(f'Percentage Error: |Predicted - Reference| / |Reference| × 100% (capped at 5%)', 
                         fontsize=14, fontweight='bold')
        axes[2].set_xlabel('X [m]', fontsize=12)
        axes[2].set_ylabel('Y [m]', fontsize=12)
        axes[2].set_aspect('equal')
        axes[2].grid(True, alpha=0.3)
        # Colorbar with percentage label, fixed range 0-5%
        cbar3 = plt.colorbar(im3, ax=axes[2], label='Error [%]', fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=10)
        cbar3.set_ticks(np.linspace(0, 5, 6))  # Show ticks at 0, 1, 2, 3, 4, 5%
        
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

