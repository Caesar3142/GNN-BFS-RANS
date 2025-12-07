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


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {
            'hidden_dim': 128,
            'num_layers': 4,
            'layer_type': 'GCN'
        }
    
    model = FlowGNN(
        input_dim=3,
        hidden_dim=config.get('hidden_dim', 128),
        output_dim=7,
        num_layers=config.get('num_layers', 4),
        layer_type=config.get('layer_type', 'GCN'),
        use_edge_attr=True,
        dropout=0.0,
        use_batch_norm=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    return model, config


def predict_fields(model, graph_data, device='cpu'):
    """Predict flow fields from graph data."""
    model.eval()
    
    with torch.no_grad():
        x = graph_data.x.to(device)
        edge_index = graph_data.edge_index.to(device)
        edge_attr = graph_data.edge_attr.to(device)
        
        output = model(x, edge_index, edge_attr)
        fields = model.predict_fields(output)
        
        for key in fields:
            fields[key] = fields[key].cpu().numpy()
    
    return fields


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
        
        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Use triangulation-based contouring for unstructured mesh (removes artifacts)
        x = cell_centers[:, 0]
        y = cell_centers[:, 1]
        
        from matplotlib.tri import Triangulation
        
        # Create triangulation from cell centers
        tri = Triangulation(x, y)
        
        # Predicted field - use tricontourf for smooth unstructured mesh visualization
        im1 = axes[0].tricontourf(tri, pred_mag, levels=20, cmap=config['cmap'], extend='both')
        axes[0].set_title(f'Predicted {config["name"]}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('X [m]')
        axes[0].set_ylabel('Y [m]')
        axes[0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0], label=config['unit'])
        
        # Reference - use tricontourf for smooth unstructured mesh visualization
        im2 = axes[1].tricontourf(tri, ref_mag, levels=20, cmap=config['cmap'], extend='both')
        axes[1].set_title(f'Reference {config["name"]}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('X [m]')
        axes[1].set_ylabel('Y [m]')
        axes[1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[1], label=config['unit'])
        
        # Error - use tricontourf for smooth unstructured mesh visualization
        error = np.abs(pred_mag - ref_mag)
        im3 = axes[2].tricontourf(tri, error, levels=20, cmap='hot', extend='both')
        axes[2].set_title(f'Absolute Error', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('X [m]')
        axes[2].set_ylabel('Y [m]')
        axes[2].set_aspect('equal')
        plt.colorbar(im3, ax=axes[2], label=config['unit'])
        
        plt.tight_layout()
        output_path = output_dir / f'{field_name}_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot: {output_path}")
        plt.close()


def visualize_predictions(checkpoint_path, case_path, reference_time, output_dir, device='cpu'):
    """Main visualization function."""
    print("Loading model...")
    model, config = load_model(checkpoint_path, device)
    
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
    predicted_fields = predict_fields(model, graph, device)
    
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
    cell_centers = mesh_data['cell_centers'][:n_internal]
    
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

