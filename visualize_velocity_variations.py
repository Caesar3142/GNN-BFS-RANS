"""
Visualize predicted flow fields for different velocity variations.
Creates contour plots for each velocity case and comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import argparse
import os
from scipy.interpolate import griddata
from matplotlib.tri import Triangulation

from openfoam_loader import OpenFOAMLoader


def load_predicted_fields(predictions_path: str):
    """
    Load predicted fields from NumPy file.
    
    Args:
        predictions_path: Path to predictions.npz file
    
    Returns:
        Dictionary of field arrays
    """
    data = np.load(predictions_path)
    fields = {}
    for key in data.files:
        fields[key] = data[key]
    return fields


def collapse_to_2d(cell_centers, field, tol=1e-6):
    """
    Collapse 3D field to 2D by averaging over z dimension.
    
    Args:
        cell_centers: [n_cells, 3] array of cell center coordinates
        field: [n_cells] or [n_cells, 3] array of field values
        tol: Tolerance for z-coordinate matching
    
    Returns:
        Tuple of (cell_centers_2d, field_2d)
    """
    # Get unique z values
    z_values = np.unique(cell_centers[:, 2])
    
    if len(z_values) <= 1:
        # Already 2D or single z-plane
        if len(field.shape) > 1 and field.shape[1] > 1:
            return cell_centers[:, :2], field
        return cell_centers[:, :2], field.flatten()
    
    # Filter to z >= 0 (positive z side)
    z_mask = cell_centers[:, 2] >= 0
    cell_centers_2d = cell_centers[z_mask, :2]
    
    if len(field.shape) > 1:
        field_2d = field[z_mask]
    else:
        field_2d = field[z_mask]
    
    return cell_centers_2d, field_2d


def create_contour_plot(cell_centers, field_values, field_name, title, 
                        output_path=None, cmap='viridis', levels=50):
    """
    Create 2D contour plot from cell-centered data using triangulation.
    
    Args:
        cell_centers: [n_cells, 2] array of 2D cell center coordinates
        field_values: [n_cells] array of field values
        field_name: Name of the field
        title: Plot title
        output_path: Path to save figure
        cmap: Colormap
        levels: Number of contour levels
    """
    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    
    # Create triangulation
    tri = Triangulation(x, y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Contour plot using triangulation
    if field_name in ['p']:
        # Use symmetric colormap for pressure
        vmin, vmax = field_values.min(), field_values.max()
        vcenter = (vmin + vmax) / 2
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        contour = ax.tricontourf(tri, field_values, levels=levels, cmap=cmap, norm=norm)
    else:
        contour = ax.tricontourf(tri, field_values, levels=levels, cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(field_name, fontsize=12)
    
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    
    plt.close(fig)


def visualize_velocity_variation(predictions_path: str, case_path: str, 
                                 velocity: float, output_dir: str):
    """
    Visualize predicted fields for a single velocity variation.
    
    Args:
        predictions_path: Path to predictions.npz file
        case_path: Path to OpenFOAM case directory (for mesh data)
        velocity: Inlet velocity value
        output_dir: Output directory for plots
    """
    print(f"\nVisualizing velocity = {velocity:.1f} m/s...")
    
    # Load predicted fields
    predicted_fields = load_predicted_fields(predictions_path)
    
    # Load mesh to get cell centers
    loader = OpenFOAMLoader(case_path)
    mesh_data = loader.load_mesh()
    cell_centers = mesh_data['cell_centers']
    
    # Get number of internal cells from predicted fields
    n_internal = predicted_fields['U'].shape[0]
    cell_centers_internal = cell_centers[:n_internal]
    
    # Collapse to 2D
    cell_centers_2d, _ = collapse_to_2d(cell_centers_internal, 
                                        cell_centers_internal[:, 0])
    
    # Create output directory
    vel_dir = os.path.join(output_dir, f"velocity_{velocity:.1f}")
    os.makedirs(vel_dir, exist_ok=True)
    
    # Field configurations
    field_configs = {
        'U': {
            'name': 'Velocity Magnitude',
            'cmap': 'viridis',
            'process': lambda f: np.linalg.norm(f, axis=1) if len(f.shape) > 1 else f
        },
        'p': {
            'name': 'Pressure',
            'cmap': 'RdBu_r',
            'process': lambda f: f.flatten() if len(f.shape) > 1 else f
        },
        'k': {
            'name': 'Turbulent Kinetic Energy',
            'cmap': 'plasma',
            'process': lambda f: f.flatten() if len(f.shape) > 1 else f
        },
        'epsilon': {
            'name': 'Dissipation Rate',
            'cmap': 'hot',
            'process': lambda f: f.flatten() if len(f.shape) > 1 else f
        },
        'nut': {
            'name': 'Turbulent Viscosity',
            'cmap': 'coolwarm',
            'process': lambda f: f.flatten() if len(f.shape) > 1 else f
        }
    }
    
    # Visualize each field
    for field_name, config in field_configs.items():
        if field_name not in predicted_fields:
            continue
        
        # Process field (get magnitude for vectors, flatten for scalars)
        field_2d = config['process'](predicted_fields[field_name])
        
        # Collapse to 2D
        _, field_2d = collapse_to_2d(cell_centers_internal, field_2d)
        
        # Create plot
        title = f"{config['name']} - Velocity = {velocity:.1f} m/s"
        output_path = os.path.join(vel_dir, f"{field_name}.png")
        
        create_contour_plot(
            cell_centers_2d, field_2d, field_name, title,
            output_path=output_path, cmap=config['cmap']
        )
        
        # Print statistics
        print(f"  {field_name}: min={field_2d.min():.4e}, max={field_2d.max():.4e}, "
              f"mean={field_2d.mean():.4e}")


def create_comparison_plot(predictions_paths: dict, case_path: str, 
                          field_name: str, output_dir: str):
    """
    Create comparison plot showing how a field changes with velocity.
    
    Args:
        predictions_paths: Dictionary mapping velocity to predictions path
        case_path: Path to OpenFOAM case directory
        field_name: Name of field to compare
        output_dir: Output directory for plots
    """
    # Load mesh
    loader = OpenFOAMLoader(case_path)
    mesh_data = loader.load_mesh()
    cell_centers = mesh_data['cell_centers']
    
    # Field configurations
    field_configs = {
        'U': {'name': 'Velocity Magnitude', 'cmap': 'viridis', 
              'process': lambda f: np.linalg.norm(f, axis=1) if len(f.shape) > 1 else f},
        'p': {'name': 'Pressure', 'cmap': 'RdBu_r',
              'process': lambda f: f.flatten() if len(f.shape) > 1 else f},
        'k': {'name': 'Turbulent Kinetic Energy', 'cmap': 'plasma',
              'process': lambda f: f.flatten() if len(f.shape) > 1 else f},
        'epsilon': {'name': 'Dissipation Rate', 'cmap': 'hot',
                    'process': lambda f: f.flatten() if len(f.shape) > 1 else f},
        'nut': {'name': 'Turbulent Viscosity', 'cmap': 'coolwarm',
                'process': lambda f: f.flatten() if len(f.shape) > 1 else f}
    }
    
    if field_name not in field_configs:
        return
    
    config = field_configs[field_name]
    velocities = sorted(predictions_paths.keys())
    n_velocities = len(velocities)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_velocities, figsize=(6*n_velocities, 6))
    if n_velocities == 1:
        axes = [axes]
    
    # Get cell centers for first velocity (all should be same)
    pred_fields_0 = load_predicted_fields(predictions_paths[velocities[0]])
    n_internal = pred_fields_0['U'].shape[0]
    cell_centers_internal = cell_centers[:n_internal]
    cell_centers_2d, _ = collapse_to_2d(cell_centers_internal, 
                                        cell_centers_internal[:, 0])
    
    # Common scale for all subplots
    all_values = []
    for velocity in velocities:
        pred_fields = load_predicted_fields(predictions_paths[velocity])
        if field_name not in pred_fields:
            continue
        field_2d = config['process'](pred_fields[field_name])
        _, field_2d = collapse_to_2d(cell_centers_internal, field_2d)
        all_values.append(field_2d)
    
    if not all_values:
        return
    
    vmin = min(v.min() for v in all_values)
    vmax = max(v.max() for v in all_values)
    levels = np.linspace(vmin, vmax, 50)
    
    # Create triangulation
    x = cell_centers_2d[:, 0]
    y = cell_centers_2d[:, 1]
    tri = Triangulation(x, y)
    
    # Plot each velocity
    for i, velocity in enumerate(velocities):
        pred_fields = load_predicted_fields(predictions_paths[velocity])
        if field_name not in pred_fields:
            continue
        
        field_2d = config['process'](pred_fields[field_name])
        _, field_2d = collapse_to_2d(cell_centers_internal, field_2d)
        
        # Contour plot
        if field_name == 'p':
            vcenter = (vmin + vmax) / 2
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            contour = axes[i].tricontourf(tri, field_2d, levels=levels, 
                                         cmap=config['cmap'], norm=norm)
        else:
            contour = axes[i].tricontourf(tri, field_2d, levels=levels, 
                                         cmap=config['cmap'])
        
        axes[i].set_title(f'{config["name"]}\nVelocity = {velocity:.1f} m/s', 
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('X [m]', fontsize=10)
        axes[i].set_ylabel('Y [m]', fontsize=10)
        axes[i].set_aspect('equal')
        axes[i].grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(contour, ax=axes[i])
        cbar.set_label(field_name, fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f"{field_name}_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved comparison plot: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize predicted flow fields for velocity variations'
    )
    parser.add_argument(
        '--predictions_dir',
        type=str,
        default='velocity_variations',
        help='Directory containing velocity variation predictions'
    )
    parser.add_argument(
        '--case_path',
        type=str,
        default='OpenFOAM-data',
        help='Path to OpenFOAM case directory (for mesh data)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='velocity_variations_plots',
        help='Directory to save visualization plots'
    )
    parser.add_argument(
        '--velocities',
        type=float,
        nargs='+',
        default=None,
        help='List of velocities to visualize (if None, auto-detect)'
    )
    parser.add_argument(
        '--comparison',
        action='store_true',
        help='Create comparison plots showing all velocities side-by-side'
    )
    
    args = parser.parse_args()
    
    # Find velocity directories
    predictions_dir = Path(args.predictions_dir)
    if not predictions_dir.exists():
        print(f"Error: Predictions directory not found: {predictions_dir}")
        return
    
    # Auto-detect velocities if not specified
    if args.velocities is None:
        velocity_dirs = sorted(predictions_dir.glob('velocity_*'))
        velocities = []
        for vel_dir in velocity_dirs:
            try:
                vel_str = vel_dir.name.replace('velocity_', '')
                velocity = float(vel_str)
                velocities.append(velocity)
            except ValueError:
                continue
        velocities = sorted(velocities)
    else:
        velocities = sorted(args.velocities)
    
    if not velocities:
        print("Error: No velocity variations found!")
        return
    
    print(f"Found {len(velocities)} velocity variations: {velocities} m/s")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize each velocity variation
    predictions_paths = {}
    for velocity in velocities:
        vel_dir = predictions_dir / f"velocity_{velocity:.1f}"
        predictions_file = vel_dir / "predictions.npz"
        
        if not predictions_file.exists():
            print(f"Warning: Predictions file not found: {predictions_file}")
            continue
        
        predictions_paths[velocity] = str(predictions_file)
        
        # Visualize individual velocity
        visualize_velocity_variation(
            str(predictions_file), args.case_path, velocity, args.output_dir
        )
    
    # Create comparison plots
    if args.comparison and len(predictions_paths) > 1:
        print(f"\n{'='*60}")
        print("Creating comparison plots...")
        
        field_names = ['U', 'p', 'k', 'epsilon', 'nut']
        for field_name in field_names:
            # Check if field exists in predictions
            test_pred = load_predicted_fields(list(predictions_paths.values())[0])
            if field_name in test_pred:
                create_comparison_plot(
                    predictions_paths, args.case_path, field_name, args.output_dir
                )
    
    print(f"\n{'='*60}")
    print(f"Visualization complete! Plots saved to {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
