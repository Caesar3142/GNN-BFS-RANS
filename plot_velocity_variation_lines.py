"""
Extract and plot field values along specific lines for velocity variations.
Creates plots and CSV files for X = 0.150 (vertical) and Y = 0.005 (horizontal).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import csv

from openfoam_loader import OpenFOAMLoader


def load_predicted_fields(predictions_path: str):
    """Load predicted fields from NumPy file."""
    data = np.load(predictions_path)
    fields = {}
    for key in data.files:
        fields[key] = data[key]
    return fields


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
    
    if x_line is not None:
        # Vertical line at X = x_line
        mask = np.abs(x - x_line) < tol
        if np.sum(mask) == 0:
            # If no exact match, find closest points
            distances = np.abs(x - x_line)
            mask = distances < (np.min(distances) + tol)
        
        positions = y[mask]
        if len(field_values.shape) == 1:
            values = field_values[mask]
        else:
            values = field_values[mask]
        
        # Sort by Y
        sort_idx = np.argsort(positions)
        positions = positions[sort_idx]
        if len(field_values.shape) == 1:
            values = values[sort_idx]
        else:
            values = values[sort_idx]
        
        return positions, values
    
    elif y_line is not None:
        # Horizontal line at Y = y_line
        mask = np.abs(y - y_line) < tol
        if np.sum(mask) == 0:
            # If no exact match, find closest points
            distances = np.abs(y - y_line)
            mask = distances < (np.min(distances) + tol)
        
        positions = x[mask]
        if len(field_values.shape) == 1:
            values = field_values[mask]
        else:
            values = field_values[mask]
        
        # Sort by X
        sort_idx = np.argsort(positions)
        positions = positions[sort_idx]
        if len(field_values.shape) == 1:
            values = values[sort_idx]
        else:
            values = values[sort_idx]
        
        return positions, values
    
    else:
        raise ValueError("Either x_line or y_line must be specified")


def plot_line_data(velocities_data, line_type, line_value, output_path, fields_to_plot=None):
    """
    Plot field values along a line for all velocity variations.
    
    Args:
        velocities_data: Dictionary mapping velocity to (positions, fields_dict)
        line_type: 'vertical' or 'horizontal'
        line_value: X or Y coordinate value
        output_path: Path to save plot
        fields_to_plot: List of field names to plot (default: ['U', 'p'])
    """
    if fields_to_plot is None:
        fields_to_plot = ['U', 'p']
    
    # Determine position label
    if line_type == 'vertical':
        position_label = 'Y [m]'
        line_label = f'X = {line_value:.3f}'
    else:
        position_label = 'X [m]'
        line_label = f'Y = {line_value:.3f}'
    
    # Create subplots for each field
    n_fields = len(fields_to_plot)
    fig, axes = plt.subplots(1, n_fields, figsize=(8*n_fields, 6))
    if n_fields == 1:
        axes = [axes]
    
    # Color map for different velocities
    velocities = sorted(velocities_data.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(velocities)))
    
    for field_idx, field_name in enumerate(fields_to_plot):
        ax = axes[field_idx]
        
        for vel_idx, velocity in enumerate(velocities):
            positions, fields_dict = velocities_data[velocity]
            
            if field_name not in fields_dict:
                continue
            
            field_values = fields_dict[field_name]
            
            # Handle vector fields (take magnitude)
            if len(field_values.shape) > 1 and field_values.shape[1] > 1:
                if field_values.shape[1] == 3:
                    # Velocity magnitude
                    field_values = np.linalg.norm(field_values, axis=1)
                else:
                    field_values = field_values[:, 0]
            else:
                field_values = field_values.flatten()
            
            # Plot
            ax.plot(positions, field_values, 
                   color=colors[vel_idx], 
                   linewidth=2, 
                   marker='o', 
                   markersize=4,
                   label=f'V = {velocity:.1f} m/s',
                   alpha=0.8)
        
        # Formatting
        if field_name == 'U':
            field_label = 'Velocity Magnitude [m/s]'
        elif field_name == 'p':
            field_label = 'Pressure [m²/s²]'
        elif field_name == 'k':
            field_label = 'Turbulent Kinetic Energy [m²/s²]'
        elif field_name == 'epsilon':
            field_label = 'Dissipation Rate [m²/s³]'
        elif field_name == 'nut':
            field_label = 'Turbulent Viscosity [m²/s]'
        else:
            field_label = field_name
        
        ax.set_xlabel(position_label, fontsize=12)
        ax.set_ylabel(field_label, fontsize=12)
        ax.set_title(f'{field_label} along {line_label}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved plot: {output_path}")
    plt.close(fig)


def save_line_data_csv(velocities_data, line_type, line_value, output_path, fields_to_save=None):
    """
    Save line data to CSV file.
    
    Args:
        velocities_data: Dictionary mapping velocity to (positions, fields_dict)
        line_type: 'vertical' or 'horizontal'
        line_value: X or Y coordinate value
        output_path: Path to save CSV
        fields_to_save: List of field names to save (default: all available)
    """
    velocities = sorted(velocities_data.keys())
    
    # Determine position column name
    if line_type == 'vertical':
        pos_col = 'Y'
    else:
        pos_col = 'X'
    
    # Get field names
    if fields_to_save is None:
        # Get fields from first velocity
        if velocities:
            fields_to_save = list(velocities_data[velocities[0]][1].keys())
        else:
            return
    
    # Build header
    header = ['Velocity_m_s', pos_col]
    for field_name in fields_to_save:
        if field_name == 'U':
            # Velocity: add magnitude and components
            header.extend(['U_magnitude', 'U_x', 'U_y', 'U_z'])
        else:
            header.append(field_name)
    
    # Collect all rows
    all_rows = []
    for velocity in velocities:
        positions, fields_dict = velocities_data[velocity]
        
        for i, pos in enumerate(positions):
            row = [velocity, pos]
            
            for field_name in fields_to_save:
                if field_name not in fields_dict:
                    # Fill with NaN if field not available
                    if field_name == 'U':
                        row.extend([np.nan, np.nan, np.nan, np.nan])
                    else:
                        row.append(np.nan)
                    continue
                
                field_values = fields_dict[field_name]
                
                # Handle vector fields
                if len(field_values.shape) > 1 and field_values.shape[1] > 1:
                    if field_values.shape[1] == 3 and field_name == 'U':
                        # Velocity: save magnitude and components
                        row.append(np.linalg.norm(field_values[i]))
                        row.append(float(field_values[i, 0]))
                        row.append(float(field_values[i, 1]))
                        row.append(float(field_values[i, 2]))
                    else:
                        row.append(float(field_values[i, 0]))
                else:
                    val = field_values[i]
                    if isinstance(val, np.ndarray):
                        val = val.item()
                    row.append(float(val))
            
            all_rows.append(row)
    
    # Sort by velocity then position
    all_rows.sort(key=lambda x: (x[0], x[1]))
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)
    
    print(f"  Saved CSV: {output_path}")


def process_velocity_variations(predictions_dir, case_path, x_line, y_line, 
                               output_dir, tol=1e-4):
    """
    Process all velocity variations and extract line data.
    
    Args:
        predictions_dir: Directory containing velocity variation predictions
        case_path: Path to OpenFOAM case directory (for mesh data)
        x_line: X coordinate for vertical line
        y_line: Y coordinate for horizontal line
        output_dir: Output directory for plots and CSV files
        tol: Tolerance for line extraction
    """
    predictions_dir = Path(predictions_dir)
    
    # Find velocity directories
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
    
    if not velocities:
        print("Error: No velocity variations found!")
        return
    
    print(f"Found {len(velocities)} velocity variations: {velocities} m/s")
    
    # Load mesh to get cell centers
    print("Loading mesh data...")
    loader = OpenFOAMLoader(case_path)
    mesh_data = loader.load_mesh()
    cell_centers = mesh_data['cell_centers']
    
    # Filter to z >= 0 (2D plane)
    z_mask = cell_centers[:, 2] >= 0
    cell_centers_2d = cell_centers[z_mask]
    
    # Process vertical line (X = x_line)
    print(f"\nProcessing vertical line at X = {x_line:.3f}...")
    vertical_data = {}
    
    for velocity in velocities:
        vel_dir = predictions_dir / f"velocity_{velocity:.1f}"
        predictions_file = vel_dir / "predictions.npz"
        
        if not predictions_file.exists():
            print(f"  Warning: Skipping velocity {velocity:.1f} - file not found")
            continue
        
        # Load predictions
        predicted_fields = load_predicted_fields(str(predictions_file))
        
        # Get number of internal cells
        n_internal = predicted_fields['U'].shape[0]
        cell_centers_internal = cell_centers[:n_internal]
        
        # Filter to z >= 0
        z_mask_internal = cell_centers_internal[:, 2] >= 0
        cell_centers_internal_2d = cell_centers_internal[z_mask_internal]
        
        # Filter fields to z >= 0
        fields_2d = {}
        for field_name, field_data in predicted_fields.items():
            if len(field_data.shape) > 1:
                fields_2d[field_name] = field_data[z_mask_internal]
            else:
                fields_2d[field_name] = field_data[z_mask_internal]
        
        # Extract line data
        try:
            positions, _ = extract_line_data(
                cell_centers_internal_2d, 
                fields_2d['U'], 
                x_line=x_line, 
                tol=tol
            )
            
            # Extract all fields along the line
            line_fields = {}
            for field_name in fields_2d:
                _, field_values = extract_line_data(
                    cell_centers_internal_2d,
                    fields_2d[field_name],
                    x_line=x_line,
                    tol=tol
                )
                line_fields[field_name] = field_values
            
            vertical_data[velocity] = (positions, line_fields)
            print(f"  Velocity {velocity:.1f} m/s: {len(positions)} points")
            
        except Exception as e:
            print(f"  Error processing velocity {velocity:.1f}: {e}")
            continue
    
    # Process horizontal line (Y = y_line)
    print(f"\nProcessing horizontal line at Y = {y_line:.3f}...")
    horizontal_data = {}
    
    for velocity in velocities:
        vel_dir = predictions_dir / f"velocity_{velocity:.1f}"
        predictions_file = vel_dir / "predictions.npz"
        
        if not predictions_file.exists():
            continue
        
        # Load predictions
        predicted_fields = load_predicted_fields(str(predictions_file))
        
        # Get number of internal cells
        n_internal = predicted_fields['U'].shape[0]
        cell_centers_internal = cell_centers[:n_internal]
        
        # Filter to z >= 0
        z_mask_internal = cell_centers_internal[:, 2] >= 0
        cell_centers_internal_2d = cell_centers_internal[z_mask_internal]
        
        # Filter fields to z >= 0
        fields_2d = {}
        for field_name, field_data in predicted_fields.items():
            if len(field_data.shape) > 1:
                fields_2d[field_name] = field_data[z_mask_internal]
            else:
                fields_2d[field_name] = field_data[z_mask_internal]
        
        # Extract line data
        try:
            positions, _ = extract_line_data(
                cell_centers_internal_2d,
                fields_2d['U'],
                y_line=y_line,
                tol=tol
            )
            
            # Extract all fields along the line
            line_fields = {}
            for field_name in fields_2d:
                _, field_values = extract_line_data(
                    cell_centers_internal_2d,
                    fields_2d[field_name],
                    y_line=y_line,
                    tol=tol
                )
                line_fields[field_name] = field_values
            
            horizontal_data[velocity] = (positions, line_fields)
            print(f"  Velocity {velocity:.1f} m/s: {len(positions)} points")
            
        except Exception as e:
            print(f"  Error processing velocity {velocity:.1f}: {e}")
            continue
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots and CSV files
    print(f"\n{'='*60}")
    print("Creating plots and CSV files...")
    
    # Vertical line plots
    if vertical_data:
        plot_path = os.path.join(output_dir, f'line_X_{x_line:.3f}.png')
        plot_line_data(vertical_data, 'vertical', x_line, plot_path, ['U', 'p'])
        
        csv_path = os.path.join(output_dir, f'line_X_{x_line:.3f}.csv')
        save_line_data_csv(vertical_data, 'vertical', x_line, csv_path, ['U', 'p', 'k', 'epsilon', 'nut'])
    
    # Horizontal line plots
    if horizontal_data:
        plot_path = os.path.join(output_dir, f'line_Y_{y_line:.3f}.png')
        plot_line_data(horizontal_data, 'horizontal', y_line, plot_path, ['U', 'p'])
        
        csv_path = os.path.join(output_dir, f'line_Y_{y_line:.3f}.csv')
        save_line_data_csv(horizontal_data, 'horizontal', y_line, csv_path, ['U', 'p', 'k', 'epsilon', 'nut'])
    
    print(f"\n{'='*60}")
    print(f"Complete! Results saved to {output_dir}/")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract and plot field values along lines for velocity variations'
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
        '--x_line',
        type=float,
        default=0.150,
        help='X coordinate for vertical line (default: 0.150)'
    )
    parser.add_argument(
        '--y_line',
        type=float,
        default=0.005,
        help='Y coordinate for horizontal line (default: 0.005)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='line_plots',
        help='Directory to save plots and CSV files'
    )
    parser.add_argument(
        '--tol',
        type=float,
        default=1e-4,
        help='Tolerance for line extraction (default: 1e-4)'
    )
    
    args = parser.parse_args()
    
    process_velocity_variations(
        args.predictions_dir,
        args.case_path,
        args.x_line,
        args.y_line,
        args.output_dir,
        args.tol
    )


if __name__ == '__main__':
    main()
