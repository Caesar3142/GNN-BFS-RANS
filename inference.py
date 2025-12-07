"""
Inference script for trained GNN model.
Predicts flow fields and optionally saves results in OpenFOAM format.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import json
import os

from openfoam_loader import OpenFOAMLoader
from graph_constructor import GraphConstructor
from gnn_model import FlowGNN
from normalization import FieldNormalizer
import pickle


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default config
        config = {
            'hidden_dim': 256,
            'num_layers': 6,
            'layer_type': 'GCN'
        }
    
    # Create model
    model = FlowGNN(
        input_dim=3,
        hidden_dim=config.get('hidden_dim', 256),
        output_dim=7,  # U(3) + p(1) + k(1) + epsilon(1) + nut(1)
        num_layers=config.get('num_layers', 6),
        layer_type=config.get('layer_type', 'GCN'),
        use_edge_attr=True,
        dropout=0.0,  # No dropout during inference
        use_batch_norm=True
    )
    
    # Load weights
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
        # Move to device
        x = graph_data.x.to(device)
        edge_index = graph_data.edge_index.to(device)
        edge_attr = graph_data.edge_attr.to(device)
        
        # Predict
        output = model(x, edge_index, edge_attr)
        
        # Parse into fields
        fields = model.predict_fields(output)
        
        # Move back to CPU and convert to numpy
        fields_numpy = {}
        for key in fields:
            fields_numpy[key] = fields[key].cpu().numpy()
        
        # Denormalize if normalizer available
        if normalizer is not None:
            fields_numpy = normalizer.inverse_transform(fields_numpy)
        
        return fields_numpy


def save_fields_openfoam_format(fields: dict, output_dir: str, time_dir: str = 'predicted'):
    """
    Save predicted fields in OpenFOAM format.
    
    Args:
        fields: Dictionary of predicted field arrays
        output_dir: Output directory path
        time_dir: Time directory name
    """
    output_path = Path(output_dir) / time_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    n_cells = fields['U'].shape[0]
    
    # Save velocity field
    if 'U' in fields:
        with open(output_path / 'U', 'w') as f:
            f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
            f.write("| =========                 |                                                 |\n")
            f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
            f.write("|  \\\\    /   O peration     | Version:  v2406                                 |\n")
            f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
            f.write("|    \\\\/     M anipulation  |                                                 |\n")
            f.write("\\*---------------------------------------------------------------------------*/\n")
            f.write("FoamFile\n")
            f.write("{\n")
            f.write("    version     2.0;\n")
            f.write("    format      ascii;\n")
            f.write("    class       volVectorField;\n")
            f.write(f"    location    \"{time_dir}\";\n")
            f.write("    object      U;\n")
            f.write("}\n")
            f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
            f.write("dimensions      [0 1 -1 0 0 0 0];\n\n")
            f.write("internalField   nonuniform List<vector>\n")
            f.write(f"{n_cells}\n")
            f.write("(\n")
            for i in range(n_cells):
                u = fields['U'][i]
                f.write(f"({u[0]:.6e} {u[1]:.6e} {u[2]:.6e})\n")
            f.write(")\n")
            f.write(";\n\n")
            f.write("boundaryField\n")
            f.write("{\n")
            f.write("    // Placeholder - boundary conditions not predicted\n")
            f.write("}\n\n")
            f.write("// ************************************************************************* //\n")
    
    # Save scalar fields
    scalar_fields = {
        'p': {'dimensions': '[0 2 -2 0 0 0 0]', 'class': 'volScalarField'},
        'k': {'dimensions': '[0 2 -2 0 0 0 0]', 'class': 'volScalarField'},
        'epsilon': {'dimensions': '[0 2 -3 0 0 0 0]', 'class': 'volScalarField'},
        'nut': {'dimensions': '[0 2 -1 0 0 0 0]', 'class': 'volScalarField'},
    }
    
    for field_name, field_info in scalar_fields.items():
        if field_name in fields:
            with open(output_path / field_name, 'w') as f:
                f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
                f.write("| =========                 |                                                 |\n")
                f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
                f.write("|  \\\\    /   O peration     | Version:  v2406                                 |\n")
                f.write("|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n")
                f.write("|    \\\\/     M anipulation  |                                                 |\n")
                f.write("\\*---------------------------------------------------------------------------*/\n")
                f.write("FoamFile\n")
                f.write("{\n")
                f.write("    version     2.0;\n")
                f.write("    format      ascii;\n")
                f.write(f"    class       {field_info['class']};\n")
                f.write(f"    location    \"{time_dir}\";\n")
                f.write(f"    object      {field_name};\n")
                f.write("}\n")
                f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
                f.write(f"dimensions      {field_info['dimensions']};\n\n")
                f.write("internalField   nonuniform List<scalar>\n")
                f.write(f"{n_cells}\n")
                f.write("(\n")
                for i in range(n_cells):
                    val = fields[field_name][i, 0] if len(fields[field_name].shape) > 1 else fields[field_name][i]
                    f.write(f"{val:.6e}\n")
                f.write(")\n")
                f.write(";\n\n")
                f.write("boundaryField\n")
                f.write("{\n")
                f.write("    // Placeholder - boundary conditions not predicted\n")
                f.write("}\n\n")
                f.write("// ************************************************************************* //\n")


def compare_with_reference(predicted_fields: dict, reference_fields: dict):
    """Compare predicted fields with reference (ground truth) fields."""
    print("\n=== Field Comparison ===")
    
    field_names = ['U', 'p', 'k', 'epsilon', 'nut']
    
    for field_name in field_names:
        if field_name not in predicted_fields or field_name not in reference_fields:
            continue
        
        pred = predicted_fields[field_name]
        ref = reference_fields[field_name]
        
        if field_name == 'U':
            # Vector field
            error = np.linalg.norm(pred - ref, axis=1)
            mae = np.mean(error)
            rmse = np.sqrt(np.mean(error**2))
            max_error = np.max(error)
            print(f"{field_name}:")
            print(f"  MAE:  {mae:.6e}")
            print(f"  RMSE: {rmse:.6e}")
            print(f"  Max:  {max_error:.6e}")
        else:
            # Scalar field
            if len(pred.shape) > 1:
                pred = pred.flatten()
            if len(ref.shape) > 1:
                ref = ref.flatten()
            
            error = np.abs(pred - ref)
            mae = np.mean(error)
            rmse = np.sqrt(np.mean(error**2))
            max_error = np.max(error)
            rel_error = mae / (np.abs(ref).mean() + 1e-10)
            
            print(f"{field_name}:")
            print(f"  MAE:        {mae:.6e}")
            print(f"  RMSE:       {rmse:.6e}")
            print(f"  Max Error:  {max_error:.6e}")
            print(f"  Rel Error:  {rel_error:.4%}")


def main():
    parser = argparse.ArgumentParser(description='Inference with trained GNN model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--case_path', type=str, default='OpenFOAM-data',
                       help='Path to OpenFOAM case directory')
    parser.add_argument('--output_dir', type=str, default='predictions',
                       help='Directory to save predictions')
    parser.add_argument('--reference_time', type=str, default=None,
                       help='Time directory for reference comparison (e.g., "282")')
    parser.add_argument('--save_format', type=str, default='numpy',
                       choices=['numpy', 'openfoam', 'both'],
                       help='Format to save predictions')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config, normalizer = load_model(args.checkpoint, args.device)
    print("Model loaded successfully!")
    if normalizer is not None:
        print("Normalizer found - will denormalize predictions")
    
    # Load mesh and construct graph
    print("Loading mesh and constructing graph...")
    loader = OpenFOAMLoader(args.case_path)
    mesh_data = loader.load_mesh()
    graph_constructor = GraphConstructor(mesh_data)
    
    # Build graph (using only geometry as input)
    graph = graph_constructor.build_graph(node_features=mesh_data['cell_centers'])
    print(f"Graph constructed: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
    
    # Predict
    print("Running inference...")
    predicted_fields = predict_fields(model, graph, args.device, normalizer)
    print("Prediction completed!")
    
    # Save predictions
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.save_format in ['numpy', 'both']:
        np.savez(os.path.join(args.output_dir, 'predictions.npz'), **predicted_fields)
        print(f"Saved predictions to {args.output_dir}/predictions.npz")
    
    if args.save_format in ['openfoam', 'both']:
        save_fields_openfoam_format(predicted_fields, args.output_dir, 'predicted')
        print(f"Saved predictions in OpenFOAM format to {args.output_dir}/predicted/")
    
    # Compare with reference if provided
    if args.reference_time:
        print(f"\nLoading reference data from time {args.reference_time}...")
        try:
            reference_fields = loader.load_fields(args.reference_time)
            # Convert to same format
            ref_dict = {
                'U': reference_fields['U'],
                'p': reference_fields['p'].reshape(-1, 1),
                'k': reference_fields['k'].reshape(-1, 1),
                'epsilon': reference_fields['epsilon'].reshape(-1, 1),
                'nut': reference_fields['nut'].reshape(-1, 1),
            }
            compare_with_reference(predicted_fields, ref_dict)
        except Exception as e:
            print(f"Warning: Could not load reference data: {e}")
    
    print("\nInference completed!")


if __name__ == '__main__':
    main()

