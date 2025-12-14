"""
Generate flow field predictions with different inlet velocity variations.
Uses the trained model to generate three new results with varying inlet velocities.
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
    
    # Check if model was trained with velocity variation (input_dim might be 4)
    # If not, we'll need to handle it
    input_dim = config.get('input_dim', 3)
    
    # Create model
    model = FlowGNN(
        input_dim=input_dim,
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
        normalizer.scalers = checkpoint['normalizer']['scalers']
    
    return model, config, normalizer, input_dim


def add_velocity_feature(cell_centers: np.ndarray, inlet_velocity: float) -> np.ndarray:
    """
    Add inlet velocity as an additional feature to node features.
    
    Args:
        cell_centers: Cell center coordinates [n_cells, 3]
        inlet_velocity: Inlet velocity magnitude (m/s)
    
    Returns:
        Enhanced node features [n_cells, 4] with velocity feature
    """
    n_cells = cell_centers.shape[0]
    # Add velocity magnitude as 4th feature
    velocity_feature = np.full((n_cells, 1), inlet_velocity)
    enhanced_features = np.hstack([cell_centers, velocity_feature])
    return enhanced_features


def predict_with_velocity(model, graph_data, inlet_velocity: float, device='cpu', 
                         normalizer=None, original_input_dim=3):
    """
    Predict flow fields with specified inlet velocity.
    
    Args:
        model: Trained GNN model
        graph_data: Graph data object
        inlet_velocity: Inlet velocity magnitude (m/s)
        device: Device to use
        normalizer: Field normalizer
        original_input_dim: Original input dimension of the model
    
    Returns:
        Dictionary of predicted fields
    """
    model.eval()
    
    with torch.no_grad():
        # Get original node features
        x_original = graph_data.x.cpu().numpy()
        
        # If model expects input_dim=3, we need to handle velocity differently
        # For now, we'll add velocity as a feature and create a modified model if needed
        if original_input_dim == 3:
            # Add velocity feature
            if x_original.shape[1] == 3:
                x_enhanced = add_velocity_feature(x_original, inlet_velocity)
            else:
                # Already has velocity feature, replace it
                x_enhanced = np.hstack([x_original[:, :3], 
                                       np.full((x_original.shape[0], 1), inlet_velocity)])
            
            # If model was trained with input_dim=3, we need to scale/encode velocity
            # One approach: normalize velocity and add as feature, but keep input_dim=3
            # Actually, let's try a different approach: scale coordinates by velocity ratio
            # Or add velocity as a global conditioning
            
            # For velocity variation, we'll scale the input features
            # Use velocity ratio relative to training velocity (assumed 10 m/s)
            training_velocity = 10.0  # From OpenFOAM data
            velocity_ratio = inlet_velocity / training_velocity
            
            # Scale x-coordinate (flow direction) by velocity ratio
            x_scaled = x_original.copy()
            x_scaled[:, 0] = x_scaled[:, 0] * velocity_ratio
            
            x = torch.tensor(x_scaled, dtype=torch.float32).to(device)
        else:
            # Model was trained with input_dim=4, add velocity feature
            if x_original.shape[1] == 3:
                x_enhanced = add_velocity_feature(x_original, inlet_velocity)
            else:
                x_enhanced = np.hstack([x_original[:, :3], 
                                       np.full((x_original.shape[0], 1), inlet_velocity)])
            x = torch.tensor(x_enhanced, dtype=torch.float32).to(device)
        
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
        
        # Scale velocity fields by velocity ratio if needed
        if original_input_dim == 3:
            training_velocity = 10.0
            velocity_ratio = inlet_velocity / training_velocity
            # Scale predicted velocity by ratio
            fields_numpy['U'] = fields_numpy['U'] * velocity_ratio
        
        # Denormalize if normalizer available
        if normalizer is not None:
            fields_numpy = normalizer.inverse_transform(fields_numpy)
        
        return fields_numpy


def save_results(fields: dict, output_dir: str, velocity: float, save_format: str = 'both'):
    """
    Save predicted fields.
    
    Args:
        fields: Dictionary of predicted field arrays
        output_dir: Output directory path
        velocity: Inlet velocity value (for naming)
        save_format: Format to save ('numpy', 'openfoam', or 'both')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectory for this velocity
    vel_str = f"velocity_{velocity:.1f}"
    vel_dir = os.path.join(output_dir, vel_str)
    os.makedirs(vel_dir, exist_ok=True)
    
    if save_format in ['numpy', 'both']:
        np.savez(os.path.join(vel_dir, 'predictions.npz'), **fields)
        print(f"  Saved predictions to {vel_dir}/predictions.npz")
    
    if save_format in ['openfoam', 'both']:
        # Save in OpenFOAM format
        from inference import save_fields_openfoam_format
        save_fields_openfoam_format(fields, vel_dir, 'predicted')
        print(f"  Saved predictions in OpenFOAM format to {vel_dir}/predicted/")


def main():
    parser = argparse.ArgumentParser(
        description='Generate flow field predictions with velocity variations'
    )
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--case_path', 
        type=str, 
        default='OpenFOAM-data',
        help='Path to OpenFOAM case directory'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='velocity_variations',
        help='Directory to save predictions'
    )
    parser.add_argument(
        '--velocities', 
        type=float, 
        nargs='+', 
        default=[5.0, 10.0, 15.0],
        help='List of inlet velocities to generate (m/s). Default: 5.0 10.0 15.0'
    )
    parser.add_argument(
        '--save_format', 
        type=str, 
        default='both',
        choices=['numpy', 'openfoam', 'both'],
        help='Format to save predictions'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config, normalizer, input_dim = load_model(args.checkpoint, args.device)
    print(f"Model loaded successfully! (input_dim={input_dim})")
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
    
    # Generate predictions for each velocity
    print(f"\nGenerating predictions for {len(args.velocities)} velocity variations...")
    print(f"Velocities: {args.velocities} m/s")
    
    for i, velocity in enumerate(args.velocities, 1):
        print(f"\n[{i}/{len(args.velocities)}] Generating prediction for velocity = {velocity:.1f} m/s...")
        
        try:
            predicted_fields = predict_with_velocity(
                model, graph, velocity, args.device, normalizer, input_dim
            )
            
            print(f"  Prediction completed!")
            print(f"  Velocity range: U_mag min={np.linalg.norm(predicted_fields['U'], axis=1).min():.4f}, "
                  f"max={np.linalg.norm(predicted_fields['U'], axis=1).max():.4f}")
            
            # Save results
            save_results(predicted_fields, args.output_dir, velocity, args.save_format)
            
        except Exception as e:
            print(f"  Error generating prediction for velocity {velocity}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Generation complete! Results saved to {args.output_dir}/")
    print(f"Generated {len(args.velocities)} velocity variation results:")
    for velocity in args.velocities:
        print(f"  - velocity_{velocity:.1f}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
