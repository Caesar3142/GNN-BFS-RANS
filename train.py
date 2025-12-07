"""
Training script for GNN-based RANS flow field predictor.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import os

from openfoam_loader import OpenFOAMLoader
from graph_constructor import GraphConstructor
from gnn_model import FlowGNN
from normalization import FieldNormalizer, WeightedMSELoss


class OpenFOAMDataset(Dataset):
    """Dataset for OpenFOAM flow field data."""
    
    def __init__(self, case_path: str, time_dirs: list, use_fields_as_input: bool = False, 
                 normalizer: FieldNormalizer = None, normalize: bool = True):
        """
        Initialize dataset.
        
        Args:
            case_path: Path to OpenFOAM case directory
            time_dirs: List of time directories to load (e.g., ['0', '100', '200', '282'])
            use_fields_as_input: If True, use field data as input features
            normalizer: FieldNormalizer instance (if None, will be created)
            normalize: Whether to normalize fields
        """
        self.case_path = case_path
        self.time_dirs = time_dirs
        self.use_fields_as_input = use_fields_as_input
        self.normalize = normalize
        self.normalizer = normalizer
        
        # Load mesh once
        loader = OpenFOAMLoader(case_path)
        self.mesh_data = loader.load_mesh()
        self.graph_constructor = GraphConstructor(self.mesh_data)
        
        # First pass: collect all fields to fit normalizer
        all_fields = {}
        field_data_list = []
        
        for time_dir in time_dirs:
            try:
                fields = loader.load_fields(time_dir)
                field_data_list.append(fields)
                
                # Collect for normalizer fitting
                for field_name, field_data in fields.items():
                    if field_name not in all_fields:
                        all_fields[field_name] = []
                    all_fields[field_name].append(field_data)
            except Exception as e:
                print(f"Warning: Could not load time directory {time_dir}: {e}")
                continue
        
        # Fit normalizer if needed
        if self.normalize and self.normalizer is None:
            print("Fitting field normalizer...")
            # Concatenate all fields for fitting
            fit_fields = {}
            for field_name, field_list in all_fields.items():
                fit_fields[field_name] = np.concatenate(field_list, axis=0)
            
            self.normalizer = FieldNormalizer()
            self.normalizer.fit(fit_fields)
            print("Normalizer fitted")
        
        # Second pass: load and normalize data
        self.data_samples = []
        for idx, time_dir in enumerate(time_dirs):
            try:
                fields = loader.load_fields(time_dir)
                
                # Normalize fields if needed
                if self.normalize and self.normalizer is not None:
                    fields = self.normalizer.transform(fields)
                
                # Determine number of internal cells from field data
                n_internal = None
                for field_name in ['U', 'p', 'k', 'epsilon', 'nut']:
                    if field_name in fields:
                        if field_name == 'U':
                            n_internal = fields[field_name].shape[0]
                        else:
                            n_internal = len(fields[field_name])
                        break
                
                if n_internal is None:
                    print(f"Warning: No fields found in {time_dir}, skipping")
                    continue
                
                # Build graph filtered to exactly n_internal cells (matching field data)
                graph = self.graph_constructor.build_graph(
                    node_features=self.mesh_data['cell_centers'],
                    filter_internal=True,
                    n_internal_cells=n_internal
                )
                
                # Verify sizes match
                if graph.num_nodes != n_internal:
                    print(f"Warning: Graph has {graph.num_nodes} nodes but fields have {n_internal} values, adjusting...")
                    graph.x = graph.x[:n_internal]
                    valid_edges = (graph.edge_index[0] < n_internal) & (graph.edge_index[1] < n_internal)
                    graph.edge_index = graph.edge_index[:, valid_edges]
                    graph.edge_attr = graph.edge_attr[valid_edges]
                    graph.num_nodes = n_internal
                
                # Prepare target fields
                target_fields = []
                field_names = ['U', 'p', 'k', 'epsilon', 'nut']
                for field_name in field_names:
                    if field_name in fields:
                        if field_name == 'U':
                            target_fields.append(fields[field_name])  # [n_cells, 3]
                        else:
                            target_fields.append(fields[field_name].reshape(-1, 1))  # [n_cells, 1]
                
                # Concatenate targets: U (3) + p (1) + k (1) + epsilon (1) + nut (1) = 7
                target = np.hstack(target_fields)
                
                # Ensure target matches graph size
                if target.shape[0] != graph.num_nodes:
                    target = target[:graph.num_nodes]
                
                # Store
                graph.y = torch.tensor(target, dtype=torch.float32)
                self.data_samples.append(graph)
                
            except Exception as e:
                print(f"Warning: Could not load time directory {time_dir}: {e}")
                continue
        
        print(f"Loaded {len(self.data_samples)} samples")
    
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        return self.data_samples[idx]


def collate_fn(batch):
    """Custom collate function for batching graphs."""
    return Batch.from_data_list(batch)


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        batch = batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        # Compute loss
        loss = criterion(output, batch.y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(output, batch.y)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def compute_field_errors(pred, target, field_names):
    """Compute per-field errors."""
    errors = {}
    idx = 0
    
    for field_name in field_names:
        if field_name == 'U':
            field_dim = 3
            pred_field = pred[:, idx:idx+field_dim]
            target_field = target[:, idx:idx+field_dim]
            # L2 error for velocity
            error = torch.mean(torch.norm(pred_field - target_field, dim=1))
        else:
            field_dim = 1
            pred_field = pred[:, idx:idx+field_dim]
            target_field = target[:, idx:idx+field_dim]
            # L2 error for scalars
            error = torch.mean(torch.abs(pred_field - target_field))
        
        errors[field_name] = error.item()
        idx += field_dim
    
    return errors


def evaluate_detailed(model, dataloader, device):
    """Detailed evaluation with per-field metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            # Collect predictions and targets
            all_preds.append(output.cpu())
            all_targets.append(batch.y.cpu())
    
    # Concatenate all
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute errors
    field_names = ['U', 'p', 'k', 'epsilon', 'nut']
    errors = compute_field_errors(all_preds, all_targets, field_names)
    
    return errors


def main():
    parser = argparse.ArgumentParser(description='Train GNN for RANS flow field prediction')
    parser.add_argument('--case_path', type=str, default='OpenFOAM-data',
                       help='Path to OpenFOAM case directory')
    parser.add_argument('--time_dirs', type=str, nargs='+', 
                       default=['0', '100', '200', '282'],
                       help='Time directories to use for training')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension for GNN')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of GNN layers')
    parser.add_argument('--layer_type', type=str, default='GCN',
                       choices=['GCN', 'GAT', 'GIN', 'Transformer'],
                       help='Type of GNN layer')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (usually 1 for single mesh)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load dataset with normalization
    print("Loading dataset...")
    dataset = OpenFOAMDataset(
        args.case_path,
        args.time_dirs,
        use_fields_as_input=False,
        normalize=True
    )
    
    # Save normalizer for inference
    if dataset.normalizer is not None:
        import pickle
        normalizer_path = os.path.join(args.output_dir, 'normalizer.pkl')
        with open(normalizer_path, 'wb') as f:
            pickle.dump(dataset.normalizer, f)
        print(f"Saved normalizer to {normalizer_path}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Create model
    print("Creating model...")
    model = FlowGNN(
        input_dim=3,  # Cell center coordinates
        hidden_dim=args.hidden_dim,
        output_dim=7,  # U(3) + p(1) + k(1) + epsilon(1) + nut(1)
        num_layers=args.num_layers,
        layer_type=args.layer_type,
        use_edge_attr=True,
        dropout=0.1,
        use_batch_norm=True
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    # Use weighted loss to account for different field magnitudes
    criterion = WeightedMSELoss(field_weights={
        'U': 1.0,      # Velocity - most important
        'p': 1.0,      # Pressure - important
        'k': 0.5,      # Turbulence fields - less critical
        'epsilon': 0.5,
        'nut': 0.5
    })
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, dataloader, optimizer, criterion, args.device, epoch)
        
        # Validate (using same data for now, can split later)
        val_loss = validate(model, dataloader, criterion, args.device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Detailed evaluation
        if epoch % 10 == 0:
            field_errors = evaluate_detailed(model, dataloader, args.device)
            print(f"\nEpoch {epoch} - Field Errors:")
            for field, error in field_errors.items():
                print(f"  {field}: {error:.6f}")
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save normalizer with checkpoint
            normalizer_data = None
            if dataset.normalizer is not None:
                normalizer_data = {
                    'field_stats': dataset.normalizer.field_stats,
                    'scalers': dataset.normalizer.scalers  # Already in correct format
                }
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': vars(args),
                'normalizer': normalizer_data
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"Saved best model (val_loss={val_loss:.6f})")
        
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    print("Training completed!")


if __name__ == '__main__':
    main()

