"""
Graph Neural Network Model for RANS Flow Field Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.nn import GCNConv, GATConv, GINConv, TransformerConv
from torch_geometric.nn import BatchNorm
from typing import Optional


class FlowGNN(nn.Module):
    """
    Graph Neural Network for predicting RANS flow fields.
    Predicts velocity (U), pressure (p), and turbulence fields (k, epsilon, nut).
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # Cell center coordinates
        hidden_dim: int = 128,
        output_dim: int = 8,  # U (3) + p (1) + k (1) + epsilon (1) + nut (1) + residual (1)
        num_layers: int = 4,
        layer_type: str = 'GCN',  # 'GCN', 'GAT', 'GIN', 'Transformer'
        use_edge_attr: bool = True,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize FlowGNN model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Hidden dimension for GNN layers
            output_dim: Output dimension (number of predicted fields)
            num_layers: Number of GNN layers
            layer_type: Type of GNN layer to use
            use_edge_attr: Whether to use edge attributes
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(FlowGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.use_edge_attr = use_edge_attr
        self.use_batch_norm = use_batch_norm
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        for i in range(num_layers):
            if layer_type == 'GCN':
                layer = GCNConv(hidden_dim, hidden_dim)
            elif layer_type == 'GAT':
                layer = GATConv(
                    hidden_dim, hidden_dim,
                    heads=4, concat=False, dropout=dropout
                )
            elif layer_type == 'GIN':
                nn_gin = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                layer = GINConv(nn_gin)
            elif layer_type == 'Transformer':
                layer = TransformerConv(
                    hidden_dim, hidden_dim,
                    heads=4, concat=False, dropout=dropout
                )
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            
            self.gnn_layers.append(layer)
            
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            batch: Batch assignment for nodes
        
        Returns:
            Predicted flow fields [num_nodes, output_dim]
        """
        # Input projection
        x = self.input_proj(x)
        
        # GNN layers
        for i, layer in enumerate(self.gnn_layers):
            # Apply GNN layer
            if self.layer_type in ['GCN', 'GIN']:
                x_new = layer(x, edge_index)
            elif self.layer_type == 'GAT':
                x_new = layer(x, edge_index)
            elif self.layer_type == 'Transformer':
                x_new = layer(x, edge_index, edge_attr=edge_attr)
            else:
                x_new = layer(x, edge_index)
            
            # Residual connection
            x = x + x_new
            
            # Batch normalization
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            # Activation and dropout
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output projection
        output = self.output_proj(x)
        
        return output
    
    def predict_fields(self, output: torch.Tensor) -> dict:
        """
        Parse model output into individual fields.
        
        Args:
            output: Model output [num_nodes, output_dim]
        
        Returns:
            Dictionary of predicted fields
        """
        fields = {
            'U': output[:, :3],  # Velocity vector
            'p': output[:, 3:4],  # Pressure
            'k': output[:, 4:5],  # Turbulent kinetic energy
            'epsilon': output[:, 5:6],  # Dissipation rate
            'nut': output[:, 6:7],  # Turbulent viscosity
        }
        
        if output.shape[1] > 7:
            fields['residual'] = output[:, 7:8]
        
        return fields


class FlowGNNSurrogate(nn.Module):
    """
    Surrogate model that can predict flow fields from boundary conditions.
    Uses encoder-decoder architecture with boundary condition encoding.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        layer_type: str = 'GCN',
        use_edge_attr: bool = True,
        dropout: float = 0.1
    ):
        super(FlowGNNSurrogate, self).__init__()
        
        # Encoder: processes mesh geometry
        self.encoder = FlowGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers // 2,
            layer_type=layer_type,
            use_edge_attr=use_edge_attr,
            dropout=dropout
        )
        
        # Decoder: predicts flow fields
        self.decoder = FlowGNN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=8,
            num_layers=num_layers // 2,
            layer_type=layer_type,
            use_edge_attr=use_edge_attr,
            dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        boundary_conditions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with boundary conditions.
        
        Args:
            x: Node features (cell centers)
            edge_index: Edge connectivity
            edge_attr: Edge attributes
            boundary_conditions: Boundary condition features [num_nodes, bc_dim]
        
        Returns:
            Predicted flow fields
        """
        # Encode geometry
        encoded = self.encoder(x, edge_index, edge_attr)
        
        # Add boundary conditions if provided
        if boundary_conditions is not None:
            encoded = encoded + boundary_conditions
        
        # Decode to flow fields
        output = self.decoder(encoded, edge_index, edge_attr)
        
        return output

