"""
Graph Construction Module
Builds graph structure from OpenFOAM mesh for GNN training.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, Optional


class GraphConstructor:
    """Constructs graph from OpenFOAM mesh data."""
    
    def __init__(self, mesh_data: Dict):
        """
        Initialize with mesh data.
        
        Args:
            mesh_data: Dictionary from OpenFOAMLoader.load_mesh()
        """
        self.mesh_data = mesh_data
        self.owner = mesh_data['owner']
        self.neighbour = mesh_data['neighbour']
        self.cell_centers = mesh_data['cell_centers']
        self.n_cells = mesh_data['n_cells']
    
    def build_edge_index(self) -> torch.Tensor:
        """
        Build edge index from owner-neighbour connectivity.
        Creates bidirectional edges for cell-to-cell connections.
        
        Returns:
            Edge index tensor of shape [2, num_edges]
        """
        edges = []
        
        # Internal faces connect owner and neighbour cells
        for i in range(len(self.neighbour)):
            owner_cell = self.owner[i]
            neighbour_cell = self.neighbour[i]
            
            # Bidirectional edges
            edges.append([owner_cell, neighbour_cell])
            edges.append([neighbour_cell, owner_cell])
        
        # Boundary faces: owner cells connect to themselves (self-loops)
        # or we can skip them. For now, we'll add self-loops for boundary cells
        boundary_start = len(self.neighbour)
        for i in range(boundary_start, len(self.owner)):
            owner_cell = self.owner[i]
            # Add self-loop for boundary cells
            edges.append([owner_cell, owner_cell])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def compute_edge_attributes(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute edge attributes (e.g., distance, direction vector).
        
        Args:
            edge_index: Edge connectivity tensor
        
        Returns:
            Edge attributes tensor of shape [num_edges, edge_dim]
        """
        edge_attr = []
        
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            
            if src == dst:
                # Self-loop: zero vector
                edge_attr.append([0.0, 0.0, 0.0, 0.0])
            else:
                # Compute distance and direction
                src_pos = self.cell_centers[src]
                dst_pos = self.cell_centers[dst]
                
                direction = dst_pos - src_pos
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    direction = direction / distance
                
                edge_attr.append([direction[0], direction[1], direction[2], distance])
        
        return torch.tensor(edge_attr, dtype=torch.float32)
    
    def build_graph(self, field_data: Optional[Dict] = None, 
                   node_features: Optional[np.ndarray] = None,
                   filter_internal: bool = False,
                   n_internal_cells: Optional[int] = None) -> Data:
        """
        Build PyTorch Geometric Data object from mesh and fields.
        
        Args:
            field_data: Dictionary of field arrays from OpenFOAMLoader
            node_features: Optional pre-computed node features
            filter_internal: If True, only include internal cells
            n_internal_cells: If provided, use exactly this many cells (cells 0 to n-1)
        
        Returns:
            PyTorch Geometric Data object
        """
        # Get internal cell mask if filtering
        if filter_internal:
            if n_internal_cells is not None:
                # Use exact number of cells from field data
                n_nodes = n_internal_cells
                internal_indices = np.arange(n_internal_cells)
                internal_mask = np.zeros(self.n_cells, dtype=bool)
                internal_mask[:n_internal_cells] = True
            elif 'internal_mask' in self.mesh_data:
                internal_mask = self.mesh_data['internal_mask']
                internal_indices = np.where(internal_mask)[0]
                n_nodes = len(internal_indices)
            else:
                # Fallback: use all cells
                internal_mask = None
                internal_indices = np.arange(self.n_cells)
                n_nodes = self.n_cells
                filter_internal = False
            
            # Create mapping from old cell index to new index
            old_to_new = np.full(self.n_cells, -1, dtype=np.int32)
            old_to_new[internal_indices] = np.arange(n_nodes)
        else:
            internal_mask = None
            internal_indices = np.arange(self.n_cells)
            old_to_new = np.arange(self.n_cells)
            n_nodes = self.n_cells
        
        # Build edge connectivity (only for internal cells if filtering)
        if filter_internal and internal_mask is not None:
            edges = []
            # Only include edges between internal cells
            for i in range(len(self.neighbour)):
                owner_cell = self.owner[i]
                neighbour_cell = self.neighbour[i]
                
                if internal_mask[owner_cell] and internal_mask[neighbour_cell]:
                    # Map to new indices
                    new_owner = old_to_new[owner_cell]
                    new_neighbour = old_to_new[neighbour_cell]
                    # Ensure valid indices
                    if new_owner >= 0 and new_neighbour >= 0:
                        # Bidirectional edges
                        edges.append([new_owner, new_neighbour])
                        edges.append([new_neighbour, new_owner])
            
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = self.build_edge_index()
            # Remap edge indices if filtering
            if filter_internal and internal_mask is not None:
                # Convert edge_index to numpy for indexing, then back to torch
                edge_index_np = edge_index.cpu().numpy()
                edge_index_remapped = old_to_new[edge_index_np]
                edge_index = torch.from_numpy(edge_index_remapped).long()
                # Remove edges that map to -1 (non-internal cells)
                valid_edges = (edge_index[0] >= 0) & (edge_index[1] >= 0)
                edge_index = edge_index[:, valid_edges]
        
        # Validate edge_index: ensure all indices are within valid range
        if edge_index.shape[1] > 0:
            max_idx = edge_index.max().item()
            if max_idx >= n_nodes:
                # Filter out invalid edges
                valid_edges = (edge_index[0] < n_nodes) & (edge_index[1] < n_nodes)
                edge_index = edge_index[:, valid_edges]
        
        # Add self-loops for isolated nodes to ensure connectivity
        if edge_index.shape[1] > 0 and n_nodes > 0:
            # Find nodes that have at least one edge
            connected_nodes = torch.unique(edge_index.flatten()).cpu().numpy()
            # Find isolated nodes (nodes with no edges)
            all_nodes = np.arange(n_nodes)
            isolated_mask = ~np.isin(all_nodes, connected_nodes)
            isolated_nodes = all_nodes[isolated_mask]
            
            # Add self-loops for isolated nodes
            if len(isolated_nodes) > 0:
                self_loops = torch.tensor([isolated_nodes, isolated_nodes], dtype=torch.long)
                edge_index = torch.cat([edge_index, self_loops], dim=1)
        
        # Compute edge attributes
        if edge_index.shape[1] > 0:
            # Get cell centers for edge attributes
            if filter_internal and internal_mask is not None:
                cell_centers = self.cell_centers[internal_indices]
            else:
                cell_centers = self.cell_centers
            
            edge_attr = []
            for i in range(edge_index.shape[1]):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                
                # Validate indices
                if src < 0 or src >= len(cell_centers) or dst < 0 or dst >= len(cell_centers):
                    # Invalid edge - use zero attributes
                    edge_attr.append([0.0, 0.0, 0.0, 0.0])
                    continue
                
                if src == dst:
                    edge_attr.append([0.0, 0.0, 0.0, 0.0])
                else:
                    src_pos = cell_centers[src]
                    dst_pos = cell_centers[dst]
                    direction = dst_pos - src_pos
                    distance = np.linalg.norm(direction)
                    if distance > 0:
                        direction = direction / distance
                    edge_attr.append([direction[0], direction[1], direction[2], distance])
            
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            # If no edges, add self-loops for all nodes to ensure connectivity
            if n_nodes > 0:
                self_loops = torch.arange(n_nodes, dtype=torch.long).repeat(2, 1)
                edge_index = self_loops
                edge_attr = torch.zeros((n_nodes, 4), dtype=torch.float32)
            else:
                edge_attr = torch.empty((0, 4), dtype=torch.float32)
        
        # Node features: cell center coordinates + optional field data
        if node_features is None:
            if filter_internal and internal_mask is not None:
                node_features = self.cell_centers[internal_indices].copy()
            else:
                node_features = self.cell_centers.copy()
        else:
            if filter_internal and internal_mask is not None:
                node_features = node_features[internal_indices].copy()
            else:
                node_features = node_features.copy()
        
        # Add field data as node features if provided
        if field_data is not None:
            feature_list = [node_features]
            
            # Add velocity components
            if 'U' in field_data:
                feature_list.append(field_data['U'])
            
            # Add scalar fields
            for field_name in ['p', 'k', 'epsilon', 'nut']:
                if field_name in field_data:
                    # Reshape scalar to [n_cells, 1]
                    scalar = field_data[field_name].reshape(-1, 1)
                    feature_list.append(scalar)
            
            node_features = np.hstack(feature_list)
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=n_nodes
        )
        
        return data
    
    def get_boundary_mask(self, boundary_name: str) -> np.ndarray:
        """
        Get boolean mask for cells on a specific boundary.
        
        Args:
            boundary_name: Name of boundary patch
        
        Returns:
            Boolean array of shape [n_cells]
        """
        if boundary_name not in self.mesh_data['boundaries']:
            raise ValueError(f"Boundary {boundary_name} not found")
        
        boundary_info = self.mesh_data['boundaries'][boundary_name]
        start_face = boundary_info['startFace']
        n_faces = boundary_info['nFaces']
        
        mask = np.zeros(self.n_cells, dtype=bool)
        
        for i in range(start_face, start_face + n_faces):
            if i < len(self.owner):
                cell_idx = self.owner[i]
                mask[cell_idx] = True
        
        return mask

