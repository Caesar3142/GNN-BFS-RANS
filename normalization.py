"""
Feature normalization utilities for flow field data.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple


class FieldNormalizer:
    """Normalizes flow fields to improve training."""
    
    def __init__(self):
        self.scalers = {}
        self.field_stats = {}
    
    def fit(self, fields_dict: Dict[str, np.ndarray]):
        """
        Fit normalizers on field data.
        
        Args:
            fields_dict: Dictionary of field arrays
        """
        for field_name, field_data in fields_dict.items():
            if field_name == 'U':
                # For velocity, normalize each component separately
                field_flat = field_data.flatten()
            else:
                field_flat = field_data.flatten()
            
            # Compute statistics
            mean = np.mean(field_flat)
            std = np.std(field_flat)
            
            self.field_stats[field_name] = {
                'mean': mean,
                'std': std,
                'min': np.min(field_flat),
                'max': np.max(field_flat)
            }
            
            # Store mean and std for normalization
            self.scalers[field_name] = {
                'mean': mean,
                'std': std if std > 1e-10 else 1.0  # Avoid division by zero
            }
    
    def transform(self, fields_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize fields."""
        normalized = {}
        for field_name, field_data in fields_dict.items():
            if field_name not in self.scalers:
                normalized[field_name] = field_data
                continue
            
            scaler = self.scalers[field_name]
            mean = scaler['mean']
            std = scaler['std']
            
            # Normalize: (x - mean) / std
            normalized[field_name] = (field_data - mean) / std
        
        return normalized
    
    def inverse_transform(self, fields_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Denormalize fields."""
        denormalized = {}
        for field_name, field_data in fields_dict.items():
            if field_name not in self.scalers:
                denormalized[field_name] = field_data
                continue
            
            scaler = self.scalers[field_name]
            mean = scaler['mean']
            std = scaler['std']
            
            # Denormalize: x * std + mean
            denormalized[field_name] = field_data * std + mean
        
        return denormalized


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss for different flow fields."""
    
    def __init__(self, field_weights: Dict[str, float] = None):
        """
        Initialize weighted loss.
        
        Args:
            field_weights: Dictionary of weights for each field
                          Default: U=1.0, p=1.0, k=0.5, epsilon=0.5, nut=0.5
        """
        super(WeightedMSELoss, self).__init__()
        
        if field_weights is None:
            # Default weights - turbulence fields get lower weight
            field_weights = {
                'U': 1.0,      # Velocity is most important
                'p': 1.0,      # Pressure is important
                'k': 0.5,      # Turbulence fields less critical
                'epsilon': 0.5,
                'nut': 0.5
            }
        
        self.field_weights = field_weights
        
        # Create weight tensor: [U(3), p(1), k(1), epsilon(1), nut(1)]
        self.weights = torch.tensor([
            field_weights.get('U', 1.0),
            field_weights.get('U', 1.0),
            field_weights.get('U', 1.0),
            field_weights.get('p', 1.0),
            field_weights.get('k', 0.5),
            field_weights.get('epsilon', 0.5),
            field_weights.get('nut', 0.5)
        ], dtype=torch.float32)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MSE loss.
        
        Args:
            pred: Predicted fields [batch_size, 7] or [num_nodes, 7]
            target: Target fields [batch_size, 7] or [num_nodes, 7]
        
        Returns:
            Weighted MSE loss
        """
        # Move weights to same device as pred
        if self.weights.device != pred.device:
            self.weights = self.weights.to(pred.device)
        
        # Compute squared error
        squared_error = (pred - target) ** 2
        
        # Apply field weights
        weighted_error = squared_error * self.weights.unsqueeze(0)
        
        # Return mean weighted MSE
        return weighted_error.mean()

