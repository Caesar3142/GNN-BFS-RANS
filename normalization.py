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
                # For velocity, normalize each component (Ux, Uy, Uz) separately
                # This is important because components may have different scales
                if len(field_data.shape) == 2 and field_data.shape[1] == 3:
                    # Store per-component statistics
                    self.scalers[field_name] = {
                        'mean': np.mean(field_data, axis=0),  # [3] array
                        'std': np.std(field_data, axis=0),     # [3] array
                        'per_component': True
                    }
                    # Also store overall stats for reference
                    field_flat = field_data.flatten()
                    self.field_stats[field_name] = {
                        'mean': np.mean(field_flat),
                        'std': np.std(field_flat),
                        'min': np.min(field_flat),
                        'max': np.max(field_flat),
                        'per_component_mean': self.scalers[field_name]['mean'].tolist(),
                        'per_component_std': self.scalers[field_name]['std'].tolist()
                    }
                    # Ensure std > 0
                    self.scalers[field_name]['std'] = np.where(
                        self.scalers[field_name]['std'] > 1e-10,
                        self.scalers[field_name]['std'],
                        1.0
                    )
                else:
                    # Fallback: flatten if not in expected format
                    field_flat = field_data.flatten()
                    mean = np.mean(field_flat)
                    std = np.std(field_flat)
                    self.field_stats[field_name] = {
                        'mean': mean,
                        'std': std,
                        'min': np.min(field_flat),
                        'max': np.max(field_flat)
                    }
                    self.scalers[field_name] = {
                        'mean': mean,
                        'std': std if std > 1e-10 else 1.0,
                        'per_component': False
                    }
            else:
                # Scalar fields: normalize as usual
                field_flat = field_data.flatten()
                mean = np.mean(field_flat)
                std = np.std(field_flat)
                
                self.field_stats[field_name] = {
                    'mean': mean,
                    'std': std,
                    'min': np.min(field_flat),
                    'max': np.max(field_flat)
                }
                
                self.scalers[field_name] = {
                    'mean': mean,
                    'std': std if std > 1e-10 else 1.0,
                    'per_component': False
                }
    
    def transform(self, fields_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize fields."""
        normalized = {}
        for field_name, field_data in fields_dict.items():
            if field_name not in self.scalers:
                normalized[field_name] = field_data
                continue
            
            scaler = self.scalers[field_name]
            
            if field_name == 'U' and scaler.get('per_component', False):
                # Normalize each velocity component separately
                mean = scaler['mean']  # [3] array
                std = scaler['std']    # [3] array
                # Broadcast subtraction and division
                normalized[field_name] = (field_data - mean) / std
            else:
                # Standard normalization for scalars
                mean = scaler['mean']
                std = scaler['std']
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
            
            if field_name == 'U' and scaler.get('per_component', False):
                # Denormalize each velocity component separately
                mean = scaler['mean']  # [3] array
                std = scaler['std']    # [3] array
                # Broadcast multiplication and addition
                denormalized[field_name] = field_data * std + mean
            else:
                # Standard denormalization for scalars
                mean = scaler['mean']
                std = scaler['std']
                denormalized[field_name] = field_data * std + mean
        
        return denormalized


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss for different flow fields with field-wise computation."""
    
    def __init__(self, field_weights: Dict[str, float] = None, use_fieldwise: bool = True):
        """
        Initialize weighted loss.
        
        Args:
            field_weights: Dictionary of weights for each field
                          Default: U=1.0, p=1.0, k=0.5, epsilon=0.5, nut=0.5
            use_fieldwise: If True, compute loss per field and then combine (better for normalized fields)
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
        self.use_fieldwise = use_fieldwise
        
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
        if self.use_fieldwise:
            # Compute loss per field, then combine with weights
            # This is better when fields are normalized independently
            field_losses = []
            
            # U (3 components)
            u_pred = pred[:, 0:3]
            u_target = target[:, 0:3]
            u_loss = torch.mean((u_pred - u_target) ** 2)
            field_losses.append(self.field_weights.get('U', 1.0) * u_loss)
            
            # p (1 component)
            p_pred = pred[:, 3:4]
            p_target = target[:, 3:4]
            p_loss = torch.mean((p_pred - p_target) ** 2)
            field_losses.append(self.field_weights.get('p', 1.0) * p_loss)
            
            # k (1 component)
            k_pred = pred[:, 4:5]
            k_target = target[:, 4:5]
            k_loss = torch.mean((k_pred - k_target) ** 2)
            field_losses.append(self.field_weights.get('k', 0.5) * k_loss)
            
            # epsilon (1 component)
            eps_pred = pred[:, 5:6]
            eps_target = target[:, 5:6]
            eps_loss = torch.mean((eps_pred - eps_target) ** 2)
            field_losses.append(self.field_weights.get('epsilon', 0.5) * eps_loss)
            
            # nut (1 component)
            nut_pred = pred[:, 6:7]
            nut_target = target[:, 6:7]
            nut_loss = torch.mean((nut_pred - nut_target) ** 2)
            field_losses.append(self.field_weights.get('nut', 0.5) * nut_loss)
            
            # Sum weighted field losses
            total_loss = sum(field_losses)
            return total_loss
        else:
            # Original method: element-wise weighting
            # Move weights to same device as pred
            if self.weights.device != pred.device:
                self.weights = self.weights.to(pred.device)
            
            # Compute squared error
            squared_error = (pred - target) ** 2
            
            # Apply field weights
            weighted_error = squared_error * self.weights.unsqueeze(0)
            
            # Return mean weighted MSE
            return weighted_error.mean()

