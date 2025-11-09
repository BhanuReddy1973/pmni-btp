"""
Simplified PMNI - Lightweight Neural SDF with Normal Supervision
Built from scratch for resource-constrained environments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleSDF(nn.Module):
    """Lightweight SDF network - plain MLP, no fancy encoding"""
    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input: xyz (3) -> hidden
        self.input_layer = nn.Linear(3, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        
        # Output: SDF value (1)
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Initialize with geometric prior (inside negative, outside positive)
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Geometric initialization - bias network toward sphere
        for layer in self.hidden_layers:
            nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(self.hidden_dim))
            nn.init.constant_(layer.bias, 0.0)
        
        # Output layer - bias toward zero level set
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.0001)
        nn.init.constant_(self.output_layer.bias, 0.3)
    
    def forward(self, x):
        """
        Args:
            x: (N, 3) xyz coordinates
        Returns:
            sdf: (N, 1) signed distance values
        """
        h = F.relu(self.input_layer(x))
        
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        
        sdf = self.output_layer(h)
        return sdf
    
    def gradient(self, x):
        """Compute SDF gradient (normal) using autodiff"""
        x.requires_grad_(True)
        y = self.forward(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradient = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        return gradient


class VarianceNetwork(nn.Module):
    """Learns surface thickness (variance) for volume rendering"""
    def __init__(self, init_val=0.3):
        super().__init__()
        self.variance = nn.Parameter(torch.tensor(init_val))
    
    def forward(self, x):
        return torch.ones_like(x[..., :1]) * torch.exp(self.variance * 10.0)


class SimplePMNI(nn.Module):
    """Complete simplified PMNI model"""
    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()
        
        self.sdf_network = SimpleSDF(hidden_dim, num_layers)
        self.variance_network = VarianceNetwork(init_val=0.3)
        
    def forward(self, points):
        """
        Args:
            points: (N, 3) 3D points
        Returns:
            sdf: (N, 1) signed distance
            normals: (N, 3) surface normals
        """
        sdf = self.sdf_network(points)
        normals = self.sdf_network.gradient(points)
        normals = F.normalize(normals, dim=-1)
        return sdf, normals
    
    def get_variance(self, points):
        return self.variance_network(points)


if __name__ == "__main__":
    # Quick test
    model = SimplePMNI(hidden_dim=32, num_layers=3)
    points = torch.randn(100, 3) * 0.5
    sdf, normals = model(points)
    print(f"SDF shape: {sdf.shape}, range: [{sdf.min():.3f}, {sdf.max():.3f}]")
    print(f"Normals shape: {normals.shape}, norm: {normals.norm(dim=-1).mean():.3f}")
    print("✓ Model test passed!")
