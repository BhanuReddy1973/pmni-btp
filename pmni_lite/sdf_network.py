"""
SDF Network with optional positional encoding and normal prediction.
Inspired by models/fields.py from PMNI but CPU-friendly.
"""
import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """Simple sin/cos positional encoding"""
    def __init__(self, num_freqs=6, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        
    def forward(self, x):
        """
        x: (*, 3)
        returns: (*, encoding_dim)
        """
        if self.num_freqs == 0:
            return x
            
        freq_bands = 2.0 ** torch.linspace(0, self.num_freqs - 1, self.num_freqs, device=x.device)
        out = []
        if self.include_input:
            out.append(x)
        for freq in freq_bands:
            out.append(torch.sin(freq * np.pi * x))
            out.append(torch.cos(freq * np.pi * x))
        return torch.cat(out, dim=-1)
    
    @property
    def output_dim(self):
        dim = 0
        if self.include_input:
            dim += 3
        dim += 2 * self.num_freqs * 3
        return dim


class SDFNetwork(nn.Module):
    """
    SDF network with geometric initialization (SIREN-like) and optional encoding.
    
    Architecture:
    - Input: xyz (optionally encoded)
    - Hidden layers with Softplus activation
    - Output: SDF value + optional surface normal prediction
    """
    def __init__(self, 
                 d_in=3,
                 d_hidden=256,
                 n_layers=8,
                 skip_in=[4],
                 multires=6,
                 bias=0.5,
                 geometric_init=True,
                 weight_norm=True,
                 predict_normals=False):
        super().__init__()
        
        self.predict_normals = predict_normals
        self.skip_in = skip_in
        
        # Positional encoding
        self.encoding = PositionalEncoding(num_freqs=multires, include_input=True) if multires > 0 else None
        d_in_enc = self.encoding.output_dim if self.encoding is not None else d_in
        
        # SDF layers
        dims = [d_in_enc] + [d_hidden] * n_layers + [1]
        self.num_layers = len(dims)
        
        for l in range(self.num_layers - 1):
            if l + 1 in skip_in:
                in_dim = dims[l] + d_in_enc
            else:
                in_dim = dims[l]
            out_dim = dims[l + 1]
            
            lin = nn.Linear(in_dim, out_dim)
            
            # Geometric initialization (inspired by SIREN/IGR)
            if geometric_init:
                if l == self.num_layers - 2:
                    # Last layer
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    # First layer with encoding
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)  # zero init for sin/cos features
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in skip_in:
                    # Skip connection layer
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(3 + 2 * multires * 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            
            if weight_norm and l < self.num_layers - 2:
                lin = nn.utils.weight_norm(lin)
                
            setattr(self, f"lin{l}", lin)
        
        self.activation = nn.Softplus(beta=100)
        
        # Optional normal prediction head
        if predict_normals:
            self.normal_head = nn.Sequential(
                nn.Linear(d_hidden, d_hidden // 2),
                nn.Softplus(beta=100),
                nn.Linear(d_hidden // 2, 3)
            )
    
    def forward(self, x):
        """
        x: (*, 3) input coordinates
        returns: (*, 1) SDF values
        """
        if self.encoding is not None:
            x_enc = self.encoding(x)
        else:
            x_enc = x
            
        h = x_enc
        for l in range(self.num_layers - 1):
            lin = getattr(self, f"lin{l}")

            # Note: constructor uses `if l + 1 in skip_in` when sizing layers,
            # so we must match that convention here when concatenating the encoding.
            if (l + 1) in self.skip_in:
                h = torch.cat([h, x_enc], dim=-1)
            
            h = lin(h)
            
            if l < self.num_layers - 2:
                h = self.activation(h)
        
        sdf = h
        return sdf
    
    def forward_with_nablas(self, x):
        """
        Compute SDF and gradients (surface normals).
        x: (*, 3)
        returns: sdf (*, 1), nabla (*, 3)
        """
        x.requires_grad_(True)
        sdf = self.forward(x)
        
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        nabla = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        return sdf, nabla
    
    def gradient(self, x):
        """Alias for forward_with_nablas for compatibility"""
        return self.forward_with_nablas(x)
