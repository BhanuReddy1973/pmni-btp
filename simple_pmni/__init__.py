"""
Simplified PMNI - Lightweight neural SDF with normal supervision
"""

from .model import SimplePMNI, SimpleSDF, VarianceNetwork
from .renderer import generate_rays, render_normals, simple_volume_rendering
from .dataset import DiLiGentDataset

__version__ = "0.1.0"
__all__ = [
    'SimplePMNI',
    'SimpleSDF', 
    'VarianceNetwork',
    'DiLiGentDataset',
    'generate_rays',
    'render_normals',
    'simple_volume_rendering'
]
