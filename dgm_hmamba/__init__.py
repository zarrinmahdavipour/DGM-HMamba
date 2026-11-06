"""DGM-HMamba: Domain-adaptive Diffusion-Guided Hierarchical Mamba
for Cross-Domain Few-Shot Hyperspectral Image Classification.

Clean re-implementation based on the Neurocomputing manuscript.
"""

__version__ = "1.0.0"

from .config import DGMHMambaConfig, get_default_config
from .model import DGMHMamba
from .data import get_dataset, make_cross_domain_split

__all__ = ["DGMHMambaConfig", "get_default_config", "DGMHMamba", "get_dataset", "make_cross_domain_split"]
