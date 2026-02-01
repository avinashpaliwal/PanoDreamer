"""
Depth estimation using Depth Anything V2.
"""

import os
import sys
import numpy as np
import torch

# Add Depth-Anything-V2 to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DAV2_PATH = os.path.join(os.path.dirname(_SCRIPT_DIR), 'Depth-Anything-V2')
if _DAV2_PATH not in sys.path:
    sys.path.insert(0, _DAV2_PATH)

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Lazy loading - model initialized on first use
_model = None
_encoder = 'vitl'


def _get_model():
    """Load model on first use."""
    global _model
    if _model is None:
        checkpoint_path = os.path.join(os.path.dirname(_SCRIPT_DIR), 'checkpoints', f'depth_anything_v2_{_encoder}.pth')
        print(f'[INFO] Loading Depth Anything V2 ({_encoder}) from {checkpoint_path}')
        _model = DepthAnythingV2(**MODEL_CONFIGS[_encoder])
        _model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        _model = _model.to(DEVICE).eval()
    return _model


def estimate_depth(img, mode='test'):
    """
    Estimate depth from an image.
    
    Args:
        img: PIL Image or numpy array (RGB)
        mode: Unused, kept for compatibility
        
    Returns:
        depth: HxW numpy array of depth values
    """
    model = _get_model()
    img = np.asarray(img)[:, :, [2, 1, 0]]  # RGB to BGR

    with torch.no_grad():
        depth = model.infer_image(img)  # HxW raw depth map in numpy

    return depth