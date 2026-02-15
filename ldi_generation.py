"""
Layered Depth Image (LDI) Generation

Creates a multi-layer representation from a panorama and its depth:
1. Splits depth into multiple layers (bins)
2. For each layer, inpaints occluded background regions
3. Outputs RGBA + depth for each layer

This is the first step toward 3D Gaussian Splatting scene creation.

Requirements:
- Clone 3d-moments repo for inpainting networks:
  git clone https://github.com/google-research/3d-moments.git
- Download inpainting checkpoints (see 3d-moments/download.sh)
- Place checkpoints in inpainting_ckpts/

Paper: PanoDreamer - https://people.engr.tamu.edu/nimak/Papers/PanoDreamer/
"""

import os
import sys
import math
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Import our utils first before adding 3d-moments to path
from utils.depth_layering import get_depth_bins
from utils.depth import colorize

# Add 3d-moments to path for inpainting networks
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_3D_MOMENTS_PATH = os.path.join(_SCRIPT_DIR, '3d-moments')
if os.path.exists(_3D_MOMENTS_PATH) and _3D_MOMENTS_PATH not in sys.path:
    sys.path.insert(0, _3D_MOMENTS_PATH)

# Optional: bilateral filtering for depth refinement
try:
    from utils_backup.bilateral_filtering import sparse_bilateral_filtering
    HAS_BILATERAL = True
except ImportError:
    HAS_BILATERAL = False


def load_inpainter(device='cuda'):
    """
    Load the inpainting model from 3d-moments submodule.
    
    Returns:
        Inpainter instance or None if not available
    """
    try:
        from core.inpainter import Inpainter
        return Inpainter(args=None, device=device)
    except Exception as e:
        print(f"[WARNING] Could not load Inpainter: {e}")
        print("[INFO] LDI generation will skip inpainting (foreground layer only)")
        return None


def generate_ldi(image, depth, num_layers=4, inpainter=None, save_dir=None, debug=False):
    """
    Generate Layered Depth Image from panorama RGB and depth.
    
    Args:
        image: RGB image as numpy array [H, W, 3] in range [0, 1]
        depth: Depth map as numpy array [H, W]
        num_layers: Number of depth layers
        inpainter: Inpainter instance (optional)
        save_dir: Directory to save outputs
        debug: Save intermediate visualizations
        
    Returns:
        rgba_layers: [num_layers, H, W, 4] - RGBA for each layer
        depth_layers: [num_layers, H, W] - Depth for each layer
        mask_layers: [num_layers, H, W] - Original masks before inpainting
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h, w = image.shape[:2]
    
    # Convert to tensors
    rgb_tensor = torch.tensor(image).permute(2, 0, 1)[None].float().to(device)  # [1, 3, H, W]
    depth_tensor = torch.tensor(depth)[None, None].float().to(device)  # [1, 1, H, W]
    
    # Get depth bins using clustering
    print(f'[INFO] Computing depth bins for {num_layers} layers...')
    depth_bins = get_depth_bins(depth=depth_tensor, num_bins=num_layers)
    print(f'[INFO] Depth bins: {[f"{b:.2f}" for b in depth_bins]}')
    
    if inpainter is not None:
        # Use 3d-moments inpainting for full LDI
        print(f'[INFO] Running sequential inpainting...')
        rgba_layers, depth_layers, mask_layers = inpainter.sequential_inpainting(
            rgb_tensor, depth_tensor, depth_bins
        )
        
        # Convert to numpy [N, H, W, C]
        rgba_layers = rgba_layers.squeeze(1).permute(0, 2, 3, 1).cpu().numpy()
        depth_layers = depth_layers.squeeze(1).squeeze(1).cpu().numpy()
        mask_layers = mask_layers.squeeze(1).squeeze(1).cpu().numpy()
    else:
        # Fallback: simple layer separation without inpainting
        print(f'[INFO] Creating layers without inpainting...')
        rgba_layers = []
        depth_layers = []
        mask_layers = []
        
        for i in range(num_layers):
            # Create mask for this depth range
            if i == num_layers - 1:
                mask = (depth >= depth_bins[i]) & (depth <= depth_bins[i+1])
            else:
                mask = (depth >= depth_bins[i]) & (depth < depth_bins[i+1])
            
            mask = mask.astype(np.float32)
            
            # Create RGBA with alpha from mask
            rgba = np.concatenate([image * mask[..., None], mask[..., None]], axis=-1)
            depth_layer = depth * mask
            
            rgba_layers.append(rgba)
            depth_layers.append(depth_layer)
            mask_layers.append(mask)
        
        rgba_layers = np.stack(rgba_layers)
        depth_layers = np.stack(depth_layers)
        mask_layers = np.stack(mask_layers)
    
    # Save outputs
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(f'{save_dir}/rgba_ldi.npy', rgba_layers)
        np.save(f'{save_dir}/depth_ldi.npy', depth_layers)
        np.save(f'{save_dir}/mask_ldi.npy', mask_layers)
        
        print(f'[INFO] Saved: {save_dir}/rgba_ldi.npy')
        print(f'[INFO] Saved: {save_dir}/depth_ldi.npy')
        print(f'[INFO] Saved: {save_dir}/mask_ldi.npy')
        
        # Save visualizations
        for i in range(len(rgba_layers)):
            rgba_img = (rgba_layers[i] * 255).astype(np.uint8)
            Image.fromarray(rgba_img).save(f'{save_dir}/layer_{i:02d}_rgba.png')
            
            if debug:
                depth_vis = colorize(depth_layers[i], cmap='turbo')
                cv2.imwrite(f'{save_dir}/layer_{i:02d}_depth.png', depth_vis[..., :3][..., ::-1])
                
                mask_vis = (mask_layers[i] * 255).astype(np.uint8)
                cv2.imwrite(f'{save_dir}/layer_{i:02d}_mask.png', mask_vis)
    
    return rgba_layers, depth_layers, mask_layers


def main():
    parser = argparse.ArgumentParser(
        description='Generate Layered Depth Image (LDI) from panorama'
    )
    parser.add_argument('--input_image', type=str, required=True,
                        help='Path to input panorama image')
    parser.add_argument('--input_depth', type=str, required=True,
                        help='Path to depth .npy file')
    parser.add_argument('--output_dir', type=str, default='output_ldi',
                        help='Output directory')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of depth layers')
    parser.add_argument('--refine_depth', action='store_true',
                        help='Apply bilateral filtering to refine depth')
    parser.add_argument('--debug', action='store_true',
                        help='Save debug visualizations')
    
    args = parser.parse_args()
    
    # Load inputs
    print(f'[INFO] Loading image: {args.input_image}')
    image = np.array(Image.open(args.input_image).convert('RGB')).astype(np.float32) / 255.
    
    print(f'[INFO] Loading depth: {args.input_depth}')
    depth = np.load(args.input_depth).astype(np.float32)
    
    print(f'[INFO] Image size: {image.shape[1]}x{image.shape[0]}')
    print(f'[INFO] Depth range: {depth.min():.2f} - {depth.max():.2f}')
    
    # Optional: refine depth with bilateral filtering
    if args.refine_depth and HAS_BILATERAL:
        print(f'[INFO] Applying bilateral filtering to depth...')
        depth_rel = 1 / depth
        config = {'sparse_iter': 5}  # Simplified config
        _, vis_depths = sparse_bilateral_filtering(
            depth_rel.copy(), 
            (image * 255).astype(np.uint8), 
            config, 
            num_iter=config['sparse_iter'], 
            spdb=False
        )
        depth = 1 / vis_depths[-1]
    
    # Load inpainter
    inpainter = load_inpainter()
    
    # Generate LDI
    rgba_layers, depth_layers, mask_layers = generate_ldi(
        image=image,
        depth=depth,
        num_layers=args.num_layers,
        inpainter=inpainter,
        save_dir=args.output_dir,
        debug=args.debug
    )
    
    print(f'[INFO] Generated {len(rgba_layers)} layers')
    print(f'[INFO] Done!')


if __name__ == '__main__':
    main()
