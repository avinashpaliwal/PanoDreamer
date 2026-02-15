"""
3D Gaussian Splatting Scene Optimization from Panorama LDI

Initializes a 3DGS scene from panorama Layered Depth Images and optimizes it
to match perspective views extracted from the panorama.

Pipeline:
1. Load panorama LDI layers (RGBA + depth)
2. Extract perspective views via cylindrical projection
3. Unproject to 3D point cloud and initialize Gaussians
4. Optimize Gaussians with rendering loss
5. Save optimized scene as PLY

Usage:
    python train_gsplat.py --ldi_dir output_ldi --output scene.ply
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import kornia
from kornia.utils import create_meshgrid

from gsplat import rasterization


def fov2focal(fov_radians, pixels):
    """Convert field of view to focal length."""
    return pixels / (2 * math.tan(fov_radians / 2))


def cyl_proj(img, f):
    """
    Apply cylindrical projection to convert panorama to perspective view.
    
    Args:
        img: [B, C, H, W] tensor in panorama space
        f: Focal length
    
    Returns:
        Perspective image [B, C, H, W]
    """
    temp = create_meshgrid(img.shape[2], img.shape[3], normalized_coordinates=False, device=img.device)
    y, x = temp[..., 0], temp[..., 1]
    h, w = img.shape[2:]
    center_x = w // 2
    center_y = h // 2
    
    x_shifted = (x - center_x)
    y_shifted = (y - center_y)
    
    theta = torch.arctan(x_shifted / f)
    height = y_shifted / torch.sqrt(x_shifted ** 2 + f ** 2)
    
    x_cyl = (f * theta + center_x)
    y_cyl = (height * f + center_y)
    
    img_cyl = kornia.geometry.transform.remap(img, torch.flip(x_cyl, dims=(1, 2)), y_cyl, mode='nearest', align_corners=True)
    img_cyl = torch.rot90(img_cyl, k=3, dims=(2, 3))
    return img_cyl


def load_panorama_ldi(ldi_dir):
    """
    Load panorama LDI layers from directory.
    
    Returns:
        rgba_pano: [num_layers, H, W, 4] numpy array (panorama)
        depth_pano: [num_layers, H, W] numpy array (panorama)
    """
    rgba_path = os.path.join(ldi_dir, 'rgba_ldi.npy')
    depth_path = os.path.join(ldi_dir, 'depth_ldi.npy')
    
    print(f'[INFO] Loading panorama LDI from {ldi_dir}')
    rgba_pano = np.load(rgba_path)
    depth_pano = np.load(depth_path)
    
    print(f'[INFO] Panorama LDI shape: {rgba_pano.shape}, depth: {depth_pano.shape}')
    
    # Duplicate horizontally for seamless 360 wrapping
    rgba_pano = np.concatenate([rgba_pano, rgba_pano], axis=2)[::-1].copy()
    depth_pano = np.concatenate([depth_pano, depth_pano], axis=2)[::-1].copy()
    
    return rgba_pano, depth_pano


def extract_perspective_views(rgba_pano, depth_pano, num_views=16, fov_deg=44.702, view_size=512, device='cuda'):
    """
    Extract perspective views from panorama LDI by cropping and applying cylindrical projection.
    
    Args:
        rgba_pano: [num_layers, H, W, 4] panorama RGBA
        depth_pano: [num_layers, H, W] panorama depth
        num_views: Number of views to extract around 360 degrees
        fov_deg: Field of view in degrees
        view_size: Output view size (H=W)
        device: Device
    
    Returns:
        views: List of dicts with 'rgba' [num_layers, H, W, 4] and 'depth' [num_layers, H, W]
    """
    num_layers, h, w = rgba_pano.shape[:3]
    focal = fov2focal(fov_deg * math.pi / 180, view_size)
    
    # Step size for panorama sampling
    step = w // (2 * num_views)
    
    views = []
    
    print(f'[INFO] Extracting {num_views} perspective views from panorama...')
    for view_idx in tqdm(range(num_views), desc="Extracting views"):
        # Crop window from panorama
        center = w // 2 + step * view_idx
        crop_rgba = rgba_pano[:, :, center - view_size // 2:center + view_size // 2]
        crop_depth = depth_pano[:, :, center - view_size // 2:center + view_size // 2]
        
        # Apply cylindrical projection to convert to perspective
        rgba_tensor = torch.tensor(crop_rgba, device=device).permute(0, 3, 1, 2).float() / 255.0
        depth_tensor = torch.tensor(crop_depth, device=device).unsqueeze(1).float()
        
        rgba_persp = cyl_proj(rgba_tensor, focal)
        depth_persp = cyl_proj(depth_tensor, focal)
        
        # Convert back to numpy
        rgba_persp = rgba_persp.permute(0, 2, 3, 1).cpu().numpy()
        depth_persp = depth_persp.squeeze(1).cpu().numpy()
        
        views.append({
            'rgba': rgba_persp,
            'depth': depth_persp
        })
    
    return views


def unproject_ldi_to_points(rgba_layers, depth_layers, focal, device='cuda'):
    """
    Unproject LDI layers to 3D point cloud.
    
    Args:
        rgba_layers: [num_layers, H, W, 4]
        depth_layers: [num_layers, H, W]
        focal: Focal length
        device: Device
    
    Returns:
        points: [N, 3] 3D positions
        colors: [N, 3] RGB colors
        alphas: [N] alpha values
    """
    num_layers, H, W, _ = rgba_layers.shape
    
    # Camera intrinsics
    cx, cy = W / 2, H / 2
    
    all_points = []
    all_colors = []
    all_alphas = []
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    
    for layer_idx in range(num_layers):
        rgba = rgba_layers[layer_idx]
        depth = depth_layers[layer_idx]
        
        # Get valid pixels (alpha > threshold)
        alpha = rgba[..., 3].flatten()
        valid = alpha > 0.1
        
        if valid.sum() == 0:
            continue
        
        # Get depth and colors for valid pixels
        d = depth.flatten()[valid]
        rgb = rgba[..., :3].reshape(-1, 3)[valid] / 255.0
        a = alpha[valid]
        
        # Unproject to 3D (pinhole camera model)
        x = (u[valid] - cx) * d / focal
        y = (v[valid] - cy) * d / focal
        z = d
        
        xyz = np.stack([x, y, z], axis=-1)
        
        all_points.append(xyz)
        all_colors.append(rgb)
        all_alphas.append(a)
    
    if len(all_points) == 0:
        raise ValueError("No valid points in LDI")
    
    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    alphas = np.concatenate(all_alphas, axis=0)
    
    print(f'[INFO] Unprojected {len(points):,} points from {num_layers} layers')
    
    return (
        torch.tensor(points, dtype=torch.float32, device=device),
        torch.tensor(colors, dtype=torch.float32, device=device),
        torch.tensor(alphas, dtype=torch.float32, device=device)
    )


def initialize_gaussians(points, colors, alphas, device='cuda'):
    """
    Initialize Gaussian parameters from point cloud.
    
    Returns:
        Dictionary with: means, scales, quats, opacities, colors (all as nn.Parameter)
    """
    N = points.shape[0]
    
    # Positions
    means = nn.Parameter(points.clone())
    
    # Colors (RGB)
    colors_param = nn.Parameter(colors.clone())
    
    # Scales (log space, small initial size)
    init_scale = 0.01
    scales = nn.Parameter(torch.full((N, 3), math.log(init_scale), device=device))
    
    # Rotations (quaternions, identity)
    quats = nn.Parameter(torch.zeros((N, 4), device=device))
    quats.data[:, 0] = 1.0
    
    # Opacities (logit space, initialize from alpha)
    opacities = nn.Parameter(torch.logit(alphas.clamp(0.01, 0.99)))
    
    print(f'[INFO] Initialized {N:,} Gaussians')
    
    return {
        'means': means,
        'scales': scales,
        'quats': quats,
        'opacities': opacities,
        'colors': colors_param,
    }


def create_training_cameras(num_views=16, radius=4.0, focal=582.69, H=512, W=512, device='cuda'):
    """
    Create training camera poses (circular path around origin).
    
    Returns:
        List of dicts with: w2c, K, H, W
    """
    cameras = []
    
    for i in range(num_views):
        theta = i * (2 * np.pi) / num_views
        
        # Camera position
        eye = np.array([radius * np.cos(theta), 0, radius * np.sin(theta)])
        lookat = np.array([0, 0, 0])
        up = np.array([0, 1, 0])
        
        # Camera axes
        w = lookat - eye
        w = w / np.linalg.norm(w)
        u = np.cross(up, w)
        u = u / np.linalg.norm(u)
        v = np.cross(w, u)
        
        # Camera-to-world
        c2w = np.eye(4)
        c2w[:3, 0] = u
        c2w[:3, 1] = v
        c2w[:3, 2] = w
        c2w[:3, 3] = eye
        
        # World-to-camera
        w2c = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)
        
        # Intrinsics
        K = torch.tensor([
            [focal, 0, W / 2],
            [0, focal, H / 2],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        
        cameras.append({'w2c': w2c, 'K': K, 'H': H, 'W': W})
    
    return cameras


def render_gaussians(gaussians, camera, background=None):
    """
    Render Gaussians from a camera view.
    
    Args:
        gaussians: Dict with Gaussian parameters
        camera: Dict with w2c, K, H, W
        background: Background color [3] or None for white
    
    Returns:
        rgb: [H, W, 3]
        depth: [H, W]
        alpha: [H, W]
    """
    if background is None:
        background = torch.ones(3, device=gaussians['means'].device)
    
    # Get parameters
    means = gaussians['means']
    scales = torch.exp(gaussians['scales'])
    quats = F.normalize(gaussians['quats'], dim=-1)
    opacities = torch.sigmoid(gaussians['opacities'])
    colors = gaussians['colors']
    
    # Rasterize with depth
    render_colors, render_alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=camera['w2c'][None],
        Ks=camera['K'][None],
        width=camera['W'],
        height=camera['H'],
        packed=False,
        render_mode="RGB+ED",
        backgrounds=background[None],
    )
    
    rgb = render_colors[0, ..., :3]
    depth = render_colors[0, ..., 3]
    alpha = render_alphas[0, ..., 0]
    
    return rgb, depth, alpha


def compute_loss(rendered_rgb, target_rgb, rendered_depth, target_depth, mask=None, depth_weight=0.005):
    """
    Compute photometric and depth loss.
    
    Args:
        rendered_rgb: [H, W, 3] rendered image
        target_rgb: [H, W, 3] target image
        rendered_depth: [H, W] rendered depth
        target_depth: [H, W] target depth
        mask: [H, W] optional mask
        depth_weight: Weight for depth loss
    
    Returns:
        loss: scalar
    """
    if mask is not None:
        rendered_rgb = rendered_rgb * mask[..., None]
        target_rgb = target_rgb * mask[..., None]
        rendered_depth = rendered_depth * mask
        target_depth = target_depth * mask
    
    # L1 loss on RGB
    rgb_loss = F.l1_loss(rendered_rgb, target_rgb)
    
    # L2 loss on depth
    depth_loss = F.mse_loss(rendered_depth, target_depth)
    
    # Combined loss
    loss = rgb_loss + depth_weight * depth_loss
    
    return loss


def save_gaussians_ply(gaussians, output_path):
    """Save Gaussians to PLY file."""
    from plyfile import PlyData, PlyElement
    
    means = gaussians['means'].detach().cpu().numpy()
    scales = gaussians['scales'].detach().cpu().numpy()
    quats = gaussians['quats'].detach().cpu().numpy()
    opacities = gaussians['opacities'].detach().cpu().numpy()
    colors = gaussians['colors'].detach().cpu().numpy()
    
    # Convert colors to SH DC component
    SH_C0 = 0.28209479177387814
    f_dc = (colors - 0.5) / SH_C0
    
    N = means.shape[0]
    normals = np.zeros_like(means)
    
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]
    
    elements = np.empty(N, dtype=dtype)
    elements['x'] = means[:, 0]
    elements['y'] = means[:, 1]
    elements['z'] = means[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]
    elements['opacity'] = opacities
    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]
    elements['rot_0'] = quats[:, 0]
    elements['rot_1'] = quats[:, 1]
    elements['rot_2'] = quats[:, 2]
    elements['rot_3'] = quats[:, 3]
    
    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(output_path)
    
    print(f'[INFO] Saved {N:,} Gaussians to {output_path}')


def train(gaussians, cameras, target_images, target_depths, num_iterations=300, lr=0.001):
    """
    Optimize Gaussians to match target views.
    
    Args:
        gaussians: Dict with Gaussian parameters (nn.Parameter)
        cameras: List of camera dicts
        target_images: List of target images [H, W, 3]
        target_depths: List of target depths [H, W]
        num_iterations: Number of optimization steps
        lr: Learning rate
    """
    # Setup optimizer
    params = [
        {'params': [gaussians['means']], 'lr': lr, 'name': 'means'},
        {'params': [gaussians['scales']], 'lr': lr * 0.1, 'name': 'scales'},
        {'params': [gaussians['quats']], 'lr': lr * 0.1, 'name': 'quats'},
        {'params': [gaussians['opacities']], 'lr': lr, 'name': 'opacities'},
        {'params': [gaussians['colors']], 'lr': lr * 0.5, 'name': 'colors'},
    ]
    optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
    
    background = torch.ones(3, device=gaussians['means'].device)
    
    print(f'[INFO] Starting optimization for {num_iterations} iterations')
    
    pbar = tqdm(range(num_iterations), desc="Training")
    for iteration in pbar:
        # Random camera selection
        cam_idx = np.random.randint(len(cameras))
        camera = cameras[cam_idx]
        target_rgb = target_images[cam_idx]
        target_depth = target_depths[cam_idx]
        
        # Render
        rendered_rgb, rendered_depth, alpha = render_gaussians(gaussians, camera, background)
        
        # Compute loss
        loss = compute_loss(rendered_rgb, target_rgb, rendered_depth, target_depth)
        
        # Backward
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Update progress
        if iteration % 10 == 0:
            pbar.set_description(f"Training (loss: {loss.item():.4f})")
    
    print(f'[INFO] Optimization complete')


def main():
    parser = argparse.ArgumentParser(
        description='Train 3DGS scene from panorama LDI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ldi_dir', type=str, required=True,
                        help='Path to panorama LDI directory (containing rgba_ldi.npy, depth_ldi.npy)')
    parser.add_argument('--output', type=str, default='scene_optimized.ply',
                        help='Output PLY file path')
    parser.add_argument('--num_iterations', type=int, default=300,
                        help='Number of optimization iterations')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_views', type=int, default=16,
                        help='Number of training views to extract from panorama')
    parser.add_argument('--focal', type=float, default=582.69,
                        help='Focal length for rendering')
    parser.add_argument('--radius', type=float, default=4.0,
                        help='Camera orbit radius for rendering')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load panorama LDI
    rgba_pano, depth_pano = load_panorama_ldi(args.ldi_dir)
    
    # Extract perspective views from panorama
    views = extract_perspective_views(
        rgba_pano, depth_pano, 
        num_views=args.num_views,
        fov_deg=44.702,
        view_size=512,
        device=device
    )
    
    # Use the first view to initialize point cloud
    # (In practice, we could use all views, but first view is sufficient)
    first_view = views[0]
    H, W = first_view['rgba'].shape[1:3]
    
    # Unproject first view to point cloud
    points, colors, alphas = unproject_ldi_to_points(
        first_view['rgba'], first_view['depth'], args.focal, device
    )
    
    # Initialize Gaussians
    gaussians = initialize_gaussians(points, colors, alphas, device)
    
    # Create training cameras (circular orbit matching the extracted views)
    cameras = create_training_cameras(
        args.num_views, args.radius, args.focal, H, W, device
    )
    
    # Prepare target images from extracted views
    # For each view, composite layers to create full target image
    print('[INFO] Preparing target images from LDI layers...')
    target_images = []
    target_depths = []
    
    for view in tqdm(views, desc="Compositing targets"):
        # Composite layers (back to front)
        rgba = view['rgba']
        depth = view['depth']
        
        # Start with first layer
        target_rgba = rgba[0].copy()
        target_depth = depth[0].copy()
        
        # Composite subsequent layers
        for layer_idx in range(1, rgba.shape[0]):
            mask = rgba[layer_idx, ..., 3:4] > 0
            target_rgba = np.where(mask, rgba[layer_idx], target_rgba)
            target_depth = np.where(mask[..., 0], depth[layer_idx], target_depth)
        
        # Convert to tensors
        rgb_tensor = torch.tensor(target_rgba[..., :3], dtype=torch.float32, device=device)
        depth_tensor = torch.tensor(target_depth, dtype=torch.float32, device=device)
        
        target_images.append(rgb_tensor)
        target_depths.append(depth_tensor)
    
    # Optimize
    train(gaussians, cameras, target_images, target_depths, args.num_iterations, args.lr)
    
    # Save
    save_gaussians_ply(gaussians, args.output)
    
    print('[INFO] Done!')


if __name__ == '__main__':
    main()
