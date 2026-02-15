"""
3D Gaussian Splatting Renderer using gsplat

Renders videos from a 3DGS PLY file with RGB, depth, and alpha outputs.

Usage:
    python render_gsplat.py --ply gaussians.ply --output renders
    python render_gsplat.py --ply gaussians.ply --output renders --num_frames 120 --radius 4.0
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import imageio
import cv2
from matplotlib import colormaps

from gsplat import rasterization


def load_3dgs_ply(ply_path, device='cuda'):
    """
    Load a 3DGS PLY file.
    
    Args:
        ply_path: Path to PLY file
        device: Device to load tensors to
    
    Returns:
        Dictionary with Gaussian parameters: means, scales, quats, opacities, colors
    """
    from plyfile import PlyData
    
    print(f'[INFO] Loading PLY: {ply_path}')
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    # Positions
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    N = xyz.shape[0]
    
    # Scales (stored in log space)
    scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=-1)
    
    # Rotations (quaternions)
    quats = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=-1)
    
    # Opacities (stored in logit space)
    opacities = np.array(vertex['opacity'])
    
    # Colors from spherical harmonics DC component
    SH_C0 = 0.28209479177387814
    f_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=-1)
    colors = np.clip(SH_C0 * f_dc + 0.5, 0, 1)
    
    print(f'[INFO] Loaded {N:,} Gaussians')
    
    return {
        'means': torch.tensor(xyz, dtype=torch.float32, device=device),
        'scales': torch.tensor(scales, dtype=torch.float32, device=device),
        'quats': torch.tensor(quats, dtype=torch.float32, device=device),
        'opacities': torch.tensor(opacities, dtype=torch.float32, device=device),
        'colors': torch.tensor(colors, dtype=torch.float32, device=device),
    }


def normalize(x):
    """Normalize vectors along last axis."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def create_camera_path(num_frames=120, radius=4.0):
    """
    Create circular camera path looking at origin.
    
    Args:
        num_frames: Number of camera positions
        radius: Distance from origin
    
    Returns:
        poses: [num_frames, 4, 4] camera-to-world matrices
    """
    up = np.array([0, 1, 0])
    poses = []
    
    for i in range(num_frames):
        theta = i * (2 * np.pi) / num_frames
        
        # Camera position on circle in XZ plane
        eye = np.array([radius * np.cos(theta), 0, radius * np.sin(theta)])
        lookat = np.array([0, 0, 0])
        
        # Construct camera axes
        w = normalize(lookat - eye)  # forward
        u = normalize(np.cross(up, w))  # right
        v = np.cross(w, u)  # up
        
        # Camera-to-world matrix
        c2w = np.eye(4)
        c2w[:3, 0] = u
        c2w[:3, 1] = v
        c2w[:3, 2] = w
        c2w[:3, 3] = eye
        
        poses.append(c2w)
    
    return np.stack(poses)


def render_frame(gaussians, c2w, H=512, W=512, focal=582.69, device='cuda'):
    """
    Render a single frame.
    
    Args:
        gaussians: Dictionary of Gaussian parameters
        c2w: [4, 4] camera-to-world matrix
        H, W: Image dimensions
        focal: Focal length
        device: Device
    
    Returns:
        rgb: [H, W, 3] RGB image
        depth: [H, W] depth map
        alpha: [H, W] alpha/opacity map
    """
    # World-to-camera transform
    w2c = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)
    
    # Camera intrinsics
    K = torch.tensor([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    # Get Gaussian parameters
    means = gaussians['means']
    scales = torch.exp(gaussians['scales'])
    quats = F.normalize(gaussians['quats'], dim=-1)
    opacities = torch.sigmoid(gaussians['opacities'])
    colors = gaussians['colors']
    
    # Rasterize with RGB+ED mode (RGB + Expected Depth)
    # Output: render_colors is [B, H, W, 4] where last channel is depth
    render_colors, render_alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=w2c[None],
        Ks=K[None],
        width=W,
        height=H,
        packed=False,
        render_mode="RGB+ED",
    )
    
    # Extract RGB (first 3 channels) and depth (4th channel)
    rgb = render_colors[0, ..., :3].clamp(0, 1)
    depth = render_colors[0, ..., 3]
    alpha = render_alphas[0, ..., 0]
    
    return rgb, depth, alpha


@torch.no_grad()
def render_video(gaussians, output_dir, num_frames=120, fps=60, radius=4.0,
                 H=512, W=512, focal=582.69):
    """
    Render RGB, depth, and alpha videos.
    
    Args:
        gaussians: Dictionary of Gaussian parameters
        output_dir: Output directory
        num_frames: Number of frames
        fps: Video frame rate
        radius: Camera orbit radius
        H, W: Image dimensions
        focal: Focal length
    """
    print(f'[INFO] Rendering {num_frames} frames at {fps} fps')
    print(f'[INFO] Camera orbit radius: {radius}')
    
    poses = create_camera_path(num_frames, radius)
    
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    rgb_path = os.path.join(output_dir, 'video.mp4')
    depth_path = os.path.join(output_dir, 'video_depth.mp4')
    alpha_path = os.path.join(output_dir, 'video_alpha.mp4')
    
    writer_rgb = imageio.get_writer(rgb_path, fps=fps)
    writer_depth = imageio.get_writer(depth_path, fps=fps)
    writer_alpha = imageio.get_writer(alpha_path, fps=fps)
    
    turbo = colormaps.get_cmap('turbo')
    
    for i in tqdm(range(num_frames), desc="Rendering"):
        c2w = poses[i]
        rgb, depth, alpha = render_frame(gaussians, c2w, H=H, W=W, focal=focal)
        
        # RGB frame
        rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
        writer_rgb.append_data(rgb_np)
        cv2.imwrite(os.path.join(frames_dir, f'{i:04d}.png'), rgb_np[:, :, ::-1])
        
        # Depth visualization (turbo colormap)
        depth_np = depth.cpu().numpy()
        if depth_np.max() > depth_np.min():
            depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        else:
            depth_norm = np.zeros_like(depth_np)
        depth_color = (turbo(depth_norm)[..., :3] * 255).astype(np.uint8)
        writer_depth.append_data(depth_color)
        
        # Alpha frame
        alpha_np = (alpha.cpu().numpy() * 255).astype(np.uint8)
        writer_alpha.append_data(alpha_np)
    
    writer_rgb.close()
    writer_depth.close()
    writer_alpha.close()
    
    print(f'[INFO] Saved:')
    print(f'       {rgb_path}')
    print(f'       {depth_path}')
    print(f'       {alpha_path}')
    print(f'       {frames_dir}/')


def main():
    parser = argparse.ArgumentParser(
        description='Render 3DGS PLY file using gsplat',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ply', type=str, required=True,
                        help='Path to 3DGS PLY file')
    parser.add_argument('--output', type=str, default='renders',
                        help='Output directory')
    parser.add_argument('--num_frames', type=int, default=720,
                        help='Number of frames')
    parser.add_argument('--fps', type=int, default=60,
                        help='Video frame rate')
    parser.add_argument('--focal', type=float, default=2*582.69,
                        help='Focal length')
    parser.add_argument('--radius', type=float, default=2.0,
                        help='Camera orbit radius')
    parser.add_argument('--height', type=int, default=512,
                        help='Image height')
    parser.add_argument('--width', type=int, default=512,
                        help='Image width')
    
    args = parser.parse_args()
    
    gaussians = load_3dgs_ply(args.ply)
    
    render_video(
        gaussians,
        args.output,
        num_frames=args.num_frames,
        fps=args.fps,
        radius=args.radius,
        H=args.height,
        W=args.width,
        focal=args.focal
    )
    
    print('[INFO] Done!')


if __name__ == '__main__':
    main()
