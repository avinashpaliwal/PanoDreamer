"""
Cylindrical Panorama Generation with Perspective Projection

Generates panoramas in cylindrical space (equirectangular projection) using
iterative inpainting with MultiDiffusion. Handles projection between perspective
and cylindrical coordinates for seamless 360° panoramas.

Paper: PanoDreamer - https://people.engr.tamu.edu/nimak/Papers/PanoDreamer/
"""

import numpy as np
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import argparse
from PIL import Image
from tqdm import tqdm
import kornia
from kornia.utils import create_meshgrid
from kornia.morphology import dilation
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# Suppress partial model loading warnings
logging.set_verbosity_error()

def seed_everything(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def cyl_proj(img, focal_length):
    """
    Project perspective image to cylindrical coordinates.
    
    Args:
        img: Input image tensor [B, C, H, W]
        focal_length: Focal length for projection
        
    Returns:
        Cylindrical projection of image
    """
    grid = create_meshgrid(img.shape[2], img.shape[3], normalized_coordinates=False, device=img.device)
    y, x = grid[..., 0], grid[..., 1]
    h, w = img.shape[2:]
    center_x = w // 2
    center_y = h // 2
    
    # Shift to center
    x_shifted = x - center_x
    y_shifted = y - center_y
    
    # Perspective to cylindrical
    theta = torch.arctan(x_shifted / focal_length)
    height = y_shifted / torch.sqrt(x_shifted ** 2 + focal_length ** 2)
    
    x_cyl = focal_length * theta + center_x
    y_cyl = height * focal_length + center_y
    
    # Remap and rotate
    img_cyl = kornia.geometry.transform.remap(
        img, torch.flip(x_cyl, dims=(1, 2)), y_cyl, 
        mode='nearest', align_corners=True
    )
    img_cyl = torch.rot90(img_cyl, k=3, dims=(2, 3))
    
    return img_cyl


def cyl_proj_inv(img, focal_length):
    """
    Project cylindrical image back to perspective coordinates.
    
    Args:
        img: Cylindrical image tensor [B, C, H, W]
        focal_length: Focal length for projection
        
    Returns:
        Perspective projection of image
    """
    grid = create_meshgrid(img.shape[2], img.shape[3], normalized_coordinates=False, device=img.device)
    y_cyl, x_cyl = grid[..., 0], grid[..., 1]
    h, w = img.shape[2:]
    center_x = w // 2
    center_y = h // 2
    
    # Cylindrical to perspective
    theta = (x_cyl - center_x) / focal_length
    height = (y_cyl - center_y) / focal_length
    
    x_shifted = torch.tan(theta) * focal_length
    y_shifted = height * torch.sqrt(x_shifted ** 2 + focal_length ** 2)
    
    x = x_shifted + center_x
    y = y_shifted + center_y
    
    # Remap and rotate
    img_persp = kornia.geometry.transform.remap(
        img, torch.flip(x, dims=(1, 2)), y,
        mode='nearest', align_corners=True
    )
    img_persp = torch.rot90(img_persp, k=3, dims=(2, 3))
    
    return img_persp


def fov2focal(fov_radians, pixels):
    """Convert field of view to focal length."""
    return pixels / (2 * math.tan(fov_radians / 2))

class CylindricalPanorama(nn.Module):
    """
    Cylindrical panorama generation with perspective-to-cylindrical projection.
    
    Generates 360° panoramas in cylindrical/equirectangular space by:
    1. Converting input perspective image to cylindrical coordinates
    2. Iteratively inpainting surrounding regions
    3. Projecting back to perspective for each view
    4. Accumulating results with MultiDiffusion
    """
    
    def __init__(self, device, model_key="sd2-community/stable-diffusion-2-inpainting"):
        super().__init__()
        
        self.device = device
        self.model_key = model_key
        
        print(f'[INFO] Loading Stable Diffusion inpainting model: {model_key}')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        print(f'[INFO] Model loaded successfully!')
    
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        """
        Encode text prompts to embeddings for classifier-free guidance.
        
        Args:
            prompt: Positive text prompt
            negative_prompt: Negative text prompt
            
        Returns:
            text_embeddings: Concatenated [negative, positive] embeddings
        """
        # Tokenize and encode positive prompt
        text_input = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
        # Tokenize and encode negative prompt
        uncond_input = self.tokenizer(
            negative_prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt'
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings
    
    @torch.no_grad()
    def decode_latents(self, latents):
        """Decode latents to image space."""
        latents = latents / 0.18215
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image
    
    @torch.no_grad()
    def decode_latents_with_padding(self, latents: torch.Tensor, padding: int = 8) -> torch.Tensor:
        """
        Decode the given latents with padding for circular inference.

        Args:
            latents (torch.Tensor): The input latents to decode.
            padding (int, optional): The number of latents to add on each side for padding. Defaults to 8.

        Returns:
            torch.Tensor: The decoded image with padding removed.

        Notes:
            - The padding is added to remove boundary artifacts and improve the output quality.
            - This would slightly increase the memory usage.
            - The padding pixels are then removed from the decoded image.

        """
        latents = 1 / 0.18215 * latents
        latents_left = latents[..., :padding]
        latents_right = latents[..., -padding:]
        latents = torch.cat((latents_right, latents, latents_left), axis=-1)
        image = self.vae.decode(latents, return_dict=False)[0]
        padding_pix = 8 * padding
        image = image[..., padding_pix:-padding_pix]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image
    

    @torch.no_grad()
    def image_to_cylindrical_panorama(self, scene, input_image, prompt, negative_prompt='', 
                                       height=512, width=3912, num_inference_steps=50,
                                       guidance_scale=7.5, num_iterations=15, save_dir='output',
                                       debug=False):
        """
        Generate cylindrical 360° panorama from input image.
        
        Uses iterative inpainting with perspective-cylindrical projection:
        1. Projects input to cylindrical space
        2. For each iteration:
           - Generates perspective views around the panorama
           - Projects views to cylindrical space
           - Denoises with MultiDiffusion
           - Accumulates results
        3. Context propagates from center outward in cylindrical space
        
        Args:
            scene: Scene name for output files
            input_image: PIL Image (will be projected to cylindrical)
            prompt: Text prompt describing the scene
            negative_prompt: Negative prompt
            height: Panorama height
            width: Panorama width (typically 3912 for 360°)
            num_inference_steps: Denoising steps per iteration
            guidance_scale: Classifier-free guidance scale
            num_iterations: Number of iterative refinement steps
            save_dir: Output directory
            debug: If True, save debug visualizations
            
        Returns:
            Final cylindrical panorama as PIL Image
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Ensure prompts are strings
        if isinstance(prompt, list):
            prompt = prompt[0]
        if isinstance(negative_prompt, list):
            negative_prompt = negative_prompt[0]
        
        print(f'[INFO] Encoding text prompt...')
        text_embeds = self.get_text_embeds(prompt, negative_prompt)

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

        # Load and prepare input image
        inp_im = input_image.resize((512, 512), resample=Image.LANCZOS).convert("RGB")
        image = transform(inp_im).to(self.device).unsqueeze(0).float()
        original_image = image.clone()

        # Initialize cylindrical panorama canvas
        print(f'[INFO] Setting up cylindrical projection (360° panorama)...')
        input_cylinder = torch.zeros((1, 3, height, width), device=self.device)
        input_fov = 44.701948991275390  # Field of view in degrees
        focal_length = fov2focal(input_fov * math.pi / 180, 512)
        
        # Project input perspective image to cylindrical and place in center
        cyl_image = cyl_proj_inv(image, focal_length)
        input_cylinder[..., width//2 - 256:width//2 + 256] = cyl_image
        original_cylinder = input_cylinder.clone()
        
        # Create cylindrical mask (1 = fill, 0 = keep)
        mask_cylinder = torch.zeros((1, 1, height, width), device=self.device)
        cyl_mask = cyl_proj_inv(torch.ones_like(image)[:, :1], focal_length)
        mask_cylinder[..., width//2 - 256:width//2 + 256] = cyl_mask
        mask_cylinder = 1 - mask_cylinder
        mask_cylinder1 = mask_cylinder.clone()
        
        # Tile mask for seamless 360° wrapping
        mask_cylinder = torch.cat([mask_cylinder, mask_cylinder, mask_cylinder], dim=3)

        # Initialize random latents
        latent_init = torch.randn((1, 4, height//8, width//8), device=self.device).float()
        
        # Create perspective view masks for left/right regions
        temp_mask = torch.zeros((1, 1, 512, 512), device=self.device)
        cyl_temp = cyl_proj_inv(torch.ones_like(temp_mask), focal_length)
        
        # Create left/right perspective view masks (for iterative expansion)
        left_mask = torch.zeros((1, 1, 512, 512), device=self.device)
        left_mask[..., 300:] = cyl_temp[..., :212]
        left_mask = 1 - left_mask
        left_mask = cyl_proj(left_mask, focal_length)
        
        right_mask = torch.zeros((1, 1, 512, 512), device=self.device)
        right_mask[..., :212] = cyl_temp[..., 300:]
        right_mask = 1 - right_mask
        right_mask = cyl_proj(right_mask, focal_length)
        
        # Save debug visualizations (if enabled)
        if debug:
            T.ToPILImage()((left_mask[0].cpu()).clamp(0, 1)).save(f"{save_dir}/debug_left_mask.jpg")
            T.ToPILImage()((right_mask[0].cpu()).clamp(0, 1)).save(f"{save_dir}/debug_right_mask.jpg")
            T.ToPILImage()((input_cylinder[0].cpu() / 2 + 0.5).clamp(0, 1)).save(f"{save_dir}/debug_input_cylinder.jpg")
            T.ToPILImage()((mask_cylinder[0].cpu()).clamp(0, 1)).save(f"{save_dir}/debug_mask_cylinder.jpg")


        print(f'[INFO] Starting {num_iterations} iterations of cylindrical MultiConDiffusion...')
        
        # Iterative refinement
        for x in range(num_iterations):
            print(f'[INFO] Iteration {x + 1}/{num_iterations}')
            
            # Reset latents for this iteration
            latent = latent_init.clone()
            count = torch.zeros_like(latent)
            value = torch.zeros_like(latent)
            
            # Setup scheduler
            self.scheduler.set_timesteps(num_inference_steps)
            
            # Tile cylindrical panorama for 360° wrapping
            input_cylinder = torch.cat([input_cylinder, input_cylinder, input_cylinder], dim=3)
            
            # Save canvas at start of iteration (if debug mode)
            if debug:
                cylinder_img = T.ToPILImage()((input_cylinder[0].cpu() / 2 + 0.5).clamp(0, 1))
                cylinder_img.save(f"{save_dir}/iter_{x:02d}_original.jpg")
            
            # MultiDiffusion parameters
            step_size = 8

            # Denoising loop
            with torch.autocast('cuda'):
                for i, t in enumerate(tqdm(self.scheduler.timesteps, desc=f"Iter {x+1}")):
                    count.zero_()
                    value.zero_()

                    latent = torch.cat([latent, latent, latent], dim=3)
                    value = torch.cat([value, value, value], dim=3)
                    count = torch.cat([count, count, count], dim=3)

                    for w_start in range(width//16 + width//8 - 32, width//8 + width//8 + 33 - 64, step_size):
                        w_end = w_start + 64

                        latent_view = latent[:, :, :, w_start:w_end]
                        latent_proj = cyl_proj(latent_view, fov2focal(input_fov * math.pi / 180, 64))

                        mask_view = mask_cylinder[:, :, :, w_start*8:w_end*8]
                        mask_proj = cyl_proj(mask_view, fov2focal(input_fov * math.pi / 180, 512))
                        if x > 0:
                            mask_proj = mask_proj * right_mask
                        mask_proj = torch.where(mask_proj > 0.5, 1.0, 0.).float()

                        image_view = input_cylinder[:, :, :, w_start*8:w_end*8]
                        image_proj = cyl_proj(image_view, fov2focal(input_fov * math.pi / 180, 512)) * (mask_proj < 0.5)

                        conditionings = self.vae.encode(image_proj).latent_dist.sample()
                        conditionings = conditionings * 0.18215
                        
                        mask_interp = F.interpolate(mask_proj, size=conditionings.shape[2:])

                        inputs = torch.cat([latent_proj, mask_interp, conditionings], dim=1)
                        latent_model_input = torch.cat([inputs] * 2)

                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                        latents_view_denoised = self.scheduler.step(noise_pred, t, latent_proj)['prev_sample']

                        latents_denoised_cyl = cyl_proj_inv(latents_view_denoised, fov2focal(input_fov * math.pi / 180, 64))
                        count_cyl = cyl_proj_inv(torch.ones_like(latents_view_denoised), fov2focal(input_fov * math.pi / 180, 64))
                        mask_interp_cyl = cyl_proj_inv(mask_interp, fov2focal(input_fov * math.pi / 180, 64))

                        if w_start == width//16 + width//8 - 32 or w_start == width//16 + width//8 - 24:
                            value[:, :, :, w_start:w_end] += latents_denoised_cyl
                            count[:, :, :, w_start:w_end] += count_cyl
                        else:
                            mask_interp_cyl[..., 56:] = 0
                            latents_denoised_cyl = latents_denoised_cyl * (mask_interp_cyl)
                            value[:, :, :, w_start:w_end] += latents_denoised_cyl
                            count[:, :, :, w_start:w_end] += count_cyl * (mask_interp_cyl)

                    for w_start in range(-32 + width//8, width//16 + width//8 - 31, step_size):
                        w_end = w_start + 64

                        latent_view = latent[:, :, :, w_start:w_end]
                        latent_proj = cyl_proj(latent_view, fov2focal(input_fov * math.pi / 180, 64))

                        mask_view = mask_cylinder[:, :, :, w_start*8:w_end*8]
                        mask_proj = cyl_proj(mask_view, fov2focal(input_fov * math.pi / 180, 512))
                        if x > 0:
                            mask_proj = mask_proj * left_mask
                        mask_proj = torch.where(mask_proj > 0.5, 1.0, 0.).float()

                        image_view = input_cylinder[:, :, :, w_start*8:w_end*8]
                        image_proj = cyl_proj(image_view, fov2focal(input_fov * math.pi / 180, 512)) * (mask_proj < 0.5)

                        conditionings = self.vae.encode(image_proj).latent_dist.sample()
                        conditionings = conditionings * 0.18215
                        
                        mask_interp = F.interpolate(mask_proj, size=conditionings.shape[2:])

                        inputs = torch.cat([latent_proj, mask_interp, conditionings], dim=1)
                        latent_model_input = torch.cat([inputs] * 2)

                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                        latents_view_denoised = self.scheduler.step(noise_pred, t, latent_proj)['prev_sample']

                        latents_denoised_cyl = cyl_proj_inv(latents_view_denoised, fov2focal(input_fov * math.pi / 180, 64))
                        count_cyl = cyl_proj_inv(torch.ones_like(latents_view_denoised), fov2focal(input_fov * math.pi / 180, 64))
                        mask_interp_cyl = cyl_proj_inv(mask_interp, fov2focal(input_fov * math.pi / 180, 64))

                        if w_start == width//16 + width//8 - 32:
                            value[:, :, :, w_start:w_end] += latents_denoised_cyl
                            count[:, :, :, w_start:w_end] += count_cyl
                        else:
                            mask_interp_cyl[..., :8] = 0
                            latents_denoised_cyl = latents_denoised_cyl * (mask_interp_cyl)
                            value[:, :, :, w_start:w_end] += latents_denoised_cyl
                            count[:, :, :, w_start:w_end] += count_cyl * (mask_interp_cyl)

                    value = sum(value.chunk(3, dim=3))
                    count = sum(count.chunk(3, dim=3))
                    
                    count = torch.clamp(count, 1)
                    latent = value / count

            imgs = self.decode_latents_with_padding(latent)
            img = T.ToPILImage()(imgs[0].cpu())
            
            # Update cylinder for next iteration
            input_cylinder = transform(img).to(self.device).unsqueeze(0).float()
            input_cylinder = mask_cylinder1 * input_cylinder + (1 - mask_cylinder1) * original_cylinder
            
            # Save iteration result
            img.save(f"{save_dir}/iter_{x:02d}.png")
            
            # Save rolled view (if debug mode)
            if debug:
                img_roll = T.ToPILImage()(imgs[0].cpu().roll(256, dims=2))
                img_roll.save(f"{save_dir}/iter_{x:02d}_roll.jpg")
        
        return img


def main():
    parser = argparse.ArgumentParser(
        description='Generate cylindrical panoramas with perspective projection'
    )
    parser.add_argument('--prompt_file', type=str, required=True,
                        help='Path to text file containing the prompt')
    parser.add_argument('--input_image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--negative', type=str, default='',
                        help='Negative prompt')
    parser.add_argument('--H', type=int, default=512,
                        help='Output height (default: 512)')
    parser.add_argument('--W', type=int, default=3912,
                        help='Output width for cylindrical panorama (default: 3912)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--steps', type=int, default=50,
                        help='Denoising steps per iteration')
    parser.add_argument('--iterations', type=int, default=15,
                        help='Number of refinement iterations')
    parser.add_argument('--guidance', type=float, default=7.5,
                        help='Guidance scale')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--debug', action='store_true',
                        help='Save debug visualizations')
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print('[WARNING] CUDA not available, using CPU (will be very slow)')
    
    # Load prompt
    with open(args.prompt_file, 'r') as f:
        prompt = f.readline().strip()
    
    # Load input image
    input_image = Image.open(args.input_image).convert('RGB')
    
    # Set default negative prompt
    if not args.negative:
        args.negative = "caption, subtitle, text, blur, lowres, bad anatomy, bad hands, cropped, worst quality, watermark"
    
    # Extract scene name
    scene = os.path.splitext(os.path.basename(args.prompt_file))[0]
    save_dir = f'{args.output_dir}/{scene}_seed{args.seed}'
    
    print(f'[INFO] Scene: {scene}')
    print(f'[INFO] Prompt: {prompt}')
    print(f'[INFO] Negative: {args.negative}')
    print(f'[INFO] Resolution: {args.W}x{args.H}')
    print(f'[INFO] Iterations: {args.iterations}, Steps: {args.steps}')
    print(f'[INFO] Output directory: {save_dir}')
    
    # Initialize model
    model = CylindricalPanorama(device)
    
    # Generate panorama
    panorama = model.image_to_cylindrical_panorama(
        scene=scene,
        input_image=input_image,
        prompt=prompt,
        negative_prompt=args.negative,
        height=args.H,
        width=args.W,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        num_iterations=args.iterations,
        save_dir=save_dir,
        debug=args.debug
    )
    
    # Save final output
    output_path = f"{save_dir}/final_output_{scene}.png"
    panorama.save(output_path)
    
    print(f'[INFO] Panorama saved to: {output_path}')
    print(f'[INFO] Done!')


if __name__ == '__main__':
    main()
