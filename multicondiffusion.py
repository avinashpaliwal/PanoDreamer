"""
MultiConDiffusion: Image-to-Panorama Generation with Iterative Inpainting

This implementation uses an iterative inpainting-conditioned MultiDiffusion approach.
Unlike text-to-panorama MultiDiffusion, this method progressively expands an input image
outward by iteratively inpainting the surrounding regions, allowing the center context
to propagate throughout the panorama.

Paper: PanoDreamer - https://people.engr.tamu.edu/nimak/Papers/PanoDreamer/
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import argparse
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# Suppress partial model loading warnings
logging.set_verbosity_error()


def seed_everything(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


class MultiConDiffusion(nn.Module):
    """
    MultiConDiffusion model for image-to-panorama generation.
    
    Iteratively applies inpainting-conditioned MultiDiffusion to expand an input image
    into a panorama. Each iteration:
    1. Uses the current panorama as conditioning for the inpainting model
    2. Applies MultiDiffusion with sliding windows to denoise overlapping views
    3. Averages overlapping regions for seamless blending
    4. The result becomes the input for the next iteration
    
    This allows context from the center image to progressively propagate outward.
    """
    
    def __init__(self, device, model_key="sd2-community/stable-diffusion-2-inpainting"):
        """
        Initialize MultiConDiffusion model.
        
        Args:
            device: torch device (cuda/cpu)
            model_key: HuggingFace model identifier (must be an inpainting model)
        """
        super().__init__()
        
        self.device = device
        self.model_key = model_key
        
        print(f'[INFO] Loading Stable Diffusion inpainting model: {model_key}')
        
        # Load model components
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(device)
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
    def image_to_panorama(self, scene, input_image, prompt, negative_prompt='',
                          height=512, width=2048, num_inference_steps=50,
                          guidance_scale=7.5, num_iterations=15, save_dir='output',
                          debug=False):
        """
        Generate panorama from input image using iterative MultiConDiffusion.
        
        The input image is placed in the center and progressively expanded outward
        through multiple iterations. Each iteration uses the current result as
        conditioning for the next, allowing context to propagate from center.
        
        Args:
            scene: Scene name for output files
            input_image: PIL Image to place in center
            prompt: Text prompt describing the scene
            negative_prompt: Negative prompt
            height: Panorama height
            width: Panorama width
            num_inference_steps: Denoising steps per iteration
            guidance_scale: Classifier-free guidance scale
            num_iterations: Number of iterative refinement steps
            save_dir: Output directory
            debug: If True, save intermediate visualizations (original, roll)
            
        Returns:
            Final panorama as PIL Image
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Ensure prompts are strings
        if isinstance(prompt, list):
            prompt = prompt[0]
        if isinstance(negative_prompt, list):
            negative_prompt = negative_prompt[0]
        
        print(f'[INFO] Encoding text prompt...')
        text_embeds = self.get_text_embeds(prompt, negative_prompt)
        
        # Image transform
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        
        # Load and prepare input image
        inp_im = input_image.resize((512, 512), resample=Image.LANCZOS).convert("RGB")
        image = transform(inp_im).to(self.device).unsqueeze(0).float()
        original_image = image.clone()
        
        # Initialize panorama canvas
        image = torch.zeros((1, 3, height, width), device=self.device).float()
        image[..., width//2 - 256:width//2 + 256] = original_image
        
        # Create mask (1 = fill, 0 = keep original)
        mask = torch.ones((1, 1, height, width), device=self.device).float()
        mask[..., width//2 - 256:width//2 + 256] = 0
        
        # Initialize random latents
        latent_init = torch.randn((1, 4, height//8, width//8), device=self.device).float()
        
        # Tile mask for edge wrapping
        mask = torch.cat([mask, mask, mask], dim=3)
        
        # MultiDiffusion parameters
        step_size = 8
        view_size = 64
        
        print(f'[INFO] Starting {num_iterations} iterations of MultiConDiffusion...')
        
        # Iterative refinement
        for x in range(num_iterations):
            print(f'[INFO] Iteration {x + 1}/{num_iterations}')
            
            # Reset latents each iteration
            latent = latent_init.clone()
            count = torch.zeros_like(latent)
            value = torch.zeros_like(latent)
            
            # Setup scheduler
            self.scheduler.set_timesteps(num_inference_steps)
            
            # Tile image for edge wrapping
            image = torch.cat([image, image, image], dim=3)
            
            # Save canvas at start of iteration (if debug mode)
            if debug:
                img = T.ToPILImage()(((image / 2 + 0.5).clamp(0, 1))[0].cpu())
                img.save(f"{save_dir}/iter_{x:02d}_original.jpg")
            
            # Denoising loop
            with torch.autocast('cuda'):
                for i, t in enumerate(tqdm(self.scheduler.timesteps, desc=f"Iter {x+1}")):
                    count.zero_()
                    value.zero_()
                    
                    # Tile latents for this timestep
                    latent = torch.cat([latent, latent, latent], dim=3)
                    value = torch.cat([value, value, value], dim=3)
                    count = torch.cat([count, count, count], dim=3)
                    
                    # Right-side views
                    for w_start in range(width//16 + width//8 - 32, width//8 + width//8 + 17 - view_size, step_size):
                        w_end = w_start + view_size
                        
                        latent_view = latent[:, :, :, w_start:w_end]
                        mask_inp = mask[:, :, :, w_start*8:w_end*8]
                        mask_view = torch.ones_like(mask_inp).float() * mask_inp
                        
                        # Mask out center region in later iterations
                        if x > 0:
                            mask_view[:, :, :, :256] = 0
                        
                        # Get masked image for conditioning
                        image1 = image[:, :, :, w_start*8:w_end*8] * (mask_view < 0.5)
                        conditionings = self.vae.encode(image1).latent_dist.sample() * 0.18215
                        mask_view = F.interpolate(mask_view, size=conditionings.shape[2:])
                        
                        # Concatenate inputs for inpainting UNet: [latent, mask, conditioning]
                        inputs = torch.cat([latent_view, mask_view, conditionings], dim=1)
                        latent_model_input = torch.cat([inputs] * 2)
                        
                        # Predict noise
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']
                        
                        # Classifier-free guidance
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        
                        # Denoise step
                        latents_view_denoised = self.scheduler.step(noise_pred, t, latent_view)['prev_sample']
                        
                        # Accumulate with special handling for boundary views
                        if w_start in [width//16 + width//8 - 32, width//16 + width//8 - 24, width//16 + width//8 - 16]:
                            value[:, :, :, w_start:w_end] += latents_view_denoised
                            count[:, :, :, w_start:w_end] += 1
                        else:
                            value[:, :, :, w_start:w_end] += latents_view_denoised * mask_view
                            count[:, :, :, w_start:w_end] += mask_view
                    
                    # Left-side views
                    for w_start in range(-16 + width//8, width//16 + width//8 - 31, step_size):
                        w_end = w_start + view_size
                        
                        latent_view = latent[:, :, :, w_start:w_end]
                        mask_inp = mask[:, :, :, w_start*8:w_end*8]
                        mask_view = torch.ones_like(mask_inp).float() * mask_inp
                        
                        # Mask out center region in later iterations
                        if x > 0:
                            mask_view[:, :, :, 256:] = 0
                        
                        # Get masked image for conditioning
                        image1 = image[:, :, :, w_start*8:w_end*8] * (mask_view < 0.5)
                        conditionings = self.vae.encode(image1).latent_dist.sample() * 0.18215
                        mask_view = F.interpolate(mask_view, size=conditionings.shape[2:])
                        
                        # Concatenate inputs for inpainting UNet
                        inputs = torch.cat([latent_view, mask_view, conditionings], dim=1)
                        latent_model_input = torch.cat([inputs] * 2)
                        
                        # Predict noise
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']
                        
                        # Classifier-free guidance
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        
                        # Denoise step
                        latents_view_denoised = self.scheduler.step(noise_pred, t, latent_view)['prev_sample']
                        
                        # Accumulate with special handling for boundary views
                        if w_start in [width//16 + width//8 - 32, width//16 + width//8 - 40, width//16 + width//8 - 48]:
                            value[:, :, :, w_start:w_end] += latents_view_denoised
                            count[:, :, :, w_start:w_end] += 1
                        else:
                            value[:, :, :, w_start:w_end] += latents_view_denoised * mask_view
                            count[:, :, :, w_start:w_end] += mask_view
                    
                    # Extract middle section from tiled tensors
                    value = value.chunk(3, dim=3)[1]
                    count = count.chunk(3, dim=3)[1]
                    
                    # Average overlapping regions
                    latent = torch.where(count > 0, value / count, value)
            
            # Decode latents to image
            imgs = self.decode_latents(latent)
            img = T.ToPILImage()(imgs[0].cpu())
            
            # Convert back to normalized tensor for next iteration
            image = transform(img).to(self.device).unsqueeze(0).float()
            
            # Preserve original center
            image[..., width//2 - 256:width//2 + 256] = original_image
            
            # Save iteration result
            img.save(f"{save_dir}/iter_{x:02d}.jpg")
            
            # Save rolled view (if debug mode)
            if debug:
                img_roll = T.ToPILImage()(imgs[0].cpu().roll(256, dims=2))
                img_roll.save(f"{save_dir}/iter_{x:02d}_roll.jpg")
        
        return img


def main():
    parser = argparse.ArgumentParser(
        description='MultiConDiffusion: Image-to-panorama generation with iterative inpainting'
    )
    parser.add_argument('--prompt_file', type=str, required=True,
                        help='Path to text file containing the prompt')
    parser.add_argument('--input_image', type=str, required=True,
                        help='Path to input image (will be placed in center)')
    parser.add_argument('--negative', type=str, default='',
                        help='Negative prompt (default: auto-generated)')
    parser.add_argument('--H', type=int, default=512,
                        help='Output height (default: 512)')
    parser.add_argument('--W', type=int, default=2048,
                        help='Output width (default: 2048)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of denoising steps per iteration')
    parser.add_argument('--iterations', type=int, default=15,
                        help='Number of refinement iterations')
    parser.add_argument('--guidance', type=float, default=7.5,
                        help='Classifier-free guidance scale')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--debug', action='store_true',
                        help='Save debug visualizations (original, roll images)')
    
    args = parser.parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print('[WARNING] CUDA not available, using CPU (will be very slow)')
    
    # Load prompt
    with open(args.prompt_file, 'r') as f:
        prompt = f.readline().strip()
    
    # Load input image
    input_image = Image.open(args.input_image).convert('RGB')
    
    # Set default negative prompt if not provided
    if not args.negative:
        args.negative = "caption, subtitle, text, blur, lowres, bad anatomy, bad hands, cropped, worst quality, watermark"
    
    # Extract scene name
    scene = os.path.splitext(os.path.basename(args.prompt_file))[0]
    save_dir = f'{args.output_dir}/{scene}_seed{args.seed}'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f'[INFO] Scene: {scene}')
    print(f'[INFO] Prompt: {prompt}')
    print(f'[INFO] Negative: {args.negative}')
    print(f'[INFO] Resolution: {args.W}x{args.H}')
    print(f'[INFO] Iterations: {args.iterations}, Steps: {args.steps}')
    print(f'[INFO] Output directory: {save_dir}')
    
    # Initialize MultiConDiffusion
    model = MultiConDiffusion(device)
    
    # Generate panorama from input image
    panorama = model.image_to_panorama(
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
