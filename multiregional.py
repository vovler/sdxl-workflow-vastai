"""
MultiDiffusion SDXL Regional Generation
Complete implementation for generating multiple characters in different regions
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import argparse
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
import torchvision.transforms as T


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_views(panorama_height, panorama_width, window_size=64, stride=8):
    """Generate overlapping windows for MultiDiffusion"""
    panorama_height //= 8
    panorama_width //= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views


def create_character_masks(width=1024, height=1024, left_bbox=None, right_bbox=None):
    """Create masks for left and right characters"""
    if left_bbox is None:
        left_bbox = [50, 100, 450, 900]  # [x1, y1, x2, y2]
    if right_bbox is None:
        right_bbox = [574, 100, 974, 900]
    
    # Create left character mask
    left_mask = Image.new('L', (width, height), 0)  # Black background
    draw = ImageDraw.Draw(left_mask)
    draw.rectangle(left_bbox, fill=255)  # White = masked region
    
    # Create right character mask  
    right_mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(right_mask)
    draw.rectangle(right_bbox, fill=255)
    
    return left_mask, right_mask


def create_custom_masks(width=1024, height=1024, bboxes=None):
    """Create multiple custom masks from bounding boxes"""
    if bboxes is None:
        # Default: left, right, center
        bboxes = [
            [50, 100, 350, 900],    # Left
            [374, 100, 674, 900],   # Center  
            [698, 100, 974, 900]    # Right
        ]
    
    masks = []
    for bbox in bboxes:
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(bbox, fill=255)
        masks.append(mask)
    
    return masks


@torch.no_grad()
def preprocess_mask(mask_pil, h, w, device):
    """Convert PIL mask to tensor format for MultiDiffusion"""
    mask = np.array(mask_pil).astype(np.float32) / 255.0
    mask = mask[None, None]  # Add batch and channel dims
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    # Resize to latent space dimensions
    mask = torch.nn.functional.interpolate(mask, size=(h//8, w//8), mode='nearest')
    return mask


class MultiDiffusionSDXL_Regional:
    def __init__(self, model_path="John6666/wai-nsfw-illustrious-v130-sdxl", device="cuda"):
        self.device = device
        
        print(f"Loading SDXL model: {model_path}")
        
        # Load SDXL pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(device)
        
        # Extract components
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.tokenizer_2 = self.pipe.tokenizer_2
        self.text_encoder = self.pipe.text_encoder
        self.text_encoder_2 = self.pipe.text_encoder_2
        
        # Use DDIM scheduler for better control
        self.scheduler = DDIMScheduler.from_pretrained(
            model_path, 
            subfolder="scheduler"
        )
        self.pipe.scheduler = self.scheduler
        
        print("SDXL model loaded successfully!")
    
    @torch.no_grad()
    def get_random_background(self, n_samples, height, width):
        """Generate random background latents for bootstrapping"""
        
        all_latents = []
        # Process one by one to avoid OOM with VAE
        for _ in tqdm(range(n_samples), desc="Generating background latents"):
            background = torch.rand(1, 3, height, width, device=self.device, dtype=torch.float16)
            background = background * 2 - 1  # Normalize to [-1, 1]
            
            # Encode to latent space
            latents = self.vae.encode(background).latent_dist.sample() * self.vae.config.scaling_factor
            all_latents.append(latents)
            
        return torch.cat(all_latents, dim=0)
    
    @torch.no_grad()
    def generate_regional(self, mask_images, prompts, negative_prompts=None, 
                         width=1024, height=1024, num_inference_steps=30,
                         guidance_scale=7.5, stride=64, bootstrapping=20):
        """
        Generate image with regional control using masks
        
        Args:
            mask_images: List of PIL Images defining regions (excluding background)
            prompts: List of prompts for [background, region1, region2, ...]
            negative_prompts: List of negative prompts (optional)
            width, height: Output dimensions
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            stride: Stride for MultiDiffusion windows
            bootstrapping: Number of steps to use background bootstrapping
        """
        
        if negative_prompts is None:
            negative_prompts = [""] * len(prompts)
        
        # Process masks
        fg_masks = []
        for mask_img in mask_images:
            mask_tensor = preprocess_mask(mask_img, height, width, self.device)
            fg_masks.append(mask_tensor)
        
        if fg_masks:
            fg_masks = torch.cat(fg_masks, dim=0)
            # Background = everything not covered by foreground masks
            bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
            bg_mask = torch.clamp(bg_mask, 0, 1)
            masks = torch.cat([bg_mask, fg_masks], dim=0)
        else:
            # No foreground masks, just background
            masks = torch.ones(1, 1, height//8, width//8, device=self.device)
        
        print(f"Created {len(masks)} masks: 1 background + {len(fg_masks)} foreground")
        
        # Encode all prompts using SDXL's dual text encoders
        all_embeds = []
        all_pooled = []
        
        for i, (prompt, neg_prompt) in enumerate(zip(prompts, negative_prompts)):
            print(f"Encoding prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
                self.pipe.encode_prompt(
                    prompt=prompt,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=neg_prompt
                )
            
            # Concatenate for CFG
            combined_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            combined_pooled = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
            
            all_embeds.append(combined_embeds)
            all_pooled.append(combined_pooled)
        
        # Stack all embeddings
        text_embeddings = torch.stack(all_embeds)  # [num_prompts, 2, seq_len, dim]
        pooled_embeddings = torch.stack(all_pooled)  # [num_prompts, 2, pooled_dim]
        
        # Setup time conditioning for SDXL
        original_size = target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=torch.float16, device=self.device)
        add_time_ids = add_time_ids.repeat(2, 1)  # For CFG
        
        # Initialize latents and noise for bootstrapping
        latents = torch.randn((1, 4, height//8, width//8), 
                             device=self.device, dtype=torch.float16)
        
        if bootstrapping > 0:
            print(f"Generating {bootstrapping} random backgrounds for bootstrapping...")
            bootstrapping_backgrounds = self.get_random_background(bootstrapping, height, width)
            noise = torch.randn_like(latents).repeat(len(prompts)-1, 1, 1, 1)
        
        # MultiDiffusion views
        views = get_views(height, width, stride=stride)
        print(f"Using {len(views)} overlapping views with stride {stride}")
        
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        print("Starting regional generation...")
        with torch.autocast('cuda'):
            for step_idx, t in enumerate(tqdm(self.scheduler.timesteps)):
                count = torch.zeros_like(latents)
                value = torch.zeros_like(latents)
                
                for view_idx, (h_start, h_end, w_start, w_end) in enumerate(views):
                    masks_view = masks[:, :, h_start:h_end, w_start:w_end]  # [num_regions, 1, h, w]
                    latent_view = latents[:, :, h_start:h_end, w_start:w_end]
                    
                    # Apply bootstrapping for foreground regions
                    if step_idx < bootstrapping and len(prompts) > 1:
                        bg_indices = torch.randint(0, bootstrapping, (len(prompts)-1,))
                        bg_latents = bootstrapping_backgrounds[bg_indices, :, h_start:h_end, w_start:w_end]
                        bg_latents = self.scheduler.add_noise(
                            bg_latents, 
                            noise[:, :, h_start:h_end, w_start:w_end], 
                            t
                        )
                        
                        # Blend background with masked regions
                        latent_view_expanded = latent_view.repeat(len(prompts), 1, 1, 1)
                        latent_view_expanded[1:] = latent_view_expanded[1:] * masks_view[1:] + \
                                                 bg_latents * (1 - masks_view[1:])
                    else:
                        latent_view_expanded = latent_view.repeat(len(prompts), 1, 1, 1)
                    
                    # Expand latent for CFG
                    latent_input = torch.cat([latent_view_expanded] * 2)
                    latent_input = self.scheduler.scale_model_input(latent_input, t)
                    
                    # Prepare text embeddings for batch processing
                    batch_text_embeds = text_embeddings.view(-1, text_embeddings.shape[-2], text_embeddings.shape[-1])
                    batch_pooled = pooled_embeddings.view(-1, pooled_embeddings.shape[-1])
                    batch_time_ids = add_time_ids.repeat(len(prompts), 1)
                    
                    # UNet forward pass for all regions at once
                    noise_pred = self.unet(
                        latent_input,
                        t,
                        encoder_hidden_states=batch_text_embeds,
                        added_cond_kwargs={
                            "text_embeds": batch_pooled,
                            "time_ids": batch_time_ids
                        },
                        return_dict=False
                    )[0]
                    
                    # CFG for each region
                    noise_pred = noise_pred.view(len(prompts), 2, *noise_pred.shape[1:])
                    noise_pred_uncond = noise_pred[:, 0]
                    noise_pred_text = noise_pred[:, 1] 
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # Prevent NaNs from unstable models in float16
                    noise_pred = torch.nan_to_num(noise_pred)
                    
                    # Scheduler step for each region
                    latents_view_denoised = []
                    for region_idx in range(len(prompts)):
                        denoised = self.scheduler.step(
                            noise_pred[region_idx], 
                            t, 
                            latent_view_expanded[region_idx:region_idx+1], 
                            return_dict=False
                        )[0]
                        latents_view_denoised.append(denoised * masks_view[region_idx])
                    
                    # Combine all regions for this view
                    combined_output = torch.sum(torch.cat(latents_view_denoised), dim=0, keepdim=True)
                    value[:, :, h_start:h_end, w_start:w_end] += combined_output
                    count[:, :, h_start:h_end, w_start:w_end] += masks_view.sum(dim=0, keepdim=True)
                
                # MultiDiffusion step: average overlapping regions
                latents = torch.where(count > 0, value / count, value)
        
        print("Decoding final image...")
        # Decode final image
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        
        return image


def main():
    parser = argparse.ArgumentParser(description='MultiDiffusion SDXL Regional Generation')
    parser.add_argument('--model', type=str, default='John6666/wai-nsfw-illustrious-v130-sdxl',
                       help='SDXL model path')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--steps', type=int, default=12, help='Number of inference steps')
    parser.add_argument('--guidance', type=float, default=1.2, help='Guidance scale')
    parser.add_argument('--stride', type=int, default=64, help='MultiDiffusion stride')
    parser.add_argument('--bootstrapping', type=int, default=20, help='Bootstrapping steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='regional_output.png', help='Output filename')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    seed_everything(args.seed)
    
    # Initialize generator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = MultiDiffusionSDXL_Regional(args.model, device)
    
    # Create masks for two characters
    print("Creating character masks...")
    left_mask, right_mask = create_character_masks(args.width, args.height)
    
    # Save masks for reference
    left_mask.save("left_mask.png")
    right_mask.save("right_mask.png")
    print("Saved mask references: left_mask.png, right_mask.png")
    
    # Define prompts for each region
    prompts = [
        # Background prompt
        "masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery",
        
        # Left character prompt  
        "1girl, aqua_(konosuba), smiling",
        
        # Right character prompt
        "1girl, megumin, sad"
    ]
    
    negative_prompts = [
        # Background negative
        "blurry, low quality, bad composition, artifacts",
        
        # Left character negative
        "blurry, bad anatomy, deformed face, low quality, multiple people, merged characters",
        
        # Right character negative  
        "blurry, bad anatomy, deformed face, low quality, multiple people, merged characters"
    ]
    
    print("Prompts:")
    for i, prompt in enumerate(prompts):
        print(f"  {i}: {prompt}")
    
    # Generate image
    print(f"\nGenerating {args.width}x{args.height} image...")
    result = generator.generate_regional(
        mask_images=[left_mask, right_mask],
        prompts=prompts,
        negative_prompts=negative_prompts,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        stride=args.stride,
        bootstrapping=args.bootstrapping
    )
    
    # Save result
    pil_image = T.ToPILImage()(result[0].cpu())
    pil_image.save(args.output)
    print(f"Saved result: {args.output}")
    
    # Create a combined visualization
    print("Creating visualization...")
    
    # Resize masks to match output
    left_vis = left_mask.resize((args.width//4, args.height//4))
    right_vis = right_mask.resize((args.width//4, args.height//4))
    result_vis = pil_image.resize((args.width//2, args.height//2))
    
    # Create combined image
    combined = Image.new('RGB', (args.width, args.height//2), (255, 255, 255))
    combined.paste(result_vis, (0, 0))
    combined.paste(left_vis.convert('RGB'), (args.width//2, 0))
    combined.paste(right_vis.convert('RGB'), (args.width//2 + args.width//4, 0))
    
    vis_filename = args.output.replace('.png', '_visualization.png')
    combined.save(vis_filename)
    print(f"Saved visualization: {vis_filename}")
    

if __name__ == "__main__":
    main()
