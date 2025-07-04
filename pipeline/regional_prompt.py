import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple

# --- 1. Helper Function to Find Token Indices ---
# This is crucial for knowing which tokens in the full prompt correspond to our regional prompts.
def find_token_indices(tokenizer, prompt: str, sub_prompt: str) -> List[int]:
    """Finds the indices of a sub-prompt's tokens within a larger prompt."""
    # Tokenize the full prompt and the sub-prompt
    tokens_prompt = tokenizer.tokenize(prompt)
    tokens_sub_prompt = tokenizer.tokenize(sub_prompt)

    # Find the starting index of the sub-prompt sequence
    for i in range(len(tokens_prompt) - len(tokens_sub_prompt) + 1):
        if tokens_prompt[i:i + len(tokens_sub_prompt)] == tokens_sub_prompt:
            # We must account for the starting BOS (Beginning of Sequence) token
            # which is always at index 0.
            return list(range(i + 1, i + len(tokens_sub_prompt) + 1))
    raise ValueError(f"Sub-prompt '{sub_prompt}' not found in prompt '{prompt}'.")

# --- 2. The Core Logic: Custom Attention Processor ---
class RegionalAttnProcessor:
    """
    A custom attention processor that applies different prompt parts to different image regions.
    """
    def __init__(self, regions: Dict[str, Dict]):
        self.regions = regions
        # Store the original attention processor to restore it later
        self.original_attn_processor = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        # Let the UNet use the full, combined prompt embeddings
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Reshape for attention calculation
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        # This is the standard attention score calculation
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # ------------------------------------------------------------------
        # --- ATTENTION COUPLING MODIFICATION ---
        # ------------------------------------------------------------------
        
        # `attention_probs` has shape (batch_size * num_heads, sequence_length, num_tokens)
        # `sequence_length` corresponds to the flattened latent image patches (e.g., 64*64=4096 for a 512x512 image)
        # `num_tokens` is the length of the tokenized prompt (e.g., 77 for SD 1.5)

        # Get the latent shape (e.g., 64x64)
        latent_h = int(np.sqrt(attention_probs.shape[1]))
        latent_w = latent_h

        # Clone the original probabilities to modify them
        modified_attention_probs = torch.zeros_like(attention_probs)

        # Get all token indices that are part of any regional prompt
        all_regional_indices = set()
        for region_data in self.regions.values():
            all_regional_indices.update(region_data["indices"])

        # Get "global" token indices (e.g., for style, quality, connectors like 'AND')
        num_tokens = attention_probs.shape[-1]
        global_indices = [i for i in range(num_tokens) if i not in all_regional_indices]

        # First, apply the global prompt tokens to the entire image
        modified_attention_probs[:, :, global_indices] = attention_probs[:, :, global_indices]
        
        # Now, apply regional prompts to their designated areas
        for region_data in self.regions.values():
            mask = region_data["mask"]
            indices = region_data["indices"]

            # Resize the user-provided mask (e.g., 1024x1024) to the latent space size (e.g., 32x32 for SDXL)
            # and flatten it to match the `sequence_length` dimension of the attention map.
            resized_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(latent_h, latent_w), mode='bilinear')
            resized_mask = resized_mask.squeeze().view(-1).to(attention_probs.device) # Shape: (sequence_length,)
            
            # Select the pixels (rows) that correspond to this mask
            mask_rows = resized_mask > 0.5
            
            # For the masked pixels, copy the attention scores but ONLY for the tokens of the regional prompt
            modified_attention_probs[:, mask_rows, indices] = attention_probs[:, mask_rows, indices]
            
        # Re-normalize the attention probabilities so each row sums to 1 again
        # This is a critical step to ensure the output is valid.
        row_sums = modified_attention_probs.sum(dim=-1, keepdim=True)
        # Avoid division by zero for rows that might have all-zero attention
        row_sums[row_sums == 0] = 1.0
        normalized_attention_probs = modified_attention_probs / row_sums
        # ------------------------------------------------------------------
        # --- END OF MODIFICATION ---
        # ------------------------------------------------------------------

        # Compute the final hidden states using the modified attention map
        hidden_states = torch.bmm(normalized_attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


# --- 3. Main Script Setup and Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    # Use a faster VAE for SDXL
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "socks22/sdxl-wai-nsfw-illustriousv14",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")

    # Define our regional prompts
    prompt_left = "masterpiece, best quality, absurdres, cinematic lighting, ultra-detailed, aqua_(konosuba), smiling"
    prompt_right = "masterpiece, best quality, absurdres, cinematic lighting, ultra-detailed, megumin, sad"
    
    # A global prompt can add style but is not strictly necessary. The 'AND' acts as a separator.
    # The key is that the full prompt must contain the regional prompts verbatim.
    full_prompt = f"{prompt_left} AND {prompt_right}, cinematic, 8k, masterpiece"
    negative_prompt = "blurry, ugly, deformed, text, watermark"
    
    width, height = 1024, 1024
    
    # --- Find Token Indices for each region ---
    print("Finding token indices...")
    # SDXL uses two tokenizers, we'll use the first one which is standard.
    tokenizer = pipe.tokenizer
    
    indices_left = find_token_indices(tokenizer, full_prompt, prompt_left)
    indices_right = find_token_indices(tokenizer, full_prompt, prompt_right)

    print(f"Left prompt '{prompt_left}' indices: {indices_left}")
    print(f"Right prompt '{prompt_right}' indices: {indices_right}")

    # --- Create Spatial Masks ---
    # These masks define which pixels are controlled by which prompt.
    # They should be the same size as the final image.
    print("Creating spatial masks...")
    mask_left = torch.zeros((height, width))
    mask_left[:, :width // 2] = 1.0  # Left half of the image

    mask_right = torch.zeros((height, width))
    mask_right[:, width // 2:] = 1.0 # Right half of the image
    
    # --- Prepare for Injection ---
    # This dictionary will be passed to our custom attention processor
    regions_config = {
        "left": {"mask": mask_left, "indices": indices_left},
        "right": {"mask": mask_right, "indices": indices_right}
    }

    # --- Inject the Custom Attention Processor ---
    print("Injecting custom attention processor...")
    # The UNet has many attention blocks. We'll replace all of them.
    # We need to import torch.nn.functional to be used inside the processor
    import torch.nn.functional as F
    
    original_processors = {}
    for name, module in pipe.unet.named_modules():
        if "attn2" in name and "processor" in name: # Target cross-attention blocks
            original_processors[name] = module.processor
            module.processor = RegionalAttnProcessor(regions=regions_config)

    # --- Generate the Image ---
    print("Generating image with attention coupling...")
    generator = torch.Generator("cuda").manual_seed(42)
    image = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=7.5,
        num_inference_steps=30,
        generator=generator,
    ).images[0]
    
    print("Image generation complete.")
    image.save("multi_character_composition_sdxl.png")
    print("Saved image to 'multi_character_composition_sdxl.png'")
    
    # --- Restore Original Processors (Good Practice) ---
    print("Restoring original attention processors...")
    for name, module in pipe.unet.named_modules():
        if "attn2" in name and "processor" in name:
            module.processor = original_processors[name]