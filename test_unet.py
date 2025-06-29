import torch
from diffusers import StableDiffusionXLPipeline

class UNetTypeInspector(torch.nn.Module):
    """
    A wrapper for a UNet model that intercepts the forward pass
    to print the data types of all its inputs.
    """
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.device = unet.device

    def print_types(self, name, tensor):
        if isinstance(tensor, torch.Tensor):
            print(f"  - Arg '{name}': {tensor.dtype} on device '{tensor.device}'")
        elif isinstance(tensor, dict):
            print(f"  - Arg '{name}' (dict):")
            for k, v in tensor.items():
                self.print_types(f"    - '{k}'", v)
        else:
            print(f"  - Arg '{name}': {type(tensor)}")

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        print("\n--- Inspecting UNet.forward() input types (from pipeline call) ---")
        self.print_types("sample", sample)
        self.print_types("timestep", timestep)
        self.print_types("encoder_hidden_states", encoder_hidden_states)
        
        # In a real pipeline call, added_cond_kwargs is inside kwargs
        if 'added_cond_kwargs' in kwargs:
            self.print_types("added_cond_kwargs", kwargs['added_cond_kwargs'])
        
        print("------------------------------------------------------------------\n")
        
        # Call the original UNet's forward method
        return self.unet.forward(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )

def main():
    """
    Loads the SDXL UNet, wraps it to inspect input types, and runs a test pass
    by calling the main diffusers pipeline.
    """
    model_id = "socks22/sdxl-wai-nsfw-illustriousv14"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading SDXL model: {model_id}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        use_safetensors=True
    )
    
    print("Loading and fusing DMD2 LoRA...")
    pipe.load_lora_weights("tianweiy/DMD2", weight_name="dmd2_sdxl_4step_lora_fp16.safetensors")
    pipe.fuse_lora(lora_scale=0.8)
    
    pipe.to(device)
    
    unet = pipe.unet
    unet.eval()
    
    print(f"UNet's main dtype: {unet.dtype}")

    # Wrap the original UNet and replace it inside the pipeline
    print("\nReplacing UNet in pipeline with type inspector...")
    pipe.unet = UNetTypeInspector(unet)

    print("\nCalling pipeline to trigger the inspector...")
    
    # We only need a few steps to see the types, and can skip the VAE
    try:
        with torch.no_grad():
            pipe(
                "a cinematic test prompt", 
                num_inference_steps=2, 
                output_type="latent",
                guidance_scale=1.0
            )
        print("\nPipeline test call completed successfully.")
    except Exception as e:
        print(f"\nAn error occurred during the pipeline call: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 