import torch
from diffusers import AutoencoderKL
import onnxruntime
import numpy as np
from PIL import Image
import math

# --- Configuration ---
VAE_PATH = "madebyollin/sdxl-vae-fp16-fix"
DECODER_PATH = "decoder.onnx"
TEST_IMAGE_PATH = "test.png"
OUTPUT_IMAGE_PATH = "testDiffusersOnnx.png"

# --- Helper Functions (for PyTorch) ---
def preprocess_for_diffusers(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    p_w = math.ceil(w / 8) * 8
    p_h = math.ceil(h / 8) * 8
    canvas = Image.new("RGB", (p_w, p_h), (0, 0, 0))
    canvas.paste(img, (0, 0))
    arr = np.array(canvas).astype(np.float32) # Diffusers prefers float32 for preproc
    arr = (arr / 127.5) - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

if __name__ == '__main__':
    print("--- Testing Diffusers Encoder -> ONNX Decoder Pipeline ---")
    
    # Load Diffusers VAE and ONNX decoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    
    print(f"Loading Diffusers VAE from {VAE_PATH}...")
    # Load model on CPU first to avoid meta tensor issue, then handle device transfer manually
    vae = AutoencoderKL.from_pretrained(VAE_PATH, torch_dtype=torch.float16)
    
    # Move to device using to_empty() if it has meta tensors, otherwise use regular to()
    try:
        vae = vae.to(device)
    except NotImplementedError as e:
        if "meta tensor" in str(e):
            print("Handling meta tensor - using to_empty() instead of to()")
            vae = vae.to_empty(device=device)
        else:
            raise e
    
    print(f"Loading ONNX decoder from {DECODER_PATH}...")
    decoder_sess = onnxruntime.InferenceSession(DECODER_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # Preprocess image for Diffusers
    print(f"Loading and preprocessing {TEST_IMAGE_PATH}...")
    image_tensor = preprocess_for_diffusers(TEST_IMAGE_PATH).to(device, dtype=dtype)
    
    # Encode with Diffusers
    print("Running Diffusers encoder...")
    with torch.no_grad():
        latent_dist = vae.encode(image_tensor).latent_dist
        # Use the deterministic mode, not random sample
        latents = latent_dist.mode() 
    
    # Scale latents (this is done inside vae.decode in diffusers, so we replicate it)
    #latents_for_decoder = latents * vae.config.scaling_factor
    
    # Convert to NumPy for ONNX Runtime
    latents_np = latents.cpu().numpy()
    
    # Decode with ONNX
    print("Running ONNX decoder...")
    decoder_inputs = {decoder_sess.get_inputs()[0].name: latents_np}
    reconstructed_image = decoder_sess.run(None, decoder_inputs)[0]
    
    # Convert ONNX output back to torch tensor for post-processing
    image_tensor = torch.from_numpy(reconstructed_image).to(device)
    
    # Post-process and save using the RUN_INFERENCE.py approach
    print("Post-processing and saving image...")
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_np = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
    
    # Print stats before casting to uint8
    print(f"Image (min/max/mean): {image_np.min():.4f}, {image_np.max():.4f}, {image_np.mean():.4f}. Contains NaNs: {np.isnan(image_np).any()}")

    image_pil = Image.fromarray((image_np[0] * 255).round().astype("uint8"))

    output_image_path = "output_image.png"
    image_pil.save(output_image_path)
    print(f"✓ Image saved to {output_image_path}")
    
    print("--- Test complete! ---")