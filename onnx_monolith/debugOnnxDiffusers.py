import torch
from diffusers import AutoencoderKL
import onnxruntime
import numpy as np
from PIL import Image
import math

# --- Configuration ---
VAE_PATH = "madebyollin/sdxl-vae-fp16-fix"
ENCODER_PATH = "encoder.onnx"
TEST_IMAGE_PATH = "test.png"
OUTPUT_IMAGE_PATH = "testOnnxDiffusers.png"

# --- Helper Functions ---
def preprocess_for_onnx(image_path: str, target_dtype: np.dtype) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    p_w = math.ceil(w / 8) * 8
    p_h = math.ceil(h / 8) * 8
    canvas = Image.new("RGB", (p_w, p_h), (0, 0, 0))
    canvas.paste(img, (0, 0))
    arr = np.array(canvas).astype(target_dtype)
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr.transpose(2, 0, 1), 0)

def postprocess_torch_image(tensor: torch.Tensor) -> Image.Image:
    # Diffusers output is already in range [0, 1] if not clamped
    img = tensor.squeeze(0) # Remove batch dim
    img = (img / 2 + 0.5).clamp(0, 1) # Denormalize from [-1, 1] to [0, 1]
    img = img.permute(1, 2, 0).cpu().numpy() # CHW to HWC
    return Image.fromarray((img * 255).astype(np.uint8))


if __name__ == '__main__':
    print("--- Testing ONNX Encoder -> Diffusers Decoder Pipeline ---")
    
    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    
    print(f"Loading ONNX encoder from {ENCODER_PATH}...")
    encoder_sess = onnxruntime.InferenceSession(ENCODER_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    print(f"Loading Diffusers VAE from {VAE_PATH}...")
    vae = AutoencoderKL.from_pretrained(VAE_PATH, torch_dtype=torch.float16).to(device)
    

    # Preprocess image for ONNX
    print(f"Loading and preprocessing {TEST_IMAGE_PATH}...")
    image_np = preprocess_for_onnx(TEST_IMAGE_PATH, np.float16)

    # Encode with ONNX
    print("Running ONNX encoder...")
    encoder_inputs = {encoder_sess.get_inputs()[0].name: image_np}
    latent_sample_np = encoder_sess.run(None, encoder_inputs)[0]

    # Convert to PyTorch tensor for Diffusers
    latents_for_decoder = torch.from_numpy(latent_sample_np).to(device, dtype=dtype)
    
    # The scaling factor is applied *before* the decoder in Diffusers
    latents_for_decoder = latents_for_decoder / vae.config.scaling_factor
    
    # Decode with Diffusers
    print("Running Diffusers decoder...")
    with torch.no_grad():
        reconstructed_image = vae.decode(latents_for_decoder).sample

    # Postprocess and save
    print(f"Saving result to {OUTPUT_IMAGE_PATH}...")
    final_image = postprocess_torch_image(reconstructed_image)
    final_image.save(OUTPUT_IMAGE_PATH)

    print("--- Test complete! ---")
