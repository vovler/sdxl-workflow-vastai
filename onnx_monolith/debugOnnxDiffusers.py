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
    image_np = preprocess_for_onnx(TEST_IMAGE_PATH, np.dtype(np.float16))

    # Encode with ONNX
    print("Running ONNX encoder...")
    encoder_inputs = {encoder_sess.get_inputs()[0].name: image_np}
    latent_sample_np = encoder_sess.run(None, encoder_inputs)[0]

    # Convert to PyTorch tensor for Diffusers
    latents_for_decoder = torch.from_numpy(latent_sample_np).to(device, dtype=dtype)
    
    # The scaling factor is applied *before* the decoder in Diffusers
    #latents_for_decoder = latents_for_decoder / vae.config.scaling_factor
    
    # Decode with Diffusers
    print("Running Diffusers decoder...")
    with torch.no_grad():
        image_tensor = vae.decode(latents_for_decoder).sample

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
