import torch
import onnxruntime
import numpy as np
from PIL import Image
import math
import json

# --- Configuration ---
ENCODER_PATH = "encoder.onnx"
DECODER_PATH = "decoder.onnx"
CONFIG_PATH = "/lab/model/vae/config.json"
TEST_IMAGE_PATH = "test.png"
OUTPUT_IMAGE_PATH = "testOnnxOnnx.png"

# --- Helper Functions ---
def preprocess_image(image_path: str, target_dtype: np.dtype) -> np.ndarray:
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
    print("--- Testing ONNX Encoder -> ONNX Decoder Pipeline ---")
    
    # Load config for scaling_factor
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    inv_scaling_factor = 1.0 / config["scaling_factor"]

    # Load ONNX models
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    encoder_sess = onnxruntime.InferenceSession(ENCODER_PATH, providers=providers)
    decoder_sess = onnxruntime.InferenceSession(DECODER_PATH, providers=providers)
    
    # Set up device for post-processing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Preprocess image
    print(f"Loading and preprocessing {TEST_IMAGE_PATH}...")
    image_np = preprocess_image(TEST_IMAGE_PATH, np.dtype(np.float16))
    
    # Encode
    print("Running ONNX encoder...")
    encoder_inputs = {encoder_sess.get_inputs()[0].name: image_np}
    latent_sample = encoder_sess.run(None, encoder_inputs)[0]
    
    # Scale latents
    latents_for_decoder = latent_sample * inv_scaling_factor
    
    # Decode
    print("Running ONNX decoder...")
    decoder_inputs = {decoder_sess.get_inputs()[0].name: latents_for_decoder}
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