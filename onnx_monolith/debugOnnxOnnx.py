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

def postprocess_image(tensor: np.ndarray) -> Image.Image:
    img = tensor[0]
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0, 255)
    return Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))

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
    
    # Preprocess image
    print(f"Loading and preprocessing {TEST_IMAGE_PATH}...")
    image_np = preprocess_image(TEST_IMAGE_PATH, np.float16)
    
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
    
    # Postprocess and save
    print(f"Saving result to {OUTPUT_IMAGE_PATH}...")
    final_image = postprocess_image(reconstructed_image)
    final_image.save(OUTPUT_IMAGE_PATH)
    
    print("--- Test complete! ---")