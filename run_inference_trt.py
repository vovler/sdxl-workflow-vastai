import torch
import tensorrt as trt
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, UNet2DConditionModel
from accelerate import init_empty_weights
import numpy as np
import os
import json
from compel import Compel, ReturnedEmbeddingsType
from io import BytesIO
from PIL import Image
import re

class _SDXLTRTPipeline:
    def __init__(self):
        self.pipe = None
        self.compel = None
        self.trt_context = None
        self.trt_engine = None
        self.trt_engine_blob = None
        self.profile_map = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_name = "conv2d_50"
        self._initialize()

    def _initialize(self):
        """
        Loads all necessary models, the TensorRT engine, and the profile map.
        """
        engine_file_path = "unet.engine"
        profile_map_path = "profile_map.json"
        model_id = "socks22/sdxl-wai-nsfw-illustriousv14"

        if not os.path.exists(engine_file_path):
            raise FileNotFoundError(f"Engine file not found at {engine_file_path}. Please run unet_onnx_tensorrt.py first.")
        
        if not os.path.exists(profile_map_path):
            raise FileNotFoundError(f"Profile map not found at {profile_map_path}. Please run unet_onnx_tensorrt.py first.")

        print("Loading profile map...")
        with open(profile_map_path, "r") as f:
            self.profile_map = json.load(f)

        print("Creating dummy UNet to avoid loading original...")
        unet_config = UNet2DConditionModel.load_config(model_id, subfolder="unet")
        with init_empty_weights():
            unet = UNet2DConditionModel.from_config(unet_config)

        print("Loading SDXL pipeline...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            unet=unet,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        print("Moving VAE to CPU to save VRAM...")
        self.pipe.vae.to("cpu")
        self.pipe.text_encoder.to("cuda")
        self.pipe.text_encoder_2.to("cuda")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

        self.compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )

        print(f"Loading serialized TensorRT engine from: {engine_file_path}")
        with open(engine_file_path, 'rb') as f:
            self.trt_engine_blob = f.read()

        self.pipe.unet.forward = self._trt_unet_forward
        print("UNet has been replaced with TensorRT engine.")
        print("SDXL TensorRT Pipeline initialized.")

    def _get_profile_id(self, batch_size, height, width, num_tokens):
        """
        Finds the index of the optimization profile for a given set of runtime parameters.
        """
        # Find the closest valid prompt length
        valid_prompt_lengths = sorted(list(set(p['num_tokens'] for p in self.profile_map)))
        closest_num_tokens = min(valid_prompt_lengths, key=lambda x: x if x >= num_tokens else float('inf'))

        try:
            runtime_config = {
                "batch_size": batch_size,
                "height": height,
                "width": width,
                "num_tokens": closest_num_tokens
            }
            profile_id = self.profile_map.index(runtime_config)
            print(f"Selected profile #{profile_id} for shape: (batch={batch_size}, size={height}x{width}, tokens={num_tokens})")
            return profile_id
        except ValueError:
            print(f"Error: No profile found for configuration: {runtime_config}")
            print("Please ensure your runtime parameters match one of the built profiles.")
            return -1

    def _trt_unet_forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        """
        Manages the lifecycle of the TRT engine for a single forward pass to minimize VRAM.
        It deserializes, runs, and cleans up the engine on each call.
        """
        print("--- TRT UNet Forward Step ---")
        # 1. Deserialize engine and create context
        print("Deserializing TRT engine for this step...")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(self.trt_engine_blob)
        context = engine.create_execution_context()
        print("Engine deserialized.")

        added_cond = kwargs["added_cond_kwargs"]
        
        stream = torch.cuda.Stream()
        
        # 3. Determine runtime dimensions & select profile
        unet_batch_size = sample.shape[0]
        height = sample.shape[2] * 8
        width = sample.shape[3] * 8
        num_tokens = encoder_hidden_states.shape[1]

        profile_id = self._get_profile_id(unet_batch_size, height, width, num_tokens)
        if profile_id == -1:
            raise ValueError("Could not find a valid optimization profile for the current input.")
        
        print(f"Selected profile #{profile_id} for shape: (batch={unet_batch_size}, size={height}x{width}, tokens={num_tokens})")
        context.set_optimization_profile_async(profile_id, stream.cuda_stream)
        
        # 4. Prepare tensors and buffers
        batch_size = sample.shape[0]
        timestep = timestep.expand(batch_size).to(device=self.device, dtype=torch.float16)

        input_tensors = {
            "sample": sample.contiguous(),
            "timestep": timestep.contiguous(),
            "encoder_hidden_states": encoder_hidden_states.contiguous(),
            "text_embeds": added_cond["text_embeds"].contiguous(),
            "time_ids": added_cond["time_ids"].contiguous()
        }
        
        for name, tensor in input_tensors.items():
            context.set_input_shape(name, tensor.shape)
            
        output_buffers = {}
        for binding in engine:
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
                shape = input_tensors["sample"].shape
                dtype = torch.from_numpy(np.array([], dtype=trt.nptype(engine.get_tensor_dtype(binding)))).dtype
                output_buffers[binding] = torch.empty(shape, dtype=dtype, device=self.device).contiguous()

        for name, tensor in input_tensors.items():
            context.set_tensor_address(name, tensor.data_ptr())
        for name, buffer in output_buffers.items():
            context.set_tensor_address(name, buffer.data_ptr())
        
        # 5. Execute engine
        print("Executing TRT engine...")
        success = context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not success:
            raise RuntimeError("Failed to execute TensorRT engine.")
        stream.synchronize()
        print("TRT engine execution complete.")

        from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
        
        # 6. Keep output on GPU
        output_tensor = output_buffers[self.output_name]
        
        # 7. Clean up GPU resources for UNet
        print("Cleaning up TRT engine resources...")
        del context
        del engine
        del runtime

        # 8. Move VAE to GPU in preparation for the final decoding step
        # This is done here to ensure VAE is ready after the last UNet step.
        if self.pipe.vae.device.type == 'cpu':
             print("Moving VAE to GPU...")
             self.pipe.vae.to(self.device)
        
        print("--- TRT UNet Forward Step Complete ---")

        return UNet2DConditionOutput(sample=output_tensor)

    def generate(self, prompt, batch_size=1, height=768, width=1152, seed=None, steps=4):
        full_prompt = prompt

        # --- VRAM Management: Text Encoding ---
        print("\n--- Prompt Encoding Step ---")

        print("Moving Text Encoders to GPU...")
        self.pipe.text_encoder.to(self.device)
        self.pipe.text_encoder_2.to(self.device)

        prompt_embeds, pooled_prompt_embeds = self.compel(full_prompt)
        print(f"prompt_embeds size: {prompt_embeds.size()}")
        print(f"pooled_prompt_embeds size: {pooled_prompt_embeds.size()}")
        
        print("Moving Text Encoders and embeddings to CPU...")
        self.pipe.text_encoder.to('cpu')
        self.pipe.text_encoder_2.to('cpu')
        print("--- Prompt Encoding Complete ---\n")
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # The main pipeline call will handle denoising. The _trt_unet_forward function
        # will now manage moving the VAE to the GPU at the appropriate time.
        print("--- Starting Denoising and Decoding Pipeline ---")
        pipe_kwargs = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "num_inference_steps": steps,
            "num_images_per_prompt": batch_size,
            "guidance_scale": 1.0,
            "height": height,
            "width": width,
            "generator": generator,
        }

        result = self.pipe(**pipe_kwargs)
        images = result.images
        print("--- Pipeline Complete ---")

        # Move VAE back to CPU to conserve VRAM for the next run
        print("Moving VAE back to CPU...")
        self.pipe.vae.to("cpu")
        
        image_byte_arrays = []
        for image in images:
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            image_byte_arrays.append(img_bytes)

        return image_byte_arrays, images

# --- Singleton instance and interface functions ---
_pipeline_instance = None

def initialize_pipeline():
    """
    Initializes the singleton pipeline instance.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = _SDXLTRTPipeline()

def generate_image(prompt: str):
    """
    Parses the prompt for parameters and generates an image.
    """
    if _pipeline_instance is None:
        raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")

    seed_match = re.search(r'--seed\s+(\d+)', prompt)
    steps_match = re.search(r'--steps\s+(\d+)', prompt)
    batch_match = re.search(r'--batch\s+(\d+)', prompt)
    height_match = re.search(r'--height\s+(\d+)', prompt)
    width_match = re.search(r'--width\s+(\d+)', prompt)
    
    seed = int(seed_match.group(1)) if seed_match else None
    steps = int(steps_match.group(1)) if steps_match else 8
    batch_size = int(batch_match.group(1)) if batch_match else 1
    height = int(height_match.group(1)) if height_match else 768
    width = int(width_match.group(1)) if width_match else 1152
    
    # Remove args from prompt
    clean_prompt = re.sub(r'--\w+\s+\d+', '', prompt).strip()
    
    print(f"Generating image with prompt: '{clean_prompt}', seed: {seed}, steps: {steps}, batch: {batch_size}, resolution: {height}x{width}")
    
    return _pipeline_instance.generate(clean_prompt, batch_size, height, width, seed, steps)

if __name__ == "__main__":
    initialize_pipeline()
    # Example: masterpiece,best quality,amazing quality, anime, aqua_(konosuba), beautiful landscape --seed 123 --steps 20 --batch 1 --height 768 --width 1152
    generate_image("masterpiece,best quality,amazing quality, anime, aqua_(konosuba), beautiful landscape --seed 12345") 