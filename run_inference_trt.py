import torch
import tensorrt as trt
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, UNet2DConditionModel
from accelerate import init_empty_weights
import numpy as np
import os
from compel import Compel, ReturnedEmbeddingsType
from io import BytesIO
from PIL import Image
import re
import time

class _SDXLTRTPipeline:
    def __init__(self):
        self.pipe = None
        self.compel = None
        self.trt_context = None
        self.trt_engine = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_name = "conv2d_50"  # From the onnx export log
        self._initialize()

    def _initialize(self):
        """
        Loads all necessary models and the TensorRT engine.
        """
        engine_file_path = "unet.engine"
        model_id = "socks22/sdxl-wai-nsfw-illustriousv14"

        if not os.path.exists(engine_file_path):
            raise FileNotFoundError(f"Engine file not found at {engine_file_path}. Please run unet_onnx_tensorrt.py first.")

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

        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

        self.compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )

        # Move models to CPU to save VRAM. They will be moved to GPU when needed.
        self.pipe.vae.to("cpu")
        self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder_2.to("cpu")

        print(f"Loading TensorRT engine from: {engine_file_path}")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())
        
        self.trt_context = self.trt_engine.create_execution_context()

        self.pipe.unet.forward = self._trt_unet_forward
        print("UNet has been replaced with TensorRT engine.")
        print("SDXL TensorRT Pipeline initialized.")

    def _trt_unet_forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        stream = torch.cuda.Stream()
        
        batch_size = sample.shape[0]
        timestep = timestep.expand(batch_size).to(dtype=torch.float16)
        added_cond = kwargs["added_cond_kwargs"]

        input_tensors = {
            "sample": sample.contiguous(),
            "timestep": timestep.contiguous(),
            "encoder_hidden_states": encoder_hidden_states.contiguous(),
            "text_embeds": added_cond["text_embeds"].contiguous(),
            "time_ids": added_cond["time_ids"].contiguous()
        }
        
        for name, tensor in input_tensors.items():
            self.trt_context.set_input_shape(name, tensor.shape)
            
        output_buffers = {}
        for binding in self.trt_engine:
            if self.trt_engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
                shape = input_tensors["sample"].shape
                dtype = torch.from_numpy(np.array([], dtype=trt.nptype(self.trt_engine.get_tensor_dtype(binding)))).dtype
                output_buffers[binding] = torch.empty(shape, dtype=dtype, device=self.device).contiguous()

        for name, tensor in input_tensors.items():
            self.trt_context.set_tensor_address(name, tensor.data_ptr())
        for name, buffer in output_buffers.items():
            self.trt_context.set_tensor_address(name, buffer.data_ptr())
        
        self.trt_context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

        from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
        return UNet2DConditionOutput(sample=output_buffers[self.output_name])

    def generate(self, prompt, seed=None, steps=4):
        height = 768
        width = 1152
        batch_size = 1
        num_images_per_prompt = 1

        # 1. CLIP text encoding
        t0 = time.time()
        prompt_prefix = "masterpiece,best quality,amazing quality, anime, aqua_(konosuba)"
        full_prompt = f"{prompt_prefix}, {prompt}"
        
        self.pipe.text_encoder.to(self.device)
        self.pipe.text_encoder_2.to(self.device)
        prompt_embeds, pooled_prompt_embeds = self.compel(full_prompt)
        self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()
        
        t1 = time.time()
        print(f"CLIP encoding took: {t1 - t0:.2f}s")
        
        # 2. Prepare latents
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        latents_shape = (
            batch_size * num_images_per_prompt,
            self.pipe.unet.config.in_channels,
            height // self.pipe.vae_scale_factor,
            width // self.pipe.vae_scale_factor,
        )
        latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=self.pipe.vae.dtype)
        
        # 3. Denoising loop (UNet)
        t2 = time.time()
        self.pipe.scheduler.set_timesteps(steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        latents = latents * self.pipe.scheduler.init_noise_sigma

        # Prepare additional conditioning signals
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=prompt_embeds.dtype, device=self.device)
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        for t in timesteps:
            latent_model_input = self.pipe.scheduler.scale_model_input(latents, t)
            
            noise_pred = self.pipe.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
            
            latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            del latent_model_input
            del noise_pred
            
            
            
        t3 = time.time()
        print(f"UNet denoising loop ({len(timesteps)} steps) took: {t3 - t2:.2f}s")

        # 4. VAE decoding
        t4 = time.time()
        self.pipe.vae.to(self.device)
        latents = latents / self.pipe.vae.config.scaling_factor
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        self.pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        t5 = time.time()
        print(f"VAE decoding took: {t5 - t4:.2f}s")

        # 5. Post-processing
        image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]

        # 6. Convert to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        return img_bytes, image

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
    
    seed = int(seed_match.group(1)) if seed_match else None
    steps = int(steps_match.group(1)) if steps_match else 4
    
    # Remove args from prompt
    clean_prompt = re.sub(r'--seed\s+\d+', '', prompt).strip()
    clean_prompt = re.sub(r'--steps\s+\d+', '', clean_prompt).strip()
    
    print(f"Generating image with prompt: '{clean_prompt}', seed: {seed}, steps: {steps}")
    
    return _pipeline_instance.generate(clean_prompt, seed, steps)

if __name__ == "__main__":
    initialize_pipeline()
    generate_image("masterpiece,best quality,amazing quality, anime, aqua_(konosuba), beautiful landscape") 