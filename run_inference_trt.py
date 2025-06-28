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

        self.pipe.vae.to(self.device)
        self.pipe.text_encoder.to(self.device)
        self.pipe.text_encoder_2.to(self.device)

        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

        self.compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )

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
        prompt_prefix = "masterpiece,best quality,amazing quality, anime, aqua_(konosuba)"
        full_prompt = f"{prompt_prefix}, {prompt}"
        negative_prompt = "lowres, bad anatomy, bad hands, blurry, text, watermark, signature"
        
        prompt_embeds, pooled_prompt_embeds = self.compel(full_prompt)
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        pipe_kwargs = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "num_inference_steps": steps,
            "guidance_scale": 1.0,
            "height": 768,
            "width": 1152,
            "generator": generator,
        }

        result = self.pipe(**pipe_kwargs)
        image = result.images[0]
        
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