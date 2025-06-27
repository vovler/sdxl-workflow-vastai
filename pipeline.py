#
# Copyright 2025 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# This script is a heavily modified version of the original Stable Diffusion TensorRT pipeline
# to support the architecture of Stable Diffusion XL (SDXL).
# Key changes include:
# - Support for dual text encoders (CLIP ViT-L and OpenCLIP ViT-bigG).
# - Updated UNet model definition to handle larger embeddings and additional conditioning (add_text_embeds, add_time_ids).
# - A two-stage denoising process to support an optional refiner model.
# - Reworked pipeline logic to correctly process and feed data to the SDXL components.
#

import gc
import os
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import numpy as np
import onnx
from tensorrt.tools import onnx_graphsurgeon as gs
import PIL.Image
import tensorrt as trt
import torch
from cuda import cudart
from huggingface_hub import snapshot_download
from huggingface_hub.utils import validate_hf_hub_args
from onnx import shape_inference
from packaging import version
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.onnx.loader import fold_constants
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import Attention
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value: key for (key, value) in numpy_to_torch_dtype_dict.items()}


class Engine:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.stream = None

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def build(self, onnx_path, fp16, input_profile=None, enable_all_tactics=False, timing_cache=None, workspace_size=0):
        logger.info(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])
        
        config_kwargs = {}
        if version.parse(trt.__version__) >= version.parse("10.1.0"):
             config_kwargs['memory_pool_limits'] = {trt.MemoryPoolType.WORKSPACE: workspace_size}
        else:
             config_kwargs['max_workspace_size'] = workspace_size
             
        engine = engine_from_network(
            network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
            config=CreateConfig(
                fp16=fp16, 
                profiles=[p], 
                tactic_sources=None if enable_all_tactics else [],
                **config_kwargs
            ),
            save_timing_cache=timing_cache
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        logger.info(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        for binding in self.engine:
            if binding in self.tensors:
                continue
            
            shape = self.engine.get_binding_shape(binding)
            # If we have dynamic shape, we need to update shape from shape_dict
            if -1 in shape:
                if shape_dict and binding in shape_dict:
                    shape = shape_dict[binding]
                else: # Use opt shape if no shape is provided
                    shape = self.engine.get_profile_shape(0, binding)[1]
            
            if not shape:
                logger.warning(f"Binding {binding} has empty shape. Skipping buffer allocation.")
                continue

            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[binding] = tensor

    def infer(self, feed_dict, stream):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)
        
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise ValueError("ERROR: inference failed.")

        return self.tensors


class Optimizer:
    def __init__(self, onnx_graph):
        self.graph = gs.import_onnx(onnx_graph)

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 4 * 1024 * 1024 * 1024: # 4GB
            logger.warning("ONNX graph size is larger than 4GB, skipping shape inference.")
            return onnx_graph if return_onnx else None
        
        onnx_graph = shape_inference.infer_shapes(onnx_graph)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph


class BaseModel:
    def __init__(self, model_name, model, device="cuda", fp16=False, max_batch_size=1, text_maxlen=77):
        self.name = model_name
        self.model = model
        self.device = device
        self.fp16 = fp16
        self.max_batch_size = max_batch_size
        self.text_maxlen = text_maxlen
        self.min_batch = 1
        
        self.min_image_dim = 256
        self.max_image_dim = 1024
        self.min_latent_dim = self.min_image_dim // 8
        self.max_latent_dim = self.max_image_dim // 8

    def get_model(self): return self.model
    def get_input_names(self): pass
    def get_output_names(self): pass
    def get_dynamic_axes(self): return None
    def get_sample_input(self, batch_size, image_height, image_width): pass
    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape): return None
    def get_shape_dict(self, batch_size, image_height, image_width): return None

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph)
        opt.cleanup()
        opt.fold_constants()
        opt.infer_shapes()
        return opt.cleanup(return_onnx=True)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch_size
        
        min_h = image_height if static_shape else self.min_image_dim
        max_h = image_height if static_shape else self.max_image_dim
        min_w = image_width if static_shape else self.min_image_dim
        max_w = image_width if static_shape else self.max_image_dim
        
        min_latent_h = min_h // 8
        max_latent_h = max_h // 8
        min_latent_w = min_w // 8
        max_latent_w = max_w // 8
        
        return (min_batch, max_batch, min_h, max_h, min_w, max_w, 
                min_latent_h, max_latent_h, min_latent_w, max_latent_w)


class CLIP1(BaseModel):
    def __init__(self, model_name, model, device, max_batch_size, text_maxlen):
        super().__init__(model_name, model, device, fp16=True, max_batch_size=max_batch_size, text_maxlen=text_maxlen)
        self.embedding_dim = 768

    def get_input_names(self): return ['input_ids']
    def get_output_names(self): return ['text_embeddings']
    def get_dynamic_axes(self): return {'input_ids': {0: 'B'}, 'text_embeddings': {0: 'B'}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {'input_ids': [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]}
    
    def get_shape_dict(self, batch_size, image_height, image_width):
        return {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.text_maxlen, self.embedding_dim)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph)
        opt.select_outputs([0], names=['text_embeddings'])
        return super().optimize(gs.export_onnx(opt.graph))


class CLIP2(BaseModel):
    def __init__(self, model_name, model, device, max_batch_size, text_maxlen):
        super().__init__(model_name, model, device, fp16=True, max_batch_size=max_batch_size, text_maxlen=text_maxlen)
        self.embedding_dim = 1280

    def get_input_names(self): return ['input_ids']
    def get_output_names(self): return ['text_embeddings', 'pooled_output']
    def get_dynamic_axes(self): return {'input_ids': {0: 'B'}, 'text_embeddings': {0: 'B'}, 'pooled_output': {0: 'B'}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {'input_ids': [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]}

    def get_shape_dict(self, batch_size, image_height, image_width):
        return {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.text_maxlen, self.embedding_dim),
            'pooled_output': (batch_size, self.embedding_dim)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph)
        # The original model returns a tuple (last_hidden_state, pooler_output, ...). We only need the first two.
        opt.select_outputs([0, 1], names=['text_embeddings', 'pooled_output'])
        return super().optimize(gs.export_onnx(opt.graph))


class UNetXL(BaseModel):
    def __init__(self, model_name, model, device, max_batch_size, text_maxlen):
        super().__init__(model_name, model, device, fp16=True, max_batch_size=max_batch_size, text_maxlen=text_maxlen)
        self.text_embed_dim = 2048  # 768 + 1280
        self.time_embed_dim = 2816 # Not directly used but good to know
        self.add_embed_dim = 1280

    def get_input_names(self): return ['sample', 'timestep', 'encoder_hidden_states', 'add_text_embeds', 'add_time_ids']
    def get_output_names(self): return ['latent']
    
    def get_dynamic_axes(self):
        return {
            'sample': {0: '2B', 2: 'H', 3: 'W'},
            'encoder_hidden_states': {0: '2B'},
            'add_text_embeds': {0: '2B'},
            'latent': {0: '2B', 2: 'H', 3: 'W'},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch, max_batch, _, _, _, _, min_latent_h, max_latent_h, min_latent_w, max_latent_w = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        
        latent_h = image_height // 8
        latent_w = image_width // 8
        
        return {
            'sample': [(2 * min_batch, 4, min_latent_h, min_latent_w), (2 * batch_size, 4, latent_h, latent_w), (2 * max_batch, 4, max_latent_h, max_latent_w)],
            'encoder_hidden_states': [(2 * min_batch, self.text_maxlen, self.text_embed_dim), (2 * batch_size, self.text_maxlen, self.text_embed_dim), (2 * max_batch, self.text_maxlen, self.text_embed_dim)],
            'add_text_embeds': [(2 * min_batch, self.add_embed_dim), (2 * batch_size, self.add_embed_dim), (2 * max_batch, self.add_embed_dim)],
        }
        
    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_h = image_height // 8
        latent_w = image_width // 8
        return {
            'sample': (2 * batch_size, 4, latent_h, latent_w),
            'timestep': (1,),
            'encoder_hidden_states': (2 * batch_size, self.text_maxlen, self.text_embed_dim),
            'add_text_embeds': (2 * batch_size, self.add_embed_dim),
            'add_time_ids': (2 * batch_size, 6),
            'latent': (2 * batch_size, 4, latent_h, latent_w)
        }
    
    def get_sample_input(self, batch_size, image_height, image_width):
        latent_h = image_height // 8
        latent_w = image_width // 8
        dtype = torch.float16
        return (
            torch.randn(2 * batch_size, 4, latent_h, latent_w, dtype=dtype, device=self.device),
            torch.tensor([1.0], dtype=torch.float32, device=self.device),
            torch.randn(2 * batch_size, self.text_maxlen, self.text_embed_dim, dtype=dtype, device=self.device),
            torch.randn(2 * batch_size, self.add_embed_dim, dtype=dtype, device=self.device),
            torch.randn(2 * batch_size, 6, dtype=dtype, device=self.device),
        )


class UNetRefiner(UNetXL): # Refiner has the same architecture as the base UNet
    pass


class VAEDecoder(BaseModel):
    def __init__(self, model_name, model, device, max_batch_size):
        super().__init__(model_name, model, device, fp16=True, max_batch_size=max_batch_size)
        
    def get_input_names(self): return ['latent']
    def get_output_names(self): return ['images']
    def get_dynamic_axes(self): return {'latent': {0: 'B', 2: 'H', 3: 'W'}, 'images': {0: 'B', 2: '8H', 3: '8W'}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch, max_batch, _, _, _, _, min_latent_h, max_latent_h, min_latent_w, max_latent_w = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        
        latent_h = image_height // 8
        latent_w = image_width // 8
        
        return {'latent': [(min_batch, 4, min_latent_h, min_latent_w), (batch_size, 4, latent_h, latent_w), (max_batch, 4, max_latent_h, max_latent_w)]}
        
    def get_shape_dict(self, batch_size, image_height, image_width):
        return {
            'latent': (batch_size, 4, image_height // 8, image_width // 8),
            'images': (batch_size, 3, image_height, image_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        return torch.randn(batch_size, 4, image_height // 8, image_width // 8, dtype=torch.float16, device=self.device)


def get_path(name, directory, opt=True):
    return os.path.join(directory, name + (".opt" if opt else "") + ".onnx")


def build_engines(
    models: dict,
    engine_dir,
    onnx_dir,
    onnx_opset,
    opt_image_height,
    opt_image_width,
    opt_batch_size=1,
    force_engine_rebuild=False,
    static_batch=False,
    static_shape=True,
    enable_all_tactics=False,
    timing_cache=None,
    workspace_size=0
):
    built_engines = {}
    if not os.path.isdir(onnx_dir):
        os.makedirs(onnx_dir)
    if not os.path.isdir(engine_dir):
        os.makedirs(engine_dir)

    for model_name, model_obj in models.items():
        engine_path = os.path.join(engine_dir, model_name + ".plan")
        if force_engine_rebuild or not os.path.exists(engine_path):
            logger.info(f"Building engine for {model_name}, this may take a while...")
            onnx_path = get_path(model_name, onnx_dir, opt=False)
            onnx_opt_path = get_path(model_name, onnx_dir, opt=True)
            
            if force_engine_rebuild or not os.path.exists(onnx_opt_path):
                if force_engine_rebuild or not os.path.exists(onnx_path):
                    logger.info(f"Exporting model to ONNX: {onnx_path}")
                    model = model_obj.get_model()
                    with torch.inference_mode(), torch.autocast("cuda"):
                        inputs = model_obj.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
                        # The VAE decode has a single input, so we need to handle it differently
                        if model_name == 'vae':
                            inputs = (inputs,)
                        
                        torch.onnx.export(
                            model,
                            inputs,
                            onnx_path,
                            export_params=True,
                            opset_version=onnx_opset,
                            do_constant_folding=True,
                            input_names=model_obj.get_input_names(),
                            output_names=model_obj.get_output_names(),
                            dynamic_axes=model_obj.get_dynamic_axes(),
                        )
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    logger.info(f"Found cached ONNX model: {onnx_path}")

                logger.info(f"Optimizing ONNX model: {onnx_opt_path}")
                onnx_opt_graph = model_obj.optimize(onnx.load(onnx_path))
                onnx.save(onnx_opt_graph, onnx_opt_path)
            else:
                logger.info(f"Found cached optimized ONNX model: {onnx_opt_path}")

            engine = Engine(engine_path)
            engine.build(
                onnx_opt_path,
                fp16=model_obj.fp16,
                input_profile=model_obj.get_input_profile(
                    opt_batch_size, opt_image_height, opt_image_width, static_batch, static_shape
                ),
                timing_cache=timing_cache,
                workspace_size=workspace_size
            )
        else:
            logger.info(f"Found cached engine: {engine_path}")
            
        built_engines[model_name] = Engine(engine_path)

    for model_name, engine in built_engines.items():
        engine.load()
        engine.activate()

    return built_engines


class TensorRTStableDiffusionXLPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DPMSolverMultistepScheduler,
        # Refiner components
        unet_refiner: Optional[UNet2DConditionModel] = None,
        vae_refiner: Optional[AutoencoderKL] = None, # Assumes same VAE for now
        # Engine build options
        image_height: int = 1024,
        image_width: int = 1024,
        max_batch_size: int = 1,
        onnx_opset: int = 13,
        onnx_dir: str = "onnx_xl",
        engine_dir: str = "engine_xl",
        force_engine_rebuild: bool = False,
        timing_cache: str = "timing_cache_xl",
        static_batch: bool = False,
        static_shape: bool = True,
        workspace_size: int = 4 * 1024 * 1024 * 1024, # 4GB workspace
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2,
            unet=unet, scheduler=scheduler, unet_refiner=unet_refiner
        )
        
        self.has_refiner = unet_refiner is not None
        self.image_height, self.image_width = image_height, image_width
        self.max_batch_size = max_batch_size
        self.onnx_opset = onnx_opset
        self.onnx_dir = onnx_dir
        self.engine_dir = engine_dir
        self.force_engine_rebuild = force_engine_rebuild
        self.timing_cache = timing_cache
        self.build_static_batch = static_batch
        self.build_dynamic_shape = not static_shape
        self.workspace_size = workspace_size
        
        self.stream = None
        self.models = {}
        self.engine = {}
        self.torch_device = "cuda"

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def to(self, torch_device: Union[str, torch.device] = "cuda", silence_dtype_warnings: bool = True):
        self.torch_device = torch_device
        
        # Move models to device before engine build.
        self.text_encoder.to(torch_device)
        self.text_encoder_2.to(torch_device)
        self.unet.to(torch_device)
        if self.has_refiner:
            self.unet_refiner.to(torch_device)
        self.vae.to(torch_device)

        # Update paths to be relative to the cached model directory
        if hasattr(self, "cached_folder"):
            self.onnx_dir = os.path.join(self.cached_folder, self.onnx_dir)
            self.engine_dir = os.path.join(self.cached_folder, self.engine_dir)
            self.timing_cache = os.path.join(self.cached_folder, self.timing_cache)
        
        logger.info(f"Running on device: {self.torch_device}")
        
        # Load models for engine build
        self.__load_models()

        # Build engines
        self.engine = build_engines(
            self.models, self.engine_dir, self.onnx_dir, self.onnx_opset,
            self.image_height, self.image_width, self.max_batch_size,
            self.force_engine_rebuild, self.build_static_batch, not self.build_dynamic_shape,
            timing_cache=self.timing_cache, workspace_size=self.workspace_size
        )
        
        # Free up model memory after engine build
        self.text_encoder = None
        self.text_encoder_2 = None
        self.unet = None
        self.unet_refiner = None
        self.vae = None
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("TensorRT engines built and loaded.")
        return self

    def __load_models(self):
        text_maxlen = self.tokenizer.model_max_length
        self.models['clip1'] = CLIP1("clip1", self.text_encoder, self.torch_device, self.max_batch_size, text_maxlen)
        self.models['clip2'] = CLIP2("clip2", self.text_encoder_2, self.torch_device, self.max_batch_size, text_maxlen)
        self.models['unet'] = UNetXL("unet", self.unet, self.torch_device, self.max_batch_size, text_maxlen)
        if self.has_refiner:
            self.models['unet_refiner'] = UNetRefiner("unet_refiner", self.unet_refiner, self.torch_device, self.max_batch_size, text_maxlen)
        self.models['vae'] = VAEDecoder("vae", self.vae.decode, self.torch_device, self.max_batch_size)
    
    def __load_resources(self, batch_size):
        if self.stream:
            return
            
        self.stream = cudart.cudaStreamCreate()[1]
        for model_name, obj in self.models.items():
            if model_name in self.engine:
                self.engine[model_name].allocate_buffers(
                    obj.get_shape_dict(batch_size, self.image_height, self.image_width), self.torch_device
                )

    def __encode_prompt(self, prompt, prompt_2, negative_prompt, negative_prompt_2):
        prompts = [prompt, negative_prompt]
        prompts_2 = [prompt_2, negative_prompt_2]
        
        # Tokenize with CLIP 1
        text_input_ids_1 = self.tokenizer(
            prompts, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).input_ids.type(torch.int32).to(self.torch_device)

        # Tokenize with CLIP 2
        text_input_ids_2 = self.tokenizer_2(
            prompts_2, padding="max_length", max_length=self.tokenizer_2.model_max_length,
            truncation=True, return_tensors="pt"
        ).input_ids.type(torch.int32).to(self.torch_device)

        # Run CLIP 1 Engine
        text_embeddings_1 = self.engine['clip1'].infer({'input_ids': text_input_ids_1}, self.stream)['text_embeddings']
        
        # Run CLIP 2 Engine
        clip2_out = self.engine['clip2'].infer({'input_ids': text_input_ids_2}, self.stream)
        text_embeddings_2 = clip2_out['text_embeddings']
        pooled_output = clip2_out['pooled_output']

        # Combine embeddings
        text_embeddings = torch.cat([text_embeddings_1, text_embeddings_2], dim=-1)
        
        # Split into conditional and unconditional
        prompt_embeds, negative_prompt_embeds = text_embeddings.chunk(2)
        add_text_embeds, add_negative_text_embeds = pooled_output.chunk(2)
        
        return prompt_embeds, negative_prompt_embeds, add_text_embeds, add_negative_text_embeds

    def __denoise_latent(
        self, latents, text_embeddings, add_text_embeds, add_time_ids,
        guidance_scale, timesteps, engine_name
    ):
        for timestep in self.progress_bar(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
            
            feed_dict = {
                'sample': latent_model_input,
                'timestep': timestep.float(),
                'encoder_hidden_states': text_embeddings,
                'add_text_embeds': add_text_embeds,
                'add_time_ids': add_time_ids
            }
            
            noise_pred = self.engine[engine_name].infer(feed_dict, self.stream)['latent']
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample
            
        return latents

    def __decode_latent(self, latents):
        # The VAE config scaling factor is important for SDXL
        latents = latents / self.vae.config.scaling_factor
        images = self.engine['vae'].infer({'latent': latents}, self.stream)['images']
        images = (images / 2 + 0.5).clamp(0, 1)
        return images.cpu().permute(0, 2, 3, 1).float().numpy()

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        denoising_end: Optional[float] = None, # For refiner
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
    ):
        if isinstance(prompt, str):
            prompt = [prompt]
        if prompt_2 is None:
            prompt_2 = prompt
        if isinstance(prompt_2, str):
            prompt_2 = [prompt_2]
        
        batch_size = len(prompt)
        
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        if negative_prompt_2 is None:
            negative_prompt_2 = negative_prompt
        if isinstance(negative_prompt_2, str):
            negative_prompt_2 = [negative_prompt_2]
        
        # Load CUDA resources
        self.__load_resources(batch_size)

        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            # 1. Encode prompts
            prompt_embeds, negative_prompt_embeds, add_text_embeds, add_negative_text_embeds = \
                self.__encode_prompt(prompt, prompt_2, negative_prompt, negative_prompt_2)
            
            # Combine for classifier-free guidance
            text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])
            add_text_embeds_combined = torch.cat([add_negative_text_embeds, add_text_embeds])

            # 2. Prepare timesteps and latents
            self.scheduler.set_timesteps(num_inference_steps, device=self.torch_device)
            timesteps = self.scheduler.timesteps

            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size, num_channels_latents, self.image_height, self.image_width,
                text_embeddings.dtype, self.torch_device, generator
            )

            # 3. Prepare `add_time_ids`
            original_size = (self.image_height, self.image_width)
            target_size = (self.image_height, self.image_width)
            add_time_ids = torch.tensor([original_size + (0,0) + target_size], dtype=text_embeddings.dtype)
            add_time_ids = torch.cat([add_time_ids] * batch_size)
            add_time_ids_combined = torch.cat([add_time_ids, add_time_ids]).to(self.torch_device)

            # 4. Denoising loop
            denoising_end_step = int(len(timesteps) * denoising_end) if denoising_end else len(timesteps)
            
            # Base model pass
            latents = self.__denoise_latent(
                latents, text_embeddings, add_text_embeds_combined, add_time_ids_combined,
                guidance_scale, timesteps[:denoising_end_step], 'unet'
            )

            # Refiner pass
            if self.has_refiner and denoising_end is not None and denoising_end < 1.0:
                logger.info("Running refiner...")
                # The refiner uses the same prompts but needs its own timesteps
                self.scheduler.set_timesteps(num_inference_steps, device=self.torch_device)
                
                # We need to add noise to the latents produced by the base model
                noise = randn_tensor(latents.shape, generator=generator, device=self.torch_device, dtype=latents.dtype)
                refiner_start_timestep = timesteps[denoising_end_step-1]
                latents = self.scheduler.add_noise(latents, noise, refiner_start_timestep.unsqueeze(0))
                
                latents = self.__denoise_latent(
                    latents, text_embeddings, add_text_embeds_combined, add_time_ids_combined,
                    guidance_scale, timesteps[denoising_end_step-1:], 'unet_refiner'
                )

            # 5. Decode latents
            images = self.__decode_latent(latents)

        # 6. Post-process and return
        if output_type == "pil":
            images = self.numpy_to_pil(images)
            
        return StableDiffusionXLPipelineOutput(images=images)