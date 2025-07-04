# debug_native.py
import torch
import time
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, AutoencoderTiny, UNet2DConditionModel
from huggingface_hub import snapshot_download
import os
from compel import Compel, ReturnedEmbeddingsType

# Add necessary imports for modelopt quantization
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto


# Ensure you use the exact same settings as your pipeline
prompt = "masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, night, 1girl, aqua_(konosuba), anal sex, ahegao, heart shaped eyes, (tree:0.2)"
height = 1024
width = 1024
num_inference_steps = 8
seed = 42
guidance_scale = 1.0 # This is the key to matching your CFG-less setup

device = "cuda:0"
base_model_id = "socks22/sdxl-wai-nsfw-illustriousv14"


# Load native pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_id, # Your base model
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True],
    device="cuda"
)

# Set the exact same scheduler
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config
)

print("\n" + "="*20 + " NATIVE SCHEDULER CONFIG " + "="*20)
for key, value in pipe.scheduler.config.items():
    print(f"{key}: {value}")
print("="*67 + "\n")


# We will monkey-patch the unet forward pass to capture the inputs
original_unet_forward = pipe.unet.forward

def new_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    added_cond_kwargs = kwargs.get("added_cond_kwargs", {})
    text_embeds = added_cond_kwargs.get("text_embeds")
    time_ids = added_cond_kwargs.get("time_ids")

    print("\n" + "="*20 + f" NATIVE DIFFUSERS - CAPTURING TENSORS (Timestep: {timestep.item()}) " + "="*20)
    # 1. Latent Model Input
    print(f"LATENT_MODEL_INPUT | Shape: {sample.shape} | Dtype: {sample.dtype}")
    print(f"LATENT_MODEL_INPUT | Mean: {sample.mean():.6f} | Std: {sample.std():.6f} | Sum: {sample.sum():.6f}")

    # 2. Prompt Embeds
    cond_embeds = encoder_hidden_states
    print(f"PROMPT_EMBEDS (cond) | Shape: {cond_embeds.shape} | Dtype: {cond_embeds.dtype}")
    print(f"PROMPT_EMBEDS (cond) | Mean: {cond_embeds.mean():.6f} | Std: {cond_embeds.std():.6f} | Sum: {cond_embeds.sum():.6f}")

    # 3. Pooled Prompt Embeds
    cond_pooled = text_embeds
    print(f"POOLED_EMBEDS (cond) | Shape: {cond_pooled.shape} | Dtype: {cond_pooled.dtype}")
    print(f"POOLED_EMBEDS (cond) | Mean: {cond_pooled.mean():.6f} | Std: {cond_pooled.std():.6f} | Sum: {cond_pooled.sum():.6f}")
    
    # 4. Time IDs
    cond_time = time_ids
    print(f"TIME_IDS (cond) | Shape: {cond_time.shape} | Dtype: {cond_time.dtype}")
    print(f"TIME_IDS (cond) | Values: {cond_time.flatten().tolist()}")

    # 5. The Timestep
    print(f"TIMESTEP | Value: {timestep.item()} | Dtype of tensor: {timestep.dtype}")
    print("="*78 + "\n")
    
    # Call the original UNet
    noise_pred_output = original_unet_forward(sample, timestep, encoder_hidden_states, **kwargs)
    
    print("\n" + "="*20 + f" NATIVE DIFFUSERS - CAPTURING UNET OUTPUT " + "="*20)
    # In CFG-less mode (guidance_scale=1.0), the UNet output is the final noise prediction
    # The UNet can return a tuple, so we take the first element.
    # Diffusers `UNet2DConditionOutput`, `Quantizedunetoutput` from modelopt... let's check for `.sample`
    if hasattr(noise_pred_output, "sample"):
        final_noise_pred = noise_pred_output.sample
    elif isinstance(noise_pred_output, tuple):
        final_noise_pred = noise_pred_output[0]
    else:
        final_noise_pred = noise_pred_output
    
    print(f"NOISE_PRED (final) | Shape: {final_noise_pred.shape} | Dtype: {final_noise_pred.dtype}")
    print(f"NOISE_PRED (final) | Mean: {final_noise_pred.mean():.6f} | Std: {final_noise_pred.std():.6f} | Sum: {final_noise_pred.sum():.6f}")
    print("="*68 + "\n")
    
    return noise_pred_output

# --- Run 1: Native (un-quantized) ---
print("\n" + "="*30 + " RUNNING NATIVE (FP16) " + "="*30 + "\n")
# Patch the UNet
pipe.unet.forward = new_unet_forward

# Run the pipeline.
prompt_embeds, pooled_prompt_embeds = compel(prompt)
generator=torch.Generator("cpu").manual_seed(0x7A35D)
start_time = time.time()
images_native = pipe(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    height=height,
    width=width,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale, # Set to 1.0 to mimic your pipeline
    generator=generator,
).images
end_time = time.time()
print(f"Native pipeline execution time: {end_time - start_time:.2f} seconds")

# Restore original unet forward
pipe.unet.forward = original_unet_forward

images_native[0].save("debug_native_output_native.png")
print("Saved native image to debug_native_output_native.png")


# --- Run 2: With modelopt INT8 Quantized UNET ---
print("\n" + "="*20 + " RUNNING WITH MODELOPT INT8 UNET " + "="*20 + "\n")

# Reload the pipeline to start fresh
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config
)

compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True],
    device="cuda"
)


# Path to the quantized UNet
model_path = snapshot_download(base_model_id)
int8_unet_path = os.path.join(model_path, "unet_int8.safetensors")
print(f"Loading INT8 UNet from: {int8_unet_path}")

# Load the quantized weights.
# mto.restore modifies the UNet in-place, swapping layers with their
# quantized counterparts and loading the INT8 weights.
mto.restore(pipe.unet, int8_unet_path)
print("INT8 UNet loaded and restored successfully.")


# Patch the UNet again for the second run
pipe.unet.forward = new_unet_forward

# Run the pipeline, resetting the generator to get the same initial noise
prompt_embeds, pooled_prompt_embeds = compel(prompt)
generator=torch.Generator("cpu").manual_seed(0x7A35D)
start_time = time.time()
images_quantized = pipe(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    height=height,
    width=width,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale, # Set to 1.0 to mimic your pipeline
    generator=generator,
).images
end_time = time.time()
print(f"INT8 Quantized pipeline execution time: {end_time - start_time:.2f} seconds")


# Restore original unet forward
pipe.unet.forward = original_unet_forward

images_quantized[0].save("debug_native_output_int8.png")
print("Saved INT8 quantized image to debug_native_output_int8.png")


print("\nNative debug script finished.") 