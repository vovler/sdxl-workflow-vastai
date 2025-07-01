import gc
import torch

def _prepare_latents(scheduler, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
    print("\n--- In _prepare_latents ---")
    print(f"batch_size: {batch_size}")
    print(f"num_channels_latents: {num_channels_latents}")
    print(f"height: {height}")
    print(f"width: {width}")
    print(f"dtype: {dtype}")
    print(f"device: {device}")
    print(f"generator: {generator}")
    print(f"latents (input): {latents}")
    
    shape = (
        batch_size,
        num_channels_latents,
        int(height) // 8,
        int(width) // 8,
    )
    print(f"Calculated shape: {shape}")

    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        print("latents is None, creating new latents with torch.randn")
        latents = torch.randn(*shape, generator=generator, device=device, dtype=dtype)
    else:
        print("latents is not None, moving to device")
        latents = latents.to(device)

    print(f"Latents before scaling: shape={latents.shape}, dtype={latents.dtype}, device={latents.device}")
    print(f"Latents before scaling | Mean: {latents.mean():.6f} | Std: {latents.std():.6f} | Sum: {latents.sum():.6f}")

    initial_sigma = scheduler.sigmas[0] 
    
    print(f"Using initial_sigma from scheduler.sigmas[0]: {initial_sigma.item()}")
    
    latents = latents * initial_sigma.to(device)
    
    print(f"Latents after scaling: shape={latents.shape}, dtype={latents.dtype}, device={latents.device}")
    print(f"Latents after scaling | Mean: {latents.mean():.6f} | Std: {latents.std():.6f} | Sum: {latents.sum():.6f}")
    print("--- Exiting _prepare_latents ---")
    
    return latents

def _get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids

def _clear_memory():
    print("--- In _clear_memory ---")
    gc.collect()
