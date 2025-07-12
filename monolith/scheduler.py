import torch
import torch.nn as nn

class EulerAncestralDiscreteScheduler(nn.Module):
    def __init__(self, num_inference_steps: int, dtype, num_train_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012, beta_schedule: str = "scaled_linear", timestep_spacing: str = "linspace"):
        super().__init__()
        
        self.num_inference_steps = num_inference_steps
        
        # Beta schedule (using torch)
        if beta_schedule == "scaled_linear":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented.")
            
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Timesteps (using torch)
        if timestep_spacing == "linspace":
            timesteps = torch.linspace(0, num_train_timesteps - 1, num_inference_steps, dtype=torch.float32).flip(0)
        elif timestep_spacing == "trailing":
            step_ratio = num_train_timesteps / self.num_inference_steps
            timesteps = (torch.arange(num_train_timesteps, 0, -step_ratio, dtype=torch.float32) - 1).round()
        else:
            raise NotImplementedError(f"{timestep_spacing} is not supported.")
        
        # Sigmas (using torch)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        
        # Interpolation of sigmas - PyTorch equivalent of np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        from_timesteps = torch.arange(num_train_timesteps, device=sigmas.device)

        # Find the indices for interpolation
        indices = torch.searchsorted(from_timesteps, timesteps, right=True)
        indices = torch.clamp(indices, 1, num_train_timesteps - 1)

        # Get the adjacent timesteps and sigmas for linear interpolation
        t_low = from_timesteps[indices - 1]
        t_high = from_timesteps[indices]
        s_low = sigmas[indices - 1]
        s_high = sigmas[indices]

        # Linearly interpolate
        w_denom = t_high.float() - t_low.float()
        w_denom[w_denom == 0] = 1.0  # Avoid division by zero
        w = (timesteps.float() - t_low.float()) / w_denom
        interpolated_sigmas = s_low.float() + w * (s_high.float() - s_low.float())
        
        # Concatenate with zero sigma for the last step
        sigmas_buffer = torch.cat([interpolated_sigmas, torch.tensor([0.0])])
        
        if timestep_spacing in ["linspace", "trailing"]:
             init_noise_sigma_val = sigmas_buffer.max()
        else: # "leading"
             init_noise_sigma_val = (sigmas_buffer.max() ** 2 + 1) ** 0.5
        
        self.register_buffer('sigmas', sigmas_buffer.to(dtype=dtype))
        self.register_buffer('timesteps', timesteps)
        self.register_buffer('init_noise_sigma', init_noise_sigma_val.clone().detach().to(dtype=dtype))
        

    @torch.no_grad()
    def step(
        self,
        noise_pred: torch.Tensor,
        latents: torch.Tensor,
        i: int,
        all_noises: torch.Tensor
    ) -> torch.Tensor:
        sigma = self.sigmas[i]

        pred_original_sample = latents - sigma * noise_pred
        derivative = (latents - pred_original_sample) / sigma
        
        sigma_from = self.sigmas[i]
        sigma_to = self.sigmas[i + 1]
        sigma_up = torch.sqrt((sigma_to**2 * (sigma_from**2 - sigma_to**2)) / sigma_from**2)
        sigma_down = torch.sqrt(sigma_to**2 - sigma_up**2)
        
        dt = sigma_down - sigma
        denoised_latents = latents + derivative * dt
        
        noise = all_noises[i]
        latents = denoised_latents + noise * sigma_up
            
        return latents