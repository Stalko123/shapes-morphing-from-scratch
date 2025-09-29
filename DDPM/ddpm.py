import torch, math
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DDPM:

    def __init__(self, args):

        # model
        self.denoiser_name: str = args.model_name
        self.denoiser: nn.Module = args.model

        if hasattr(args, "image_shape"):
            self.image_shape = args.image_shape
        if hasattr(args, "t_max"):
            self.t_max = args.t_max
        if hasattr(args, "num_trials"):
            self.num_trials = args.num_trials
        
        # Beta schedule parameters
        self.beta_schedule = getattr(args, "beta_schedule", "cosine")
        self.beta_start = getattr(args, "beta_start", 1e-4)
        self.beta_end = getattr(args, "beta_end", 0.02)
        
        # Initialize alpha and alpha_bar based on schedule
        if self.beta_schedule == "linear":
            self.alphas, self.alphas_bar = self.linear_beta_schedule()
        else:  # cosine (default)
            self.alphas, self.alphas_bar = self.cosine_alpha_bar()

    
    def linear_beta_schedule(self):
        """
        Linear beta schedule from beta_start to beta_end over T timesteps.
        
        Returns:
            alphas: [T] tensor where alpha_t = 1 - beta_t
            alphas_bar: [T+1] tensor where alpha_bar_t = prod(alpha_1, ..., alpha_t)
        """
        T = self.t_max
        # Linear interpolation from beta_start to beta_end
        betas = torch.linspace(self.beta_start, self.beta_end, T, dtype=torch.float64)
        alphas = 1.0 - betas
        
        # Compute cumulative product for alpha_bar
        # alpha_bar[0] = 1.0 (at t=0, no noise)
        # alpha_bar[t] = prod(alpha_1, ..., alpha_t) for t >= 1
        alpha_bar = torch.zeros(T + 1, dtype=torch.float64)
        alpha_bar[0] = 1.0
        alpha_bar[1:] = torch.cumprod(alphas, dim=0)
        
        return alphas.float(), alpha_bar.float()

    def cosine_alpha_bar(self, s: float = 0.08):
        """
        Nichol & Dhariwal alphas : enable linear decay of signal to noise ratio
        """
        T = self.t_max
        t = torch.linspace(0, T, T+1, dtype=torch.float64) / T
        f = torch.cos(( (t + s) / (1 + s) ) * math.pi / 2) ** 2
        alpha_bar = (f / f[0]).clamp(min=1e-12, max=1.0)
        alphas = (alpha_bar[1:] / alpha_bar[:-1]).clamp(min=1e-6, max=1-1e-6)
        return alphas.float(), alpha_bar.float()

    def blurData(
        self, 
        x: torch.Tensor,
        t_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          x_noisy: [B*num_trials, C, H, W]
          white_noise: [B*num_trials, C, H, W]
        """
        device = x.device
        t_tensor = t_tensor.to(device=device, dtype=torch.long)

        alpha_bar_t = self.alphas_bar.to(device).index_select(0, t_tensor)
        alpha_bar_t = alpha_bar_t.view(-1, *([1] * (x.ndim - 1)))

        white_noise = torch.randn_like(x)
        x_noisy = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * white_noise
        return x_noisy, white_noise

    def computeLoss(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns: scalar loss tensor
        """
        B = x.size(0)
        t = torch.randint(0, self.t_max, (B * self.num_trials,), device=x.device)
        x_rep = x.repeat_interleave(self.num_trials, dim=0)
        x_blurred, white_noise = self.blurData(x_rep, t)
        preds = self.denoiser(x_blurred, t)
        loss = F.mse_loss(preds, white_noise)
        return loss

    @torch.no_grad()
    def generate_one_sample(
        self, 
        return_intermediates: bool = True,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        DDPM reverse process.
        """
        device = device or self.alphas_bar.device
        dtype = self.alphas_bar.dtype

        beta_t: torch.Tensor = 1.0 - self.alphas
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alphas_bar)
        inv_sqrt_alpha = torch.rsqrt(self.alphas)

        x_t = torch.randn(
            1,
            self.image_shape[0],
            self.image_shape[1],
            self.image_shape[2],
            device=device,
            dtype=dtype,
        )

        if return_intermediates:
            frames = [x_t[0].detach().cpu().clone()]

        for t in reversed(range(self.t_max)):                
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)

            eps_theta = self.denoiser(x_t, t_batch)
            c1 = inv_sqrt_alpha[t]
            c2 = beta_t[t] / (sqrt_one_minus_alpha_bar[t] + 1e-12)
            mean = c1 * (x_t - c2 * eps_theta)

            if t > 0:
                sigma = torch.sqrt(beta_t[t])
                z = torch.randn_like(x_t)
                x_t = mean + sigma * z
                
            else:
                x_t = mean
            
            if return_intermediates:
                frames.append(x_t[0].detach().cpu().clone())
        
        if return_intermediates:
            return frames

        return x_t