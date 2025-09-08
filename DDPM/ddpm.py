import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from utils.parsing.args import args
from utils.viz.visualizer import Visualizer


class DDPM:

    def __init__(self, args=args):
        # data
        self.image_shape = args.image_shape

        # model
        self.denoiser: nn.Module = args.model

        # loss
        self.alphas: torch.Tensor = args.alphas
        self.alphas_bar: torch.Tensor = args.alphas_bar
        self.num_trials: int = args.num_trials
        self.t_max: int = args.t_max

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
    def generate(
        self, 
        n_samples: int, 
        visualise: bool = False,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        DDPM reverse process.
        Args:
          n_samples: number of images to generate
          device: device for computation (default: alphas_bar.device)
        Returns:
          x0 samples: [n_samples, C, H, W]
        """
        device = device or self.alphas_bar.device
        dtype = self.alphas_bar.dtype

        beta_t: torch.Tensor = 1.0 - self.alphas
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alphas_bar)
        inv_sqrt_alpha = torch.rsqrt(self.alphas)

        x_t = torch.randn(
            n_samples,
            self.image_shape[0],
            self.image_shape[1],
            self.image_shape[2],
            device=device,
            dtype=dtype,
        )

        if visualise:
            visualiser = Visualizer(args, self)
            frames = [x_t]

        for t in reversed(range(self.t_max)):                
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

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
            
            if visualise:
                frames.append(x_t)
        
        if visualise:
            visualiser.save_gif(frames, args.output_dir)

        return x_t