#!/usr/bin/env python3
"""
Compact DDPM visualization script that handles [-1,1] tensor values.
"""

import torch
import sys
import os
import argparse
from PIL import Image
import numpy as np

# Add the project root to the path
sys.path.append('.')

from DDPM.denoisers.denoisermlp.denoisermlp import DenoiserMLP
from DDPM.denoisers.denoiserunet.denoiserunet import DenoiserUNet


class Visualizer:
    """Compact visualizer that follows repo conventions."""
    
    def __init__(self, args):
        # Use args attributes that match the existing args system
        self.checkpoint = getattr(args, 'path_to_weights', 'checkpoints/best.pth')
        self.output_dir = args.output_dir
        self.n_samples = getattr(args, 'n_samples', 2)
        self.fps = args.fps
        self.save_every = getattr(args, 'save_every', 10)
        
        # Load model and create DDMP
        self.denoiser, self.img_shape, self.device = self._load_model()
        self.ddpm = self._create_ddpm()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_model(self):
        """Load model from checkpoint."""
        print(f"Loading model from: {self.checkpoint}")
        
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
        
        checkpoint = torch.load(self.checkpoint, map_location='cpu')
        img_shape = checkpoint['img_shape']
        
        # Extract alpha schedule parameters from checkpoint (with fallbacks for older checkpoints)
        self.t_max = checkpoint.get('t_max', 1000)
        self.alpha_min = checkpoint.get('alpha_min', 0.95)
        self.alpha_max = checkpoint.get('alpha_max', 1.0)
        self.alpha_interp = checkpoint.get('alpha_interp', 'linear')
        
                # Infer model architecture from checkpoint
        state_dict = checkpoint['model_state_dict']
        
        # Infer hidden sizes from state dict
        first_hidden = state_dict['in_proj.weight'].shape[0]
        hidden_sizes = [first_hidden]
        
        i = 0
        while f'blocks.{i}.weight' in state_dict:
            hidden_size = state_dict[f'blocks.{i}.weight'].shape[0]
            hidden_sizes.append(hidden_size)
            i += 1
        
        # Infer time dimensions from state dict
        time_hidden = state_dict['time_embed.mlp.0.weight'].shape[0]
        time_base_dim = state_dict['time_embed.mlp.0.weight'].shape[1]
        time_output_dim = state_dict['time_embed.mlp.2.weight'].shape[0]
        
        denoiser = DenoiserMLP(
            img_shape=img_shape,
            hidden_sizes=hidden_sizes,
            time_base_dim=time_base_dim,
            time_output_dim=time_output_dim,
            time_hidden=time_hidden,
            activation='silu',
            norm='layer',
            dropout=0.0,
            init_scheme='xavier'
        )
        
        denoiser.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        denoiser.to(device).eval()
        
        print(f"Model loaded on device: {device}")
        return denoiser, img_shape, device

    def _create_ddpm(self):
        """Create DDPM-like object using checkpoint parameters."""
        # Now using parameters from checkpoint instead of hardcoded values
        
        # Generate alphas using the same logic as in args.py
        if self.alpha_interp == "linear":
            alphas = torch.linspace(self.alpha_max, self.alpha_min, self.t_max)
        else:
            raise ValueError(f"Argument error : interpolation method {self.alpha_interp} not implemented")
        
        alphas = alphas.to(self.device)
        
        # Create simple DDPM-like object
        ddpm = type('DDPM', (), {
            'denoiser': self.denoiser,
            'image_shape': self.img_shape,
            'alphas': alphas,
            'alphas_bar': torch.cumprod(alphas, dim=0),
            't_max': self.t_max
        })()
        
        return ddpm

    def _tensor_to_pil(self, tensor):
        """Convert [-1,1] tensor to PIL Image."""
        # Convert tensor to numpy
        if hasattr(tensor, 'detach'):
            np_array = tensor.detach().cpu().numpy()
        else:
            np_array = tensor.cpu().numpy() if hasattr(tensor, 'cpu') else tensor
        
        # Handle tensor shape
        if np_array.ndim == 3:
            if np_array.shape[0] == 1:  # Grayscale [1, H, W] -> [H, W]
                np_array = np_array.squeeze(0)
            elif np_array.shape[0] == 3:  # RGB [3, H, W] -> [H, W, 3]
                np_array = np_array.transpose(1, 2, 0)
        
        # Convert [-1,1] to [0,1] then to [0,255]
        np_array = (np_array + 1) / 2  # [-1,1] -> [0,1]
        np_array = np.clip(np_array, 0, 1)  # Ensure bounds
        
        # Handle NaN values and convert safely to uint8
        np_array = np.nan_to_num(np_array, nan=0.0, posinf=1.0, neginf=0.0)
        np_array = (np_array * 255).clip(0, 255).astype(np.uint8)
        
        # Create PIL image
        if np_array.ndim == 2:  # Grayscale
            return Image.fromarray(np_array)
        elif np_array.ndim == 3 and np_array.shape[2] == 3:  # RGB
            return Image.fromarray(np_array)
        else:
            raise ValueError(f"Cannot convert array shape {np_array.shape} to PIL Image")

    def save_gif(self, frames, filename):
        """Save frames as GIF."""
        pil_frames = [self._tensor_to_pil(frame) for frame in frames]
        duration = int(1000 / self.fps)
        
        path = os.path.join(self.output_dir, filename)
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved: {path}")

    def save_png(self, initial, final, filename):
        """Save side-by-side comparison PNG."""
        initial_pil = self._tensor_to_pil(initial)
        final_pil = self._tensor_to_pil(final)
        
        # Create side-by-side comparison
        width, height = initial_pil.size
        comparison = Image.new(initial_pil.mode, (width * 2, height))
        comparison.paste(initial_pil, (0, 0))
        comparison.paste(final_pil, (width, 0))
        
        path = os.path.join(self.output_dir, filename)
        comparison.save(path)
        print(f"PNG saved: {path}")

    def visualize(self):
        """Generate and visualize denoising process."""
        print(f"Generating {self.n_samples} samples...")
        
        # Initialize noise
        x_t = torch.randn(
            self.n_samples,
            self.img_shape[0],
            self.img_shape[1],
            self.img_shape[2],
            device=self.device,
            dtype=self.ddpm.alphas.dtype
        )
        
        frames = [x_t.clone()]
        
        # Denoising process
        beta_t = 1.0 - self.ddpm.alphas
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.ddpm.alphas_bar)
        inv_sqrt_alpha = torch.rsqrt(self.ddpm.alphas)
        
        for t in reversed(range(self.ddpm.t_max)):
            t_batch = torch.full((self.n_samples,), t, device=self.device, dtype=torch.long)
            
            eps_theta = self.ddpm.denoiser(x_t, t_batch)
            c1 = inv_sqrt_alpha[t]
            c2 = beta_t[t] / (sqrt_one_minus_alpha_bar[t] + 1e-12)
            mean = c1 * (x_t - c2 * eps_theta)
            
            if t > 0:
                sigma = torch.sqrt(beta_t[t])
                z = torch.randn_like(x_t)
                x_t = mean + sigma * z
            else:
                x_t = mean
            
            # Save frame at intervals
            if t % self.save_every == 0 or t == 0:
                frames.append(x_t.clone())
        
        # Save visualizations for each sample
        for i in range(self.n_samples):
            sample_frames = [f[i] for f in frames]
            self.save_gif(sample_frames, f'denoising_sample_{i}.gif')
            self.save_png(sample_frames[0], sample_frames[-1], f'comparison_sample_{i}.png')
        
        print(f"Visualization complete! Files saved to: {self.output_dir}")


def main():
    from utils.parsing.args import args
    visualizer = Visualizer(args)
    visualizer.visualize()


if __name__ == '__main__':
    main() 