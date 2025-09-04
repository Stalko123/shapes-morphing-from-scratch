#!/usr/bin/env python
"""
Script to load a trained DDPM model and visualize the denoising process
using real FashionMNIST samples from the existing dataloader.
"""

import torch
import sys
import os
import argparse
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DDPM.ddpm import DDPM
from DDPM.denoisers.denoisermlp import DenoiserMLP
from dataloader import Dataloader


def load_model(checkpoint_path):
    """Load the trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract information from checkpoint
    img_shape = checkpoint['img_shape']
    saved_args = checkpoint['args']
    
    print(f"Model trained on image shape: {img_shape}")
    
    # Create the denoiser model
    denoiser = DenoiserMLP(
        img_shape=img_shape,
        hidden_sizes=getattr(saved_args, 'hidden_sizes', [1024, 1024]),
        time_dim=getattr(saved_args, 'time_dim', 128),
        activation=getattr(saved_args, 'activation', 'silu'),
        norm=getattr(saved_args, 'norm', 'layer'),
        dropout=getattr(saved_args, 'dropout', 0.0)
    )
    
    # Load the trained weights
    denoiser.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser.to(device)
    denoiser.eval()
    
    print(f"Model loaded on device: {device}")
    
    return denoiser, img_shape, saved_args, device


def load_fashionmnist_samples(n_samples=2):
    """Load FashionMNIST samples using the existing dataloader."""
    print(f"Loading {n_samples} FashionMNIST samples using existing dataloader...")
    
    # Create args for dataloader
    args = type('Args', (), {
        'name_dataset': 'FashionMNIST',
        'batch_size': n_samples,
        'n_workers': 1,
        'n_epochs': 1  # Just to make it work
    })()
    
    # Use existing dataloader
    dataloader = Dataloader(args)
    
    # Get a batch of samples
    for batch_data, batch_labels in dataloader.dataloader:
        samples = batch_data
        labels = batch_labels
        break  # Just take the first batch
    
    print(f"Loaded samples shape: {samples.shape}")
    print(f"Sample labels: {labels.tolist()}")
    
    return samples, labels


def create_ddpm_args(denoiser, img_shape, saved_args, device):
    """Create DDPM arguments for generation."""
    # Create dataset attribute object
    dataset = type('Dataset', (), {
        'C': img_shape[0],
        'H': img_shape[1], 
        'W': img_shape[2]
    })()
    
    # Create alphas schedule
    if getattr(saved_args, 'alphas', 'linear') == "linear":
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, saved_args.t_max)
        alphas = 1.0 - betas
    else:
        # Default to linear if not specified
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, saved_args.t_max)
        alphas = 1.0 - betas
    
    alphas = alphas.to(device)
    
    # Create DDPM args
    ddpm_args = type('Args', (), {
        'dataset': dataset,
        'denoiser': denoiser,
        'alphas': alphas,
        'num_trials': getattr(saved_args, 'num_trials', 100),
        't_max': getattr(saved_args, 't_max', 1000)
    })()
    
    return ddpm_args


def visualize_denoising_process(ddpm, samples, device, output_dir):
    """Visualize the denoising process for given samples."""
    print("Starting denoising visualization...")
    
    # Move samples to device
    samples = samples.to(device)
    
    # Add noise to the samples (forward process)
    t_max = ddpm.t_max
    t_noise = torch.randint(0, t_max, (samples.size(0),), device=device)
    
    # Apply forward diffusion to add noise
    x_noisy, white_noise = ddpm.blurData(samples, t_noise)
    
    print(f"Added noise at timestep: {t_noise.cpu().numpy()}")
    print(f"Noisy samples shape: {x_noisy.shape}")
    
    # Now denoise using the model (reverse process)
    with torch.no_grad():
        # Start from the noisy samples
        x_t = x_noisy.clone()
        frames = [x_t.cpu()]
        
        # Denoise step by step
        for t in reversed(range(t_noise.max().item() + 1)):
            t_batch = torch.full((samples.size(0),), t, device=device, dtype=torch.long)
            
            # Predict noise
            eps_theta = ddpm.denoiser(x_t, t_batch)
            
            # Compute denoising step
            beta_t = 1.0 - ddpm.alphas
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - ddpm.alphas_bar)
            inv_sqrt_alpha = torch.rsqrt(ddpm.alphas)
            
            c1 = inv_sqrt_alpha[t]
            c2 = beta_t[t] / (sqrt_one_minus_alpha_bar[t] + 1e-12)
            mean = c1 * (x_t - c2 * eps_theta)
            
            if t > 0:
                sigma = torch.sqrt(beta_t[t])
                z = torch.randn_like(x_t)
                x_t = mean + sigma * z
            else:
                x_t = mean
            
            # Store frame for visualization
            frames.append(x_t.cpu())
            
            if t % 50 == 0:  # Print progress every 50 steps
                print(f"Denoising step {t}/{t_noise.max().item()}")
    
    print(f"Denoising completed! Generated {len(frames)} frames")
    
    # Save the visualization frames
    frames_tensor = torch.stack(frames)
    print(f"Frames tensor shape: {frames_tensor.shape}")
    
    # Save as individual images or create a GIF
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the final denoised samples
    final_samples = frames[-1]
    torch.save(final_samples, os.path.join(output_dir, 'denoised_samples.pth'))
    
    print(f"Denoised samples saved to: {os.path.join(output_dir, 'denoised_samples.pth')}")
    
    return frames


def main():
    """Main function to load model and visualize denoising."""
    parser = argparse.ArgumentParser(description='Visualize DDPM denoising process with FashionMNIST')
    parser.add_argument('--checkpoint', 
                       default='/users/eleves-b/2023/oussama.akar/Projects/shapes-morphing-from-scratch/checkpoints/final_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--n_samples', type=int, default=2, 
                       help='Number of FashionMNIST samples to use')
    parser.add_argument('--output_dir', default='./denoising_visualization',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("DDPM Denoising Visualization with FashionMNIST")
    print("=" * 50)
    
    # Load the model
    denoiser, img_shape, saved_args, device = load_model(args.checkpoint)
    
    # Create DDPM instance
    ddpm_args = create_ddpm_args(denoiser, img_shape, saved_args, device)
    ddpm = DDPM(ddpm_args)
    
    # Load FashionMNIST samples using existing dataloader
    samples, labels = load_fashionmnist_samples(args.n_samples)
    
    # Visualize denoising process
    frames = visualize_denoising_process(ddpm, samples, device, args.output_dir)
    
    print(f"\nVisualization completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Original samples labels: {labels.tolist()}")


if __name__ == '__main__':
    main()
