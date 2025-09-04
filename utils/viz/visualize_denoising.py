#!/usr/bin/env python
"""
Script to load a trained DDPM model and visualize the denoising process.
"""

import torch
import sys
import os
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DDPM.ddpm import DDPM
from DDPM.denoisers.denoisermlp.denoisermlp import DenoiserMLP


def load_model(checkpoint_path):
    """Load the trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract information from checkpoint
    img_shape = checkpoint['img_shape']
    saved_args = checkpoint['args']
    
    print(f"Model trained on image shape: {img_shape}")
    print(f"Model arguments: {vars(saved_args)}")
    
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


def main():
    """Main function to load model and generate samples."""
    parser = argparse.ArgumentParser(description='Visualize DDPM denoising process')
    parser.add_argument('--checkpoint', 
                       default='/users/eleves-b/2023/oussama.akar/Projects/shapes-morphing-from-scratch/checkpoints/final_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--n_samples', type=int, default=2, 
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', default='./denoising_visualization',
                       help='Output directory for GIFs')
    
    args = parser.parse_args()
    
    print("DDPM Denoising Visualization")
    print("=" * 40)
    
    # Load the model
    denoiser, img_shape, saved_args, device = load_model(args.checkpoint)
    
    # Create DDPM instance
    ddpm_args = create_ddpm_args(denoiser, img_shape, saved_args, device)
    ddpm = DDPM(ddpm_args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nGenerating {args.n_samples} samples with denoising visualization...")
    
    # Generate samples with visualization
    with torch.no_grad():
        samples = ddpm.generate(
            n_samples=args.n_samples,
            visualise=True,
            device=device
        )
    
    print(f"Generation completed!")
    print(f"Generated samples shape: {samples.shape}")
    print(f"Visualization saved to: {args.output_dir}")
    
    # Print some statistics about the generated samples
    print(f"\nSample statistics:")
    print(f"  Min value: {samples.min().item():.4f}")
    print(f"  Max value: {samples.max().item():.4f}")
    print(f"  Mean value: {samples.mean().item():.4f}")
    print(f"  Std value: {samples.std().item():.4f}")


if __name__ == '__main__':
    main()