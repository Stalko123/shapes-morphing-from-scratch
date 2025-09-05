#!/usr/bin/env python
"""
Script to load a trained DDPM model and visualize the denoising process.
"""

import torch
import sys
import os
from configargparse import ArgParser

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DDPM.ddpm import DDPM
from DDPM.denoisers.denoisermlp.denoisermlp import DenoiserMLP


def load_model(checkpoint_path):
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    img_shape = checkpoint['img_shape']
    saved_args = checkpoint['args']

    print(f"Model trained on image shape: {img_shape}")
    try:
        print(f"Model arguments: {vars(saved_args)}")
    except Exception:
        print("Model arguments loaded (non-dict-like object).")

    denoiser = DenoiserMLP(
        img_shape=img_shape,
        hidden_sizes=getattr(saved_args, 'hidden_sizes', [1024, 1024]),
        time_dim=getattr(saved_args, 'time_dim', 128),
        activation=getattr(saved_args, 'activation', 'silu'),
        norm=getattr(saved_args, 'norm', 'layer'),
        dropout=getattr(saved_args, 'dropout', 0.0)
    )
    denoiser.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser.to(device).eval()
    print(f"Model loaded on device: {device}")
    return denoiser, img_shape, saved_args, device


def create_ddpm_args(denoiser, img_shape, saved_args, device):
    dataset = type('Dataset', (), {'C': img_shape[0], 'H': img_shape[1], 'W': img_shape[2]})()
    if getattr(saved_args, 'alphas', 'linear') == "linear":
        beta_start, beta_end = 1e-4, 0.02
        betas = torch.linspace(beta_start, beta_end, saved_args.t_max)
        alphas = 1.0 - betas
    else:
        beta_start, beta_end = 1e-4, 0.02
        betas = torch.linspace(beta_start, beta_end, saved_args.t_max)
        alphas = 1.0 - betas
    alphas = alphas.to(device)
    ddpm_args = type('Args', (), {
        'dataset': dataset,
        'denoiser': denoiser,
        'alphas': alphas,
        'num_trials': getattr(saved_args, 'num_trials', 100),
        't_max': getattr(saved_args, 't_max', 1000)
    })()
    return ddpm_args


def main():
    parser = ArgParser(
        description='Visualize DDPM denoising process',
        default_config_files=['configs/visualize.yaml'],  # optional default
        env_prefix="DDPM_"  # allow env vars like DDPM_N_SAMPLES=4
    )
    # Let users pass a config file explicitly: -c path/to/file.yaml
    parser.add('-c', '--config', is_config_file=True, help='Path to a config file')

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
    print("Effective arguments:", vars(args))

    denoiser, img_shape, saved_args, device = load_model(args.checkpoint)
    ddpm_args = create_ddpm_args(denoiser, img_shape, saved_args, device)
    ddpm = DDPM(ddpm_args)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nGenerating {args.n_samples} samples with denoising visualization...")

    with torch.no_grad():
        samples = ddpm.generate(n_samples=args.n_samples, visualise=True, device=device)

    print("Generation completed!")
    print(f"Generated samples shape: {samples.shape}")
    print(f"Visualization saved to: {args.output_dir}")
    print("\nSample statistics:")
    print(f"  Min value: {samples.min().item():.4f}")
    print(f"  Max value: {samples.max().item():.4f}")
    print(f"  Mean value: {samples.mean().item():.4f}")
    print(f"  Std value: {samples.std().item():.4f}")


if __name__ == '__main__':
    main()