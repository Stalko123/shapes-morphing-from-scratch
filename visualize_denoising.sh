#!/bin/bash

# Bash script to visualize DDPM denoising process
# This script passes arguments to the Python script

source ~/miniconda3/bin/activate
conda activate venv

python -c "
import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the path
sys.path.append('.')

from DDPM.ddpm import DDPM
from DDPM.denoisers.denoisermlp import DenoiserMLP

def load_model(checkpoint_path):
    print(f'Loading model from: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    img_shape = checkpoint['img_shape']
    saved_args = checkpoint['args']
    print(f'Model trained on image shape: {img_shape}')
    
    denoiser = DenoiserMLP(
        img_shape=img_shape,
        hidden_sizes=getattr(saved_args, 'hidden_sizes', [1024, 1024]),
        time_dim=getattr(saved_args, 'time_dim', 128),
        activation=getattr(saved_args, 'activation', 'silu'),
        norm=getattr(saved_args, 'norm', 'layer'),
        dropout=getattr(saved_args, 'dropout', 0.0)
    )
    
    denoiser.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    denoiser.to(device)
    denoiser.eval()
    print(f'Model loaded on device: {device}')
    
    return denoiser, img_shape, saved_args, device

def create_ddpm_args(denoiser, img_shape, saved_args, device):
    dataset = type('Dataset', (), {
        'C': img_shape[0],
        'H': img_shape[1], 
        'W': img_shape[2]
    })()
    
    if getattr(saved_args, 'alphas', 'linear') == 'linear':
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, saved_args.t_max)
        alphas = 1.0 - betas
    else:
        beta_start = 1e-4
        beta_end = 0.02
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

def generate_with_fixed_noise(ddpm, initial_noise, device):
    '''Generate samples using the provided initial noise and save frames for GIF'''
    from visualiser import Visualiser
    
    visualizer_args = type('Args', (), {
        'fps': 5,
        'output_dir': '.output',
        'denoiser_weights': None
    })()
    
    visualiser = Visualiser(visualizer_args)
    
    n_samples = initial_noise.shape[0]
    x_t = initial_noise.clone()
    frames = [x_t.clone()]
    
    # Manual denoising process to capture frames
    beta_t = 1.0 - ddpm.alphas
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - ddpm.alphas_bar)
    inv_sqrt_alpha = torch.rsqrt(ddpm.alphas)
    
    for t in reversed(range(ddpm.t_max)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        eps_theta = ddpm.denoiser(x_t, t_batch)
        c1 = inv_sqrt_alpha[t]
        c2 = beta_t[t] / (sqrt_one_minus_alpha_bar[t] + 1e-12)
        mean = c1 * (x_t - c2 * eps_theta)
        
        if t > 0:
            sigma = torch.sqrt(beta_t[t])
            z = torch.randn_like(x_t)
            x_t = mean + sigma * z
        else:
            x_t = mean
        
        frames.append(x_t.clone())
    
    # Save individual GIFs for each sample
    for i in range(n_samples):
        sample_frames = [f[i] for f in frames]
        visualiser.save_gif(sample_frames, f'{visualizer_args.output_dir}/sample_{i}.gif')
    
    return x_t

def save_png_images(initial_noise, final_results, output_dir):
    '''Save PNG images of initial noise and final results'''
    os.makedirs(output_dir, exist_ok=True)
    
    # Save initial noise
    fig, axes = plt.subplots(1, initial_noise.shape[0], figsize=(10, 3))
    if initial_noise.shape[0] == 1:
        axes = [axes]
    
    for i in range(initial_noise.shape[0]):
        img = initial_noise[i, 0].detach().cpu().numpy()  # Use detach() to remove gradients
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Initial Noise {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/initial_noise.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save final results
    fig, axes = plt.subplots(1, final_results.shape[0], figsize=(10, 3))
    if final_results.shape[0] == 1:
        axes = [axes]
    
    for i in range(final_results.shape[0]):
        img = final_results[i, 0].detach().cpu().numpy()  # Use detach() to remove gradients
        # Normalize to [0, 1] for better visualization
        img = (img - img.min()) / (img.max() - img.min())
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Final Result {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/final_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'PNG images saved to {output_dir}/')

def save_tensor(tensor, filepath):
    '''Save a tensor to a file'''
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(tensor.detach().cpu(), filepath)
    print(f'Tensor saved to: {filepath}')
    print(f'Tensor shape: {tensor.shape}')
    print(f'Tensor dtype: {tensor.dtype}')

# Main execution
print('Generating GIF and PNG images using consistent initial noise')
print('=' * 60)

# Use the specific checkpoint path
checkpoint_path = '/users/eleves-b/2023/oussama.akar/Projects/shapes-morphing-from-scratch/checkpoints/MLP/final_model.pth'

if not os.path.exists(checkpoint_path):
    print(f'Checkpoint not found at: {checkpoint_path}')
    exit(1)

denoiser, img_shape, saved_args, device = load_model(checkpoint_path)
ddpm_args = create_ddpm_args(denoiser, img_shape, saved_args, device)
ddpm = DDPM(ddpm_args)

print('Generating 2 samples with consistent denoising visualization...')

# Generate initial noise ONCE
n_samples = 2
initial_noise = torch.randn(n_samples, img_shape[0], img_shape[1], img_shape[2], device=device)

# Save the initial noise tensor
save_tensor(initial_noise, '.output/initial_noise_tensor.pt')

# Generate samples using the SAME initial noise for both GIF and PNG
final_samples = generate_with_fixed_noise(ddpm, initial_noise, device)

# Save the final samples tensor
save_tensor(final_samples, '.output/final_samples_tensor.pt')

# Save PNG images using the SAME samples
save_png_images(initial_noise, final_samples, '.output')

print(f'Generation completed!')
print(f'Generated samples shape: {final_samples.shape}')
print(f'Check .output directory for GIFs, PNG images, and tensor files!')
print('Now the GIFs and PNGs should show the SAME samples!')
"

echo "Visualization completed! Check the .output directory for the generated GIFs, PNG images, and tensor files."
