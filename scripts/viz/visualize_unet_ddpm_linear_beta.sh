#!/bin/bash

# Example visualization script that demonstrates using a trained model with linear beta schedule
# This script will load the beta schedule parameters from the saved YAML config automatically

python utils/viz/visualizer.py \
    --model "u-net" \
    --dataset_name "CIFAR10" \
    --version 1 \
    --fps 10 \
    --generate_gifs 1 \
    --viz_noising \
    --viz_progressive_denoising \
    --viz_denoising_from_t 100,400,700,1000 \
    --beta_schedule "linear" \
    --beta_start 1e-4 \
    --beta_end 0.02
