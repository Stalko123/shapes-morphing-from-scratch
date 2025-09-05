#!/bin/bash

# Training script for DDPM shapes morphing
# Run with: bash train.sh

python trainers/trainer.py \
    --n_epochs 100 \
    --batch_size 10 \
    --learning_rate 1e-4 \
    --dataset "MNIST" \
    --model "MLP" \
    --t_max 500 \
    --num_trials 150 \
    --alpha_interp "linear" \
    --hidden_sizes 1024 1024 \
    --time_dim 128 \
    --activation "silu" \
    --norm "layer" \
    --dropout 0.2 \
    --n_workers 4 \
    --log_dir "./logs" \
    --checkpoint_dir "./checkpoints/MLP" \
    --save_frequency 10 \
    --verbose \
    --fps 5 \
    --output_dir "./outputs"
