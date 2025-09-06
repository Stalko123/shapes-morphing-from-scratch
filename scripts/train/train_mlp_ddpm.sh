#!/bin/bash

# Training script for DDPM shapes morphing
# Run with: bash train.sh

python trainers/trainer.py \
    --t_max 1000 \
    --n_epochs 100 \
    --verbose \
    --model "mlp" \
    --dataset "MNIST" \
    --batch_size 8 \
    --n_workers 4 \
    --num_trials 100 \
    --alpha_min 0.95 \
    --alpha_max 1.0 \
    --alpha_interp "linear" \
    --learning_rate 1e-5 \
    --dropout 0.2 \
    --optimizer "adam" \
    --time_base_dim 128 \
    --time_hidden 512 \
    --time_output_dim 256 \
    --activation "silu" \
    --norm_1d "layer" \
    --init_scheme "auto" \
    --hidden_sizes "2048, 2048, 1024, 1024" \
    --exp_name "mnist_mlp_experiment" \
    --log_dir "./logs" \
    --checkpoint_dir "./checkpoints" \
    --save_frequency 50 \
    --output_dir "./outputs" \
    --fps 5
