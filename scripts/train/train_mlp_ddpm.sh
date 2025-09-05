#!/bin/bash

# Training script for DDPM shapes morphing
# Run with: bash train.sh

python trainers/trainer.py \
    --t_max 1000 \
    --n_epochs 100 \
    --verbose \
    --model "u-net" \
    --dataset "MNIST" \
    --batch_size 8 \
    --n_workers 4 \
    --num_trials 100 \
    --alpha_min 0.95 \
    --alpha_max 1.0 \
    --alpha_interp "linear" \
    --learning_rate 1e-3 \
    --dropout 0.2 \
    --optimizer "adam" \
    --time_dim 128 \
    --activation "silu" \
    --norm "group" \
    --init_scheme "auto" \
    --hidden_sizes "2048, 2048, 1024, 1024" \
    --exp_name "mnist_u-net_experiment" \
    --log_dir "./logs" \
    --checkpoint_dir "./checkpoints" \
    --save_frequency 10 \
    --output_dir "./outputs" \
    --fps 5
