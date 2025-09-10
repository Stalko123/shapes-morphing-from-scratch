#!/bin/bash

# Training script for DDPM shapes morphing
# Run with: bash train.sh

python trainers/trainer.py \
    --t_max 1000 \
    --n_epochs 100 \
    --verbose \
    --model "mlp" \
    --dataset "CIFAR10" \
    --batch_size 8 \
    --num_workers 4 \
    --validation \
    --test \
    --val_ratio 0.05 \
    --seed 42 \
    --num_trials 1 \
    --learning_rate 1e-5 \
    --dropout 0.2 \
    --optimizer_name "adam" \
    --grad_clip 10.0 \
    --activation "silu" \
    --time_base_dim 128 \
    --time_hidden 512 \
    --time_output_dim 256 \
    --init_scheme "auto" \
    --hidden_sizes "2048, 2048, 1024, 1024" \
    --norm_1d "layer" \
    --log_dir "./logs" \
    --checkpoint_dir "./checkpoints" \
    --save_frequency 50 \
    --output_dir "./outputs" \
    --patience 10
