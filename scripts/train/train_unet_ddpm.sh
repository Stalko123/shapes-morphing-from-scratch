#!/bin/bash

# Training script for DDPM shapes morphing
# Run with: bash train.sh

python trainers/trainer.py \
    --t_max 1000 \
    --n_epochs 1 \
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
    --activation "silu" \
    --norm_2d "group" \
    --init_scheme "auto" \
    --base_channels 64 \
    --channel_mults 1,2,2 \
    --num_res_blocks 2 \
    --upsample "nearest_conv" \
    --groups 32 \
    --num_res_blocks_in_bottleneck 2 \
    --stem_kernel 5 \
    --head_kernel 5 \
    --downsample "stride" \
    --time_base_dim 128 \
    --time_hidden 512 \
    --time_output_dim 256 \
    --exp_name "mnist_u-net_experiment" \
    --log_dir "./logs" \
    --checkpoint_dir "./checkpoints" \
    --save_frequency 1 \
    --output_dir "./outputs" \
    --fps 5
