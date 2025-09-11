#!/bin/bash

# Training script for DDPM shapes morphing
# Run with: bash train.sh

python trainers/trainer.py \
    --t_max 1000 \
    --n_epochs 1000 \
    --verbose \
    --model "u-net" \
    --dataset "CIFAR10" \
    --batch_size 16 \
    --num_workers 4 \
    --validation \
    --test \
    --val_ratio 0.05 \
    --seed 42 \
    --num_trials 1 \
    --learning_rate 1e-4 \
    --dropout 0 \
    --optimizer_name "adam" \
    --grad_clip 0.0 \
    --patience 0 \
    --activation "silu" \
    --time_base_dim 128 \
    --time_hidden 256 \
    --time_output_dim 128 \
    --init_scheme "auto" \
    --base_channels 64 \
    --channel_mults 1,2,2,4,8 \
    --num_res_blocks 2 \
    --upsample "nearest_conv" \
    --groups 32 \
    --num_res_blocks_in_bottleneck 4 \
    --norm_2d "group" \
    --stem_kernel 5 \
    --head_kernel 5 \
    --downsample "stride" \
    --log_dir "./logs" \
    --checkpoint_dir "./checkpoints" \
    --save_frequency 100 \
    --output_dir "./outputs" \
    --attn_stages false,false,false,true,true \
    --attn_num_heads 4 \
    --attn_in_bottleneck \
