#!/bin/bash

# Optimized DDPM U-Net training script for CIFAR-10
# Based on proven configurations that guarantee good results
# Run with: bash scripts/train/train_unet_ddpm.sh

export PYTHONPATH=$(pwd)
python trainers/trainer.py \
    --t_max 1000 \
    --n_epochs 800 \
    --verbose \
    --model "u-net" \
    --dataset_name "CIFAR10" \
    --batch_size 64 \
    --num_workers 8 \
    --validation \
    --test \
    --val_ratio 0.01 \
    --seed 42 \
    --num_trials 1 \
    --learning_rate 5e-5 \
    --dropout 0.1 \
    --optimizer_name "Adam" \
    --grad_clip 0.5 \
    --patience 800 \
    --activation "silu" \
    --time_base_dim 128 \
    --time_hidden 512 \
    --time_output_dim 512 \
    --init_scheme "auto" \
    --base_channels 64 \
    --channel_mults 1,2,4,4 \
    --num_res_blocks 2 \
    --upsample "nearest_conv" \
    --groups 32 \
    --num_res_blocks_in_bottleneck 3 \
    --norm_2d "group" \
    --stem_kernel 3 \
    --head_kernel 3 \
    --downsample "stride" \
    --log_dir "./logs" \
    --checkpoint_dir "./checkpoints" \
    --save_frequency 25 \
    --output_dir "./outputs" \
    --attn_stages False,False,False,True \
    --attn_num_heads 4 \
    --attn_in_bottleneck \
    --grad_accum 1 \
