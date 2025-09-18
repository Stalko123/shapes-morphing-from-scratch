#!/bin/bash

# Training script for DDPM shapes morphing
# Run with: bash train.sh
export PYTHONPATH=$(pwd)
python trainers/trainer.py \
    --t_max 1000 \
    --n_epochs 800 \
    --verbose \
    --model "u-net" \
    --dataset_name "CIFAR10" \
    --batch_size 32 \
    --num_workers 8 \
    --validation \
    --test \
    --val_ratio 0.05 \
    --seed 42 \
    --num_trials 1 \
    --learning_rate 2e-4 \
    --dropout 0.1 \
    --optimizer_name "adam" \
    --grad_clip 1.0 \
    --patience 0 \
    --use_amp \
    --activation "silu" \
    --time_base_dim 128 \
    --time_hidden 512 \
    --time_output_dim 256 \
    --init_scheme "auto" \
    --base_channels 128 \
    --channel_mults 1,2,2,2 \
    --num_res_blocks 2 \
    --upsample "nearest_conv" \
    --groups 32 \
    --num_res_blocks_in_bottleneck 2 \
    --norm_2d "group" \
    --stem_kernel 3 \
    --head_kernel 3 \
    --downsample "stride" \
    --log_dir "./logs" \
    --checkpoint_dir "./checkpoints" \
    --save_frequency 25 \
    --output_dir "./outputs" \
    --attn_stages false,true,false,false \
    --attn_num_heads 4 \
    --attn_in_bottleneck \
    --grad_accum 2 \
