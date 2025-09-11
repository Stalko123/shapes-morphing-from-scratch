#!/bin/bash

python utils/viz/visualizer.py \
    --model "u-net" \
    --dataset_name "CIFAR10" \
    --version 1 \
    --fps 10 \
    --generate_gifs 1 \
    --viz_noising \
    --viz_progressive_denoising \
    --viz_denoising_from_t 100,400,700,1000 \
