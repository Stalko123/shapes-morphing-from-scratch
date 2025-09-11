#!/bin/bash

python utils/viz/visualizer.py \
    --model "u-net" \
    --dataset_name "CIFAR10" \
    --version 0 \
    --fps 10 \
    --generate_gifs 1 \
    --viz_noising \
    --viz_denoising_from_t 100,400,700,1000 \
