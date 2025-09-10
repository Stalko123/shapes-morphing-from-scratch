#!/bin/bash

python utils/viz/visualizer.py \
    --model "u-net" \
    --dataset_name "CIFAR10" \
    --version 0 \
    --fps 10 \
    --generate_gifs 3 \
    --visualize_noising \
