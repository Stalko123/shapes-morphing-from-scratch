#!/usr/bin/env bash

set -euo pipefail

# Example usage:
#   bash scripts/fid/compute_fid.sh \
#     --model u-net \
#     --dataset_name CIFAR10 \
#     --version 0 \
#     --path_to_weights ./checkpoints/CIFAR10_u-net_experiment/version_0/best.pth \
#     --path_to_yaml ./logs/CIFAR10_u-net_experiment/version_0/config.yml \
#     --num_gen 5000 \
#     --ref_split test \
#     --batch_size 64

PYTHON=${PYTHON:-python}

${PYTHON} -m scripts.fid.run_fid "$@" | cat 