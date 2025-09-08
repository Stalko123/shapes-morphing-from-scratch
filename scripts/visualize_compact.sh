#!/bin/bash

# Compact DDPM Visualization Script
# Usage: ./visualize_compact.sh [OPTIONS]

set -e

# Default values
CHECKPOINT_PATH="/users/eleves-b/2023/oussama.akar/Projects/shapes-morphing-from-scratch/checkpoints/CIFAR10_mlp_experiment/version_1/best.pth"
OUTPUT_DIR="./outputs/visualization_output"
N_SAMPLES=1
FPS=5

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint) CHECKPOINT_PATH="$2"; shift 2 ;;
        -o|--output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        -n|--n_samples) N_SAMPLES="$2"; shift 2 ;;
        -f|--fps) FPS="$2"; shift 2 ;;
        -s|--save_every) SAVE_EVERY="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# Check checkpoint
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# Get project root and run
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

python utils/viz/visualize.py \
    --path_to_weights "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --fps "$FPS" \
    --model "mlp" \
    --n_epochs 1

 