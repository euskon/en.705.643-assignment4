#!/bin/bash
# Train a UCF50 classifier. Run `python src/prepare_data.py` once beforehand.
# Extra flags are passed through, e.g.:  bash train.sh --model_type 3dcnn --batch_size 16
# Avoid silent VRAM->RAM spill on WSL when the allocator's reserved pool fragments.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python src/run.py \
    --mode train \
    --frame_dir data/UCF50_frames \
    --model_type lrcn \
    --cnn_backbone convnext_tiny \
    --fr_per_vid 16 \
    --batch_size 8 \
    --n_epochs 20 \
    "$@"
