#!/bin/bash
# Evaluate trained checkpoint(s) on the held-out test groups.
#   bash test.sh runs/<run>/best.pt                        # one model (+ TTA)
#   bash test.sh runs/<a>/best.pt runs/<b>/best.pt         # equal-weight ensemble
#   bash test.sh runs/<run>/best.pt --wandb_mode disabled  # extra run.py flags after the paths
CKPTS=()
while [[ $# -gt 0 && "$1" != -* ]]; do CKPTS+=("$1"); shift; done
if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "usage: bash test.sh path/to/best.pt [more.pt ...] [run.py flags]" >&2
    exit 1
fi
python src/run.py \
    --mode eval \
    --frame_dir data/UCF50_frames \
    --ckpt "${CKPTS[@]}" \
    --batch_size 8 \
    "$@"
