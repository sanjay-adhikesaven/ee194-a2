#!/bin/bash
# run_remaining_experiments.sh — Train + eval the remaining 3 experiments.
#
# Completed already: equal_raw_dense, equal_curated_dense, equal_hybrid_dense, full_curated_moe
# Remaining: full_curated_dense, full_raw_dense, full_raw_moe
#
# Usage:
#   tmux new -s dq-experiments
#   bash run_remaining_experiments.sh
#   # Ctrl+B D to detach

set -euo pipefail
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================="
echo " Remaining experiments (with NCCL fix)"
echo " Started: $(date)"
echo "============================================="

TRAIN_CONFIGS=(
    configs/full_raw_moe.yaml
)

for CONFIG in "${TRAIN_CONFIGS[@]}"; do
    EXPERIMENT_ID=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment_id'])")
    OUTPUT_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['train']['output_dir'])")

    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "============================================="
    echo " [$EXPERIMENT_ID] Starting training"
    echo " Config: $CONFIG"
    echo " $(date)"
    echo "============================================="

    if torchrun --nproc_per_node=8 pretrain.py --config "$CONFIG" 2>&1 | tee "${OUTPUT_DIR}/train.log"; then
        echo "  [$EXPERIMENT_ID] Training finished successfully"
    else
        echo "  [$EXPERIMENT_ID] Training exited with error (checking for checkpoint...)"
    fi

    BEST_CKPT="${OUTPUT_DIR}/best.pt"
    if [ ! -f "$BEST_CKPT" ] || [ "$(stat -c%s "$BEST_CKPT" 2>/dev/null)" -lt 1000000 ]; then
        if [ -f "${OUTPUT_DIR}/latest.pt" ]; then
            cp "${OUTPUT_DIR}/latest.pt" "$BEST_CKPT"
            echo "  [$EXPERIMENT_ID] Copied latest.pt -> best.pt"
        fi
    fi

    if [ -f "$BEST_CKPT" ]; then
        echo "  [$EXPERIMENT_ID] Running evaluation..."
        python3 evaluate.py \
            --checkpoint "$BEST_CKPT" \
            --output "${OUTPUT_DIR}/eval_results.json" \
            2>&1 | tee "${OUTPUT_DIR}/eval.log"

        if [ -f "${OUTPUT_DIR}/eval_results.json" ]; then
            python3 save_experiment.py "$OUTPUT_DIR" "$CONFIG" 2>&1 || true
        fi
    else
        echo "  [$EXPERIMENT_ID] WARNING: No checkpoint found"
    fi

    echo "  [$EXPERIMENT_ID] Done at $(date)"
    echo "============================================="
done

echo ""
echo "============================================="
echo " All remaining experiments complete!"
echo " $(date)"
echo "============================================="
echo " Results: ls experiments/*.json"
