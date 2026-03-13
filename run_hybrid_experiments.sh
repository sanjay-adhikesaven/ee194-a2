#!/bin/bash
# run_hybrid_experiments.sh — Train + eval the 4 hybrid data experiments.
#
# Part 1 (equal-size, ~5.7B tokens each):
#   equal5b_raw_dense, equal5b_hybrid_dense
#   (equal5b_curated_dense reuses full_curated_dense)
#
# Part 2 (full hybrid, ~8.8B tokens):
#   full_hybrid_dense, full_hybrid_moe
#
# Usage:
#   tmux new -s hybrid-experiments
#   bash run_hybrid_experiments.sh
#   # Ctrl+B D to detach

set -euo pipefail
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================="
echo " Hybrid data experiments (4 runs)"
echo " Started: $(date)"
echo "============================================="

TRAIN_CONFIGS=(
    # configs/equal5b_raw_dense.yaml  # already completed
    configs/equal5b_hybrid_dense.yaml
    configs/full_hybrid_dense.yaml
    configs/full_hybrid_moe.yaml
)

COMPLETED=0
FAILED=0

for CONFIG in "${TRAIN_CONFIGS[@]}"; do
    EXPERIMENT_ID=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment_id'])")
    OUTPUT_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['train']['output_dir'])")

    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "============================================="
    echo " [$EXPERIMENT_ID] Starting training"
    echo " Config: $CONFIG"
    echo " Output: $OUTPUT_DIR"
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
        echo ""
        echo "  [$EXPERIMENT_ID] Running evaluation..."
        if python3 evaluate.py \
            --checkpoint "$BEST_CKPT" \
            --output "${OUTPUT_DIR}/eval_results.json" \
            2>&1 | tee "${OUTPUT_DIR}/eval.log"; then
            echo "  [$EXPERIMENT_ID] Eval complete"
        else
            echo "  [$EXPERIMENT_ID] WARNING: Eval failed"
        fi

        if [ -f "${OUTPUT_DIR}/eval_results.json" ]; then
            echo "  [$EXPERIMENT_ID] Saving experiment record..."
            python3 save_experiment.py "$OUTPUT_DIR" "$CONFIG" 2>&1 || true
        fi

        COMPLETED=$((COMPLETED + 1))
    else
        echo "  [$EXPERIMENT_ID] WARNING: No checkpoint found, skipping eval"
        FAILED=$((FAILED + 1))
    fi

    echo ""
    echo "  [$EXPERIMENT_ID] Done at $(date)"
    echo "  Progress: $COMPLETED completed, $FAILED failed, $((${#TRAIN_CONFIGS[@]} - COMPLETED - FAILED)) remaining"
    echo "============================================="
done

echo ""
echo "============================================="
echo " All hybrid experiments complete!"
echo " Completed: $COMPLETED / ${#TRAIN_CONFIGS[@]}"
echo " Failed: $FAILED"
echo " $(date)"
echo "============================================="
echo " Results: ls experiments/equal5b_*.json experiments/full_hybrid_*.json"
