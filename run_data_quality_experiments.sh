#!/bin/bash
# run_data_quality_experiments.sh — Data quality ablation: raw vs curated vs hybrid
#
# 7 experiments total:
#   Part 1 (equal-size, ~2h): 3 dense runs on 1.3B tokens each
#   Part 2 (full-size, ~55h): 2 dense + 2 MoE runs on full datasets
#
# Usage:
#   tmux new -s dq-experiments
#   bash run_data_quality_experiments.sh
#   # Ctrl+B D to detach; tmux attach -t dq-experiments to reconnect

set -euo pipefail
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIGS=(
    # Part 1: Equal-size controlled comparison (1.3B tokens, dense)
    configs/equal_raw_dense.yaml
    configs/equal_curated_dense.yaml
    configs/equal_hybrid_dense.yaml
    # Part 2: Full-size scaling
    configs/full_curated_dense.yaml
    configs/full_curated_moe.yaml
    configs/full_raw_dense.yaml
    configs/full_raw_moe.yaml
)

echo "============================================="
echo " Data Quality Ablation: ${#CONFIGS[@]} experiments"
echo " Started: $(date)"
echo "============================================="

COMPLETED=0
FAILED=0

for CONFIG in "${CONFIGS[@]}"; do
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
        echo "  [$EXPERIMENT_ID] WARNING: No best.pt found, skipping eval"
        FAILED=$((FAILED + 1))
    fi

    echo ""
    echo "  [$EXPERIMENT_ID] Done at $(date)"
    echo "  Progress: $COMPLETED completed, $FAILED failed, $((${#CONFIGS[@]} - COMPLETED - FAILED)) remaining"
    echo "============================================="
done

echo ""
echo "============================================="
echo " All experiments complete!"
echo " Completed: $COMPLETED / ${#CONFIGS[@]}"
echo " Failed: $FAILED"
echo " $(date)"
echo "============================================="
echo ""
echo " Results saved in experiments/*.json"
echo " To compare: ls experiments/equal_*.json experiments/full_*.json"
