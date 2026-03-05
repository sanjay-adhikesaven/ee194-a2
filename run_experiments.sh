#!/bin/bash
# run_experiments.sh — Launch all 4 Chinchilla scaling experiments sequentially.
# Each experiment: train -> eval -> save results
#
# Usage:
#   tmux new -s experiments
#   bash run_experiments.sh
#   # Ctrl+B D to detach, come back later with: tmux attach -t experiments

set -euo pipefail
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIGS=(
    configs/dense_500M.yaml
    configs/moe_500M_active.yaml
    configs/dense_1_5B.yaml
    configs/moe_1_5B_active.yaml
)

echo "============================================="
echo " Starting ${#CONFIGS[@]} experiments"
echo " $(date)"
echo "============================================="

for CONFIG in "${CONFIGS[@]}"; do
    EXPERIMENT_ID=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment_id'])")
    OUTPUT_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['train']['output_dir'])")

    echo ""
    echo "============================================="
    echo " [$EXPERIMENT_ID] Starting training"
    echo " Config: $CONFIG"
    echo " Output: $OUTPUT_DIR"
    echo " $(date)"
    echo "============================================="

    torchrun --nproc_per_node=8 pretrain.py --config "$CONFIG" 2>&1 | tee "${OUTPUT_DIR}/train.log"

    BEST_CKPT="${OUTPUT_DIR}/best.pt"
    if [ -f "$BEST_CKPT" ]; then
        echo ""
        echo "  [$EXPERIMENT_ID] Running evaluation..."
        python3 evaluate.py \
            --checkpoint "$BEST_CKPT" \
            --output "${OUTPUT_DIR}/eval_results.json" \
            2>&1 | tee "${OUTPUT_DIR}/eval.log"

        echo "  [$EXPERIMENT_ID] Saving experiment record..."
        python3 save_experiment.py "$OUTPUT_DIR" "$CONFIG" 2>&1 || true
    else
        echo "  [$EXPERIMENT_ID] WARNING: No best.pt found, skipping eval"
    fi

    echo ""
    echo "  [$EXPERIMENT_ID] Complete at $(date)"
    echo "============================================="
done

echo ""
echo "============================================="
echo " All experiments complete!"
echo " $(date)"
echo "============================================="
