#!/usr/bin/env bash
#
# run_ablation.sh — Run the data quality ablation: 3 training runs + evals.
#
# Runs sequentially:
#   1. raw_sub_57M   — Raw subsample (~328M tok), 3 epochs
#   2. curated_57M   — Curated (~328M tok), 3 epochs
#   3. raw_full_57M  — Raw full (~6.5B tok), 1 epoch
#
# Each run: train -> eval -> save experiment JSON
#
# Usage:
#   tmux new -s ablation
#   bash run_ablation.sh 2>&1 | tee ablation.log
#
# Expected total time: ~3-4 hours

set -euo pipefail
cd "$(dirname "$0")"

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
TOKENIZER_DIR="tokenizer_16k"

echo "============================================"
echo " Data Quality Ablation"
echo " $(date)"
echo "============================================"
echo "GPUs: ${NUM_GPUS}"
echo ""

CONFIGS=(
    "configs/raw_sub_57M.yaml"
    "configs/curated_57M.yaml"
    "configs/hybrid_sub_57M.yaml"
    "configs/raw_full_57M.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
    EXP_ID=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment_id'])")
    OUTPUT_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['train']['output_dir'])")
    BEST_CKPT="${OUTPUT_DIR}/best.pt"

    echo ""
    echo "============================================"
    echo " [$(date '+%H:%M:%S')] Starting: ${EXP_ID}"
    echo " Config: ${CONFIG}"
    echo " Output: ${OUTPUT_DIR}"
    echo "============================================"
    echo ""

    # --- Train ---
    echo "[TRAIN] Starting DDP training across ${NUM_GPUS} GPUs..."
    torchrun --nproc_per_node="${NUM_GPUS}" pretrain.py --config "$CONFIG"
    echo "[TRAIN] Done at $(date '+%H:%M:%S')"
    echo ""

    # --- Eval ---
    if [ ! -f "${BEST_CKPT}" ]; then
        echo "ERROR: ${BEST_CKPT} not found after training. Skipping eval."
        continue
    fi

    echo "[EVAL] Evaluating ${BEST_CKPT}..."
    python3 evaluate.py \
        --checkpoint "${BEST_CKPT}" \
        --tokenizer_dir "${TOKENIZER_DIR}" \
        --output "${OUTPUT_DIR}/eval_results.json"
    echo "[EVAL] Done at $(date '+%H:%M:%S')"
    echo ""

    # --- Save experiment record ---
    echo "[SAVE] Writing experiment JSON..."
    python3 save_experiment.py --run_dir "${OUTPUT_DIR}" --config "${CONFIG}"
    echo ""

    echo "============================================"
    echo " [$(date '+%H:%M:%S')] Completed: ${EXP_ID}"
    echo "============================================"
done

echo ""
echo "============================================"
echo " All ablation runs complete!"
echo " $(date)"
echo " Results in experiments/"
echo "============================================"
ls -la experiments/*.json
