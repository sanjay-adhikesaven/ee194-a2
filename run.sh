#!/usr/bin/env bash
#
# run.sh — Config-driven pre-training and evaluation pipeline.
#
# All model variants are defined by YAML configs in configs/.
# Checkpoints and eval results go into runs/<experiment_id>/.
#
# Prerequisites:
#   - 8x NVIDIA GPUs (tested on H100 80GB)
#   - Python 3.10+ with: torch, tokenizers, lm-eval, pyyaml
#   - Data at path specified in config YAML
#
# Usage:
#   bash run.sh configs/baseline_4M.yaml          # train + eval
#   bash run.sh configs/scaled_38M.yaml train      # train only
#   bash run.sh configs/scaled_38M.yaml eval        # eval only
#   bash run.sh all                                # train + eval ALL configs

set -euo pipefail
cd "$(dirname "$0")"

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
TOKENIZER_DIR="tokenizer_16k"

echo "============================================"
echo " Nemotron-Nano Pre-training Pipeline"
echo "============================================"
echo "GPUs detected: ${NUM_GPUS}"
echo ""

# Extract output_dir from a YAML config (lightweight, no python yaml dep needed)
get_output_dir() {
    python3 -c "import yaml; print(yaml.safe_load(open('$1'))['train']['output_dir'])"
}

get_experiment_id() {
    python3 -c "import yaml; print(yaml.safe_load(open('$1')).get('experiment_id', 'unknown'))"
}

run_single() {
    local CONFIG="$1"
    local MODE="${2:-all}"

    local EXP_ID
    EXP_ID=$(get_experiment_id "$CONFIG")
    local OUTPUT_DIR
    OUTPUT_DIR=$(get_output_dir "$CONFIG")
    local BEST_CKPT="${OUTPUT_DIR}/best.pt"

    echo "--------------------------------------------"
    echo " Experiment: ${EXP_ID}"
    echo " Config:     ${CONFIG}"
    echo " Output:     ${OUTPUT_DIR}"
    echo "--------------------------------------------"
    echo ""

    if [[ "$MODE" == "all" || "$MODE" == "train" ]]; then
        echo "[TRAIN] Starting DDP training across ${NUM_GPUS} GPUs..."
        torchrun --nproc_per_node="${NUM_GPUS}" pretrain.py --config "$CONFIG"
        echo ""
        echo "[TRAIN] Complete. Checkpoints in ${OUTPUT_DIR}/"
        echo ""
    fi

    if [[ "$MODE" == "all" || "$MODE" == "eval" ]]; then
        if [ ! -f "${BEST_CKPT}" ]; then
            echo "ERROR: ${BEST_CKPT} not found. Run training first."
            return 1
        fi

        echo "[EVAL] Evaluating ${BEST_CKPT} across ${NUM_GPUS} GPUs..."
        python3 evaluate.py \
            --checkpoint "${BEST_CKPT}" \
            --tokenizer_dir "${TOKENIZER_DIR}" \
            --output "${OUTPUT_DIR}/eval_results.json"
        echo ""
        echo "[EVAL] Results saved to ${OUTPUT_DIR}/eval_results.json"
        echo ""
    fi
}

# --- Main dispatch ---

if [[ "${1:-}" == "all" ]]; then
    # Run ALL configs in configs/ directory
    for cfg in configs/*.yaml; do
        run_single "$cfg" "all"
    done
elif [[ -f "${1:-}" ]]; then
    # Single config file
    run_single "$1" "${2:-all}"
else
    echo "Usage:"
    echo "  bash run.sh <config.yaml> [train|eval|all]"
    echo "  bash run.sh all                              # run all configs"
    echo ""
    echo "Available configs:"
    ls -1 configs/*.yaml 2>/dev/null || echo "  (none found in configs/)"
    exit 1
fi

echo "============================================"
echo " Done."
echo "============================================"
