#!/usr/bin/env bash
#
# run.sh — Reproduce pre-training and evaluation for the
#           Nemotron-Nano-V3-Micro model (EE 194/290-16, Assignment 2).
#
# Prerequisites:
#   - 8× NVIDIA GPUs (tested on H100 80GB)
#   - Python 3.10+ with: torch, tokenizers, lm-eval
#   - Data at /home/jason/ee194-a2/partb_data_designer_exports/partb_hybrid_training_data.jsonl
#
# Usage:
#   bash run.sh          # full pipeline: train tokenizer + train model + eval
#   bash run.sh train    # training only
#   bash run.sh eval     # evaluation only (requires existing checkpoint)

set -euo pipefail
cd "$(dirname "$0")"

CHECKPOINT_DIR="checkpoints"
TOKENIZER_DIR="tokenizer_16k"
BEST_CKPT="${CHECKPOINT_DIR}/best.pt"
EVAL_OUTPUT="eval_results.json"
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)

echo "============================================"
echo " Nemotron-Nano-V3-Micro Pre-training Pipeline"
echo "============================================"
echo "GPUs detected: ${NUM_GPUS}"
echo ""

run_train() {
    echo "--------------------------------------------"
    echo " Step 1: Pre-training (includes tokenizer)"
    echo "--------------------------------------------"
    echo "  Tokenizer : 16K BPE (trained on corpus)"
    echo "  Model     : 4.0M params (128h, 8L, GQA)"
    echo "  Data      : ~26M tokens, 3 epochs"
    echo "  Parallel  : DDP across ${NUM_GPUS} GPUs"
    echo ""

    torchrun --nproc_per_node="${NUM_GPUS}" pretrain.py

    echo ""
    echo "Training complete. Checkpoints in ${CHECKPOINT_DIR}/"
    echo ""
}

run_eval() {
    if [ ! -f "${BEST_CKPT}" ]; then
        echo "ERROR: ${BEST_CKPT} not found. Run training first."
        exit 1
    fi

    echo "--------------------------------------------"
    echo " Step 2: Evaluation (lm-evaluation-harness)"
    echo "--------------------------------------------"
    echo "  Checkpoint : ${BEST_CKPT}"
    echo "  Benchmarks : hellaswag, lambada_openai, piqa,"
    echo "               arc_easy, arc_challenge, winogrande,"
    echo "               sciq, boolq, wikitext"
    echo "  Few-shot   : 0"
    echo "  GPUs       : ${NUM_GPUS} (model replicated)"
    echo ""

    python3 evaluate.py --checkpoint "${BEST_CKPT}"

    echo ""
    echo "Results saved to ${EVAL_OUTPUT}"
    echo ""
}

case "${1:-all}" in
    train) run_train ;;
    eval)  run_eval ;;
    all)   run_train; run_eval ;;
    *)     echo "Usage: bash run.sh [train|eval|all]"; exit 1 ;;
esac

echo "============================================"
echo " Done."
echo "============================================"
