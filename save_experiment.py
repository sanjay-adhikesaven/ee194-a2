"""
save_experiment.py — Generate a structured experiment JSON from a completed run.

Usage:
  python save_experiment.py --run_dir runs/raw_sub_57M --config configs/raw_sub_57M.yaml
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml
import torch
import torch.distributed.tensor  # needed to load any DTensor checkpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("config")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    experiment_id = cfg.get("experiment_id", run_dir.name)
    mcfg = cfg["model"]
    tcfg = cfg["train"]

    # Load checkpoint metadata
    best_ckpt = run_dir / "best.pt"
    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)

    # Load eval results
    eval_path = run_dir / "eval_results.json"
    with open(eval_path) as f:
        eval_full = json.load(f)

    # Extract summary metrics
    def get(task, metric):
        return eval_full.get(task, {}).get(metric, None)

    eval_summary = {
        "hellaswag_acc_norm": get("hellaswag", "acc_norm,none"),
        "piqa_acc": get("piqa", "acc,none"),
        "winogrande_acc": get("winogrande", "acc,none"),
        "boolq_acc": get("boolq", "acc,none"),
        "arc_easy_acc_norm": get("arc_easy", "acc_norm,none"),
        "arc_challenge_acc_norm": get("arc_challenge", "acc_norm,none"),
        "sciq_acc_norm": get("sciq", "acc_norm,none"),
        "wikitext_word_ppl": get("wikitext", "word_perplexity,none"),
        "wikitext_bits_per_byte": get("wikitext", "bits_per_byte,none"),
        "lambada_ppl": get("lambada_openai", "perplexity,none"),
    }

    # Count params from checkpoint
    from pretrain import ModelConfig, NemotronNano
    model = NemotronNano(ModelConfig(**{k: v for k, v in ckpt["model_config"].items()
                                        if k in ModelConfig.__dataclass_fields__}))
    param_info = model.count_parameters()

    experiment = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": f"Data ablation: {experiment_id}",
        "config_path": args.config,
        "data": {
            "path": tcfg["data_path"],
        },
        "tokenizer": {
            "type": "byte-level BPE (HuggingFace tokenizers)",
            "vocab_size": mcfg["vocab_size"],
            "trained_on": "hybrid pre-training corpus",
            "dir": tcfg["tokenizer_dir"],
        },
        "model": {
            "family": "Nemotron-Nano-V3 (dense, scaled down)",
            **mcfg,
            "total_params": param_info.get("total", param_info.get("total_with_tying")),
            "active_params_per_token": param_info.get("active_per_token", param_info.get("total_with_tying")),
            "embedding_params": param_info["embedding"],
            "non_embedding_params": param_info["non_embedding"],
        },
        "training": {
            "epochs": tcfg["num_epochs"],
            "batch_size_per_gpu": tcfg["batch_size"],
            "gradient_accumulation_steps": tcfg["gradient_accumulation_steps"],
            "num_gpus": 8,
            "global_batch_size": tcfg["batch_size"] * tcfg["gradient_accumulation_steps"] * 8,
            "learning_rate": tcfg["learning_rate"],
            "weight_decay": tcfg["weight_decay"],
            "warmup_fraction": tcfg["warmup_fraction"],
            "dtype": tcfg["dtype"],
            "optimizer": f"AdamW (beta1={tcfg['adam_beta1']}, beta2={tcfg['adam_beta2']})",
            "schedule": "linear warmup + cosine decay",
            "best_train_loss": ckpt.get("loss"),
            "best_epoch": ckpt.get("epoch"),
            "total_steps": ckpt.get("global_step"),
        },
        "eval_results": eval_summary,
        "eval_results_full": eval_full,
    }

    os.makedirs("experiments", exist_ok=True)
    out_path = f"experiments/{experiment_id}.json"
    with open(out_path, "w") as f:
        json.dump(experiment, f, indent=2, default=str)
    print(f"Saved experiment record to {out_path}")


if __name__ == "__main__":
    main()
