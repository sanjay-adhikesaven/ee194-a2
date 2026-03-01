"""
evaluate.py — Evaluate any pre-trained Nemotron-Nano model checkpoint using
lm-evaluation-harness (EleutherAI) with a custom LM wrapper.

Works with the config-driven pretrain.py — reads model_config from the
checkpoint itself, so the same script handles all model sizes.

Parallelizes across all available GPUs by replicating the model.

Benchmarks (0-shot):
  - hellaswag, piqa, arc_easy, arc_challenge, winogrande
  - sciq, boolq, lambada_openai, wikitext

Usage:
  python evaluate.py --checkpoint runs/scaled_38M/best.pt
  python evaluate.py --checkpoint runs/scaled_38M/best.pt --tasks hellaswag,piqa
  python evaluate.py --checkpoint runs/scaled_38M/best.pt --output experiments/scaled_38M_eval.json
"""

import argparse
import json
import os
import sys
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from lm_eval import simple_evaluate
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

sys.path.insert(0, os.path.dirname(__file__))
from pretrain import (
    ModelConfig, NemotronNano, load_tokenizer,
    EOS_TOKEN, PAD_TOKEN,
)

# ---------------------------------------------------------------------------
# Custom LM wrapper for lm-evaluation-harness
# ---------------------------------------------------------------------------

class NemotronNanoEvalLM(LM):
    """Wraps our custom model for the lm-eval-harness evaluation interface.

    Reads model_config from the checkpoint, so it auto-adapts to any model size.
    Supports multi-GPU by replicating the model on each GPU.
    """

    def __init__(self, checkpoint_path: str, tokenizer_dir: str = "tokenizer_16k",
                 batch_size: int = 64):
        super().__init__()

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        mcfg = ModelConfig(**ckpt["model_config"])

        self.num_gpus = torch.cuda.device_count()
        self.models = []
        for gpu_id in range(self.num_gpus):
            device = torch.device(f"cuda:{gpu_id}")
            model = NemotronNano(mcfg)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device).eval()
            self.models.append(model)

        self.tokenizer = load_tokenizer(tokenizer_dir)
        self._eos_id = self.tokenizer.token_to_id(EOS_TOKEN)
        self._pad_id = self.tokenizer.token_to_id(PAD_TOKEN)
        self._max_length = mcfg.max_position_embeddings
        self._batch_size = batch_size
        self._vocab_size = mcfg.vocab_size
        self._device = torch.device("cuda:0")

        experiment_id = ckpt.get("experiment_id", "unknown")
        epoch = ckpt.get("epoch", "?")
        loss = ckpt.get("loss", "?")
        param_info = self.models[0].count_parameters()
        print(f"Loaded: {checkpoint_path}")
        print(f"  Experiment : {experiment_id}")
        print(f"  Epoch/loss : {epoch} / {loss}")
        print(f"  Model      : {mcfg.hidden_size}h, {mcfg.num_layers}L, "
              f"{mcfg.vocab_size} vocab, {self._max_length} ctx")
        print(f"  Params     : {param_info['total_with_tying']:,} total "
              f"({param_info['non_embedding']:,} non-embed)")
        print(f"  Eval GPUs  : {self.num_gpus}")

    @property
    def eot_token_id(self):
        return self._eos_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str) -> list[int]:
        return self.tokenizer.encode(string).ids

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _get_logprobs(self, input_ids_list: list[list[int]]) -> list[torch.Tensor]:
        """Compute per-token log-probs distributed across all GPUs."""
        results: list[Optional[torch.Tensor]] = [None] * len(input_ids_list)

        gpu_batches: list[list[tuple[int, list[int]]]] = [[] for _ in range(self.num_gpus)]
        for i, ids in enumerate(input_ids_list):
            gpu_batches[i % self.num_gpus].append((i, ids))

        for gpu_id, batch in enumerate(gpu_batches):
            if not batch:
                continue
            device = torch.device(f"cuda:{gpu_id}")
            model = self.models[gpu_id]

            indices = [b[0] for b in batch]
            seqs = [b[1] for b in batch]
            max_len = min(max(len(s) for s in seqs), self._max_length)

            padded = []
            lengths = []
            for s in seqs:
                s = s[-max_len:]
                lengths.append(len(s))
                padded.append(s + [self._pad_id] * (max_len - len(s)))

            input_tensor = torch.tensor(padded, dtype=torch.long, device=device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_tensor)

            for j, (idx, seq_len) in enumerate(zip(indices, lengths)):
                if seq_len < 2:
                    results[idx] = torch.tensor([], device="cpu")
                    continue
                seq_logits = logits[j, :seq_len - 1, :]
                seq_labels = input_tensor[j, 1:seq_len]
                log_probs = F.log_softmax(seq_logits.float(), dim=-1)
                token_lps = log_probs.gather(1, seq_labels.unsqueeze(1)).squeeze(1)
                results[idx] = token_lps.cpu()

        return results

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        all_ids = []
        cont_lens = []
        for req in requests:
            context, continuation = req.args
            ctx_ids = self.tok_encode(context) if context else [self._eos_id]
            cont_ids = self.tok_encode(continuation)
            full_ids = (ctx_ids + cont_ids)[-self._max_length:]
            cont_len = min(len(cont_ids), len(full_ids) - 1)
            all_ids.append(full_ids)
            cont_lens.append(cont_len)

        results = []
        for batch_start in range(0, len(all_ids), self._batch_size):
            batch_ids = all_ids[batch_start:batch_start + self._batch_size]
            batch_cont = cont_lens[batch_start:batch_start + self._batch_size]
            token_lps_list = self._get_logprobs(batch_ids)

            for token_lps, cont_len in zip(token_lps_list, batch_cont):
                if len(token_lps) == 0:
                    results.append((0.0, False))
                    continue
                cont_lps = token_lps[-cont_len:]
                ll = cont_lps.sum().item()
                results.append((ll, True))

        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        results = []
        all_chunks = []
        req_chunk_map = []

        for req_idx, req in enumerate(requests):
            (string,) = req.args
            token_ids = self.tok_encode(string)
            chunk_indices = []

            for start in range(0, len(token_ids), self._max_length - 1):
                chunk = token_ids[start:start + self._max_length]
                if len(chunk) < 2:
                    continue
                chunk_indices.append(len(all_chunks))
                all_chunks.append(chunk)

            req_chunk_map.append(chunk_indices)

        all_lps = []
        for batch_start in range(0, len(all_chunks), self._batch_size):
            batch = all_chunks[batch_start:batch_start + self._batch_size]
            batch_lps = self._get_logprobs(batch)
            all_lps.extend(batch_lps)

        for req_idx, chunk_indices in enumerate(req_chunk_map):
            total_ll = sum(all_lps[ci].sum().item() for ci in chunk_indices)
            results.append(total_ll)

        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Minimal implementation for tasks that require generation."""
        results = []
        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            ctx_ids = self.tok_encode(context)
            if len(ctx_ids) > self._max_length - max_gen:
                ctx_ids = ctx_ids[-(self._max_length - max_gen):]

            input_ids = torch.tensor([ctx_ids], device=torch.device("cuda:0"))

            generated = []
            for _ in range(max_gen):
                if input_ids.shape[1] >= self._max_length:
                    break
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.models[0](input_ids)
                next_token = logits[0, -1, :].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                generated.append(next_token.item())
                gen_text = self.tok_decode(generated)
                if any(s in gen_text for s in until) or next_token.item() == self._eos_id:
                    break

            gen_text = self.tok_decode(generated)
            for s in until:
                idx = gen_text.find(s)
                if idx >= 0:
                    gen_text = gen_text[:idx]
            results.append(gen_text)

        return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_TASKS = [
    "hellaswag",
    "lambada_openai",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "sciq",
    "boolq",
    "wikitext",
]

def main():
    parser = argparse.ArgumentParser(description="Evaluate pre-trained model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (e.g. runs/scaled_38M/best.pt)")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer_16k")
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON. Defaults to <run_dir>/eval_results.json")
    args = parser.parse_args()

    # Default output: same directory as checkpoint
    if args.output is None:
        ckpt_dir = Path(args.checkpoint).parent
        args.output = str(ckpt_dir / "eval_results.json")

    task_list = [t.strip() for t in args.tasks.split(",")]
    print(f"Tasks: {task_list}")

    t0 = time.time()
    lm = NemotronNanoEvalLM(
        checkpoint_path=args.checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        batch_size=args.batch_size,
    )

    results = simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=0,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print(f"{'Task':<25} {'Metric':<25} {'Value':>10}")
    print("=" * 70)

    for task_name, task_results in sorted(results["results"].items()):
        for metric, value in sorted(task_results.items()):
            if metric.endswith(",none") or metric.endswith("_stderr,none"):
                clean_metric = metric.replace(",none", "")
                if isinstance(value, (int, float)) and not math.isnan(value):
                    print(f"{task_name:<25} {clean_metric:<25} {value:>10.4f}")

    print("=" * 70)
    print(f"Evaluation completed in {elapsed:.1f}s")

    output_path = Path(args.output)
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results["results"], f, indent=2, default=str)
    print(f"Full results saved to {output_path}")


if __name__ == "__main__":
    main()
