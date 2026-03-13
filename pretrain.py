"""
pretrain.py — Config-driven pre-training for Nemotron-Nano-V3-style models.

Supports dense and Mixture-of-Experts (MoE) architectures with FSDP2 or DDP.

Architecture family: NVIDIA Nemotron-Nano-V3
  - Grouped-Query Attention (GQA) with Flash SDP
  - RMSNorm (pre-norm)
  - SwiGLU MLP (gate · up, then down)
  - Rotary Position Embeddings (RoPE)
  - Tied input/output embeddings
  - Optional top-k MoE routing

Parallelism: FSDP2 (fully_shard) + torch.compile for best MFU on H100s.
Falls back to DDP for small models or when FSDP2 is unavailable.

Usage:
  torchrun --nproc_per_node=8 pretrain.py --config configs/dense_500M.yaml
  torchrun --nproc_per_node=8 pretrain.py --config configs/moe_500M.yaml
"""

import argparse
import os
import json
import math
import time
import random
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 16_384
    hidden_size: int = 128
    num_layers: int = 8
    num_attention_heads: int = 8
    num_kv_heads: int = 2
    intermediate_size: int = 512
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    # MoE config (set num_experts > 0 to enable)
    num_experts: int = 0
    num_experts_per_tok: int = 2
    moe_aux_loss_coeff: float = 0.01


@dataclass
class TrainConfig:
    data_path: str = ""
    output_dir: str = "runs/default"
    tokenizer_dir: str = "tokenizer_16k"

    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.05
    num_epochs: int = 1

    dtype: str = "bfloat16"
    compile_model: bool = True
    use_fsdp: bool = True

    log_interval: int = 50
    save_interval_steps: int = 5000
    save_interval_epochs: int = 1
    wandb_project: str = "ee194-a2-pretrain"
    wandb_run_name: str = ""
    seed: int = 42

    max_tokens: int = 0  # 0 = use all data; >0 = stop after this many tokens
    resume_from: str = ""  # path to checkpoint to resume training from


def load_config(config_path: str) -> tuple[str, ModelConfig, TrainConfig]:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    experiment_id = raw.get("experiment_id", Path(config_path).stem)
    mcfg = ModelConfig(**{k: v for k, v in raw.get("model", {}).items() if k in ModelConfig.__dataclass_fields__})
    tcfg = TrainConfig(**{k: v for k, v in raw.get("train", {}).items() if k in TrainConfig.__dataclass_fields__})
    return experiment_id, mcfg, tcfg


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|eos|>"
UNK_TOKEN = "<|unk|>"
SPECIAL_TOKENS = [PAD_TOKEN, EOS_TOKEN, UNK_TOKEN]

def train_tokenizer(data_path: str, vocab_size: int, save_dir: str):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2,
        special_tokens=SPECIAL_TOKENS, show_progress=True,
    )

    def doc_iterator():
        for fpath in _iter_jsonl_files(data_path):
            with open(fpath) as f:
                for line in f:
                    yield json.loads(line)["text"]

    tokenizer.train_from_iterator(doc_iterator(), trainer=trainer)
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    test = tokenizer.encode("Hello world! This is a test.")
    print(f"Tokenizer trained: vocab_size={tokenizer.get_vocab_size()}, "
          f"test encode length={len(test.ids)}")
    return tokenizer


def load_tokenizer(save_dir: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(save_dir, "tokenizer.json"))


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(dim, max_seq_len, theta=10_000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    hd = x.shape[-1]
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    cos = cos[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class GQAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.num_groups = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(attn.transpose(1, 2).contiguous().view(B, T, -1))


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    """Top-k Mixture of Experts with load-balancing auxiliary loss."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_tok
        self.aux_loss_coeff = cfg.moe_aux_loss_coeff

        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUMLP(cfg) for _ in range(cfg.num_experts)])

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                expert_input = x_flat[mask]
                expert_out = expert(expert_input)
                weight_for_expert = topk_weights[mask] * (topk_indices[mask] == i).float()
                weight_sum = weight_for_expert.sum(dim=-1, keepdim=True)
                out[mask] += expert_out * weight_sum

        # Load-balancing auxiliary loss
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            tokens_per_expert[i] = (topk_indices == i).any(dim=-1).float().sum()
        tokens_per_expert = tokens_per_expert / (B * T)
        avg_routing = routing_weights.mean(dim=0)
        aux_loss = self.aux_loss_coeff * self.num_experts * (tokens_per_expert * avg_routing).sum()

        return out.view(B, T, D), aux_loss


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = GQAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.use_moe = cfg.num_experts > 0
        if self.use_moe:
            self.mlp = MoELayer(cfg)
        else:
            self.mlp = SwiGLUMLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        if self.use_moe:
            mlp_out, aux_loss = self.mlp(self.mlp_norm(x))
            x = x + mlp_out
            return x, aux_loss
        else:
            x = x + self.mlp(self.mlp_norm(x))
            return x, torch.tensor(0.0, device=x.device)


class NemotronNano(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        rope_cos, rope_sin = precompute_rope(
            cfg.hidden_size // cfg.num_attention_heads,
            cfg.max_position_embeddings, cfg.rope_theta,
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, aux = layer(x, self.rope_cos, self.rope_sin)
            total_aux_loss = total_aux_loss + aux
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, total_aux_loss

    def count_parameters(self):
        seen = {}
        for p in self.parameters():
            pid = id(p)
            if pid not in seen:
                seen[pid] = p.numel()
        unique = sum(seen.values())
        embed = self.embed_tokens.weight.numel()

        active = unique
        if self.cfg.num_experts > 0:
            expert_params = 0
            for layer in self.layers:
                if hasattr(layer.mlp, 'experts'):
                    for expert in layer.mlp.experts:
                        for p in expert.parameters():
                            expert_params += p.numel()
            active_expert_frac = self.cfg.num_experts_per_tok / self.cfg.num_experts
            inactive_expert_params = expert_params * (1 - active_expert_frac)
            active = unique - int(inactive_expert_params)

        return {"total": unique, "active_per_token": active,
                "embedding": embed, "non_embedding": unique - embed}


# ---------------------------------------------------------------------------
# Dataset — supports both in-memory and streaming
# ---------------------------------------------------------------------------

class PretrainDataset(Dataset):
    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        n = len(token_ids) // seq_len
        self.data = torch.tensor(token_ids[:n * seq_len], dtype=torch.long).view(n, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]


class StreamingPretrainDataset(IterableDataset):
    """Streams tokenized chunks from sharded JSONL files without loading all into memory."""

    def __init__(self, data_path: str, tokenizer, seq_len: int, eos_id: int,
                 rank: int, world_size: int, seed: int, epoch: int = 0,
                 max_tokens: int = 0):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = epoch
        self.max_tokens = max_tokens

    def __iter__(self):
        files = list(_iter_jsonl_files(self.data_path))
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(files)

        # Shard files across workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            total_workers = self.world_size * worker_info.num_workers
            worker_id = self.rank * worker_info.num_workers + worker_info.id
        else:
            total_workers = self.world_size
            worker_id = self.rank

        my_files = [f for i, f in enumerate(files) if i % total_workers == worker_id]

        buffer = []
        total_tokens_yielded = 0
        for fpath in my_files:
            with open(fpath) as f:
                for line in f:
                    doc = json.loads(line)
                    ids = self.tokenizer.encode(doc["text"]).ids
                    buffer.extend(ids)
                    buffer.append(self.eos_id)

                    while len(buffer) >= self.seq_len:
                        chunk = buffer[:self.seq_len]
                        buffer = buffer[self.seq_len:]
                        t = torch.tensor(chunk, dtype=torch.long)
                        total_tokens_yielded += self.seq_len
                        yield t[:-1], t[1:]

                        worker_info = torch.utils.data.get_worker_info()
                        num_workers = worker_info.num_workers if worker_info else 1
                        if self.max_tokens > 0 and total_tokens_yielded >= self.max_tokens // (self.world_size * num_workers):
                            return


def _iter_jsonl_files(data_path: str):
    p = Path(data_path)
    if p.is_file():
        yield p
    elif p.is_dir():
        for f in sorted(p.glob("*.jsonl")):
            yield f
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")


def load_and_tokenize(data_path: str, tokenizer, seq_len: int, eos_id: int, rank: int):
    all_ids = []
    count = 0
    for fpath in _iter_jsonl_files(data_path):
        if rank == 0:
            print(f"  Reading {fpath.name} ...", flush=True)
        with open(fpath) as f:
            for line in f:
                doc = json.loads(line)
                ids = tokenizer.encode(doc["text"]).ids
                all_ids.extend(ids)
                all_ids.append(eos_id)
                count += 1
                if rank == 0 and count % 100_000 == 0:
                    print(f"  Tokenized {count:,} docs ({len(all_ids):,} tokens)...", flush=True)
    return all_ids


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step, total_steps, warmup_steps, max_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
    else:
        rank, local_rank, world_size = 0, 0, 1
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def print0(msg, rank):
    if rank == 0:
        print(msg, flush=True)


def unwrap_model(model):
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


# ---------------------------------------------------------------------------
# Estimate data size for streaming
# ---------------------------------------------------------------------------

def estimate_tokens_from_manifest(data_path: str):
    """Estimate total tokens from manifest or file size."""
    p = Path(data_path)
    if p.is_dir():
        manifest = p / "_manifest.json"
        if manifest.exists():
            with open(manifest) as f:
                m = json.load(f)
            if "estimated_total_tokens" in m:
                return m["estimated_total_tokens"]
            total_rows = m.get("total_rows", 0)
            return total_rows * 2300
        # Fallback: estimate from total file size (~0.165 tokens per byte for JSONL)
        total_bytes = sum(f.stat().st_size for f in p.glob("*.jsonl"))
        if total_bytes > 0:
            return int(total_bytes * 0.165)
    elif p.is_file():
        return int(p.stat().st_size * 0.165)
    return 0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config_path: str):
    experiment_id, mcfg, tcfg = load_config(config_path)
    rank, local_rank, world_size = setup_distributed()

    random.seed(tcfg.seed)
    torch.manual_seed(tcfg.seed)
    torch.cuda.manual_seed_all(tcfg.seed)

    device = torch.device("cuda", local_rank)
    pt_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}[tcfg.dtype]

    print0(f"\n{'='*60}", rank)
    print0(f" Experiment: {experiment_id}", rank)
    print0(f" Config:     {config_path}", rank)
    print0(f" Parallelism: {'FSDP2' if tcfg.use_fsdp else 'DDP'} + "
           f"{'compile' if tcfg.compile_model else 'no-compile'}", rank)
    if mcfg.num_experts > 0:
        print0(f" MoE: {mcfg.num_experts} experts, top-{mcfg.num_experts_per_tok}", rank)
    print0(f"{'='*60}\n", rank)

    # ---- Tokenizer ----
    tok_path = os.path.join(tcfg.tokenizer_dir, "tokenizer.json")
    if rank == 0:
        if os.path.exists(tok_path):
            print(f"Loading existing tokenizer from {tcfg.tokenizer_dir}")
        else:
            print(f"Training new {mcfg.vocab_size}-token BPE tokenizer ...")
            train_tokenizer(tcfg.data_path, mcfg.vocab_size, tcfg.tokenizer_dir)
    if world_size > 1:
        dist.barrier()
    tokenizer = load_tokenizer(tcfg.tokenizer_dir)

    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    actual_vocab = tokenizer.get_vocab_size()
    assert actual_vocab == mcfg.vocab_size, (
        f"Tokenizer vocab {actual_vocab} != model vocab {mcfg.vocab_size}")
    print0(f"Tokenizer: vocab={actual_vocab}, eos_id={eos_id}, pad_id={pad_id}", rank)

    # ---- Dataset ----
    data_path = Path(tcfg.data_path)
    use_streaming = data_path.is_dir() and len(list(data_path.glob("*.jsonl"))) > 10

    if use_streaming:
        print0("Using streaming data loader (sharded dataset)", rank)
        estimated_tokens = estimate_tokens_from_manifest(tcfg.data_path)
        if tcfg.max_tokens > 0:
            estimated_tokens = min(estimated_tokens, tcfg.max_tokens)
        print0(f"Estimated tokens: ~{estimated_tokens:,}", rank)

        seqs_per_epoch = estimated_tokens // mcfg.max_position_embeddings
        steps_per_epoch = seqs_per_epoch // (tcfg.batch_size * tcfg.gradient_accumulation_steps * world_size)
        total_steps = steps_per_epoch * tcfg.num_epochs
        warmup_steps = int(total_steps * tcfg.warmup_fraction)
        tokens_per_step = (tcfg.batch_size * tcfg.gradient_accumulation_steps
                           * world_size * (mcfg.max_position_embeddings - 1))
        total_tokens = estimated_tokens
        dataloader = None  # created per-epoch
    else:
        print0("Tokenizing dataset (in-memory) ...", rank)
        all_ids = load_and_tokenize(tcfg.data_path, tokenizer,
                                    mcfg.max_position_embeddings, eos_id, rank)
        if tcfg.max_tokens > 0 and len(all_ids) > tcfg.max_tokens:
            print0(f"Truncating {len(all_ids):,} tokens to max_tokens={tcfg.max_tokens:,}", rank)
            all_ids = all_ids[:tcfg.max_tokens]
        total_tokens = len(all_ids)
        print0(f"Total tokens: {total_tokens:,}", rank)
        dataset = PretrainDataset(all_ids, mcfg.max_position_embeddings)

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=tcfg.seed)
        dataloader = DataLoader(
            dataset, batch_size=tcfg.batch_size, sampler=sampler,
            num_workers=4, pin_memory=True, drop_last=True,
        )
        steps_per_epoch = len(dataloader) // tcfg.gradient_accumulation_steps
        total_steps = steps_per_epoch * tcfg.num_epochs
        warmup_steps = int(total_steps * tcfg.warmup_fraction)
        tokens_per_step = (tcfg.batch_size * tcfg.gradient_accumulation_steps
                           * world_size * (mcfg.max_position_embeddings - 1))

    print0(f"Steps/epoch: ~{steps_per_epoch:,}, Total steps: ~{total_steps:,}", rank)
    print0(f"Warmup: {warmup_steps:,} steps", rank)
    print0(f"Tok/step: {tokens_per_step:,}", rank)

    # ---- Model ----
    print0("Initializing model ...", rank)
    model = NemotronNano(mcfg).to(device)
    param_info = None
    if rank == 0:
        param_info = model.count_parameters()
        print(f"Parameters: {param_info['total']:,} total, "
              f"{param_info['active_per_token']:,} active/token "
              f"({param_info['embedding']:,} embed + {param_info['non_embedding']:,} non-embed)")

    # ---- Resume from checkpoint (before FSDP/compile wrapping) ----
    resume_step = 0
    resume_epoch = 0
    best_loss = float("inf")
    if tcfg.resume_from and os.path.isfile(tcfg.resume_from):
        print0(f"Resuming from checkpoint: {tcfg.resume_from}", rank)
        ckpt = torch.load(tcfg.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        resume_step = ckpt.get("global_step", 0)
        resume_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("loss", float("inf"))
        del ckpt
        print0(f"  Resumed at step {resume_step}, epoch {resume_epoch}, loss {best_loss:.4f}", rank)

    # ---- Parallelism ----
    if tcfg.use_fsdp and world_size > 1:
        from torch.distributed.fsdp import fully_shard
        print0("Applying FSDP2 (fully_shard) ...", rank)
        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)
    elif world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])

    if tcfg.compile_model and hasattr(torch, "compile"):
        print0("Compiling model with torch.compile ...", rank)
        model = torch.compile(model)

    # ---- Optimizer ----
    no_decay = {"bias", "norm", "layernorm"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n.lower() for nd in no_decay) and p.requires_grad],
         "weight_decay": tcfg.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n.lower() for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=tcfg.learning_rate,
        betas=(tcfg.adam_beta1, tcfg.adam_beta2), eps=tcfg.adam_eps,
    )

    # ---- W&B ----
    use_wandb = False
    if rank == 0:
        try:
            import wandb
            run_name = tcfg.wandb_run_name or experiment_id
            wandb.init(project=tcfg.wandb_project, name=run_name, config={
                "experiment_id": experiment_id,
                "model": asdict(mcfg),
                "train": asdict(tcfg),
                "param_info": param_info,
                "world_size": world_size,
                "total_tokens": total_tokens,
            })
            use_wandb = True
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without logging")

    # ---- Output dirs ----
    output_dir = Path(tcfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        shutil.copy2(config_path, output_dir / "config.yaml")

    # ---- Training ----
    global_step = resume_step

    skip_micro_steps = resume_step * tcfg.gradient_accumulation_steps if resume_step > 0 else 0

    for epoch in range(tcfg.num_epochs):
        if use_streaming:
            stream_ds = StreamingPretrainDataset(
                tcfg.data_path, tokenizer, mcfg.max_position_embeddings,
                eos_id, rank, world_size, tcfg.seed, epoch, tcfg.max_tokens,
            )
            epoch_loader = DataLoader(stream_ds, batch_size=tcfg.batch_size,
                                      num_workers=4, pin_memory=True)
        else:
            sampler.set_epoch(epoch)
            epoch_loader = dataloader

        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        num_steps_this_epoch = 0
        micro_step = 0
        t0 = time.time()

        # Use an iterator so we can detect exhaustion and keep all ranks in sync.
        # FSDP forward/backward are collective ops, so every rank must execute
        # the same number of forward passes to avoid NCCL deadlocks.
        data_iter = iter(epoch_loader)
        data_exhausted = False

        while True:
            # Each rank tries to get a batch
            try:
                if data_exhausted:
                    raise StopIteration
                inputs, targets = next(data_iter)
                has_data = torch.ones(1, device=device, dtype=torch.int32)
            except StopIteration:
                has_data = torch.zeros(1, device=device, dtype=torch.int32)
                # Create dummy batch so we can still participate in the
                # collective forward/backward if other ranks still have data
                inputs = torch.zeros(tcfg.batch_size, mcfg.max_position_embeddings - 1,
                                     device=device, dtype=torch.long)
                targets = torch.zeros_like(inputs)

            # All-reduce to check if ANY rank still has data
            if world_size > 1:
                global_has_data = has_data.clone()
                dist.all_reduce(global_has_data, op=dist.ReduceOp.SUM)
                anyone_has_data = global_has_data.item() > 0
            else:
                anyone_has_data = has_data.item() > 0

            if not anyone_has_data:
                break

            if global_step >= total_steps:
                break

            if has_data.item() == 0:
                data_exhausted = True

            # Fast-forward past already-trained steps when resuming
            if skip_micro_steps > 0:
                skip_micro_steps -= 1
                micro_step += 1
                if micro_step % tcfg.gradient_accumulation_steps == 0 and rank == 0 and (micro_step // tcfg.gradient_accumulation_steps) % 5000 == 0:
                    print(f"  Skipping (resume): {micro_step // tcfg.gradient_accumulation_steps} steps fast-forwarded...")
                continue

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=pt_dtype):
                logits, aux_loss = model(inputs)
                ce_loss = F.cross_entropy(logits.view(-1, mcfg.vocab_size), targets.view(-1))
                loss = (ce_loss + aux_loss) / tcfg.gradient_accumulation_steps

            # Skip gradient update for dummy batches
            if has_data.item() > 0:
                loss.backward()
            else:
                loss.backward()
                optimizer.zero_grad(set_to_none=True)

            micro_step += 1

            if micro_step % tcfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)

                lr = get_lr(global_step, total_steps, warmup_steps, tcfg.learning_rate)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                step_loss = loss.item() * tcfg.gradient_accumulation_steps
                epoch_loss += step_loss
                epoch_tokens += tokens_per_step
                num_steps_this_epoch += 1
                global_step += 1

                if global_step % tcfg.log_interval == 0 and rank == 0:
                    elapsed = time.time() - t0
                    tok_per_sec = epoch_tokens / elapsed
                    ppl = math.exp(min(step_loss, 20.0))
                    mfu = (tok_per_sec * 6 * param_info["active_per_token"]) / (world_size * 989e12) * 100
                    print(f"  step {global_step:>6d}/{total_steps} | "
                          f"loss {step_loss:.4f} | ppl {ppl:.2f} | "
                          f"lr {lr:.2e} | {tok_per_sec:.0f} tok/s | MFU {mfu:.1f}%")
                    if use_wandb:
                        import wandb
                        wandb.log({
                            "train/loss": step_loss,
                            "train/ce_loss": ce_loss.item(),
                            "train/perplexity": ppl,
                            "train/lr": lr,
                            "train/tokens_per_sec": tok_per_sec,
                            "train/mfu": mfu,
                            "train/global_step": global_step,
                            "train/epoch": epoch + micro_step / max(steps_per_epoch * tcfg.gradient_accumulation_steps, 1),
                        }, step=global_step)

                if tcfg.save_interval_steps > 0 and global_step % tcfg.save_interval_steps == 0:
                    _gather_and_save_checkpoint(
                        model, optimizer, mcfg, tcfg, experiment_id,
                        epoch + 1, global_step, step_loss, output_dir,
                        "latest.pt", rank, tcfg.use_fsdp and world_size > 1)

        avg_loss = epoch_loss / max(num_steps_this_epoch, 1)
        avg_ppl = math.exp(min(avg_loss, 20.0))
        print0(f"Epoch {epoch+1}/{tcfg.num_epochs} -- avg loss {avg_loss:.4f}, ppl {avg_ppl:.2f}", rank)

        if avg_loss < best_loss:
            best_loss = avg_loss
            _gather_and_save_checkpoint(
                model, None, mcfg, tcfg, experiment_id,
                epoch + 1, global_step, avg_loss, output_dir,
                "best.pt", rank, tcfg.use_fsdp and world_size > 1)
            if rank == 0:
                print(f"  New best model (loss {best_loss:.4f})")

        if world_size > 1:
            dist.barrier()

    print0(f"\nTraining complete. Best loss: {best_loss:.4f}", rank)
    print0(f"Checkpoints in: {output_dir}", rank)
    if use_wandb:
        import wandb
        wandb.finish()
    cleanup_distributed()


def _gather_and_save_checkpoint(model, optimizer, mcfg, tcfg, experiment_id,
                                epoch, global_step, loss, output_dir, filename,
                                rank, use_fsdp):
    """Gather full state dict across all FSDP ranks and save on rank 0."""
    if use_fsdp:
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict, StateDictOptions)
        cpu_sd = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    else:
        raw = unwrap_model(model)
        cpu_sd = {k: v.cpu() for k, v in raw.state_dict().items()}

    if rank == 0:
        ckpt = {
            "experiment_id": experiment_id,
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": cpu_sd,
            "model_config": asdict(mcfg),
            "train_config": asdict(tcfg),
            "loss": loss,
        }
        path = Path(output_dir) / filename
        tmp_path = path.with_suffix(".pt.tmp")
        try:
            torch.save(ckpt, tmp_path)
            tmp_path.rename(path)
            print(f"  Saved checkpoint -> {path}")
        except Exception as e:
            print(f"  WARNING: Checkpoint save failed ({e}), removing partial file")
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config-driven pre-training")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)
