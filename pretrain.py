"""
pretrain.py — Config-driven pre-training for Nemotron-Nano-V3-style models.

Reads a YAML config file that specifies model architecture, training
hyperparameters, and data paths. This single script handles all model sizes
from the 4M baseline to the 57M scaled variant.

Architecture family: NVIDIA Nemotron-Nano-V3 (dense decoder-only transformer)
  - Grouped-Query Attention (GQA)
  - RMSNorm (pre-norm)
  - SwiGLU MLP (gate · up, then down)
  - Rotary Position Embeddings (RoPE)
  - Tied input/output embeddings

Usage:
  torchrun --nproc_per_node=8 pretrain.py --config configs/baseline_4M.yaml
  torchrun --nproc_per_node=8 pretrain.py --config configs/scaled_38M.yaml
  torchrun --nproc_per_node=8 pretrain.py --config configs/scaled_57M.yaml
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# ---------------------------------------------------------------------------
# Configuration (loaded from YAML)
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
    num_epochs: int = 3

    dtype: str = "bfloat16"
    compile_model: bool = True

    log_interval: int = 50
    save_interval_epochs: int = 1
    wandb_project: str = "ee194-a2-pretrain"
    wandb_run_name: str = ""
    seed: int = 42


def load_config(config_path: str) -> tuple[str, ModelConfig, TrainConfig]:
    """Load experiment config from YAML. Returns (experiment_id, mcfg, tcfg)."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    experiment_id = raw.get("experiment_id", Path(config_path).stem)
    mcfg = ModelConfig(**raw.get("model", {}))
    tcfg = TrainConfig(**raw.get("train", {}))
    return experiment_id, mcfg, tcfg


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|eos|>"
UNK_TOKEN = "<|unk|>"
SPECIAL_TOKENS = [PAD_TOKEN, EOS_TOKEN, UNK_TOKEN]

def train_tokenizer(data_path: str, vocab_size: int, save_dir: str):
    """Train a byte-level BPE tokenizer on the pre-training corpus."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    def doc_iterator():
        with open(data_path) as f:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(dim: int, max_seq_len: int, theta: float = 10_000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """x shape: (B, n_heads, T, head_dim)."""
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos = cos[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class GQAttention(nn.Module):
    """Grouped-Query Attention with RoPE."""

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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn)


class SwiGLUMLP(nn.Module):
    """SwiGLU: out = down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = GQAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLUMLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class NemotronNano(nn.Module):
    """Config-driven Nemotron-Nano-V3-style causal language model."""

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
            cfg.max_position_embeddings,
            cfg.rope_theta,
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

    def forward(self, input_ids: torch.Tensor):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        x = self.norm(x)
        return self.lm_head(x)

    def count_parameters(self):
        seen = {}
        for p in self.parameters():
            pid = id(p)
            if pid not in seen:
                seen[pid] = p.numel()
        unique = sum(seen.values())
        embed = self.embed_tokens.weight.numel()
        return {"total_with_tying": unique, "embedding": embed,
                "non_embedding": unique - embed}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PretrainDataset(Dataset):
    """Packs tokenized documents into fixed-length chunks for causal LM training."""

    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        n = len(token_ids) // seq_len
        self.data = torch.tensor(token_ids[: n * seq_len], dtype=torch.long).view(n, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]


def _iter_jsonl_files(data_path: str):
    """Yield file paths: if data_path is a file, yield it; if a directory,
    yield all .jsonl files sorted by name."""
    p = Path(data_path)
    if p.is_file():
        yield p
    elif p.is_dir():
        for f in sorted(p.glob("*.jsonl")):
            yield f
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")


def load_and_tokenize(data_path: str, tokenizer, seq_len: int, eos_id: int, rank: int):
    """Load JSONL file(s), tokenize all documents, concatenate with EOS separators.
    data_path can be a single .jsonl file or a directory of .jsonl shards."""
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
                    print(f"  Tokenized {count:,} docs ({len(all_ids):,} tokens)...",
                          flush=True)
    return all_ids


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup -> cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float) -> float:
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


def print0(msg: str, rank: int):
    if rank == 0:
        print(msg, flush=True)


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *args): pass


def open_no_sync(model, micro_step, grad_accum, world_size):
    is_sync_step = (micro_step + 1) % grad_accum == 0
    if world_size > 1 and not is_sync_step:
        return model.no_sync()
    return _NullContext()


def unwrap_model(model):
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


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
    print0(f"{'='*60}\n", rank)

    # ---- Tokenizer (rank 0 trains if needed, others wait) ----
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
    print0("Tokenizing dataset ...", rank)
    all_ids = load_and_tokenize(tcfg.data_path, tokenizer,
                                mcfg.max_position_embeddings, eos_id, rank)
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

    print0(f"Dataset : {len(dataset):,} sequences of length {mcfg.max_position_embeddings}", rank)
    print0(f"DDP     : {world_size} GPUs, {tcfg.batch_size}/GPU x "
           f"{tcfg.gradient_accumulation_steps} accum = "
           f"{tcfg.batch_size * tcfg.gradient_accumulation_steps * world_size} global batch", rank)
    print0(f"Steps   : {total_steps:,} total ({steps_per_epoch}/epoch x {tcfg.num_epochs} epochs)", rank)
    print0(f"Warmup  : {warmup_steps:,} steps", rank)
    print0(f"Tok/step: {tokens_per_step:,}", rank)

    # ---- Model ----
    print0("Initializing model ...", rank)
    model = NemotronNano(mcfg).to(device)
    param_info = None
    if rank == 0:
        param_info = model.count_parameters()
        print(f"Parameters: {param_info['total_with_tying']:,} total "
              f"({param_info['embedding']:,} embed + {param_info['non_embedding']:,} non-embed)")

    if tcfg.compile_model and hasattr(torch, "compile"):
        print0("Compiling model with torch.compile ...", rank)
        model = torch.compile(model)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

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
    scaler = torch.amp.GradScaler(enabled=(pt_dtype == torch.float16))

    # ---- W&B (rank 0 only) ----
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

    # Copy config file into the run directory for reproducibility
    if rank == 0:
        shutil.copy2(config_path, output_dir / "config.yaml")

    # ---- Training ----
    global_step = 0
    best_loss = float("inf")

    for epoch in range(tcfg.num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        num_steps_this_epoch = 0
        t0 = time.time()

        for micro_step, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with open_no_sync(model, micro_step, tcfg.gradient_accumulation_steps, world_size):
                with torch.amp.autocast("cuda", dtype=pt_dtype):
                    logits = model(inputs)
                    loss = F.cross_entropy(logits.view(-1, mcfg.vocab_size), targets.view(-1))
                    loss = loss / tcfg.gradient_accumulation_steps

                scaler.scale(loss).backward()

            if (micro_step + 1) % tcfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)

                lr = get_lr(global_step, total_steps, warmup_steps, tcfg.learning_rate)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.step(optimizer)
                scaler.update()
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
                    print(f"  step {global_step:>6d}/{total_steps} | "
                          f"loss {step_loss:.4f} | ppl {ppl:.2f} | "
                          f"lr {lr:.2e} | {tok_per_sec:.0f} tok/s")
                    if use_wandb:
                        import wandb
                        wandb.log({
                            "train/loss": step_loss,
                            "train/perplexity": ppl,
                            "train/lr": lr,
                            "train/tokens_per_sec": tok_per_sec,
                            "train/global_step": global_step,
                            "train/epoch": epoch + (micro_step + 1) / len(dataloader),
                        }, step=global_step)

        avg_loss = epoch_loss / max(num_steps_this_epoch, 1)
        avg_ppl = math.exp(min(avg_loss, 20.0))
        print0(f"Epoch {epoch+1}/{tcfg.num_epochs} -- avg loss {avg_loss:.4f}, ppl {avg_ppl:.2f}", rank)

        # Checkpointing (rank 0 only)
        if rank == 0 and (epoch + 1) % tcfg.save_interval_epochs == 0:
            raw = unwrap_model(model)
            ckpt_path = output_dir / f"epoch_{epoch+1}.pt"
            torch.save({
                "experiment_id": experiment_id,
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": raw.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": asdict(mcfg),
                "train_config": asdict(tcfg),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"  Saved checkpoint -> {ckpt_path}")

        if avg_loss < best_loss and rank == 0:
            best_loss = avg_loss
            raw = unwrap_model(model)
            best_path = output_dir / "best.pt"
            torch.save({
                "experiment_id": experiment_id,
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": raw.state_dict(),
                "model_config": asdict(mcfg),
                "train_config": asdict(tcfg),
                "loss": avg_loss,
            }, best_path)
            print(f"  New best model -> {best_path} (loss {best_loss:.4f})")

        if world_size > 1:
            dist.barrier()

    print0(f"\nTraining complete. Best loss: {best_loss:.4f}", rank)
    print0(f"Checkpoints in: {output_dir}", rank)
    if use_wandb:
        import wandb
        wandb.finish()
    cleanup_distributed()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config-driven pre-training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file (e.g. configs/scaled_38M.yaml)")
    args = parser.parse_args()
    train(args.config)
