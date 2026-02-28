"""
pretrain.py — Pre-train a scaled-down Nemotron-Nano-V3-style model from scratch.

Architecture
============
Family       : NVIDIA Nemotron-Nano-V3 (dense decoder-only transformer variant)
               The full Nemotron-Nano-V3 is a hybrid Mamba-Transformer MoE with
               52 layers (23 Mamba-2 + 23 MoE + 6 GQA), hidden=4096, vocab=131072.
               At our data scale we use a dense transformer that preserves the key
               architectural choices of the family:
                 - Grouped-Query Attention (GQA, 4:1 ratio)
                 - RMSNorm (pre-norm)
                 - SwiGLU MLP (gate · up, then down)
                 - Rotary Position Embeddings (RoPE)
                 - Tied input/output embeddings

Depth / Width
=============
  hidden_size       = 128
  num_layers        = 8
  num_attn_heads    = 8   (head_dim = 16)
  num_kv_heads      = 2   (GQA ratio 4:1, same as full model)
  intermediate_size = 512  (4× hidden, SwiGLU)
  max_position      = 2048

  Non-embedding params : ~1.90 M
  Embedding params     : ~2.10 M  (16 384 × 128, tied with LM head)
  Total params         : ~4.00 M

Tokenizer
=========
  Type       : BPE (trained from scratch with HuggingFace `tokenizers`)
  Vocab size : 16 384
  Trained on : The hybrid pre-training corpus itself (29 074 documents,
               ~24K real FineWiki + ~5K synthetic Data Designer rephrased)
  Settings   : byte-level BPE, min_frequency=2, special tokens=[<|pad|>,
               <|eos|>, <|unk|>], no pre-existing merges

  Rationale  : A 16K vocab balances compression efficiency against embedding
               table size. With only ~26M tokens of training data, a 131K
               vocab (Nemotron-Nano-V3 default) would allocate 88% of model
               parameters to an embedding table whose rows are mostly unseen.
               16K keeps ~48% of params in the transformer layers.

Data
====
  File   : /home/jason/ee194-a2/partb_data_designer_exports/partb_hybrid_training_data.jsonl
  Split  : ~24 000 real (FineWiki) + ~5 000 synthetic (Data Designer rephrased)
  Tokens : ~26 M (measured with the 16K BPE tokenizer)

Training stack
==============
  Framework    : PyTorch 2.9 + HuggingFace tokenizers
  Parallelism  : DDP across 8× NVIDIA H100 80 GB (data parallel)
  Precision    : bfloat16 (torch.amp autocast)
  Optimizer    : AdamW (β1=0.9, β2=0.95, eps=1e-8, weight_decay=0.1)
  Schedule     : Linear warmup (5% of steps) → cosine decay to 0
  Epochs       : 3 (Chinchilla-optimal: ~78M tokens seen / ~4M params ≈ 19.5 tok/param)
  Logging      : Weights & Biases (rank 0 only)

Launch
======
  torchrun --nproc_per_node=8 pretrain.py
"""

import os
import json
import math
import time
import random
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOKENIZER_DIR = "tokenizer_16k"

@dataclass
class ModelConfig:
    vocab_size: int = 16_384
    hidden_size: int = 128
    num_layers: int = 8
    num_attention_heads: int = 8        # head_dim = 16
    num_kv_heads: int = 2               # GQA 4:1
    intermediate_size: int = 512        # 4× hidden (SwiGLU)
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

@dataclass
class TrainConfig:
    data_path: str = "/home/jason/ee194-a2/partb_data_designer_exports/partb_hybrid_training_data.jsonl"
    output_dir: str = "checkpoints"
    tokenizer_dir: str = TOKENIZER_DIR

    # Per-GPU batch size; global batch = batch_size * grad_accum * world_size
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    # With 8 GPUs: global_batch = 8 * 1 * 8 = 64 sequences/step (~131K tokens)
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

    log_interval: int = 10
    save_interval_epochs: int = 1
    wandb_project: str = "ee194-a2-pretrain"
    wandb_run_name: str = "nemotron-nano-v3-micro"
    seed: int = 42


# ---------------------------------------------------------------------------
# Tokenizer training
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

    tokenizer.train_from_iterator(doc_iterator(), trainer=trainer, length=29_074)

    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

    test = tokenizer.encode("Hello world! This is a test.")
    print(f"Tokenizer trained: vocab_size={tokenizer.get_vocab_size()}, "
          f"test encode length={len(test.ids)}")
    print(f"Saved to {save_dir}/tokenizer.json")
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


class NemotronNanoMicro(nn.Module):
    """Scaled-down Nemotron-Nano-V3-style causal language model."""

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


def load_and_tokenize(data_path: str, tokenizer, seq_len: int, eos_id: int):
    """Load JSONL, tokenize all documents, concatenate with EOS separators."""
    all_ids = []
    with open(data_path) as f:
        for line in f:
            doc = json.loads(line)
            ids = tokenizer.encode(doc["text"]).ids
            all_ids.extend(ids)
            all_ids.append(eos_id)
    return all_ids


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup → cosine decay
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
    """Initialize DDP. Returns (rank, local_rank, world_size). Falls back to
    single-GPU if torchrun env vars are not set."""
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


def is_main(rank: int) -> bool:
    return rank == 0


def print0(msg: str, rank: int):
    """Print only on rank 0."""
    if is_main(rank):
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    tcfg = TrainConfig()
    mcfg = ModelConfig()

    rank, local_rank, world_size = setup_distributed()

    # Seed each rank differently for data shuffling, but same model init
    random.seed(tcfg.seed)
    torch.manual_seed(tcfg.seed)
    torch.cuda.manual_seed_all(tcfg.seed)

    device = torch.device("cuda", local_rank)
    pt_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}[tcfg.dtype]

    # ---- Tokenizer (rank 0 trains, others wait) ----
    tok_path = os.path.join(tcfg.tokenizer_dir, "tokenizer.json")
    if is_main(rank):
        if os.path.exists(tok_path):
            print(f"Loading existing tokenizer from {tcfg.tokenizer_dir} …")
        else:
            print(f"Training new {mcfg.vocab_size}-token BPE tokenizer …")
            train_tokenizer(tcfg.data_path, mcfg.vocab_size, tcfg.tokenizer_dir)
    if world_size > 1:
        dist.barrier()
    tokenizer = load_tokenizer(tcfg.tokenizer_dir)

    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    actual_vocab = tokenizer.get_vocab_size()
    assert actual_vocab == mcfg.vocab_size, (
        f"Tokenizer vocab {actual_vocab} != model vocab {mcfg.vocab_size}")
    print0(f"Tokenizer ready: vocab={actual_vocab}, eos_id={eos_id}, pad_id={pad_id}", rank)

    # ---- Dataset (every rank tokenizes — fast enough, avoids large tensor broadcast) ----
    print0("Tokenizing dataset …", rank)
    all_ids = load_and_tokenize(tcfg.data_path, tokenizer,
                                mcfg.max_position_embeddings, eos_id)
    print0(f"Total tokens: {len(all_ids):,}", rank)
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
    # Global tokens processed per optimizer step
    tokens_per_step = (tcfg.batch_size * tcfg.gradient_accumulation_steps
                       * world_size * (mcfg.max_position_embeddings - 1))

    print0(f"Dataset : {len(dataset):,} sequences of length {mcfg.max_position_embeddings}", rank)
    print0(f"DDP     : {world_size} GPUs, {tcfg.batch_size}/GPU × "
           f"{tcfg.gradient_accumulation_steps} accum = "
           f"{tcfg.batch_size * tcfg.gradient_accumulation_steps * world_size} global batch", rank)
    print0(f"Steps   : {total_steps:,} total ({steps_per_epoch}/epoch × {tcfg.num_epochs} epochs)", rank)
    print0(f"Warmup  : {warmup_steps:,} steps", rank)
    print0(f"Tok/step: {tokens_per_step:,}", rank)

    # ---- Model ----
    print0("Initializing model …", rank)
    model = NemotronNanoMicro(mcfg).to(device)
    if is_main(rank):
        param_info = model.count_parameters()
        print(f"Parameters: {param_info['total_with_tying']:,} total "
              f"({param_info['embedding']:,} embed + {param_info['non_embedding']:,} non-embed)")

    if tcfg.compile_model and hasattr(torch, "compile"):
        print0("Compiling model with torch.compile …", rank)
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
    if is_main(rank):
        try:
            import wandb
            wandb.init(project=tcfg.wandb_project, name=tcfg.wandb_run_name, config={
                "model": mcfg.__dict__, "train": tcfg.__dict__,
                "param_info": param_info, "world_size": world_size,
            })
            use_wandb = True
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without logging")

    # ---- Training ----
    os.makedirs(tcfg.output_dir, exist_ok=True)
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

            # Skip gradient all-reduce on non-sync micro-steps for efficiency
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

                if global_step % tcfg.log_interval == 0 and is_main(rank):
                    elapsed = time.time() - t0
                    tok_per_sec = epoch_tokens / elapsed
                    ppl = math.exp(min(step_loss, 20.0))
                    print(f"  step {global_step:>6d}/{total_steps} | "
                          f"loss {step_loss:.4f} | ppl {ppl:.2f} | "
                          f"lr {lr:.2e} | {tok_per_sec:.0f} tok/s")
                    if use_wandb:
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
        print0(f"Epoch {epoch+1}/{tcfg.num_epochs} — avg loss {avg_loss:.4f}, ppl {avg_ppl:.2f}", rank)

        # Checkpointing (rank 0 only)
        if is_main(rank) and (epoch + 1) % tcfg.save_interval_epochs == 0:
            raw = unwrap_model(model)
            ckpt_path = Path(tcfg.output_dir) / f"epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": raw.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": mcfg.__dict__,
                "train_config": tcfg.__dict__,
                "loss": avg_loss,
            }, ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

        if avg_loss < best_loss and is_main(rank):
            best_loss = avg_loss
            raw = unwrap_model(model)
            best_path = Path(tcfg.output_dir) / "best.pt"
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": raw.state_dict(),
                "model_config": mcfg.__dict__,
                "loss": avg_loss,
            }, best_path)
            print(f"  New best model → {best_path} (loss {best_loss:.4f})")

        if world_size > 1:
            dist.barrier()

    print0(f"\nTraining complete. Best loss: {best_loss:.4f}", rank)
    if use_wandb:
        import wandb
        wandb.finish()
    cleanup_distributed()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def unwrap_model(model):
    """Get the raw model from DDP / torch.compile wrappers."""
    m = model
    if hasattr(m, "module"):        # DDP
        m = m.module
    if hasattr(m, "_orig_mod"):     # torch.compile
        m = m._orig_mod
    return m


class _NullContext:
    """Minimal no-op context manager."""
    def __enter__(self): return self
    def __exit__(self, *args): pass


def open_no_sync(model, micro_step, grad_accum, world_size):
    """Return model.no_sync() on non-sync micro-steps under DDP, else no-op."""
    is_sync_step = (micro_step + 1) % grad_accum == 0
    if world_size > 1 and not is_sync_step:
        return model.no_sync()
    return _NullContext()


if __name__ == "__main__":
    train()
