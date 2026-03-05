"""
benchmark_strategies.py — MFU benchmark of training strategies on 8x H100.

Computes Model FLOPS Utilization (MFU) for each strategy:
  MFU = actual_flops / (peak_flops * time)

H100 SXM bf16 peak: 989 TFLOPS/GPU
Transformer FLOPS per token (forward+backward): 6 * num_params * seq_len
  (using the standard 6N approximation: 2N forward, 4N backward)

Usage:
  torchrun --nproc_per_node=8 benchmark_strategies.py
"""

import os
import gc
import time
import json
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

H100_BF16_PEAK_TFLOPS = 989.0

# ---------------------------------------------------------------------------
# Model config — ~500M params
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
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
    x1, x2 = x[..., :hd//2], x[..., hd//2:]
    cos = cos[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)


class GQAttention(nn.Module):
    def __init__(self, hidden, nheads, nkv, head_dim):
        super().__init__()
        self.num_heads = nheads
        self.num_kv_heads = nkv
        self.head_dim = head_dim
        self.num_groups = nheads // nkv
        self.q_proj = nn.Linear(hidden, nheads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, nkv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, nkv * head_dim, bias=False)
        self.o_proj = nn.Linear(nheads * head_dim, hidden, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).unflatten(-1, (-1, self.head_dim)).transpose(1, 2)
        k = self.k_proj(x).unflatten(-1, (-1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).unflatten(-1, (-1, self.head_dim)).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        nq, nk = q.shape[1], k.shape[1]
        if nq != nk:
            k = k.repeat_interleave(nq // nk, dim=1)
            v = v.repeat_interleave(nq // nk, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(attn.transpose(1, 2).contiguous().flatten(2))


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, hidden, nheads, nkv, head_dim, intermediate, eps):
        super().__init__()
        self.attn_norm = RMSNorm(hidden, eps)
        self.attn = GQAttention(hidden, nheads, nkv, head_dim)
        self.mlp_norm = RMSNorm(hidden, eps)
        self.mlp = SwiGLUMLP(hidden, intermediate)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Model(nn.Module):
    def __init__(self, vocab=16384, hidden=1024, layers=24, nheads=16, nkv=4,
                 intermediate=4096, max_pos=2048, eps=1e-5, theta=10000.0):
        super().__init__()
        self.vocab = vocab
        self.hidden = hidden
        head_dim = hidden // nheads
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([
            Block(hidden, nheads, nkv, head_dim, intermediate, eps)
            for _ in range(layers)
        ])
        self.norm = RMSNorm(hidden, eps)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.head.weight = self.embed.weight
        rc, rs = precompute_rope(head_dim, max_pos, theta)
        self.register_buffer("rc", rc, persistent=False)
        self.register_buffer("rs", rs, persistent=False)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h, self.rc, self.rs)
        return self.head(self.norm(h))


def count_params(model):
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


# ---------------------------------------------------------------------------
# Liger-kernel patching
# ---------------------------------------------------------------------------

def patch_liger(model):
    from liger_kernel.transformers.rms_norm import LigerRMSNorm

    for block in model.layers:
        for attr in ("attn_norm", "mlp_norm"):
            old = getattr(block, attr)
            new = LigerRMSNorm(old.weight.shape[0], eps=old.eps)
            new.weight.data.copy_(old.weight.data)
            setattr(block, attr, new)

    old_norm = model.norm
    new_norm = LigerRMSNorm(old_norm.weight.shape[0], eps=old_norm.eps)
    new_norm.weight.data.copy_(old_norm.weight.data)
    model.norm = new_norm
    return model


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

class FakeData(Dataset):
    def __init__(self, n, seq_len, vocab):
        self.n, self.seq_len, self.vocab = n, seq_len, vocab

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        t = torch.randint(0, self.vocab, (self.seq_len,), generator=g)
        return t[:-1], t[1:]


# ---------------------------------------------------------------------------
# MFU calculation
# ---------------------------------------------------------------------------

def compute_mfu(tokens_per_sec_global, num_params, num_gpus):
    """
    MFU = achieved_flops / peak_flops
    Transformer training FLOPS per token ≈ 6 * N (forward 2N + backward 4N)
    """
    flops_per_token = 6 * num_params
    achieved_flops = tokens_per_sec_global * flops_per_token
    peak_flops = num_gpus * H100_BF16_PEAK_TFLOPS * 1e12
    return achieved_flops / peak_flops


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def bench(model, loader, vocab, warmup=5, measure=20, grad_accum=1):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    dev = next(model.parameters()).device
    it = iter(loader)
    micro = 0

    for _ in range(warmup * grad_accum):
        try:
            inp, tgt = next(it)
        except StopIteration:
            it = iter(loader)
            inp, tgt = next(it)
        inp, tgt = inp.to(dev, non_blocking=True), tgt.to(dev, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(inp).view(-1, vocab), tgt.view(-1)) / grad_accum
        loss.backward()
        micro += 1
        if micro % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    toks = 0
    for _ in range(measure * grad_accum):
        try:
            inp, tgt = next(it)
        except StopIteration:
            it = iter(loader)
            inp, tgt = next(it)
        inp, tgt = inp.to(dev, non_blocking=True), tgt.to(dev, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(inp).view(-1, vocab), tgt.view(-1)) / grad_accum
        loss.backward()
        micro += 1
        if micro % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            toks += inp.shape[0] * inp.shape[1] * dist.get_world_size()

    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    tok_s = toks / elapsed

    del opt
    return tok_s, peak_gb, elapsed


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

def make_loader(rank, world_size, batch_size, seq_len=2048, vocab=16384):
    ds = FakeData(4096, seq_len, vocab)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      num_workers=2, pin_memory=True, drop_last=True)


def strat_ddp_compile(rank, lr, ws, bs, ga):
    cleanup()
    m = Model().cuda()
    m = torch.compile(m)
    m = DDP(m, device_ids=[lr])
    loader = make_loader(rank, ws, bs)
    r = bench(m, loader, 16384, grad_accum=ga)
    del m; cleanup()
    return r


def strat_ddp_compile_liger(rank, lr, ws, bs, ga):
    cleanup()
    m = Model().cuda()
    m = patch_liger(m)
    m = m.cuda()  # ensure all params back on GPU after liger patching
    m = torch.compile(m)
    m = DDP(m, device_ids=[lr])
    loader = make_loader(rank, ws, bs)
    r = bench(m, loader, 16384, grad_accum=ga)
    del m; cleanup()
    return r


def strat_fsdp2_compile(rank, lr, ws, bs, ga):
    from torch.distributed.fsdp import fully_shard
    cleanup()
    m = Model().cuda()
    for layer in m.layers:
        fully_shard(layer)
    fully_shard(m)
    m = torch.compile(m)
    loader = make_loader(rank, ws, bs)
    r = bench(m, loader, 16384, grad_accum=ga)
    del m; cleanup()
    return r


def strat_fsdp2_compile_liger(rank, lr, ws, bs, ga):
    from torch.distributed.fsdp import fully_shard
    cleanup()
    m = Model().cuda()
    m = patch_liger(m)
    m = m.cuda()
    for layer in m.layers:
        fully_shard(layer)
    fully_shard(m)
    m = torch.compile(m)
    loader = make_loader(rank, ws, bs)
    r = bench(m, loader, 16384, grad_accum=ga)
    del m; cleanup()
    return r


def strat_fsdp2_tp_compile_liger(rank, lr, ws, bs, ga, tp=2):
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor.parallel import (
        parallelize_module, ColwiseParallel, RowwiseParallel,
    )
    from torch.distributed.device_mesh import init_device_mesh
    cleanup()

    dp = ws // tp
    mesh = init_device_mesh("cuda", (dp, tp), mesh_dim_names=("dp", "tp"))
    m = Model().cuda()
    m = patch_liger(m)
    m = m.cuda()

    tp_mesh = mesh["tp"]
    for layer in m.layers:
        parallelize_module(layer.attn, tp_mesh, {
            "q_proj": ColwiseParallel(),
            "k_proj": ColwiseParallel(),
            "v_proj": ColwiseParallel(),
            "o_proj": RowwiseParallel(),
        })
        parallelize_module(layer.mlp, tp_mesh, {
            "gate_proj": ColwiseParallel(),
            "up_proj": ColwiseParallel(),
            "down_proj": RowwiseParallel(),
        })

    dp_mesh = mesh["dp"]
    for layer in m.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(m, mesh=dp_mesh)

    m = torch.compile(m)
    loader = make_loader(rank, ws, bs)
    r = bench(m, loader, 16384, grad_accum=ga)
    del m; cleanup()
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    lr = int(os.environ["LOCAL_RANK"])
    ws = dist.get_world_size()
    torch.cuda.set_device(lr)

    bs, ga = args.batch_size, args.grad_accum

    if rank == 0:
        nparams = count_params(Model())
        print(f"\n{'='*72}")
        print(f" MFU Benchmark: {nparams/1e6:.0f}M params | {ws} GPUs | bs={bs} ga={ga}")
        print(f" H100 bf16 peak: {H100_BF16_PEAK_TFLOPS} TFLOPS/GPU")
        print(f"{'='*72}")

    nparams = count_params(Model())

    strategies = [
        ("1. DDP + compile",                    lambda: strat_ddp_compile(rank, lr, ws, bs, ga)),
        ("2. DDP + compile + liger",            lambda: strat_ddp_compile_liger(rank, lr, ws, bs, ga)),
        ("3. FSDP2 + compile",                  lambda: strat_fsdp2_compile(rank, lr, ws, bs, ga)),
        ("4. FSDP2 + compile + liger",          lambda: strat_fsdp2_compile_liger(rank, lr, ws, bs, ga)),
        ("5. FSDP2 + TP(2) + compile + liger",  lambda: strat_fsdp2_tp_compile_liger(rank, lr, ws, bs, ga)),
    ]

    results = {}
    for name, fn in strategies:
        if rank == 0:
            print(f"\n  Running: {name} ...", flush=True)
        dist.barrier()
        try:
            tok_s, peak_gb, elapsed = fn()
            mfu = compute_mfu(tok_s, nparams, ws)
            if rank == 0:
                results[name] = {
                    "tokens_per_sec": round(tok_s),
                    "peak_memory_gb": round(peak_gb, 2),
                    "mfu": round(mfu * 100, 2),
                    "elapsed_sec": round(elapsed, 2),
                }
                print(f"    -> {tok_s:>10,.0f} tok/s | MFU {mfu*100:.1f}% | {peak_gb:.1f} GB peak", flush=True)
        except Exception as e:
            if rank == 0:
                results[name] = {"error": str(e)[:200]}
                print(f"    -> FAILED: {e}", flush=True)
            cleanup()
        dist.barrier()

    if rank == 0:
        print(f"\n{'='*72}")
        print(f" {'Strategy':<40s} {'tok/s':>10s} {'MFU':>7s} {'Mem':>7s}")
        print(f" {'-'*40} {'-'*10} {'-'*7} {'-'*7}")
        best_name, best_mfu = None, 0
        for name, r in results.items():
            if "error" in r:
                print(f" {name:<40s} {'FAILED':>10s}")
            else:
                print(f" {name:<40s} {r['tokens_per_sec']:>10,d} {r['mfu']:>6.1f}% {r['peak_memory_gb']:>5.1f}GB")
                if r["mfu"] > best_mfu:
                    best_mfu = r["mfu"]
                    best_name = name
        print(f"{'='*72}")
        if best_name:
            print(f"\n >>> WINNER: {best_name} at {best_mfu:.1f}% MFU <<<\n")

        with open("benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Saved: benchmark_results.json")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
