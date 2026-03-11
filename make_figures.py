"""Generate report figures from experiment results and training logs."""

import json
import glob
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from collections import defaultdict

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_experiments():
    exps = {}
    for f in sorted(glob.glob("experiments/*.json")):
        e = json.load(open(f))
        exps[e["experiment_id"]] = e
    return exps


def parse_train_log(path):
    steps, losses, ppls, lrs = [], [], [], []
    pattern = re.compile(
        r"step\s+(\d+)/\d+\s+\|\s+loss\s+([\d.]+)\s+\|\s+ppl\s+([\d.]+)\s+\|\s+lr\s+([\d.eE+-]+)"
    )
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                ppls.append(float(m.group(3)))
                lrs.append(float(m.group(4)))
    return {
        "steps": np.array(steps),
        "loss": np.array(losses),
        "ppl": np.array(ppls),
        "lr": np.array(lrs),
    }


def load_all_logs():
    logs = {}
    for logfile in glob.glob("runs/*/train.log"):
        name = logfile.split("/")[1]
        parsed = parse_train_log(logfile)
        if len(parsed["steps"]) > 5:
            logs[name] = parsed
    return logs


EXPS = load_experiments()
LOGS = load_all_logs()

BENCHMARKS = [
    ("hellaswag_acc_norm", "HellaSwag"),
    ("piqa_acc", "PIQA"),
    ("winogrande_acc", "WinoGrande"),
    ("boolq_acc", "BoolQ"),
    ("arc_easy_acc_norm", "ARC-Easy"),
    ("arc_challenge_acc_norm", "ARC-Challenge"),
    ("sciq_acc_norm", "SciQ"),
]

PPL_BENCHMARKS = [
    ("wikitext_word_ppl", "WikiText PPL"),
    ("lambada_ppl", "LAMBADA PPL"),
]

COLORS = {
    "equal_raw_dense":     "#1f77b4",
    "equal_curated_dense": "#ff7f0e",
    "equal_hybrid_dense":  "#2ca02c",
    "full_raw_dense":      "#d62728",
    "full_curated_dense":  "#9467bd",
    "full_raw_moe":        "#e377c2",
    "full_curated_moe":    "#8c564b",
    "dense_500M":          "#7f7f7f",
}

NICE_NAMES = {
    "equal_raw_dense":     "Raw (1.3B tok, Dense)",
    "equal_curated_dense": "Curated (1.3B tok, Dense)",
    "equal_hybrid_dense":  "Hybrid (1.3B tok, Dense)",
    "full_raw_dense":      "Raw (Full, Dense 382M)",
    "full_curated_dense":  "Curated (Full, Dense 382M)",
    "full_raw_moe":        "Raw (Full, MoE 390M/1.4B)",
    "full_curated_moe":    "Curated (Full, MoE 390M/1.4B)",
}


def get_eval(exp_id, metric):
    return EXPS[exp_id].get("eval_results", {}).get(metric, 0)


# ---------------------------------------------------------------------------
# Figure 1: Training loss curves — data quality ablation runs
# ---------------------------------------------------------------------------

def fig1_loss_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Full-scale runs (long training)
    ax = axes[0]
    full_runs = ["full_raw_dense", "full_curated_dense", "full_raw_moe", "full_curated_moe"]
    for name in full_runs:
        if name not in LOGS:
            continue
        log = LOGS[name]
        window = min(20, len(log["loss"]) // 5) or 1
        smoothed = np.convolve(log["loss"], np.ones(window)/window, mode="valid")
        steps = log["steps"][:len(smoothed)]
        ax.plot(steps, smoothed, label=NICE_NAMES.get(name, name),
                color=COLORS.get(name), linewidth=1.5, alpha=0.9)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss (smoothed)")
    ax.set_title("(a) Full-Scale Runs")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(bottom=1.5)

    # Panel B: Equal-size controlled comparison
    ax = axes[1]
    equal_runs = ["equal_raw_dense", "equal_curated_dense", "equal_hybrid_dense"]
    for name in equal_runs:
        if name not in LOGS:
            continue
        log = LOGS[name]
        window = min(5, len(log["loss"]) // 3) or 1
        smoothed = np.convolve(log["loss"], np.ones(window)/window, mode="valid")
        steps = log["steps"][:len(smoothed)]
        ax.plot(steps, smoothed, label=NICE_NAMES.get(name, name),
                color=COLORS.get(name), linewidth=1.5, alpha=0.9)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss (smoothed)")
    ax.set_title("(b) Equal-Size Comparison (1.3B tokens)")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Training Loss Curves", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig1_loss_curves.png")
    fig.savefig(f"{OUT_DIR}/fig1_loss_curves.pdf")
    plt.close(fig)
    print("  [+] fig1_loss_curves")


# ---------------------------------------------------------------------------
# Figure 2: Benchmark comparison — 7 data-quality ablation experiments
# ---------------------------------------------------------------------------

def fig2_benchmark_bars():
    ablation_ids = [
        "equal_raw_dense", "equal_curated_dense", "equal_hybrid_dense",
        "full_raw_dense", "full_curated_dense", "full_raw_moe", "full_curated_moe",
    ]
    labels = [NICE_NAMES[e] for e in ablation_ids]
    n_bench = len(BENCHMARKS)
    n_exp = len(ablation_ids)
    x = np.arange(n_bench)
    width = 0.8 / n_exp

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, eid in enumerate(ablation_ids):
        vals = [get_eval(eid, m) * 100 for m, _ in BENCHMARKS]
        bars = ax.bar(x + i * width - 0.4 + width/2, vals, width,
                      label=labels[i], color=COLORS.get(eid, f"C{i}"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([b[1] for b in BENCHMARKS], rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Benchmark Accuracy: Data Quality Ablation", fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_ylim(20, 85)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig2_benchmark_bars.png")
    fig.savefig(f"{OUT_DIR}/fig2_benchmark_bars.pdf")
    plt.close(fig)
    print("  [+] fig2_benchmark_bars")


# ---------------------------------------------------------------------------
# Figure 3: Dense vs MoE (paired comparison on same data)
# ---------------------------------------------------------------------------

def fig3_dense_vs_moe():
    pairs = [
        ("full_curated_dense", "full_curated_moe", "Curated Data"),
        ("full_raw_dense", "full_raw_moe", "Raw Data"),
    ]
    bench_names = [b[1] for b in BENCHMARKS]
    n = len(BENCHMARKS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (dense_id, moe_id, title) in zip(axes, pairs):
        dense_vals = [get_eval(dense_id, m) * 100 for m, _ in BENCHMARKS]
        moe_vals = [get_eval(moe_id, m) * 100 for m, _ in BENCHMARKS]

        x = np.arange(n)
        w = 0.35
        ax.bar(x - w/2, dense_vals, w, label="Dense (382M)", color="#4c72b0", alpha=0.85)
        ax.bar(x + w/2, moe_vals, w, label="MoE (390M/1.4B)", color="#dd8452", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(bench_names, rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(20, 85)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("Dense vs MoE on Same Data", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3_dense_vs_moe.png")
    fig.savefig(f"{OUT_DIR}/fig3_dense_vs_moe.pdf")
    plt.close(fig)
    print("  [+] fig3_dense_vs_moe")


# ---------------------------------------------------------------------------
# Figure 4: Data quality at equal token budget
# ---------------------------------------------------------------------------

def fig4_equal_size_comparison():
    ids = ["equal_raw_dense", "equal_curated_dense", "equal_hybrid_dense"]
    labels = ["Raw", "Curated", "Hybrid"]
    colors_local = ["#d62728", "#ff7f0e", "#2ca02c"]
    bench_names = [b[1] for b in BENCHMARKS]
    n = len(BENCHMARKS)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n)
    w = 0.25
    for i, (eid, lab, col) in enumerate(zip(ids, labels, colors_local)):
        vals = [get_eval(eid, m) * 100 for m, _ in BENCHMARKS]
        ax.bar(x + i*w - w, vals, w, label=lab, color=col, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(bench_names, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Data Quality at Equal Token Budget (1.3B tokens, Dense 382M)",
                 fontweight="bold")
    ax.legend()
    ax.set_ylim(20, 75)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig4_equal_size.png")
    fig.savefig(f"{OUT_DIR}/fig4_equal_size.pdf")
    plt.close(fig)
    print("  [+] fig4_equal_size")


# ---------------------------------------------------------------------------
# Figure 5: Scaling with data — equal vs full (raw dense)
# ---------------------------------------------------------------------------

def fig5_data_scaling():
    ids = ["equal_raw_dense", "full_raw_dense", "equal_curated_dense", "full_curated_dense"]
    labels = ["Raw 1.3B tok", "Raw 36.5B tok", "Curated 1.3B tok", "Curated 5.7B tok"]
    colors_local = ["#aec7e8", "#1f77b4", "#ffbb78", "#ff7f0e"]
    bench_names = [b[1] for b in BENCHMARKS]
    n = len(BENCHMARKS)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n)
    w = 0.2
    for i, (eid, lab, col) in enumerate(zip(ids, labels, colors_local)):
        vals = [get_eval(eid, m) * 100 for m, _ in BENCHMARKS]
        ax.bar(x + i*w - 1.5*w, vals, w, label=lab, color=col, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(bench_names, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Effect of Data Quantity (Dense 382M)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(20, 80)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig5_data_scaling.png")
    fig.savefig(f"{OUT_DIR}/fig5_data_scaling.pdf")
    plt.close(fig)
    print("  [+] fig5_data_scaling")


# ---------------------------------------------------------------------------
# Figure 6: Radar chart — top experiments
# ---------------------------------------------------------------------------

def fig6_radar():
    ids = [
        "full_raw_moe", "full_raw_dense", "full_curated_moe",
        "full_curated_dense", "equal_curated_dense",
    ]
    colors_local = ["#e377c2", "#d62728", "#8c564b", "#9467bd", "#ff7f0e"]
    bench_keys = [m for m, _ in BENCHMARKS]
    bench_labels = [b for _, b in BENCHMARKS]
    N = len(bench_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for eid, col in zip(ids, colors_local):
        vals = [get_eval(eid, k) * 100 for k in bench_keys]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.8, label=NICE_NAMES.get(eid, eid), color=col)
        ax.fill(angles, vals, alpha=0.08, color=col)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(bench_labels, fontsize=9)
    ax.set_ylim(20, 85)
    ax.set_title("Benchmark Profile (Top Experiments)", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig6_radar.png")
    fig.savefig(f"{OUT_DIR}/fig6_radar.pdf")
    plt.close(fig)
    print("  [+] fig6_radar")


# ---------------------------------------------------------------------------
# Figure 7: Perplexity comparison
# ---------------------------------------------------------------------------

def fig7_perplexity():
    ablation_ids = [
        "equal_raw_dense", "equal_curated_dense", "equal_hybrid_dense",
        "full_raw_dense", "full_curated_dense", "full_raw_moe", "full_curated_moe",
    ]
    labels = [NICE_NAMES[e] for e in ablation_ids]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # WikiText
    ax = axes[0]
    vals = [get_eval(e, "wikitext_word_ppl") for e in ablation_ids]
    bars = ax.barh(range(len(vals)), vals, color=[COLORS.get(e, "C0") for e in ablation_ids], alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Word Perplexity")
    ax.set_title("WikiText Perplexity (lower is better)", fontweight="bold")
    ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}", va="center", fontsize=8)

    # LAMBADA
    ax = axes[1]
    vals = [get_eval(e, "lambada_ppl") for e in ablation_ids]
    bars = ax.barh(range(len(vals)), vals, color=[COLORS.get(e, "C0") for e in ablation_ids], alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Perplexity")
    ax.set_title("LAMBADA Perplexity (lower is better)", fontweight="bold")
    ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig7_perplexity.png")
    fig.savefig(f"{OUT_DIR}/fig7_perplexity.pdf")
    plt.close(fig)
    print("  [+] fig7_perplexity")


# ---------------------------------------------------------------------------
# Figure 8: Model scaling — small models to large
# ---------------------------------------------------------------------------

def fig8_model_scaling():
    scale_ids = [
        ("baseline_29k_docs", 4e6),
        ("scaled_38M", 39e6),
        ("scaled_57M", 58e6),
        ("full_raw_dense", 382e6),
        ("full_raw_moe", 390e6),
    ]
    bench_subset = [
        ("hellaswag_acc_norm", "HellaSwag"),
        ("piqa_acc", "PIQA"),
        ("arc_easy_acc_norm", "ARC-Easy"),
        ("sciq_acc_norm", "SciQ"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    for metric, bname in bench_subset:
        params = [p for _, p in scale_ids]
        vals = [get_eval(eid, metric) * 100 for eid, _ in scale_ids]
        ax.plot(params, vals, "o-", label=bname, linewidth=2, markersize=7)

    ax.set_xscale("log")
    ax.set_xlabel("Active Parameters")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Benchmark Accuracy vs Model Scale", fontweight="bold")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    ax.set_ylim(20, 85)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig8_model_scaling.png")
    fig.savefig(f"{OUT_DIR}/fig8_model_scaling.pdf")
    plt.close(fig)
    print("  [+] fig8_model_scaling")


# ---------------------------------------------------------------------------
# Table: LaTeX summary
# ---------------------------------------------------------------------------

def table_latex():
    ablation_ids = [
        "equal_raw_dense", "equal_curated_dense", "equal_hybrid_dense",
        "full_raw_dense", "full_curated_dense", "full_raw_moe", "full_curated_moe",
    ]
    all_metrics = BENCHMARKS + PPL_BENCHMARKS

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Evaluation results for data quality ablation experiments.}")
    lines.append(r"\label{tab:eval_results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    cols = "l" + "r" * len(all_metrics)
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")

    header = "Experiment"
    for _, bname in all_metrics:
        header += f" & {bname}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for eid in ablation_ids:
        row = NICE_NAMES.get(eid, eid)
        for metric, _ in all_metrics:
            v = get_eval(eid, metric)
            if "ppl" in metric:
                row += f" & {v:.1f}"
            else:
                row += f" & {v*100:.1f}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    with open(f"{OUT_DIR}/table_eval_results.tex", "w") as f:
        f.write(tex)
    print("  [+] table_eval_results.tex")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loaded {len(EXPS)} experiments, {len(LOGS)} training logs\n")
    print("Generating figures...")
    fig1_loss_curves()
    fig2_benchmark_bars()
    fig3_dense_vs_moe()
    fig4_equal_size_comparison()
    fig5_data_scaling()
    fig6_radar()
    fig7_perplexity()
    fig8_model_scaling()
    table_latex()
    print(f"\nAll outputs saved to {OUT_DIR}/")
