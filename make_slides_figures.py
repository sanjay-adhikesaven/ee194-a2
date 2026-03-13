"""Generate clean presentation-ready figures for Groups 1 and 2."""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.3)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "axes.titlesize": 16,
    "axes.labelsize": 13,
})

OUT_DIR = "figures/slides"
os.makedirs(OUT_DIR, exist_ok=True)

def load_exp(eid):
    with open(f"experiments/{eid}.json") as f:
        return json.load(f)

BENCHMARKS_ACC = [
    ("hellaswag_acc_norm", "HellaSwag"),
    ("piqa_acc", "PIQA"),
    ("arc_easy_acc_norm", "ARC-Easy"),
    ("arc_challenge_acc_norm", "ARC-Chall."),
    ("sciq_acc_norm", "SciQ"),
    ("boolq_acc", "BoolQ"),
    ("winogrande_acc", "WinoGrande"),
]

PPL_METRICS = [
    ("wikitext_word_ppl", "WikiText"),
    ("lambada_ppl", "LAMBADA"),
]

def get_eval(exp, metric):
    return exp.get("eval_results", {}).get(metric, 0)


# ---------------------------------------------------------------------------
# Group 1: Model Scaling
# ---------------------------------------------------------------------------

def fig_group1a():
    """1a: Model size scaling (raw data, vary params)."""
    ids = [
        ("baseline_29k_docs", "4M",  "#74b9ff"),
        ("scaled_38M",        "39M", "#0984e3"),
        ("raw_full_57M",      "58M", "#2d3436"),
    ]
    exps = [(load_exp(eid), label, col) for eid, label, col in ids]

    fig, ax = plt.subplots(figsize=(13, 6))
    n = len(BENCHMARKS_ACC)
    n_exp = len(exps)
    x = np.arange(n)
    w = 0.25

    for i, (exp, label, col) in enumerate(exps):
        vals = [get_eval(exp, m) * 100 for m, _ in BENCHMARKS_ACC]
        ax.bar(x + i * w - w, vals, w, label=label, color=col,
               alpha=0.9, edgecolor="white", linewidth=0.5)
        for j, v in enumerate(vals):
            ax.text(x[j] + i * w - w, v + 0.5, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([b[1] for b in BENCHMARKS_ACC])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Effect of Model Size (Raw Wikipedia Data)", fontweight="bold")
    ax.legend(title="Parameters", title_fontsize=10, fontsize=11)
    ax.set_ylim(20, 65)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/group1a_model_size.png")
    fig.savefig(f"{OUT_DIR}/group1a_model_size.pdf")
    plt.close(fig)
    print("  [+] group1a_model_size")


def fig_group1b():
    """1b: Data source/quantity at fixed 58M model size."""
    ids = [
        ("raw_sub_57M",    "Raw (subset)",  "#e17055"),
        ("curated_57M",    "Curated",       "#fdcb6e"),
        ("hybrid_sub_57M", "Hybrid",        "#00b894"),
        ("scaled_57M",     "Scaled",        "#6c5ce7"),
        ("raw_full_57M",   "Raw (full)",    "#d63031"),
    ]
    exps = [(load_exp(eid), label, col) for eid, label, col in ids]

    fig, ax = plt.subplots(figsize=(14, 6))
    n = len(BENCHMARKS_ACC)
    n_exp = len(exps)
    x = np.arange(n)
    w = 0.8 / n_exp

    for i, (exp, label, col) in enumerate(exps):
        vals = [get_eval(exp, m) * 100 for m, _ in BENCHMARKS_ACC]
        ax.bar(x + i * w - 0.4 + w/2, vals, w, label=label, color=col,
               alpha=0.9, edgecolor="white", linewidth=0.5)
        for j, v in enumerate(vals):
            ax.text(x[j] + i * w - 0.4 + w/2, v + 0.4, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([b[1] for b in BENCHMARKS_ACC])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Effect of Data Source & Quantity (58M Model)", fontweight="bold")
    ax.legend(title="Data", title_fontsize=10, fontsize=9)
    ax.set_ylim(20, 60)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/group1b_data_source.png")
    fig.savefig(f"{OUT_DIR}/group1b_data_source.pdf")
    plt.close(fig)
    print("  [+] group1b_data_source")


# ---------------------------------------------------------------------------
# Group 2: Data Quality — Equal Token Budget (5.7B)
# ---------------------------------------------------------------------------

def fig_group2_accuracy():
    ids = [
        ("equal5b_raw_dense",     "Raw",     "#3498db"),
        ("full_curated_dense",    "Curated", "#e67e22"),
        ("equal5b_hybrid_dense",  "Hybrid",  "#2ecc71"),
    ]
    exps = [(load_exp(eid), label, col) for eid, label, col in ids]

    fig, ax = plt.subplots(figsize=(12, 6))
    n = len(BENCHMARKS_ACC)
    n_exp = len(exps)
    x = np.arange(n)
    w = 0.25

    for i, (exp, label, col) in enumerate(exps):
        vals = [get_eval(exp, m) * 100 for m, _ in BENCHMARKS_ACC]
        bars = ax.bar(x + i * w - w, vals, w, label=label, color=col,
                      alpha=0.9, edgecolor="white", linewidth=0.5)
        for j, v in enumerate(vals):
            ax.text(x[j] + i * w - w, v + 0.3, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([b[1] for b in BENCHMARKS_ACC])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Data Quality at Equal Token Budget (5.7B tokens, Dense 382M)",
                 fontweight="bold")
    ax.legend(title="Data Source", title_fontsize=10, fontsize=11)
    ax.set_ylim(20, 75)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/group2_equal_accuracy.png")
    fig.savefig(f"{OUT_DIR}/group2_equal_accuracy.pdf")
    plt.close(fig)
    print("  [+] group2_equal_accuracy")


def fig_group2_perplexity():
    ids = [
        ("equal5b_raw_dense",     "Raw",     "#3498db"),
        ("full_curated_dense",    "Curated", "#e67e22"),
        ("equal5b_hybrid_dense",  "Hybrid",  "#2ecc71"),
    ]
    exps = [(load_exp(eid), label, col) for eid, label, col in ids]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, (metric, title) in zip(axes, PPL_METRICS):
        vals = [get_eval(exp, metric) for exp, _, _ in exps]
        labels = [label for _, label, _ in exps]
        colors = [col for _, _, col in exps]
        bars = ax.bar(labels, vals, color=colors, alpha=0.9,
                      edgecolor="white", linewidth=0.5, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 1,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylabel("Perplexity")
        ax.set_title(f"{title} Perplexity ↓", fontweight="bold")

    fig.suptitle("Perplexity at Equal Token Budget (5.7B tokens, Dense 382M)",
                 fontweight="bold", fontsize=14, y=1.03)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/group2_equal_perplexity.png")
    fig.savefig(f"{OUT_DIR}/group2_equal_perplexity.pdf")
    plt.close(fig)
    print("  [+] group2_equal_perplexity")


# ---------------------------------------------------------------------------
# Group 3: Full-Scale × Architecture (3×2 grid)
# ---------------------------------------------------------------------------

def fig_group3_accuracy():
    """3x2 grid: Raw/Curated/Hybrid × Dense/MoE — accuracy benchmarks."""
    pairs = [
        ("Raw",     "full_raw_dense",     "full_raw_moe"),
        ("Curated", "full_curated_dense", "full_curated_moe"),
        ("Hybrid",  "full_hybrid_dense",  "full_hybrid_moe"),
    ]
    dense_color = "#4c72b0"
    moe_color = "#dd8452"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    bench_names = [b[1] for b in BENCHMARKS_ACC]
    n = len(BENCHMARKS_ACC)
    x = np.arange(n)
    w = 0.35

    for ax, (data_label, dense_id, moe_id) in zip(axes, pairs):
        dense_exp = load_exp(dense_id)
        moe_exp = load_exp(moe_id)
        dense_vals = [get_eval(dense_exp, m) * 100 for m, _ in BENCHMARKS_ACC]
        moe_vals = [get_eval(moe_exp, m) * 100 for m, _ in BENCHMARKS_ACC]

        bars_d = ax.bar(x - w/2, dense_vals, w, label="Dense (382M)",
                        color=dense_color, alpha=0.9, edgecolor="white", linewidth=0.5)
        bars_m = ax.bar(x + w/2, moe_vals, w, label="MoE (390M/1.4B)",
                        color=moe_color, alpha=0.9, edgecolor="white", linewidth=0.5)

        for j in range(n):
            ax.text(x[j] - w/2, dense_vals[j] + 0.4, f"{dense_vals[j]:.1f}",
                    ha="center", va="bottom", fontsize=6.5, color=dense_color, fontweight="bold")
            ax.text(x[j] + w/2, moe_vals[j] + 0.4, f"{moe_vals[j]:.1f}",
                    ha="center", va="bottom", fontsize=6.5, color=moe_color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(bench_names, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"{data_label} Data", fontweight="bold", fontsize=14)
        ax.legend(fontsize=9)
        ax.set_ylim(20, 85)

    axes[0].set_ylabel("Accuracy (%)")
    axes[0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    fig.suptitle("Full-Scale Runs: Dense vs MoE across Data Sources",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/group3_fullscale_accuracy.png")
    fig.savefig(f"{OUT_DIR}/group3_fullscale_accuracy.pdf")
    plt.close(fig)
    print("  [+] group3_fullscale_accuracy")


def fig_group3_perplexity():
    """3x2 grid perplexity comparison."""
    experiments = [
        ("full_raw_dense",     "Raw\nDense",     "#4c72b0"),
        ("full_raw_moe",       "Raw\nMoE",       "#dd8452"),
        ("full_curated_dense", "Curated\nDense",  "#4c72b0"),
        ("full_curated_moe",   "Curated\nMoE",    "#dd8452"),
        ("full_hybrid_dense",  "Hybrid\nDense",   "#4c72b0"),
        ("full_hybrid_moe",    "Hybrid\nMoE",     "#dd8452"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, (metric, title) in zip(axes, PPL_METRICS):
        vals = [get_eval(load_exp(eid), metric) for eid, _, _ in experiments]
        labels = [lab for _, lab, _ in experiments]
        colors = [col for _, _, col in experiments]
        bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.9,
                      edgecolor="white", linewidth=0.5, width=0.6)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Perplexity")
        ax.set_title(f"{title} Perplexity ↓", fontweight="bold")

        # Add group separators
        for sep_x in [1.5, 3.5]:
            ax.axvline(sep_x, color="gray", linestyle="--", alpha=0.3)

    fig.suptitle("Full-Scale Perplexity: Dense vs MoE across Data Sources",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/group3_fullscale_perplexity.png")
    fig.savefig(f"{OUT_DIR}/group3_fullscale_perplexity.pdf")
    plt.close(fig)
    print("  [+] group3_fullscale_perplexity")


def fig_group3_heatmap():
    """Compact heatmap view of the 3x2 grid — average accuracy."""
    data_sources = ["Raw", "Curated", "Hybrid"]
    archs = ["Dense", "MoE"]
    ids_grid = [
        ["full_raw_dense", "full_raw_moe"],
        ["full_curated_dense", "full_curated_moe"],
        ["full_hybrid_dense", "full_hybrid_moe"],
    ]

    fig, axes = plt.subplots(1, len(BENCHMARKS_ACC) + 1, figsize=(18, 3.5),
                              gridspec_kw={"width_ratios": [1]*len(BENCHMARKS_ACC) + [1.2]})

    for b_idx, (metric, bname) in enumerate(BENCHMARKS_ACC):
        ax = axes[b_idx]
        grid = np.array([[get_eval(load_exp(eid), metric) * 100
                          for eid in row] for row in ids_grid])
        im = ax.imshow(grid, cmap="YlOrRd", vmin=25, vmax=80, aspect="auto")
        for i in range(3):
            for j in range(2):
                ax.text(j, i, f"{grid[i,j]:.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if grid[i,j] > 55 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(archs, fontsize=8)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(data_sources if b_idx == 0 else ["", "", ""], fontsize=9)
        ax.set_title(bname, fontsize=9, fontweight="bold")

    # Average column
    ax = axes[-1]
    avg_grid = np.array([[np.mean([get_eval(load_exp(eid), m) * 100 for m, _ in BENCHMARKS_ACC])
                          for eid in row] for row in ids_grid])
    im = ax.imshow(avg_grid, cmap="YlOrRd", vmin=25, vmax=80, aspect="auto")
    for i in range(3):
        for j in range(2):
            ax.text(j, i, f"{avg_grid[i,j]:.1f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if avg_grid[i,j] > 55 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(archs, fontsize=8)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["", "", ""], fontsize=9)
    ax.set_title("Average", fontsize=9, fontweight="bold")

    fig.suptitle("Full-Scale Results: Data Source × Architecture",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/group3_heatmap.png")
    fig.savefig(f"{OUT_DIR}/group3_heatmap.pdf")
    plt.close(fig)
    print("  [+] group3_heatmap")


def parse_log(run_id):
    """Parse train.log and return list of (step, loss) tuples."""
    import re
    path = f"runs/{run_id}/train.log"
    steps = []
    with open(path) as f:
        for line in f:
            m = re.search(r'step\s+(\d+)/\s*(\d+)\s*\|\s*loss\s+([\d.]+)', line)
            if m:
                steps.append((int(m.group(1)), float(m.group(3))))
    return steps


def fig_loss_curves():
    """Training loss curves for all 6 full-scale runs."""
    runs = [
        ("full_raw_dense",     "Raw Dense",     "#3498db", "-"),
        ("full_raw_moe",       "Raw MoE",       "#3498db", "--"),
        ("full_curated_dense", "Curated Dense",  "#e67e22", "-"),
        ("full_curated_moe",   "Curated MoE",    "#e67e22", "--"),
        ("full_hybrid_dense",  "Hybrid Dense",   "#2ecc71", "-"),
        ("full_hybrid_moe",    "Hybrid MoE",     "#2ecc71", "--"),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    for run_id, label, color, ls in runs:
        data = parse_log(run_id)
        if not data:
            continue
        steps, losses = zip(*data)
        ax.plot(steps, losses, label=label, color=color, linestyle=ls,
                linewidth=2 if ls == "-" else 1.5, alpha=0.9)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves (Full-Scale Runs)", fontweight="bold")
    ax.legend(fontsize=10, ncol=2)
    ax.set_ylim(1, 10)
    ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/loss_curves_fullscale.png")
    fig.savefig(f"{OUT_DIR}/loss_curves_fullscale.pdf")
    plt.close(fig)
    print("  [+] loss_curves_fullscale")


if __name__ == "__main__":
    print("Generating slide figures...")
    fig_group1a()
    fig_group1b()
    fig_group2_accuracy()
    fig_group2_perplexity()
    fig_group3_accuracy()
    fig_group3_perplexity()
    fig_group3_heatmap()
    fig_loss_curves()
    print(f"\nAll outputs saved to {OUT_DIR}/")
