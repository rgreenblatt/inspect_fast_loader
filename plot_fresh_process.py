"""Generate plots from fresh-process benchmark results."""

import json
import os
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_latest_fresh_results():
    files = sorted(glob(os.path.join(RESULTS_DIR, "benchmark_fresh_process_*.jsonl")))
    assert files, "No fresh-process benchmark results found."
    results = []
    with open(files[-1]) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def plot_fresh_speedup_overview(results):
    """Bar chart showing speedup from fresh-process benchmarks."""
    # Pair up original/fast results
    pairs = []
    i = 0
    while i < len(results) - 1:
        orig = results[i]
        fast = results[i + 1]
        if orig["config"] == "original" and fast["config"] == "fast" and orig["operation"] == fast["operation"]:
            if fast["median"] > 0:
                speedup = orig["median"] / fast["median"]
                pairs.append((orig["operation"], speedup, orig["median"], fast["median"]))
            i += 2
        else:
            i += 1

    labels = [p[0] for p in pairs]
    speedups = [p[1] for p in pairs]

    # Color by category
    colors = []
    for label, _, _, _ in pairs:
        if "eval" in label and "full" in label:
            colors.append("#2196F3")
        elif "json" in label and "full" in label:
            colors.append("#4CAF50")
        elif "batch" in label:
            colors.append("#FF9800")
        elif "single" in label or "streaming" in label:
            colors.append("#9C27B0")
        elif "summar" in label:
            colors.append("#E91E63")
        else:
            colors.append("#607D8B")

    # Format labels for readability
    label_map = {
        "full_read eval 10 samples": ".eval full (10)",
        "full_read eval 100 samples": ".eval full (100)",
        "full_read eval 1000 samples": ".eval full (1000)",
        "full_read eval 5000 samples": ".eval full (5000)",
        "full_read json 10 samples": ".json full (10)",
        "full_read json 100 samples": ".json full (100)",
        "full_read json 1000 samples": ".json full (1000)",
        "batch_headers 10 files": "batch hdrs (10)",
        "batch_headers 25 files": "batch hdrs (25)",
        "batch_headers 50 files": "batch hdrs (50)",
        "batch_headers 60 files": "batch hdrs (60)*",
        "single_sample": "single sample",
        "single_sample_excluded": "single sample (excl)",
        "sample_summaries": "summaries",
        "streaming_samples": "streaming",
    }
    display_labels = [label_map.get(l, l) for l in labels]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(display_labels)), speedups, color=colors)
    ax.set_yticks(range(len(display_labels)))
    ax.set_yticklabels(display_labels, fontsize=10)
    ax.set_xlabel("Speedup (x)", fontsize=12)
    ax.set_title("inspect_fast_loader: Speedup (Fresh-Process Benchmark)", fontsize=14)
    ax.axvline(x=1, color="gray", linestyle="--", alpha=0.5, label="1x (no speedup)")
    ax.axvline(x=5, color="green", linestyle="--", alpha=0.3, label="5x target")

    for bar, val in zip(bars, speedups):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}x", va="center", fontsize=9)

    ax.legend(loc="lower right")
    ax.invert_yaxis()

    # Add footnote about batch_headers 60
    ax.text(0.02, 0.02,
            "* batch hdrs (60) includes one 5000-sample file with large header",
            transform=ax.transAxes, fontsize=8, style="italic", alpha=0.7)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fresh_process_speedup.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_fresh_absolute_times(results):
    """Side-by-side bar chart for key operations."""
    pairs = []
    i = 0
    while i < len(results) - 1:
        orig = results[i]
        fast = results[i + 1]
        if orig["config"] == "original" and fast["config"] == "fast" and orig["operation"] == fast["operation"]:
            pairs.append((orig["operation"], orig["median"], fast["median"]))
            i += 2
        else:
            i += 1

    # Select key operations
    key_ops = [
        "full_read eval 5000 samples",
        "full_read eval 1000 samples",
        "full_read eval 100 samples",
        "full_read json 1000 samples",
        "batch_headers 50 files",
        "single_sample",
        "sample_summaries",
        "streaming_samples",
    ]

    labels = []
    orig_vals = []
    fast_vals = []
    for op in key_ops:
        for name, o, f in pairs:
            if name == op:
                label_map = {
                    "full_read eval 5000 samples": ".eval full\n(5000)",
                    "full_read eval 1000 samples": ".eval full\n(1000)",
                    "full_read eval 100 samples": ".eval full\n(100)",
                    "full_read json 1000 samples": ".json full\n(1000)",
                    "batch_headers 50 files": "batch hdrs\n(50 files)",
                    "single_sample": "single\nsample",
                    "sample_summaries": "sample\nsummaries",
                    "streaming_samples": "streaming\nsamples",
                }
                labels.append(label_map.get(op, op))
                orig_vals.append(o)
                fast_vals.append(f)
                break

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, orig_vals, width, label="Original", color="#EF5350")
    bars2 = ax.bar(x + width / 2, fast_vals, width, label="Fast (Rust+bypass)", color="#42A5F5")

    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("Fresh-Process Benchmark: Original vs Fast", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend()
    ax.set_yscale("log")

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h,
                f"{h:.0f}ms", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h,
                f"{h:.0f}ms", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fresh_process_absolute_times.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def main():
    results = load_latest_fresh_results()
    plot_fresh_speedup_overview(results)
    plot_fresh_absolute_times(results)
    print("\nAll plots saved to plots/")


if __name__ == "__main__":
    main()
