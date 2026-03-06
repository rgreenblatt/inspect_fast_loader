"""Generate comprehensive benchmark comparison plots."""

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


def load_latest_results():
    """Load the most recent comprehensive benchmark results."""
    files = sorted(glob(os.path.join(RESULTS_DIR, "benchmark_comprehensive_*.jsonl")))
    assert files, "No benchmark results found. Run benchmark_comprehensive.py first."
    results = []
    with open(files[-1]) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def plot_speedup_overview(results):
    """Bar chart showing speedup for all operations."""
    orig = {}
    fast = {}
    for r in results:
        key = r["operation"] + ("_" + r["size"] if r.get("size") else "")
        if r["config"] == "original":
            orig[key] = r["time_ms"]
        elif r["config"] == "fast":
            fast[key] = r["time_ms"]

    # Calculate speedups
    speedups = {}
    for key in sorted(orig.keys()):
        if key in fast and fast[key] > 0:
            speedups[key] = orig[key] / fast[key]

    # Format labels
    label_map = {
        "full_read_eval_1000_samples": ".eval full (1000)",
        "full_read_eval_100_samples": ".eval full (100)",
        "full_read_eval_10_samples": ".eval full (10)",
        "full_read_eval_5_samples": ".eval full (5)",
        "full_read_json_1000_samples": ".json full (1000)",
        "full_read_json_100_samples": ".json full (100)",
        "full_read_json_10_samples": ".json full (10)",
        "full_read_json_5_samples": ".json full (5)",
        "header_only_eval": ".eval header",
        "header_only_json": ".json header",
        "batch_headers_10_files": "batch hdrs (10)",
        "batch_headers_25_files": "batch hdrs (25)",
        "batch_headers_50_files": "batch hdrs (50)",
        "batch_headers_59_files": "batch hdrs (59)",
        "single_sample": "single sample",
        "single_sample_excluded": "single sample (excl)",
        "sample_summaries": "summaries",
        "streaming_samples": "streaming",
    }

    # Order for display
    display_order = [
        "full_read_eval_1000_samples", "full_read_eval_100_samples",
        "full_read_eval_10_samples", "full_read_eval_5_samples",
        "full_read_json_1000_samples", "full_read_json_100_samples",
        "full_read_json_10_samples", "full_read_json_5_samples",
        "batch_headers_10_files", "batch_headers_25_files",
        "batch_headers_50_files", "batch_headers_59_files",
        "single_sample", "single_sample_excluded",
        "sample_summaries", "streaming_samples",
        "header_only_eval", "header_only_json",
    ]

    labels = []
    values = []
    for key in display_order:
        if key in speedups:
            labels.append(label_map.get(key, key))
            values.append(speedups[key])

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = []
    for key in display_order:
        if key not in speedups:
            continue
        if "eval" in key and "full" in key:
            colors.append("#2196F3")
        elif "json" in key and "full" in key:
            colors.append("#4CAF50")
        elif "batch" in key:
            colors.append("#FF9800")
        elif "single" in key or "streaming" in key:
            colors.append("#9C27B0")
        elif "summar" in key:
            colors.append("#E91E63")
        else:
            colors.append("#607D8B")

    bars = ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Speedup (x)", fontsize=12)
    ax.set_title("inspect_fast_loader: Speedup Over Original (All Operations)", fontsize=14)
    ax.axvline(x=1, color="gray", linestyle="--", alpha=0.5, label="1x (no speedup)")
    ax.axvline(x=5, color="green", linestyle="--", alpha=0.3, label="5x target")

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}x", va="center", fontsize=9)

    ax.legend(loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "comprehensive_speedup.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_absolute_times(results):
    """Side-by-side bar chart showing absolute times."""
    orig = {}
    fast = {}
    for r in results:
        key = r["operation"] + ("_" + r["size"] if r.get("size") else "")
        if r["config"] == "original":
            orig[key] = r["time_ms"]
        elif r["config"] == "fast":
            fast[key] = r["time_ms"]

    # Focus on the most important operations
    key_ops = [
        ("full_read_eval_1000_samples", ".eval full\n(1000 samples)"),
        ("full_read_eval_100_samples", ".eval full\n(100 samples)"),
        ("full_read_json_1000_samples", ".json full\n(1000 samples)"),
        ("batch_headers_50_files", "batch headers\n(50 files)"),
        ("single_sample", "single\nsample"),
        ("sample_summaries", "sample\nsummaries"),
        ("streaming_samples", "streaming\nsamples"),
    ]

    labels = []
    orig_vals = []
    fast_vals = []
    for key, label in key_ops:
        if key in orig and key in fast:
            labels.append(label)
            orig_vals.append(orig[key])
            fast_vals.append(fast[key])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, orig_vals, width, label="Original", color="#EF5350")
    bars2 = ax.bar(x + width / 2, fast_vals, width, label="Fast (Rust+bypass)", color="#42A5F5")

    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("Absolute Performance: Original vs Fast", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend()
    ax.set_yscale("log")

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f"{height:.0f}ms", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f"{height:.0f}ms", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "comprehensive_absolute_times.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_batch_headers_scaling(results):
    """Line chart showing batch header performance vs file count."""
    orig_points = {}
    fast_points = {}
    for r in results:
        if r["operation"] == "batch_headers":
            n = r.get("n_files", 0)
            if r["config"] == "original":
                orig_points[n] = r["time_ms"]
            elif r["config"] == "fast":
                fast_points[n] = r["time_ms"]

    if not orig_points or not fast_points:
        print("No batch header scaling data")
        return

    x_orig = sorted(orig_points.keys())
    y_orig = [orig_points[x] for x in x_orig]
    x_fast = sorted(fast_points.keys())
    y_fast = [fast_points[x] for x in x_fast]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute times
    ax1.plot(x_orig, y_orig, "o-", color="#EF5350", label="Original", linewidth=2, markersize=8)
    ax1.plot(x_fast, y_fast, "s-", color="#42A5F5", label="Fast (rayon batch)", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of .eval files", fontsize=12)
    ax1.set_ylabel("Time (ms)", fontsize=12)
    ax1.set_title("Batch Header Read: Absolute Times", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Speedup
    common_x = sorted(set(x_orig) & set(x_fast))
    speedups = [orig_points[x] / fast_points[x] for x in common_x]
    ax2.plot(common_x, speedups, "D-", color="#FF9800", linewidth=2, markersize=8)
    ax2.axhline(y=5, color="green", linestyle="--", alpha=0.5, label="5x target")
    ax2.axhline(y=10, color="green", linestyle=":", alpha=0.3, label="10x target")
    ax2.set_xlabel("Number of .eval files", fontsize=12)
    ax2.set_ylabel("Speedup (x)", fontsize=12)
    ax2.set_title("Batch Header Read: Speedup", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    for x, s in zip(common_x, speedups):
        ax2.annotate(f"{s:.1f}x", (x, s), textcoords="offset points", xytext=(0, 10), fontsize=9, ha="center")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "batch_headers_scaling.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_new_operations_speedup(results):
    """Focused plot on newly patched operations."""
    orig = {}
    fast = {}
    for r in results:
        key = r["operation"] + ("_" + r["size"] if r.get("size") else "")
        if r["config"] == "original":
            orig[key] = r["time_ms"]
        elif r["config"] == "fast":
            fast[key] = r["time_ms"]

    # New operations only
    new_ops = [
        ("single_sample", "Single Sample\nRead"),
        ("single_sample_excluded", "Single Sample\n(exclude_fields)"),
        ("sample_summaries", "Sample\nSummaries"),
        ("streaming_samples", "Streaming\nSamples"),
    ]

    labels = []
    orig_vals = []
    fast_vals = []
    speedups = []
    for key, label in new_ops:
        if key in orig and key in fast:
            labels.append(label)
            orig_vals.append(orig[key])
            fast_vals.append(fast[key])
            speedups.append(orig[key] / fast[key])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(labels))
    width = 0.35

    # Absolute times
    ax1.bar(x - width / 2, orig_vals, width, label="Original", color="#EF5350")
    ax1.bar(x + width / 2, fast_vals, width, label="Fast", color="#42A5F5")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("New Patched Operations: Absolute Times")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend()

    # Speedup
    bars = ax2.bar(x, speedups, color=["#9C27B0", "#9C27B0", "#E91E63", "#9C27B0"])
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title("New Patched Operations: Speedup")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f"{val:.1f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "new_operations_speedup.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def main():
    results = load_latest_results()
    plot_speedup_overview(results)
    plot_absolute_times(results)
    plot_batch_headers_scaling(results)
    plot_new_operations_speedup(results)
    print("\nAll plots saved to plots/")


if __name__ == "__main__":
    main()
