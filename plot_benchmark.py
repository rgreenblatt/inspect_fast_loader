"""Generate benchmark comparison plots."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os


def load_results(path: str) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def plot_speedup_bars(results: list[dict], output_path: str):
    """Create a bar chart showing speedup for each operation."""
    fig, ax = plt.subplots(figsize=(14, 6))

    names = [r["name"] for r in results]
    speedups = [r["speedup_median"] for r in results]
    colors = []
    for s in speedups:
        if s >= 1.5:
            colors.append("#2ecc71")  # green
        elif s >= 1.0:
            colors.append("#f39c12")  # orange
        else:
            colors.append("#e74c3c")  # red

    x = np.arange(len(names))
    bars = ax.bar(x, speedups, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{s:.2f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Speedup (higher is better)", fontsize=11)
    ax.set_title("Rust-Accelerated vs Python Original: Speedup by Operation", fontsize=13)
    ax.set_ylim(0, max(speedups) * 1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_absolute_times(results: list[dict], output_path: str):
    """Create a grouped bar chart showing absolute times."""
    fig, ax = plt.subplots(figsize=(14, 6))

    names = [r["name"] for r in results]
    orig_ms = [r["original_median_s"] * 1000 for r in results]
    fast_ms = [r["fast_median_s"] * 1000 for r in results]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, orig_ms, width, label="Python Original", color="#3498db", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, fast_ms, width, label="Rust Accelerated", color="#e74c3c", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Time (ms, log scale)", fontsize=11)
    ax.set_title("Absolute Read Times: Python Original vs Rust-Accelerated", fontsize=13)
    ax.set_yscale("log")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    results_path = "results/benchmark_comparison.jsonl"
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    results = load_results(results_path)
    os.makedirs("plots", exist_ok=True)

    plot_speedup_bars(results, "plots/benchmark_speedup.png")
    plot_absolute_times(results, "plots/benchmark_absolute_times.png")


if __name__ == "__main__":
    main()
