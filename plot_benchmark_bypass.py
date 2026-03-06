"""Generate benchmark plots for the Pydantic bypass optimization phase."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = "results/benchmark_bypass_final.jsonl"
PLOTS_DIR = "plots"


def load_results():
    results = []
    with open(RESULTS_FILE) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def plot_speedup_comparison(results):
    """Bar chart comparing speedups across operations."""
    full_read_results = [r for r in results if r["operation"] == "full_read"]
    batch_results = [r for r in results if r["operation"] == "batch_headers"]

    # Full reads
    names = []
    speedups = []
    colors = []
    for r in full_read_results:
        label = r["name"].replace("full_read_", "")
        names.append(label)
        speedups.append(r["speedup_median"])
        colors.append("#2196F3" if "eval" in label else "#FF9800")

    for r in batch_results:
        label = r["name"].replace("batch_headers_", "batch_")
        names.append(label)
        speedups.append(r["speedup_median"])
        colors.append("#4CAF50")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    bars = ax.bar(x, speedups, color=colors, width=0.6)

    # Add value labels
    for bar, spd in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{spd:.1f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="1.0x baseline")
    ax.axhline(y=5.0, color="red", linestyle="--", alpha=0.5, label="5.0x target")

    ax.set_ylabel("Speedup (x)", fontsize=12)
    ax.set_title("Speedup: Rust + Pydantic Bypass vs Original Python", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, max(speedups) * 1.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bypass_speedup.png"), dpi=150)
    plt.close()


def plot_absolute_times(results):
    """Grouped bar chart showing absolute times."""
    full_read_results = [r for r in results if r["operation"] == "full_read"]

    names = [r["name"].replace("full_read_", "") for r in full_read_results]
    orig_times = [r["original_median_s"] * 1000 for r in full_read_results]
    fast_times = [r["fast_median_s"] * 1000 for r in full_read_results]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, orig_times, width, label="Original Python", color="#E53935")
    bars2 = ax.bar(x + width/2, fast_times, width, label="Rust + Bypass", color="#2196F3")

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 10,
                f"{h:.0f}ms", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 10,
                f"{h:.0f}ms", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("Absolute Times: Full Read Operations", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bypass_absolute_times.png"), dpi=150)
    plt.close()


def plot_pipeline_breakdown(results):
    """Stacked bar showing pipeline component breakdown for .eval 1000 samples."""
    import time
    from inspect_fast_loader._native import read_eval_file
    from inspect_fast_loader._construct import construct_sample_fast
    from inspect_ai.log._log import EvalSample, sort_samples
    from inspect_ai._util.constants import get_deserializing_context
    from inspect_ai.log._log import EvalLog

    path = "test_logs/test_1000samples.eval"

    # Profile Rust parsing
    times_rust = []
    for _ in range(5):
        start = time.perf_counter()
        raw = read_eval_file(path)
        times_rust.append(time.perf_counter() - start)

    # Profile sample construction
    times_construct = []
    for _ in range(5):
        raw = read_eval_file(path)
        start = time.perf_counter()
        samples = [construct_sample_fast(s) for s in raw["samples"]]
        times_construct.append(time.perf_counter() - start)

    # Profile header validation
    times_header = []
    for _ in range(5):
        raw = read_eval_file(path)
        ctx = get_deserializing_context()
        start = time.perf_counter()
        header_data = raw["header"]
        header_data.pop("samples", None)
        EvalLog.model_validate(header_data, context=ctx)
        times_header.append(time.perf_counter() - start)

    # Profile original model_validate
    times_validate = []
    for _ in range(5):
        raw = read_eval_file(path)
        ctx = get_deserializing_context()
        start = time.perf_counter()
        [EvalSample.model_validate(s, context=ctx) for s in raw["samples"]]
        times_validate.append(time.perf_counter() - start)

    rust_ms = sorted(times_rust)[2] * 1000
    construct_ms = sorted(times_construct)[2] * 1000
    header_ms = sorted(times_header)[2] * 1000
    validate_ms = sorted(times_validate)[2] * 1000

    fig, ax = plt.subplots(figsize=(10, 6))

    # New pipeline
    ax.barh(1, rust_ms, color="#2196F3", label="Rust JSON + ZIP")
    ax.barh(1, construct_ms, left=rust_ms, color="#4CAF50", label="Fast construct")
    ax.barh(1, header_ms, left=rust_ms + construct_ms, color="#FF9800", label="Header validate")

    # Old pipeline
    ax.barh(0, rust_ms, color="#2196F3")
    ax.barh(0, validate_ms, left=rust_ms, color="#E53935", label="model_validate")
    ax.barh(0, header_ms, left=rust_ms + validate_ms, color="#FF9800")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Rust + model_validate\n(previous)", "Rust + Fast construct\n(new)"])
    ax.set_xlabel("Time (ms)")
    ax.set_title("Pipeline Breakdown: .eval 1000 samples")
    ax.legend(loc="upper right")

    # Add total time annotations
    total_old = rust_ms + validate_ms + header_ms
    total_new = rust_ms + construct_ms + header_ms
    ax.text(total_old + 10, 0, f"{total_old:.0f}ms", va="center")
    ax.text(total_new + 10, 1, f"{total_new:.0f}ms", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bypass_pipeline_breakdown.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    results = load_results()
    print("Generating plots...")
    plot_speedup_comparison(results)
    print("  bypass_speedup.png")
    plot_absolute_times(results)
    print("  bypass_absolute_times.png")
    plot_pipeline_breakdown(results)
    print("  bypass_pipeline_breakdown.png")
    print("Done!")
