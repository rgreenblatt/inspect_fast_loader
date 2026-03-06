"""Benchmark comparing Rust-accelerated vs Python-original log reading.

Runs the same operations with both implementations and reports speedups.
"""

import argparse
import json
import os
import time
from typing import Any


def benchmark_operation(name: str, func, n_iterations: int = 5, warmup: int = 1) -> dict[str, Any]:
    """Benchmark a single operation."""
    for _ in range(warmup):
        func()

    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "name": name,
        "n_iterations": n_iterations,
        "times": times,
        "mean_s": sum(times) / len(times),
        "min_s": min(times),
        "max_s": max(times),
        "median_s": sorted(times)[len(times) // 2],
    }


def run_comparison_benchmarks(test_logs_dir: str, n_iterations: int = 5) -> list[dict[str, Any]]:
    """Run benchmarks with both original and fast implementations."""
    from inspect_ai.log._file import read_eval_log, read_eval_log_headers
    from inspect_fast_loader import patch, unpatch

    results = []

    operations = []

    # Full reads
    for n_samples in [10, 100, 1000]:
        for fmt in ["eval", "json"]:
            path = os.path.join(test_logs_dir, f"test_{n_samples}samples.{fmt}")
            if not os.path.exists(path):
                continue
            operations.append({
                "name": f"full_read_{fmt}_{n_samples}",
                "func": lambda p=path: read_eval_log(p),
                "format": fmt,
                "n_samples": n_samples,
                "operation": "full_read",
                "file_size_bytes": os.path.getsize(path),
            })

    # Header-only reads
    for n_samples in [10, 100, 1000]:
        for fmt in ["eval", "json"]:
            path = os.path.join(test_logs_dir, f"test_{n_samples}samples.{fmt}")
            if not os.path.exists(path):
                continue
            operations.append({
                "name": f"header_only_{fmt}_{n_samples}",
                "func": lambda p=path: read_eval_log(p, header_only=True),
                "format": fmt,
                "n_samples": n_samples,
                "operation": "header_only",
                "file_size_bytes": os.path.getsize(path),
            })

    # Batch header reads
    for batch_size in [10, 50]:
        paths = [os.path.join(test_logs_dir, f"batch_{i:03d}.eval") for i in range(batch_size)]
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            continue
        operations.append({
            "name": f"batch_headers_eval_{len(paths)}",
            "func": lambda ps=paths: read_eval_log_headers(ps),
            "format": "eval",
            "n_files": len(paths),
            "operation": "batch_headers",
            "total_size_bytes": sum(os.path.getsize(p) for p in paths),
        })

    for op in operations:
        func = op.pop("func")
        name = op["name"]

        # Benchmark original
        unpatch()
        orig_result = benchmark_operation(f"{name}_original", func, n_iterations)

        # Benchmark fast
        patch()
        fast_result = benchmark_operation(f"{name}_fast", func, n_iterations)
        unpatch()

        # Combine results
        entry = {
            **op,
            "original_mean_s": orig_result["mean_s"],
            "original_median_s": orig_result["median_s"],
            "original_min_s": orig_result["min_s"],
            "fast_mean_s": fast_result["mean_s"],
            "fast_median_s": fast_result["median_s"],
            "fast_min_s": fast_result["min_s"],
            "speedup_mean": orig_result["mean_s"] / fast_result["mean_s"] if fast_result["mean_s"] > 0 else float("inf"),
            "speedup_median": orig_result["median_s"] / fast_result["median_s"] if fast_result["median_s"] > 0 else float("inf"),
        }
        results.append(entry)

    return results


def print_results(results: list[dict[str, Any]]) -> None:
    """Print comparison benchmark results."""
    print("\n" + "=" * 100)
    print("BENCHMARK COMPARISON: Original Python vs Rust-Accelerated")
    print("=" * 100)

    operations = {}
    for r in results:
        op = r["operation"]
        if op not in operations:
            operations[op] = []
        operations[op].append(r)

    for op, op_results in operations.items():
        print(f"\n--- {op.upper().replace('_', ' ')} ---")
        print(f"  {'Name':<35} {'Original':>10} {'Fast':>10} {'Speedup':>10}")
        print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")
        for r in op_results:
            name = r["name"]
            orig_ms = r["original_median_s"] * 1000
            fast_ms = r["fast_median_s"] * 1000
            speedup = r["speedup_median"]
            print(f"  {name:<35} {orig_ms:>8.1f}ms {fast_ms:>8.1f}ms {speedup:>8.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark comparison: Python vs Rust-accelerated")
    parser.add_argument("--test-logs-dir", default="test_logs", help="Directory with test logs")
    parser.add_argument("--iterations", type=int, default=5, help="Number of timing iterations")
    parser.add_argument("--output", default="results/benchmark_comparison.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    print(f"Running comparison benchmarks with {args.iterations} iterations...")
    results = run_comparison_benchmarks(args.test_logs_dir, args.iterations)

    print_results(results)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
