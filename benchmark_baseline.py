"""Benchmark baseline performance of inspect's Python log reading.

Measures current (unoptimized) performance for various operations and log sizes.
Outputs results to results/ directory as JSONL.
"""

import argparse
import json
import os
import time
from typing import Any

from generate_test_logs import generate_all_test_logs


def benchmark_operation(name: str, func, n_iterations: int = 5, warmup: int = 1) -> dict[str, Any]:
    """Benchmark a single operation, returning timing stats."""
    # Warmup
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


def run_benchmarks(test_logs_dir: str, n_iterations: int = 5) -> list[dict[str, Any]]:
    """Run all benchmarks and return results."""
    from inspect_ai.log import read_eval_log
    from inspect_ai.log._file import read_eval_log_headers, read_eval_log_sample_summaries

    results = []

    # ---------------------------------------------------------------------- #
    # Full log reads — .eval format
    # ---------------------------------------------------------------------- #
    for n_samples in [10, 100, 1000]:
        path = os.path.join(test_logs_dir, f"test_{n_samples}samples.eval")
        if not os.path.exists(path):
            continue
        result = benchmark_operation(
            f"full_read_eval_{n_samples}",
            lambda p=path: read_eval_log(p),
            n_iterations=n_iterations,
        )
        result["format"] = "eval"
        result["n_samples"] = n_samples
        result["operation"] = "full_read"
        result["file_size_bytes"] = os.path.getsize(path)
        results.append(result)

    # ---------------------------------------------------------------------- #
    # Full log reads — .json format
    # ---------------------------------------------------------------------- #
    for n_samples in [10, 100, 1000]:
        path = os.path.join(test_logs_dir, f"test_{n_samples}samples.json")
        if not os.path.exists(path):
            continue
        result = benchmark_operation(
            f"full_read_json_{n_samples}",
            lambda p=path: read_eval_log(p),
            n_iterations=n_iterations,
        )
        result["format"] = "json"
        result["n_samples"] = n_samples
        result["operation"] = "full_read"
        result["file_size_bytes"] = os.path.getsize(path)
        results.append(result)

    # ---------------------------------------------------------------------- #
    # Header-only reads — .eval format
    # ---------------------------------------------------------------------- #
    for n_samples in [10, 100, 1000]:
        path = os.path.join(test_logs_dir, f"test_{n_samples}samples.eval")
        if not os.path.exists(path):
            continue
        result = benchmark_operation(
            f"header_only_eval_{n_samples}",
            lambda p=path: read_eval_log(p, header_only=True),
            n_iterations=n_iterations,
        )
        result["format"] = "eval"
        result["n_samples"] = n_samples
        result["operation"] = "header_only"
        result["file_size_bytes"] = os.path.getsize(path)
        results.append(result)

    # ---------------------------------------------------------------------- #
    # Header-only reads — .json format
    # ---------------------------------------------------------------------- #
    for n_samples in [10, 100, 1000]:
        path = os.path.join(test_logs_dir, f"test_{n_samples}samples.json")
        if not os.path.exists(path):
            continue
        result = benchmark_operation(
            f"header_only_json_{n_samples}",
            lambda p=path: read_eval_log(p, header_only=True),
            n_iterations=n_iterations,
        )
        result["format"] = "json"
        result["n_samples"] = n_samples
        result["operation"] = "header_only"
        result["file_size_bytes"] = os.path.getsize(path)
        results.append(result)

    # ---------------------------------------------------------------------- #
    # Sample summaries — .eval format
    # ---------------------------------------------------------------------- #
    for n_samples in [10, 100, 1000]:
        path = os.path.join(test_logs_dir, f"test_{n_samples}samples.eval")
        if not os.path.exists(path):
            continue
        result = benchmark_operation(
            f"summaries_eval_{n_samples}",
            lambda p=path: read_eval_log_sample_summaries(p),
            n_iterations=n_iterations,
        )
        result["format"] = "eval"
        result["n_samples"] = n_samples
        result["operation"] = "summaries"
        result["file_size_bytes"] = os.path.getsize(path)
        results.append(result)

    # ---------------------------------------------------------------------- #
    # Batch header reading — .eval format
    # ---------------------------------------------------------------------- #
    for batch_size in [10, 50]:
        paths = [os.path.join(test_logs_dir, f"batch_{i:03d}.eval") for i in range(batch_size)]
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            continue
        result = benchmark_operation(
            f"batch_headers_eval_{len(paths)}",
            lambda ps=paths: read_eval_log_headers(ps),
            n_iterations=n_iterations,
        )
        result["format"] = "eval"
        result["n_files"] = len(paths)
        result["operation"] = "batch_headers"
        result["total_size_bytes"] = sum(os.path.getsize(p) for p in paths)
        results.append(result)

    return results


def print_results(results: list[dict[str, Any]]) -> None:
    """Print benchmark results in a readable format."""
    print("\n" + "=" * 80)
    print("BASELINE BENCHMARK RESULTS")
    print("=" * 80)

    # Group by operation
    operations = {}
    for r in results:
        op = r["operation"]
        if op not in operations:
            operations[op] = []
        operations[op].append(r)

    for op, op_results in operations.items():
        print(f"\n--- {op.upper().replace('_', ' ')} ---")
        for r in op_results:
            size_info = ""
            if "file_size_bytes" in r:
                size_mb = r["file_size_bytes"] / (1024 * 1024)
                size_info = f" [{size_mb:.2f} MB]"
            elif "total_size_bytes" in r:
                size_mb = r["total_size_bytes"] / (1024 * 1024)
                size_info = f" [{size_mb:.2f} MB total]"

            n_info = ""
            if "n_samples" in r:
                n_info = f" ({r['n_samples']} samples)"
            elif "n_files" in r:
                n_info = f" ({r['n_files']} files)"

            print(f"  {r['name']}{n_info}{size_info}:")
            print(f"    mean={r['mean_s']:.4f}s  min={r['min_s']:.4f}s  max={r['max_s']:.4f}s  median={r['median_s']:.4f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark inspect log reading performance")
    parser.add_argument("--test-logs-dir", default="test_logs", help="Directory with test logs")
    parser.add_argument("--iterations", type=int, default=5, help="Number of timing iterations")
    parser.add_argument("--generate", action="store_true", help="Generate test logs first")
    parser.add_argument("--output", default="results/benchmark_baseline.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    if args.generate or not os.path.exists(args.test_logs_dir):
        print(f"Generating test logs in {args.test_logs_dir}...")
        generate_all_test_logs(args.test_logs_dir)
        print("Done generating test logs.")

    print(f"Running benchmarks with {args.iterations} iterations...")
    results = run_benchmarks(args.test_logs_dir, args.iterations)

    print_results(results)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
