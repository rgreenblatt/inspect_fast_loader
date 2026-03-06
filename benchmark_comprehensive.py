"""Comprehensive benchmark suite for inspect_fast_loader.

Measures ALL operations across all configurations:
- Full reads: .eval and .json, varying sample counts
- Header-only reads: .eval and .json
- Batch header reads: varying file counts
- Single sample reads (new)
- Sample summaries reads (new)
- Streaming sample reads (new)

Saves results to results/ and generates comparison plots in plots/.
"""

import json
import os
import sys
import time
from datetime import datetime
from glob import glob

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def time_operation(fn, n_runs=5, warmup=1):
    """Time an operation, returning median time in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def benchmark_full_reads(eval_files_by_size, json_files_by_size, results):
    """Benchmark full reads for .eval and .json at different sizes."""
    from inspect_ai.log._file import read_eval_log

    for label, files in eval_files_by_size.items():
        if not files:
            continue
        f = files[0]
        t = time_operation(lambda f=f: read_eval_log(f))
        results.append({"operation": "full_read_eval", "size": label, "time_ms": t * 1000, "config": "current"})
        print(f"  .eval full read ({label}): {t*1000:.1f}ms")

    for label, files in json_files_by_size.items():
        if not files:
            continue
        f = files[0]
        t = time_operation(lambda f=f: read_eval_log(f))
        results.append({"operation": "full_read_json", "size": label, "time_ms": t * 1000, "config": "current"})
        print(f"  .json full read ({label}): {t*1000:.1f}ms")


def benchmark_header_reads(eval_files, json_files, results):
    """Benchmark header-only reads."""
    from inspect_ai.log._file import read_eval_log

    if eval_files:
        f = eval_files[0]
        t = time_operation(lambda f=f: read_eval_log(f, header_only=True))
        results.append({"operation": "header_only_eval", "time_ms": t * 1000, "config": "current"})
        print(f"  .eval header-only: {t*1000:.1f}ms")

    if json_files:
        f = json_files[0]
        t = time_operation(lambda f=f: read_eval_log(f, header_only=True))
        results.append({"operation": "header_only_json", "time_ms": t * 1000, "config": "current"})
        print(f"  .json header-only: {t*1000:.1f}ms")


def benchmark_batch_headers(eval_files, results):
    """Benchmark batch header reads at different file counts."""
    from inspect_ai.log._file import read_eval_log_headers

    for n_files in [10, 25, 50, len(eval_files)]:
        files = eval_files[:n_files]
        if len(files) < n_files and n_files != len(eval_files):
            continue
        actual_n = len(files)
        t = time_operation(lambda files=files: read_eval_log_headers(files))
        results.append({"operation": "batch_headers", "size": f"{actual_n}_files", "time_ms": t * 1000, "config": "current", "n_files": actual_n})
        print(f"  batch headers ({actual_n} files): {t*1000:.1f}ms")


def benchmark_single_sample(eval_files, results):
    """Benchmark single sample reads."""
    from inspect_ai.log._file import read_eval_log_sample

    if not eval_files:
        return

    f = eval_files[0]
    t = time_operation(lambda f=f: read_eval_log_sample(f, id=1, epoch=1))
    results.append({"operation": "single_sample", "time_ms": t * 1000, "config": "current"})
    print(f"  single sample read: {t*1000:.1f}ms")

    # With exclude_fields
    t = time_operation(lambda f=f: read_eval_log_sample(f, id=1, epoch=1, exclude_fields={"store", "attachments"}))
    results.append({"operation": "single_sample_excluded", "time_ms": t * 1000, "config": "current"})
    print(f"  single sample read (exclude_fields): {t*1000:.1f}ms")


def benchmark_summaries(eval_files, results):
    """Benchmark sample summaries reads."""
    from inspect_ai.log._file import read_eval_log_sample_summaries

    if not eval_files:
        return

    f = eval_files[0]
    t = time_operation(lambda f=f: read_eval_log_sample_summaries(f))
    results.append({"operation": "sample_summaries", "time_ms": t * 1000, "config": "current"})
    print(f"  sample summaries: {t*1000:.1f}ms")


def benchmark_streaming_samples(eval_files, results):
    """Benchmark streaming sample reads."""
    from inspect_ai.log._file import read_eval_log_samples

    if not eval_files:
        return

    f = eval_files[0]
    t = time_operation(lambda f=f: list(read_eval_log_samples(f)))
    results.append({"operation": "streaming_samples", "time_ms": t * 1000, "config": "current"})
    print(f"  streaming samples: {t*1000:.1f}ms")


def run_benchmarks(config_name, eval_files_by_size, json_files_by_size, all_eval_files, all_json_files):
    """Run all benchmarks for a given configuration."""
    results = []

    print(f"\n{'='*60}")
    print(f"Configuration: {config_name}")
    print(f"{'='*60}")

    print("\nFull reads:")
    benchmark_full_reads(eval_files_by_size, json_files_by_size, results)

    print("\nHeader-only reads:")
    benchmark_header_reads(all_eval_files, all_json_files, results)

    print("\nBatch headers:")
    benchmark_batch_headers(all_eval_files, results)

    print("\nSingle sample reads:")
    benchmark_single_sample(all_eval_files, results)

    print("\nSample summaries:")
    benchmark_summaries(all_eval_files, results)

    print("\nStreaming samples:")
    benchmark_streaming_samples(all_eval_files, results)

    # Tag with config name
    for r in results:
        r["config"] = config_name

    return results


def find_files_by_size(file_list):
    """Group files by approximate sample count."""
    from inspect_ai.log._file import read_eval_log
    import inspect_fast_loader
    was_patched = inspect_fast_loader.is_patched()
    if not was_patched:
        inspect_fast_loader.patch()

    by_size = {}
    for f in file_list:
        try:
            log = read_eval_log(f)
            n = len(log.samples) if log.samples else 0
        except (ValueError, KeyError, FileNotFoundError):
            continue
        if n >= 1000:
            label = "1000_samples"
        elif n >= 100:
            label = "100_samples"
        elif n >= 10:
            label = "10_samples"
        elif n >= 5:
            label = "5_samples"
        else:
            continue
        by_size.setdefault(label, []).append(f)

    if not was_patched:
        inspect_fast_loader.unpatch()

    return by_size


def main():
    import inspect_fast_loader

    test_dir = "test_logs"
    all_eval_files = sorted(glob(os.path.join(test_dir, "*.eval")))
    all_json_files = sorted(glob(os.path.join(test_dir, "*.json")))

    print(f"Found {len(all_eval_files)} .eval files, {len(all_json_files)} .json files")

    # Categorize files by size
    eval_by_size = find_files_by_size(all_eval_files)
    json_by_size = find_files_by_size(all_json_files)

    print("\nFile size categories:")
    for label, files in sorted(eval_by_size.items()):
        print(f"  .eval {label}: {len(files)} files")
    for label, files in sorted(json_by_size.items()):
        print(f"  .json {label}: {len(files)} files")

    all_results = []

    # Benchmark original (unpatched)
    inspect_fast_loader.unpatch()
    results = run_benchmarks("original", eval_by_size, json_by_size, all_eval_files, all_json_files)
    all_results.extend(results)

    # Benchmark fast (patched)
    inspect_fast_loader.patch()
    results = run_benchmarks("fast", eval_by_size, json_by_size, all_eval_files, all_json_files)
    all_results.extend(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"benchmark_comprehensive_{timestamp}.jsonl")
    with open(results_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {results_file}")

    # Print speedup summary
    print("\n" + "=" * 60)
    print("SPEEDUP SUMMARY")
    print("=" * 60)

    orig_by_op = {}
    fast_by_op = {}
    for r in all_results:
        key = (r["operation"], r.get("size", ""))
        if r["config"] == "original":
            orig_by_op[key] = r["time_ms"]
        elif r["config"] == "fast":
            fast_by_op[key] = r["time_ms"]

    for key in sorted(orig_by_op.keys()):
        if key in fast_by_op:
            op, size = key
            orig_t = orig_by_op[key]
            fast_t = fast_by_op[key]
            speedup = orig_t / fast_t if fast_t > 0 else float("inf")
            size_str = f" ({size})" if size else ""
            print(f"  {op}{size_str}: {orig_t:.1f}ms -> {fast_t:.1f}ms ({speedup:.2f}x)")

    return results_file


if __name__ == "__main__":
    main()
