"""Fresh-process benchmark: runs each benchmark in an isolated subprocess.

This avoids in-process caching artifacts (Python memory allocator, OS page cache
warming from prior reads, etc.) that can make repeated in-process benchmarks
unreliable — especially for the original implementation which benefits heavily
from warm caches.

Each measurement spawns a fresh Python process, does one warmup read, then
measures a single read. The median of N runs is reported.
"""

import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from glob import glob

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_RUNS = 5


def fresh_process_benchmark(code_template: str, config: str, n_runs: int = N_RUNS) -> dict:
    """Run a benchmark in fresh subprocesses and return timing stats."""
    code = code_template.replace("__CONFIG__", config)
    times = []
    for _ in range(n_runs):
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120,
            cwd=os.getcwd(),
        )
        if r.returncode != 0:
            print(f"  ERROR: {r.stderr[:200]}")
            continue
        times.append(float(r.stdout.strip()))
    if not times:
        return {"median": 0, "min": 0, "max": 0, "times": []}
    times.sort()
    return {
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "times": times,
    }


SETUP_CODE = '''
import time, sys
sys.path.insert(0, "inspect_fast_loader/python")
import inspect_fast_loader
if "__CONFIG__" == "fast":
    inspect_fast_loader.patch()
from inspect_ai.log._file import (
    read_eval_log, read_eval_log_headers,
    read_eval_log_sample, read_eval_log_sample_summaries,
    read_eval_log_samples,
)
'''


def make_code(body: str) -> str:
    return SETUP_CODE + body


def main():
    test_dir = "test_logs"
    eval_files = sorted(glob(os.path.join(test_dir, "*.eval")))
    json_files = sorted(glob(os.path.join(test_dir, "*.json")))

    all_results = []

    # --- Full reads ---
    for fmt, ext in [("eval", ".eval"), ("json", ".json")]:
        for n_samples in [10, 100, 1000, 5000]:
            matching = [f for f in (eval_files if fmt == "eval" else json_files)
                        if f"{n_samples}samples" in f]
            if not matching:
                continue
            f = matching[0]
            label = f"full_read {fmt} {n_samples} samples"
            print(f"\n{label}:")

            body = f'''
read_eval_log("{f}")  # warmup
t0 = time.perf_counter()
read_eval_log("{f}")
t = time.perf_counter() - t0
print(f"{{t*1000:.1f}}")
'''
            code = make_code(body)
            for config in ["original", "fast"]:
                stats = fresh_process_benchmark(code, config)
                print(f"  {config}: median={stats['median']:.1f}ms min={stats['min']:.1f}ms max={stats['max']:.1f}ms")
                all_results.append({
                    "operation": label,
                    "config": config,
                    **stats,
                })

            if all_results[-2]["median"] > 0 and all_results[-1]["median"] > 0:
                speedup = all_results[-2]["median"] / all_results[-1]["median"]
                print(f"  speedup: {speedup:.2f}x")

    # --- Batch headers ---
    for n_files in [10, 25, 50, len(eval_files)]:
        files = eval_files[:n_files]
        if len(files) < n_files and n_files != len(eval_files):
            continue
        actual_n = len(files)
        file_list_str = repr(files)
        label = f"batch_headers {actual_n} files"
        print(f"\n{label}:")

        body = f'''
files = {file_list_str}
read_eval_log_headers(files)  # warmup
t0 = time.perf_counter()
read_eval_log_headers(files)
t = time.perf_counter() - t0
print(f"{{t*1000:.1f}}")
'''
        code = make_code(body)
        for config in ["original", "fast"]:
            stats = fresh_process_benchmark(code, config)
            print(f"  {config}: median={stats['median']:.1f}ms min={stats['min']:.1f}ms max={stats['max']:.1f}ms")
            all_results.append({
                "operation": label,
                "config": config,
                "n_files": actual_n,
                **stats,
            })

        if all_results[-2]["median"] > 0 and all_results[-1]["median"] > 0:
            speedup = all_results[-2]["median"] / all_results[-1]["median"]
            print(f"  speedup: {speedup:.2f}x")

    # --- Single sample ---
    f = eval_files[0]
    label = "single_sample"
    print(f"\n{label}:")
    body = f'''
read_eval_log_sample("{f}", id=1, epoch=1)  # warmup
t0 = time.perf_counter()
read_eval_log_sample("{f}", id=1, epoch=1)
t = time.perf_counter() - t0
print(f"{{t*1000:.1f}}")
'''
    code = make_code(body)
    for config in ["original", "fast"]:
        stats = fresh_process_benchmark(code, config)
        print(f"  {config}: median={stats['median']:.1f}ms")
        all_results.append({"operation": label, "config": config, **stats})

    if all_results[-2]["median"] > 0 and all_results[-1]["median"] > 0:
        print(f"  speedup: {all_results[-2]['median'] / all_results[-1]['median']:.2f}x")

    # --- Single sample with exclude_fields ---
    label = "single_sample_excluded"
    print(f"\n{label}:")
    body = f'''
read_eval_log_sample("{f}", id=1, epoch=1, exclude_fields={{"store", "attachments"}})
t0 = time.perf_counter()
read_eval_log_sample("{f}", id=1, epoch=1, exclude_fields={{"store", "attachments"}})
t = time.perf_counter() - t0
print(f"{{t*1000:.1f}}")
'''
    code = make_code(body)
    for config in ["original", "fast"]:
        stats = fresh_process_benchmark(code, config)
        print(f"  {config}: median={stats['median']:.1f}ms")
        all_results.append({"operation": label, "config": config, **stats})

    if all_results[-2]["median"] > 0 and all_results[-1]["median"] > 0:
        print(f"  speedup: {all_results[-2]['median'] / all_results[-1]['median']:.2f}x")

    # --- Sample summaries ---
    label = "sample_summaries"
    print(f"\n{label}:")
    body = f'''
read_eval_log_sample_summaries("{f}")
t0 = time.perf_counter()
read_eval_log_sample_summaries("{f}")
t = time.perf_counter() - t0
print(f"{{t*1000:.1f}}")
'''
    code = make_code(body)
    for config in ["original", "fast"]:
        stats = fresh_process_benchmark(code, config)
        print(f"  {config}: median={stats['median']:.1f}ms")
        all_results.append({"operation": label, "config": config, **stats})

    if all_results[-2]["median"] > 0 and all_results[-1]["median"] > 0:
        print(f"  speedup: {all_results[-2]['median'] / all_results[-1]['median']:.2f}x")

    # --- Streaming samples ---
    label = "streaming_samples"
    print(f"\n{label}:")
    body = f'''
list(read_eval_log_samples("{f}"))
t0 = time.perf_counter()
list(read_eval_log_samples("{f}"))
t = time.perf_counter() - t0
print(f"{{t*1000:.1f}}")
'''
    code = make_code(body)
    for config in ["original", "fast"]:
        stats = fresh_process_benchmark(code, config)
        print(f"  {config}: median={stats['median']:.1f}ms")
        all_results.append({"operation": label, "config": config, **stats})

    if all_results[-2]["median"] > 0 and all_results[-1]["median"] > 0:
        print(f"  speedup: {all_results[-2]['median'] / all_results[-1]['median']:.2f}x")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"benchmark_fresh_process_{timestamp}.jsonl")
    with open(results_file, "w") as fout:
        for r in all_results:
            fout.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {results_file}")

    # Print summary table
    print("\n" + "=" * 70)
    print("FRESH-PROCESS BENCHMARK SUMMARY")
    print("=" * 70)
    i = 0
    while i < len(all_results) - 1:
        orig = all_results[i]
        fast = all_results[i + 1]
        if orig["config"] == "original" and fast["config"] == "fast" and orig["operation"] == fast["operation"]:
            speedup = orig["median"] / fast["median"] if fast["median"] > 0 else 0
            print(f"  {orig['operation']}: {orig['median']:.1f}ms -> {fast['median']:.1f}ms ({speedup:.2f}x)")
            i += 2
        else:
            i += 1


if __name__ == "__main__":
    main()
