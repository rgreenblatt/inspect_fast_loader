"""Benchmark inspect_fast_loader against the original inspect_ai implementation.

Usage:
    python benchmark.py              # quick benchmark
    python benchmark.py --thorough   # more iterations, more file sizes
"""

import argparse
import gc
import time

import inspect_fast_loader
from inspect_ai.log._file import read_eval_log, read_eval_log_headers


def bench(fn, n=5, warmup=2):
    """Return min time in ms over n runs (after warmup)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        gc.disable()
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
        gc.enable()
        gc.collect()
    return min(times)


def run_benchmarks(thorough: bool = False):
    from pathlib import Path
    import subprocess, sys

    # Ensure test logs exist
    test_logs = Path("test_logs")
    if not test_logs.exists() or not list(test_logs.glob("*.eval")):
        subprocess.run([sys.executable, "generate_test_logs.py"], check=True)

    if thorough:
        # Generate 5000-sample log if missing
        if not (test_logs / "test_5000samples.eval").exists():
            subprocess.run([sys.executable, "-c", """
import random; from generate_test_logs import generate_log, write_eval_log, write_json_log
rng = random.Random(5042)
log = generate_log(rng, n_samples=5000)
write_eval_log(log, 'test_logs/test_5000samples.eval')
write_json_log(log, 'test_logs/test_5000samples.json')
"""], check=True)

    print(f"inspect_fast_loader.HAS_NATIVE: {inspect_fast_loader.HAS_NATIVE}")
    print()

    eval_files = {
        "100": "test_logs/test_100samples.eval",
        "1000": "test_logs/test_1000samples.eval",
    }
    json_files = {
        "100": "test_logs/test_100samples.json",
        "1000": "test_logs/test_1000samples.json",
    }
    if thorough:
        eval_files["10"] = "test_logs/test_10samples.eval"
        eval_files["5000"] = "test_logs/test_5000samples.eval"
        json_files["5000"] = "test_logs/test_5000samples.json"

    n = 10 if thorough else 5

    # Full reads
    print("=== Full reads ===")
    print(f"{'':>20} {'Original':>10} {'Fast':>10} {'Speedup':>8}")
    print("-" * 52)

    for label, path in sorted(eval_files.items(), key=lambda x: int(x[0])):
        t_orig = bench(lambda p=path: read_eval_log(p), n=n)

        inspect_fast_loader.patch()
        t_fast = bench(lambda p=path: read_eval_log(p), n=n)
        inspect_fast_loader.unpatch()

        print(f"  .eval {label:>5} samples {t_orig:>8.0f}ms {t_fast:>8.0f}ms {t_orig/t_fast:>7.1f}x")

    for label, path in sorted(json_files.items(), key=lambda x: int(x[0])):
        t_orig = bench(lambda p=path: read_eval_log(p), n=n)

        inspect_fast_loader.patch()
        t_fast = bench(lambda p=path: read_eval_log(p), n=n)
        inspect_fast_loader.unpatch()

        print(f"  .json {label:>5} samples {t_orig:>8.0f}ms {t_fast:>8.0f}ms {t_orig/t_fast:>7.1f}x")

    # Batch headers
    print()
    print("=== Batch headers ===")
    batch_files = sorted(str(f) for f in Path("test_logs").glob("batch_*.eval"))
    for count in ([10, 25, 50] if thorough else [25]):
        paths = batch_files[:count]
        t_orig = bench(lambda p=paths: read_eval_log_headers(p), n=n)

        inspect_fast_loader.patch()
        t_fast = bench(lambda p=paths: read_eval_log_headers(p), n=n)
        inspect_fast_loader.unpatch()

        print(f"  {count:>2} files           {t_orig:>8.0f}ms {t_fast:>8.0f}ms {t_orig/t_fast:>7.1f}x")

    # Single sample read
    print()
    print("=== Single sample read ===")
    from inspect_ai.log._file import read_eval_log_sample

    path = "test_logs/test_1000samples.eval"
    t_orig = bench(lambda: read_eval_log_sample(path, id=500, epoch=1), n=n)

    inspect_fast_loader.patch()
    t_fast = bench(lambda: read_eval_log_sample(path, id=500, epoch=1), n=n)
    inspect_fast_loader.unpatch()

    print(f"  .eval 1000        {t_orig:>8.1f}ms {t_fast:>8.1f}ms {t_orig/t_fast:>7.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thorough", action="store_true")
    args = parser.parse_args()
    run_benchmarks(thorough=args.thorough)
