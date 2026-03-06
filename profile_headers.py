"""Profile batch header reading to identify bottlenecks."""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def profile_original_headers(log_files: list[str], n_runs: int = 5):
    """Profile original implementation."""
    import inspect_fast_loader
    inspect_fast_loader.unpatch()

    from inspect_ai.log._file import read_eval_log_headers

    # Warmup
    read_eval_log_headers(log_files)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        read_eval_log_headers(log_files)
        times.append(time.perf_counter() - t0)

    median = sorted(times)[len(times) // 2]
    print(f"Original batch headers ({len(log_files)} files): {median*1000:.1f}ms (median of {n_runs})")
    return median


def profile_rust_headers(log_files: list[str], n_runs: int = 5):
    """Profile our fast implementation."""
    import inspect_fast_loader
    inspect_fast_loader.patch()

    from inspect_ai.log._file import read_eval_log_headers

    # Warmup
    read_eval_log_headers(log_files)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        read_eval_log_headers(log_files)
        times.append(time.perf_counter() - t0)

    median = sorted(times)[len(times) // 2]
    print(f"Fast batch headers ({len(log_files)} files): {median*1000:.1f}ms (median of {n_runs})")
    return median


def profile_rust_header_breakdown(log_files: list[str]):
    """Break down where time is spent in the fast header path."""
    from inspect_fast_loader._native import read_eval_file
    from inspect_fast_loader._patch import _build_eval_log_from_eval_file

    # Time just the Rust ZIP read + JSON parse
    times_rust = []
    for _ in range(5):
        t0 = time.perf_counter()
        raws = [read_eval_file(f, header_only=True) for f in log_files]
        times_rust.append(time.perf_counter() - t0)
    median_rust = sorted(times_rust)[2]
    print(f"  Rust read_eval_file (header_only) x{len(log_files)}: {median_rust*1000:.1f}ms")

    # Time just the Python model_validate for headers
    raws = [read_eval_file(f, header_only=True) for f in log_files]
    times_python = []
    for _ in range(5):
        t0 = time.perf_counter()
        logs = [_build_eval_log_from_eval_file(raw, f, header_only=True) for raw, f in zip(raws, log_files)]
        times_python.append(time.perf_counter() - t0)
    median_python = sorted(times_python)[2]
    print(f"  Python _build_eval_log (header): {median_python*1000:.1f}ms")

    # Time Rust read_eval_file per-file (sequential)
    times_per_file = []
    for _ in range(3):
        file_times = []
        for f in log_files:
            t0 = time.perf_counter()
            read_eval_file(f, header_only=True)
            file_times.append(time.perf_counter() - t0)
        times_per_file.append(sum(file_times) / len(file_times))
    avg_per_file = sorted(times_per_file)[1]
    print(f"  Avg per-file Rust read: {avg_per_file*1000:.2f}ms")

    # Time with rayon parallel batch (new potential Rust function)
    print(f"  Total sequential Rust: {avg_per_file*len(log_files)*1000:.1f}ms")


def profile_original_header_breakdown(log_files: list[str]):
    """Break down where time is spent in original headers."""
    import inspect_fast_loader
    inspect_fast_loader.unpatch()

    from inspect_ai.log._file import read_eval_log
    from inspect_ai._util._async import run_coroutine

    # Time single header read
    times_single = []
    for _ in range(5):
        single_times = []
        for f in log_files[:5]:
            t0 = time.perf_counter()
            read_eval_log(f, header_only=True)
            single_times.append(time.perf_counter() - t0)
        times_single.append(sum(single_times) / len(single_times))
    avg_single = sorted(times_single)[2]
    print(f"  Original avg single header read: {avg_single*1000:.2f}ms")
    print(f"  Original sequential total ({len(log_files)} files): {avg_single*len(log_files)*1000:.1f}ms")


def main():
    from glob import glob
    test_dir = "test_logs"

    # Get all .eval files
    eval_files = sorted(glob(os.path.join(test_dir, "*.eval")))
    print(f"Found {len(eval_files)} .eval files\n")

    if not eval_files:
        print("No test logs found. Run: python generate_test_logs.py --output-dir test_logs")
        return

    # Profile with different batch sizes
    for n_files in [10, 25, len(eval_files)]:
        files = eval_files[:n_files]
        print(f"\n=== Batch of {n_files} files ===")
        t_orig = profile_original_headers(files)
        t_fast = profile_rust_headers(files)
        print(f"Speedup: {t_orig/t_fast:.2f}x")

    # Detailed breakdown for all files
    print(f"\n=== Detailed breakdown ({len(eval_files)} files) ===")
    print("\nFast implementation breakdown:")
    import inspect_fast_loader
    inspect_fast_loader.patch()
    profile_rust_header_breakdown(eval_files)

    print("\nOriginal implementation breakdown:")
    profile_original_header_breakdown(eval_files)


if __name__ == "__main__":
    main()
