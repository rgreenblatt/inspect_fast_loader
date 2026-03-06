# Inspect Fast Loader — Rust Native Extension for inspect_ai

## Project Goal
Implement a Rust native extension (PyO3/maturin) that monkey-patches inspect_ai's log reading functions for 5-10x+ performance improvements. Focus on local file reading of `.eval` (ZIP) and `.json` formats.

## Phase: documentation_scaffold_setup (Complete)
See `write_up_documentation_scaffold_setup.md` for detailed findings.

- **Baseline established**: Full read of 1000 samples takes ~2s (.eval) / ~1.2s (.json). Header-only reads are already fast (~3-5ms).
- **Main bottleneck identified**: Pydantic model_validate on EvalSample.
- **Infrastructure ready**: Rust project compiles/imports, test log generator, 28 tests passing, benchmark operational.

## Phase: core_rust_implementation (Complete)
See `write_up_core_rust_implementation.md` for detailed findings and plots.

### Key Results
- **.eval full read 1000 samples**: 2091ms → 1025ms (**2.04x speedup**)
- **.eval full read 100 samples**: 175ms → 91ms (**1.92x speedup**)
- **Batch headers (50 .eval files)**: 93ms → 33ms (**2.85x speedup**)
- **.json format**: Falls back to original (~1.0x) — pydantic_core.from_json is already Rust-backed
- **Pydantic model_validate remains the dominant bottleneck** — must be bypassed for 5x+ targets

![Benchmark Speedup](../plots/benchmark_speedup.png)

### What Was Built
- Rust NaN/Inf-safe JSON parser (pre-processing sentinel approach)
- Rust `.eval` reader: ZIP decompression + JSON parsing → Python dicts
- Rust `.json` reader: file read + JSON parsing (not used in monkey-patching, falls back to original)
- Monkey-patching: replaces 4 functions, falls back to original for IO[bytes] and .json format
- 70 tests total (42 correctness + 28 existing), all passing

## Important Choices
- Test logs generated via direct JSON/ZIP construction (simpler, verified loadable)
- Monkey-patching approach: replace 4 functions on `inspect_ai.log._file` module
- NaN/Inf: pre-processing sentinel approach (simple, fast, correct)
- .json format: fall back to original (pydantic_core.from_json is already Rust-backed and faster)
- IO[bytes] input: fall back to original (Rust functions expect file paths)
