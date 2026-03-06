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
- **.eval full read 1000 samples**: 2132ms → 973ms (**2.19x speedup**)
- **.eval full read 100 samples**: 178ms → 96ms (**1.84x speedup**)
- **Batch headers (50 .eval files)**: 98ms → 35ms (**2.84x speedup**)
- **.json full reads**: ~0.9x (slight regression — pydantic_core.from_json is already Rust-backed)
- **Pydantic model_validate remains the dominant bottleneck** — must be bypassed for 5x+ targets

![Benchmark Speedup](../plots/benchmark_speedup.png)

### What Was Built
- Rust NaN/Inf-safe JSON parser (pre-processing sentinel approach)
- Rust `.eval` reader: ZIP decompression + JSON parsing → Python dicts
- Rust `.json` reader: file read + JSON parsing → Python dict
- Monkey-patching with actual fast implementations (falls back to original for IO[bytes] and header-only .json)
- 70 tests total (42 correctness + 28 existing), all passing

## Important Choices
- Test logs generated via direct JSON/ZIP construction (simpler, verified loadable)
- Monkey-patching approach: replace 4 functions on `inspect_ai.log._file` module
- NaN/Inf: pre-processing sentinel approach (simple, fast, correct)
- Header-only .json: fall back to original (ijson streaming is faster than full parse + discard)
- IO[bytes] input: fall back to original (Rust functions expect file paths)
