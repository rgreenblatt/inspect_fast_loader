# Phase: documentation_scaffold_setup

## Goals
Set up the foundation for a Rust native extension (PyO3/maturin) that will monkey-patch inspect_ai's log reading functions with high-performance implementations. This phase covers:
1. Understanding inspect's log reading code
2. Documenting the log format and code paths
3. Setting up the Rust project skeleton
4. Building test infrastructure (log generator, tests, benchmarks)

## Key Findings

### Performance Baseline
Benchmark results (mean times, 3 iterations):

| Operation | .eval (1000 samples) | .json (1000 samples) |
|-----------|---------------------|---------------------|
| Full read | 2.08s | 1.19s |
| Header only | 0.005s | 0.003s |
| Summaries | 0.06-0.23s | N/A (uses full read) |
| Batch headers (50 files) | 0.10s | N/A |

Key observations:
- Full reads dominate — 1000 samples takes ~2s for .eval, ~1.2s for .json
- .eval is slower than .json for full reads (ZIP overhead + per-file JSON parsing + per-sample model_validate)
- Header-only reads are already fast (~3-5ms) due to streaming/targeted reads
- Batch header reads scale linearly (~2ms per file)
- **Pydantic model_validate is a major bottleneck** — each EvalSample goes through extensive validation/transformation

### Architecture Understanding
- The .eval format is a ZIP archive with individual JSON files per sample — this enables efficient header-only reads and single-sample reads
- The .json format is a monolithic JSON file — streaming header reads use ijson to avoid parsing the samples array
- NaN/Inf values in JSON are a real concern: ijson can't handle them, requiring fallback to full parsing
- 19 model_validator transformations need to be replicated if bypassing Pydantic (documented in INSPECT_LOG_FORMAT.md)
- The deserialization context mechanism (`DESERIALIZING`, `MESSAGE_CACHE`) affects eval_id generation and message deduplication

### Important Choices
- **Test log generation approach**: Used direct JSON/ZIP construction rather than inspect's writer classes. Simpler and the output is verified loadable by inspect. The downside is it may drift from inspect's actual output format in future versions, but since we verify loadability in tests, this would be caught.
- **Rust project structure**: Used maturin mixed Python/Rust project with the native module at `inspect_fast_loader._native`. Python wrappers in `inspect_fast_loader/` provide the public API and patching logic.

## What Was Done
1. Cloned inspect_ai repo and studied all key code paths (see INSPECT_LOG_FORMAT.md)
2. Wrote comprehensive INSPECT_LOG_FORMAT.md documenting data models, code paths, model validators, type hierarchies, format details
3. Set up Rust/PyO3 project with minimal functions (parse_json_bytes, list_zip_entries, read_zip_member)
4. Built test log generator producing 68 log files in both formats with edge cases
5. Created 25 tests covering all major scenarios (both formats, error/cancelled/NaN/attachments/multiepoch, patching)
6. Created benchmark script establishing performance baseline

## Current Status
Phase complete. All deliverables in place:
- `INSPECT_LOG_FORMAT.md` — comprehensive documentation
- `inspect_fast_loader/` — Rust project skeleton, compiles and imports
- `generate_test_logs.py` — deterministic test log generator
- `benchmark_baseline.py` — baseline performance measurements
- 25 tests, all passing

## Next Steps (for subsequent segments)
1. Implement Rust-accelerated JSON parsing (handle NaN/Inf)
2. Implement Rust ZIP reading for .eval format
3. Profile and optimize the Pydantic bottleneck
4. Consider hybrid approach: Rust for parsing, Python for Pydantic validation initially
