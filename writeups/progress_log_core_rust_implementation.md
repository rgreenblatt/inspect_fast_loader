# Progress Log: core_rust_implementation

## Core Rust implementation complete 03/05/2026 22:52 - commit d4fb9ca

### What was done
- Implemented NaN/Inf-safe JSON parsing in Rust via pre-processing sentinel approach
- Implemented `read_eval_file` (ZIP + JSON) and `read_json_file` in Rust
- Updated monkey-patching from passthrough wrappers to actual Rust-accelerated implementations
- Added fallbacks for IO[bytes] input and header-only .json reads
- Created 42 correctness tests comparing all fields between original and fast implementations
- Created benchmark comparison script and plots

### Details and examples
- **Rust code**: `inspect_fast_loader/src/lib.rs` — 357 lines, 5 exported functions + NaN/Inf pre-processor + JSON→Python converter + 4 Rust unit tests
- **Monkey-patching**: `inspect_fast_loader/python/inspect_fast_loader/_patch.py` — Replaces 4 functions (sync + async variants of read_eval_log and read_eval_log_headers)
- **Correctness tests**: `inspect_fast_loader/tests/test_correctness.py` — 42 tests covering both formats × 9 log types (10/100/1000 samples, multiepoch, error, cancelled, nan_inf, attachments, empty) × header-only and full read
- **Benchmark results**: `results/benchmark_comparison.jsonl`
- **Plots**: `plots/benchmark_speedup.png`, `plots/benchmark_absolute_times.png`

### Key findings
- **.eval full reads**: 1.84-2.19x speedup (scales with sample count)
- **.json full reads**: ~0.9x (slight regression) — pydantic_core.from_json is already Rust-backed, our extra dict conversion step adds overhead
- **Batch headers**: 2.7-2.8x speedup from concurrent execution
- **Pydantic model_validate is still the dominant bottleneck** — at 1000 samples, it accounts for most of the ~1s read time even with Rust parsing
- **Upstream bug found**: inspect_ai's `_read_header_streaming` crashes on error/cancelled .json logs when results=null (TypeError: EvalResults(**v) where v is None)

### Notes
- The `.json` full read regression is expected: serde_json→Python dict→model_validate vs pydantic_core.from_json→model_validate. The extra conversion step costs ~5-10% overhead.
- Header-only .json falls back to original (ijson streaming is optimal for this case)
- The 2.19x speedup for .eval 1000-sample full reads is meaningful but below the 5x target — bypassing Pydantic (Segments 3/4) is needed
