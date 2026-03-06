# Progress Log: core_rust_implementation

## Branch merge: fast-path optimization and additional tests 03/05/2026 23:48 - commit pending

### What was done
- Incorporated NaN/Inf fast-path optimization from Branch 1: try standard serde_json parse first, only fall back to preprocessing if it fails. Avoids preprocessing overhead for the common case (files without NaN/Inf).
- Added 5 additional test cases from Branch 2: location field verification, IO[bytes] fallback for both formats, auto-detection test
- Fixed critical import bug (inspect_ai._util.concurrency → inspect_ai._util._async) found during review
- Added 4 module-attribute tests to exercise sync fallback paths that weren't being tested
- Total: 79 tests, all passing

### Key findings
- Fast-path optimization doesn't change overall benchmark numbers significantly (Pydantic is the bottleneck, not JSON parsing), but avoids unnecessary work in the common case
- Final benchmark profile: .eval full reads ~2x, batch headers ~3.2x, no regressions elsewhere

## Code quality fixes and header-only fallback 03/05/2026 23:17 - commit bce53ac

### What was done
- Code quality review found several issues; all fixed
- Added fallback to original for single-file header-only .eval reads (eliminates 4.5x regression for 1000-sample files)
- Batch headers still use Rust in threads for parallelism (3.77x speedup for 50 files)
- Removed unused logging import, duplicate results file, unnecessary summaries.json parsing
- Added operator precedence parentheses, -Infinity boundary guard, fixed unused variables

### Key findings
- Single-file header-only .eval: Rust zip crate parses full central directory, slower than original's targeted reads
- Batch headers benefit enormously from true thread parallelism (asyncio.to_thread + asyncio.gather): 3.77x
- Final benchmark profile: no regressions anywhere, speedups on .eval full reads (2.12x) and batch headers (3.77x)

## Optimized .json handling: fall back to original 03/05/2026 23:05 - commit 9bea610

### What was done
- Profiled Rust read_json_file vs pydantic_core.from_json: 115ms vs 35ms for 1000-sample file
- Changed .json format to fall back to original implementation entirely (both full and header-only)
- Cleaned up unused code (_build_eval_log_from_json_file, _validate_version, read_json_file import in _patch.py)
- Re-ran benchmarks: .json reads now at ~1.0x (no regression)

### Key findings
- pydantic_core.from_json is ~3x faster at JSON→Python conversion than our serde_json→PyDict path
- This is because pydantic_core is specifically optimized for JSON→Python (it's Rust internally too)
- The Rust read_json_file function still exists and works, just not used in monkey-patching
- Final .eval speedups: 1.75-2.04x for full reads, 2.85-2.92x for batch headers

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
- **.eval full reads**: 1.75-2.04x speedup (scales with sample count)
- **.json full reads**: Falls back to original (~1.0x) — pydantic_core.from_json is already Rust-backed and faster
- **Batch headers**: 2.85-2.92x speedup from concurrent execution
- **Pydantic model_validate is still the dominant bottleneck** — at 1000 samples, it accounts for most of the ~1s read time even with Rust parsing
- **Upstream bug found**: inspect_ai's `_read_header_streaming` crashes on error/cancelled .json logs when results=null (TypeError: EvalResults(**v) where v is None)

### Notes
- The 2.04x speedup for .eval 1000-sample full reads is meaningful but below the 5x target — bypassing Pydantic (Segments 3/4) is needed
- .eval header-only for large files has regression (26ms vs 6ms for 1000 samples) due to reading full ZIP — per instructions this is acceptable
