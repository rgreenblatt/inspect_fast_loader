# Continuation Context

## Scripts and Purposes
- `generate_test_logs.py` — Generates deterministic test log files in .eval and .json formats (70 files total). Run with `python generate_test_logs.py --output-dir test_logs`.
- `benchmark_baseline.py` — Benchmarks original Python log reading performance.
- `benchmark_comparison.py` — Benchmarks original vs Rust-accelerated side by side. Run with `python benchmark_comparison.py`.
- `plot_benchmark.py` — Generates benchmark comparison plots from results/benchmark_comparison.jsonl.
- `inspect_fast_loader/src/lib.rs` — Rust extension: JSON parser (NaN/Inf safe), read_eval_file (ZIP), read_json_file, plus older utilities.
- `inspect_fast_loader/python/inspect_fast_loader/_patch.py` — Monkey-patching: replaces 4 inspect functions with Rust-accelerated implementations. Falls back to original for IO[bytes] and header-only .json.
- `inspect_fast_loader/tests/test_correctness.py` — 42 correctness tests comparing all fields between original and fast implementations.

Build Rust extension: `cd inspect_fast_loader && PATH=../.venv/bin:$PATH maturin develop --release`

## Learned Context
- PyO3 v0.23 API: `bool.into_pyobject()` returns a `Borrowed<PyBool>` — need `.to_owned().into_any().unbind()` to convert to `PyObject`.
- `EvalLogInfo` is in `inspect_ai.log._file`, NOT in `inspect_ai.log._log`.
- The cancelled log test: inspect returns `samples=None` for cancelled .json logs but `samples=[]` for cancelled .eval logs (when empty samples list is written to ZIP).
- **Upstream bug**: inspect_ai's `_read_header_streaming` crashes on error/cancelled .json logs when `results=null` (TypeError: `EvalResults(**v)` where v is None). Our tests handle this gracefully.
- The `.json` header streaming parser uses `EvalSpec(**v)` (kwargs) not `model_validate(v, context=...)` — deserialization context NOT passed, so eval_id generation differs.
- `.json` full reads with Rust show slight regression vs original because `pydantic_core.from_json()` is already Rust-backed internally. The extra serde_json→Python dict conversion step adds overhead.
- For `.eval` files, the speedup comes from faster per-sample JSON parsing and ZIP decompression in Rust, despite still using Pydantic model_validate.

## Current State
Phase `core_rust_implementation` is complete. Core Rust-accelerated reading works for .eval full reads with ~2x speedup on 1000-sample files. Falls back to original for .json format, header-only reads, and IO[bytes] input.

## Phase-specific write-ups
- `writeups/write_up_core_rust_implementation.md` — Core implementation write-up with benchmark results and plots
- `writeups/progress_log_core_rust_implementation.md` — Progress log
- `writeups/write_up_documentation_scaffold_setup.md` — Prior phase write-up
- `writeups/progress_log_documentation_scaffold_setup.md` — Prior phase progress log
- `writeups/write_up.md` — Overall project write-up (key takeaways only)
