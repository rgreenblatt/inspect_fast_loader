# Continuation Context

## Scripts and Purposes
- `generate_test_logs.py` — Generates deterministic test log files in .eval and .json formats (70 files total). Run with `python generate_test_logs.py --output-dir test_logs`.
- `benchmark_baseline.py` — Benchmarks inspect's Python log reading performance. Run with `python benchmark_baseline.py`.
- `inspect_fast_loader/` — Rust/PyO3 project. Build with `cd inspect_fast_loader && maturin develop`.
- `inspect_fast_loader/src/lib.rs` — Minimal Rust functions (parse_json_bytes, list_zip_entries, read_zip_member).
- `inspect_fast_loader/python/inspect_fast_loader/_patch.py` — Monkey-patching for both sync and async log reading functions (currently passthrough wrappers).

## Learned Context
- PyO3 v0.23 API: `bool.into_pyobject()` returns a `Borrowed<PyBool>` — need `.to_owned().into_any().unbind()` to convert to `PyObject`.
- `read_eval_log_headers` and `read_eval_log_sample_summaries` are in `inspect_ai.log._file`, not in `inspect_ai.log` public API.
- The cancelled log test: inspect returns `samples=None` for cancelled .json logs but `samples=[]` for cancelled .eval logs (when empty samples list is written to ZIP).
- inspect_ai installed via `uv pip install inspect-ai` in the project venv.
- The `.json` header streaming parser (`_read_header_streaming` in `json.py`) uses `EvalSpec(**v)` (kwargs) not `model_validate(v, context=...)` — so the deserialization context is NOT passed in that code path. This is an important subtlety: `model_post_init` on `EvalSpec` won't see `DESERIALIZING=True` when reading .json headers via the streaming path, which means `eval_id` generation uses a random UUID instead of a deterministic hash. The Rust implementation should be aware of this discrepancy.
- Patched functions have `_is_fast_loader_wrapper = True` attribute, useful for detecting whether patching is active.

## Current State
Phase `documentation_scaffold_setup` is complete. All deliverables in place and tests passing.

## Phase-specific write-ups
- `writeups/write_up_documentation_scaffold_setup.md` — Phase write-up with findings and results
- `writeups/progress_log_documentation_scaffold_setup.md` — Chronological progress log
- `writeups/write_up.md` — Overall project write-up (key takeaways only)
