# Progress Log: documentation_scaffold_setup

## Initial setup and implementation 03/05/2026 22:07 - commit 7b4d0b8

### What was done
- Cloned inspect_ai repo and studied all key log reading code paths in detail
- Studied the complete data model: EvalLog, EvalSample, EvalSampleSummary, EvalSpec, EvalConfig, EvalPlan, EvalResults, EvalStats, EvalScore, etc.
- Studied all 19 model_validator transformations that affect deserialization
- Studied the Event discriminated union (19 event types), ChatMessage union (4 types), Content union (8 types)
- Studied the deserialization context mechanism (DESERIALIZING flag, MESSAGE_CACHE)
- Studied the NaN/Inf handling in both formats (ijson limitation, pydantic_core.from_json handling)
- Wrote comprehensive INSPECT_LOG_FORMAT.md covering all of the above
- Set up Rust/PyO3 project (inspect_fast_loader/) with maturin, verified compilation and import
- Created 3 Rust functions: parse_json_bytes, list_zip_entries, read_zip_member
- Created monkey-patching skeleton (_patch.py) with passthrough wrappers
- Built test log generator (generate_test_logs.py) producing 68 files across 7 scenarios
- Created 25 tests in 2 test files (test_smoke.py, test_log_generator.py), all passing
- Created benchmark_baseline.py and ran initial benchmarks

### Details and examples
- **INSPECT_LOG_FORMAT.md**: ~600 lines of documentation covering data models with field tables, ZIP internal structure, JSON format, all code paths for reading, model validators, type hierarchies, NaN/Inf handling
- **Test log scenarios**: 10/100/1000 samples, multi-epoch (20x3), error status, cancelled status, NaN/Inf values, attachments, empty samples, batch of 50 small logs
- **Benchmark results**: Full read of 1000 samples: ~2.08s (.eval), ~1.19s (.json). Header-only: ~3-5ms. Batch 50 headers: ~100ms.

### Key findings
- Pydantic model_validate is likely the main bottleneck for full log reads (EvalSample is the heaviest)
- .eval format is slower than .json for full reads despite smaller file size — per-file ZIP extraction + per-sample JSON parse + model_validate overhead
- Header-only reads are already quite fast
- The message deduplication cache in ChatMessageBase._wrap validator is important for performance during deserialization

### Notes
- Used PyO3 v0.23 (stable, compatible with our Rust toolchain)
- test_logs/ and target/ are gitignored since they're generated artifacts
- All generated logs verified loadable by inspect's Python API

## Merge improvements 03/05/2026 22:26 - commit f70df67

### What was done
- Extended monkey-patching to cover all 4 functions (sync + async): `read_eval_log`, `read_eval_log_async`, `read_eval_log_headers`, `read_eval_log_headers_async`
- Added wrapper attribute marking (`_is_fast_loader_wrapper = True`) on patched functions for easy detection of patching status
- Added `is_patched()` function to public API
- Added string sample IDs test case and test log generation (inspect supports both int and str IDs)
- Documented important subtlety: .json streaming header parser uses `EvalSpec(**v)` (not `model_validate` with context), so deserialization context is NOT passed — `eval_id` generation differs from full-parse path
- Updated continuation_context.md with learned context from parallel branches
- Tests: 28 passing (up from 25)

### Key findings
- The .json streaming header parser (`_read_header_streaming`) bypasses the deserialization context, which means `EvalSpec.model_post_init` generates `eval_id` via random UUID rather than deterministic hash. This is a subtle but important discrepancy for the Rust implementation to be aware of.
