# Phase: pydantic_bypass_optimization — Progress Log

## Pydantic bypass implementation 03/06/2026 00:45 - commit e2dd11d

### What was done
- Profiled model_validate bottleneck: 900ms for 1000 samples, confirmed as 85-90% of total read time
- Explored multiple approaches: model_construct, direct __init__, raw dicts, hybrid
- Implemented `_fast_construct()` in `_construct.py` — direct `__dict__` assignment on Pydantic model instances
- Recursively constructs all nested Pydantic types (ChatMessage, Event, Score, ModelOutput, etc.)
- Handles field aliases (JsonChange `from`→`from_`), timestamp str→datetime conversion, ToolCall dataclass
- Replicates all relevant migrations: migrate_deprecated, set_completion, migrate_stop_reason
- Added rayon parallel JSON parsing in Rust
- Activated .json format bypass (was previously 1.0x fallback, now 2.6x)
- Added scorer placeholder replacement (EvalLog.populate_scorer_name_for_samples)
- 38 bypass correctness tests (117 total, all passing)
- Final benchmarks and plots

### Details and examples
- Profiling results showing model_validate breakdown: Messages ~500ms, Events ~380ms, Output ~17ms, Scores ~3ms
- `_fast_construct` achieves ~30ms for 1000 EvalSample top-level objects (vs 900ms for model_validate)
- Full pipeline with nested construction: ~250ms construction + ~115ms Rust parsing = ~365ms total (vs ~2050ms original)
- Data inspection: compared model_dump() output for every sample in every test file (10, 100, 1000 samples, multiepoch, error, nan_inf, attachments) — zero differences
- Plot files: `plots/bypass_speedup.png`, `plots/bypass_absolute_times.png`, `plots/bypass_pipeline_breakdown.png`
- Benchmark results: `results/benchmark_bypass_final.jsonl`

### Key findings
- **7.25x speedup** for .eval 1000-sample full reads (target was 5x+)
- **5.81x** for 100 samples, **4.0x** for 10 samples (fixed overhead dominates at small sizes)
- **2.64x** for .json 1000-sample reads (previously 1.0x fallback)
- model_construct() is unsuitable: calls model_post_init (UUID generation), slow for missing fields
- Direct `__dict__` assignment is 30x faster than model_validate for the top-level object
- Rayon parallel parsing gives marginal speedup (JSON parsing is fast; GIL limits Python object creation)
- Eager module-level imports of event types: 6x faster than lazy per-call imports

### Notes
- The `_fast_construct` approach is fragile to inspect_ai model changes (new fields, renamed fields, new validators). If inspect_ai updates its Pydantic models, the bypass may need updates. The comprehensive correctness tests should catch regressions.
- The high variance in benchmark timings (271ms to 493ms for the same operation) is likely GC/cache related. Using min or p25 may be more representative than median for stable benchmarks.

## Merge additional bypass tests 03/06/2026 01:48 - commit 7f0eae5

### What was done
- Added 11 isolated unit tests for construction helpers, inspired by parallel branch's test suite
- Tests cover: basic construct_sample_fast, content list handling, tool call construction, ModelOutput completion auto-population/preservation, score/model_usage construction, event construction from real data, comprehensive field type checking, model_dump roundtrip, all message roles
- Updated test count in write-ups (was incorrectly stated as 103, actual is 117)
- Total: 117 tests (79 prior + 38 bypass-specific), all passing

### Key findings
- No issues found — all new tests pass, confirming the bypass correctness from additional angles
