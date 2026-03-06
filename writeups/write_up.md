# Inspect Fast Loader — Rust Native Extension for inspect_ai

## Project Goal
Implement a Rust native extension (PyO3/maturin) that monkey-patches inspect_ai's log reading functions for 5-10x+ performance improvements. Focus on local file reading of `.eval` (ZIP) and `.json` formats.

## Phase: documentation_scaffold_setup (Current)
See `write_up_documentation_scaffold_setup.md` for detailed findings.

### Key Results
- **Baseline established**: Full read of 1000 samples takes ~2s (.eval) / ~1.2s (.json). Header-only reads are already fast (~3-5ms).
- **Main bottleneck identified**: Pydantic model_validate on EvalSample (complex type with many nested models, 19 validators with data transformations).
- **Comprehensive documentation written**: INSPECT_LOG_FORMAT.md covers all data models, code paths, validators, type hierarchies, and edge cases (NaN/Inf, attachments, format detection).
- **Infrastructure ready**: Rust project compiles/imports, test log generator produces valid files, 25 tests passing, benchmark script operational.

## Important Choices
- Test logs generated via direct JSON/ZIP construction (simpler than using inspect's writer classes, verified loadable)
- Monkey-patching approach: replace `read_eval_log` and `read_eval_log_headers` on `inspect_ai.log._file` module
