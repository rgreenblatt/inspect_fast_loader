# inspect_fast_loader

A Rust/Python extension that monkey-patches [inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai)'s log reading functions for 5-13x speedups.

The speedup comes from two techniques:
- **Pydantic bypass**: Constructing `EvalSample` and nested model instances via direct `__dict__` assignment instead of `model_validate()`, which is the dominant bottleneck (85-90% of read time).
- **Rust native extension** (PyO3/maturin): Faster ZIP decompression and parallel batch header reading via rayon.

## Performance

Fresh-process benchmark results (isolated subprocesses, tested against inspect_ai v0.3.188):

| Operation | Original | Fast | Speedup |
|---|---|---|---|
| .eval full read (1000 samples) | 2059ms | 376ms | **5.48x** |
| .eval full read (100 samples) | 171ms | 23ms | **7.36x** |
| .json full read (1000 samples) | 1095ms | 388ms | **2.83x** |
| batch headers (50 files) | 93ms | 9ms | **10.36x** |
| single sample read | 5.2ms | 0.4ms | **13.0x** |
| sample summaries | 3.4ms | 0.5ms | **6.80x** |

## Usage

```python
import inspect_fast_loader

inspect_fast_loader.patch()   # monkey-patches inspect_ai.log._file functions
# ... use inspect_ai as normal — log reads are now accelerated ...
inspect_fast_loader.unpatch() # restore originals (optional)
```

Nine functions are patched: `read_eval_log`, `read_eval_log_async`, `read_eval_log_headers`, `read_eval_log_headers_async`, `read_eval_log_sample`, `read_eval_log_sample_async`, `read_eval_log_sample_summaries`, `read_eval_log_sample_summaries_async`, and `read_eval_log_samples`.

## Building

Requires Rust (via rustup) and a Python venv with inspect_ai installed.

```bash
cd inspect_fast_loader
maturin develop --release
```

If `maturin develop` can't find `rustc`, set the environment explicitly:

```bash
RUSTUP_HOME=$HOME/.rustup CARGO_HOME=$HOME/.cargo PATH=$HOME/.cargo/bin:$PATH maturin develop --release
```

## Testing

Generate test logs, then run pytest:

```bash
python generate_test_logs.py --output-dir test_logs
cd inspect_fast_loader
pytest tests/ -v
```

## Benchmarking

```bash
# Primary benchmark (reliable, fresh-process isolation)
python benchmark_fresh_process.py

# Secondary benchmark (in-process, may have caching artifacts)
python benchmark_comprehensive.py
```

## Fragility / Version Compatibility

The Pydantic bypass in `_construct.py` hard-codes inspect_ai's model types and field names. If inspect_ai changes its models, the bypass may produce incorrect results. The package warns at import time if the installed inspect_ai version differs from the tested version (0.3.188). Run the test suite to verify correctness after upgrading.
