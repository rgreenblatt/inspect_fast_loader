# inspect_fast_loader

Drop-in accelerator for [inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) log reading. ~6-8x faster with zero API changes.

The speedup comes from bypassing Pydantic `model_validate()` when constructing `EvalSample` objects — the dominant bottleneck (85-90% of read time). An optional Rust native extension provides additional ~2x on `.eval` (ZIP) files via faster decompression.

## Installation

```bash
pip install git+https://github.com/rgreenblatt/inspect_fast_loader.git
```

If a Rust toolchain is available, the native extension builds automatically. If not, the package installs as pure Python — no Rust required.

To install Rust (optional): https://rustup.rs

## Usage

### CLI

The package installs a `fast-inspect` command — a drop-in replacement for `inspect` with the fast loader pre-applied:

```bash
fast-inspect log dump mylog.eval
fast-inspect eval-retry mylog.eval
fast-inspect eval-set ...
```

### Python API

```python
import inspect_fast_loader

inspect_fast_loader.patch()   # monkey-patches inspect_ai.log._file functions
# ... use inspect_ai as normal — log reads are now accelerated ...
inspect_fast_loader.unpatch() # restore originals (optional)
```

## Performance

Tested against inspect_ai v0.3.188, 1000-sample .eval file:

| | Original | Fast (pure Python) | Fast (with Rust) |
|---|---|---|---|
| .eval full read | ~900ms | ~150ms (**6x**) | ~110ms (**8x**) |
| .json full read | ~560ms | ~90ms (**6x**) | ~90ms (**6x**) |

The Rust extension helps `.eval` files (~2x faster ZIP decompression). For `.json` files there's no difference since no ZIP is involved.

## Testing

```bash
python generate_test_logs.py
pytest tests/
```

## How it works

- **`_construct.py`**: Constructs `EvalSample` and all nested Pydantic models via direct `__dict__` assignment, skipping validators and `model_post_init`. Replicates all 7 data migration validators manually.
- **`_zip.py`**: Reads `.eval` ZIP files using Python's `zipfile` module (or the Rust extension when available). JSON parsing uses Python's `json.loads` which natively handles NaN/Inf.
- **`_patch.py`**: Monkey-patches `inspect_ai.log._file` functions to route through the fast implementations, with fallback to originals for unsupported inputs.

## Version compatibility

The Pydantic bypass hard-codes inspect_ai's model types. The package warns at import if the installed inspect_ai version differs from 0.3.188. Run the test suite after upgrading inspect_ai.
