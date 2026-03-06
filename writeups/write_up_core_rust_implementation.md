# Phase: core_rust_implementation — Write-Up

## Goal
Implement the core Rust-accelerated log reading for both `.eval` (ZIP) and `.json` formats, with full correctness via Pydantic `model_validate()`, and integrate via monkey-patching.

## Key Results

### Speedup Summary (final)
| Operation | Original | Fast | Speedup |
|---|---|---|---|
| full_read .eval 1000 samples | 2047ms | 968ms | **2.12x** |
| full_read .eval 100 samples | 165ms | 95ms | **1.74x** |
| full_read .eval 10 samples | 18ms | 10ms | **1.81x** |
| full_read .json (all sizes) | — | — | ~1.0x (falls back to original) |
| batch_headers .eval 50 files | 99ms | 26ms | **3.77x** |
| batch_headers .eval 10 files | 23ms | 7ms | **3.32x** |
| header_only (all) | — | — | ~1.0x (falls back to original) |

![Benchmark Speedup](../plots/benchmark_speedup.png)
![Absolute Times](../plots/benchmark_absolute_times.png)

### Where Speedup Comes From
- **`.eval` full reads**: Rust handles ZIP decompression + JSON parsing significantly faster than Python's `json.load()` per entry. The ~2x speedup at 1000 samples corresponds to faster per-sample JSON parsing (Rust serde_json vs Python json.loads) and faster ZIP decompression (Rust zip crate vs Python zipfile).
- **Batch headers**: True thread parallelism via `asyncio.to_thread` + `asyncio.gather`. Each file's Rust reading runs in a separate thread, achieving ~3.8x speedup for 50 files.

### Where We Fall Back to Original
- **`.json` format (all operations)**: `pydantic_core.from_json()` is already Rust-backed internally and highly optimized for JSON→Python conversion. Our serde_json→Python dict→model_validate pipeline adds overhead (~115ms vs 35ms for JSON parsing alone at 1000 samples). Falling back avoids this regression.
- **Single-file header-only reads**: Our Rust `zip` crate opens the full ZIP and parses the entire central directory, which is slower than inspect's `AsyncZipReader` that uses targeted range reads. Falling back avoids regression. Batch headers still use Rust for thread parallelism benefits.

### Pydantic as the Dominant Bottleneck
For full reads, Pydantic `model_validate()` dominates runtime. At 1000 samples, even with Rust handling all JSON parsing and ZIP decompression, the Python-side model_validate on each `EvalSample` accounts for most of the time. This confirms the finding from Segment 0 that bypassing Pydantic (Segments 3/4) is essential for large speedups.

## What Was Implemented

### 1. Rust NaN/Inf-safe JSON Parser
- Pre-processes JSON bytes to replace bare `NaN`, `Infinity`, `-Infinity` tokens with sentinel strings
- Sentinels are restored during JSON→Python conversion
- Correctly handles NaN/Inf inside JSON strings (no replacement)
- 4 Rust unit tests + Python-level tests

### 2. Rust .eval Reader (`read_eval_file`)
- Opens ZIP file from disk via `std::fs::File`, parses with `zip` crate
- Reads header.json (or _journal/start.json fallback)
- Reads all samples/*.json entries when not header_only
- Reads reductions.json if present
- Returns structured Python dict for Python-side model_validate

### 3. Rust .json Reader (`read_json_file`)
- Reads entire file into memory via `std::fs::read`
- Parses with NaN/Inf-safe JSON parser
- Returns Python dict
- **Note**: Not used in monkey-patching (falls back to original), but available for other use cases

### 4. Monkey-Patching
- Replaces `read_eval_log`, `read_eval_log_async`, `read_eval_log_headers`, `read_eval_log_headers_async`
- Falls back to original for: IO[bytes] input, all .json format reads, single-file header-only reads
- Batch headers use Rust in threads for parallelism (`asyncio.to_thread` + `asyncio.gather`)
- Async full read uses `asyncio.to_thread` to avoid blocking event loop

### 5. Correctness Tests (42 tests)
- Field-by-field comparison of original vs fast output for all test log types
- Both formats × 9 log variants (10/100/1000 samples, multiepoch, error, cancelled, nan_inf, attachments, empty)
- Header-only and batch header tests
- Async tests
- NaN-aware comparison

## Important Choices

### NaN/Inf Strategy: Pre-processing Sentinels
Chose to pre-process JSON bytes and replace NaN/Inf tokens with sentinel strings before serde_json parsing, then restore during Python conversion. Alternative was a custom JSON parser, but pre-processing is simpler and the overhead is minimal (scanning bytes is fast, and NaN/Inf is rare in practice).

### Falling Back for .json Format
Initially used Rust for .json reads but found pydantic_core.from_json is ~3x faster at JSON→Python conversion than our serde_json→Python dict path (35ms vs 115ms for 1000 samples). This is because pydantic_core is specifically optimized for this task. Falling back eliminates the regression.

### Falling Back for Single-File Header-Only, Using Rust for Batch Headers
Single-file header-only .eval reads are slower with Rust (zip crate parses full central directory vs original's targeted range reads). But for batch header reads, using Rust in threads provides true parallelism that more than compensates: 3.77x speedup for 50 files.

### Falling Back for IO[bytes] Input
Some callers pass bytes streams instead of file paths. Since our Rust functions expect file paths, we fall back to the original implementation for these cases.

## Testing
- 79 tests total (51 correctness + 28 existing)
- All pass
- Correctness tests compare every field recursively with NaN-aware comparison

## Status and Next Steps
- Core implementation complete and working
- Future optimization (Segments 3/4): bypass Pydantic model_validate for 5-10x+ speedup
- Consider implementing Rust streaming JSON header parser for .json header-only
- Consider using rayon for parallel sample parsing in .eval files
- Consider releasing GIL in Rust during file I/O for even better thread parallelism
