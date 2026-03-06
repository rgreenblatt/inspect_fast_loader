"""Monkey-patching logic to replace inspect's log reading functions with fast Rust implementations.

Patches both sync and async variants:
- read_eval_log / read_eval_log_async
- read_eval_log_headers / read_eval_log_headers_async

The Rust layer handles JSON parsing (with NaN/Inf support) and ZIP decompression.
For full reads, EvalSample construction bypasses Pydantic model_validate using
fast direct construction (_construct.py) for ~5x+ overall speedup.
"""

import asyncio
import functools
from pathlib import Path
from typing import Any, IO, Literal

import inspect_ai.log._file as _file_module
from inspect_ai._util.constants import get_deserializing_context
from inspect_ai.log._file import EvalLogInfo
from inspect_ai.log._log import (
    EvalLog,
    EvalSampleReductions,
    sort_samples,
)

from inspect_fast_loader._construct import construct_sample_fast
from inspect_fast_loader._native import read_eval_file, read_json_file

SCORER_PLACEHOLDER = "88F74D2C"

# Store original functions so we can restore them
_originals: dict[str, Any] = {}
_patched = False


def _resolve_path(log_file: str | Path | EvalLogInfo) -> str:
    """Convert log_file argument to a string path."""
    if isinstance(log_file, str):
        return log_file
    elif isinstance(log_file, Path):
        return log_file.as_posix()
    elif isinstance(log_file, EvalLogInfo):
        return log_file.name
    else:
        raise TypeError(f"Unexpected log_file type: {type(log_file)}")


def _detect_format(path: str, format: str) -> str:
    """Detect log format from file extension or explicit format."""
    if format != "auto":
        return format
    if path.endswith(".eval"):
        return "eval"
    elif path.endswith(".json"):
        return "json"
    raise ValueError(f"Cannot detect format for: {path}")


def _build_eval_log_from_eval_file(raw: dict, path: str, header_only: bool) -> EvalLog:
    """Build an EvalLog from the dict returned by read_eval_file (Rust .eval reader).

    For full reads, uses construct_sample_fast to bypass Pydantic model_validate
    for EvalSample objects. Header-level objects still use model_validate (negligible
    overhead since they're validated only once per file).
    """
    header_data = raw["header"]
    has_header_json = raw["has_header_json"]
    ctx = get_deserializing_context()

    if has_header_json:
        # header.json contains a full EvalLog-like structure (minus samples).
        # We strip "samples" to prevent EvalLog.populate_scorer_name_for_samples
        # from running on header-only data (we handle it ourselves below).
        header_data.pop("samples", None)
        log = EvalLog.model_validate(header_data, context=ctx)
        log.location = path
    else:
        # Fallback: _journal/start.json has {version, eval, plan} only
        from inspect_ai.log._recorders.eval import LogStart
        start = LogStart.model_validate(header_data, context=ctx)
        log = EvalLog(version=start.version, eval=start.eval, plan=start.plan, location=path)

    # Reductions
    if raw["reductions"] is not None:
        reductions = [
            EvalSampleReductions.model_validate(r, context=ctx)
            for r in raw["reductions"]
        ]
        if log.results is not None:
            log.reductions = reductions

    # Samples — using fast construction (bypasses Pydantic model_validate)
    if not header_only and raw["samples"] is not None:
        samples = [construct_sample_fast(s) for s in raw["samples"]]
        sort_samples(samples)
        log.samples = samples

        # Replicate EvalLog.populate_scorer_name_for_samples (mode="after" validator)
        # This replaces the "88F74D2C" scorer placeholder with the actual scorer name
        _populate_scorer_placeholder(log)

    return log


def _populate_scorer_placeholder(log: EvalLog) -> None:
    """Replace scorer placeholder key in sample scores with actual scorer name.

    Replicates EvalLog.populate_scorer_name_for_samples which normally runs
    as a model_validator(mode="after"). Since we construct samples outside of
    EvalLog.model_validate, we apply this transformation manually.
    """
    if not log.samples or not log.results or not log.results.scores:
        return
    scorer_name = log.results.scores[0].name
    for sample in log.samples:
        if sample.scores and SCORER_PLACEHOLDER in sample.scores:
            sample.scores[scorer_name] = sample.scores.pop(SCORER_PLACEHOLDER)


def _build_eval_log_from_json_file(raw: dict, path: str, header_only: bool) -> EvalLog:
    """Build an EvalLog from the dict returned by read_json_file (Rust .json reader).

    For full reads, bypasses Pydantic model_validate for EvalSample construction.
    For header-only reads, falls back to original (since header is small).
    """
    ctx = get_deserializing_context()

    # Extract samples before header validation so EvalLog.populate_scorer_name_for_samples
    # doesn't run on the raw sample dicts
    sample_dicts = raw.pop("samples", None)

    # Validate the header portion with EvalLog.model_validate (negligible overhead)
    log = EvalLog.model_validate(raw, context=ctx)
    log.location = path

    # Construct samples using fast bypass
    if not header_only and sample_dicts is not None:
        samples = [construct_sample_fast(s) for s in sample_dicts]
        sort_samples(samples)
        log.samples = samples
        _populate_scorer_placeholder(log)

    return log


def _is_bytes_input(log_file: Any) -> bool:
    """Check if log_file is a bytes stream (IO[bytes]) rather than a path."""
    return not isinstance(log_file, (str, Path, EvalLogInfo))


def _fallback_to_original_sync(
    log_file: str | Path | EvalLogInfo | IO[bytes],
    header_only: bool,
    resolve_attachments: bool | Literal["full", "core"],
    format: Literal["eval", "json", "auto"],
) -> EvalLog:
    """Call the original read_eval_log_async directly, avoiding double-dispatch.

    The original sync read_eval_log calls run_coroutine(read_eval_log_async(...)),
    but since we patched read_eval_log_async on the module, that would route through
    our patched async version before falling back again. Instead, we call the original
    async function directly to avoid the extra hop.
    """
    from inspect_ai._util._async import run_coroutine
    return run_coroutine(
        _originals["read_eval_log_async"](log_file, header_only, resolve_attachments, format)
    )


def _fast_read_eval_log_impl(
    log_file: str | Path | EvalLogInfo | IO[bytes],
    header_only: bool = False,
    resolve_attachments: bool | Literal["full", "core"] = False,
    format: Literal["eval", "json", "auto"] = "auto",
) -> EvalLog:
    """Fast implementation of read_eval_log using Rust parsing.

    Falls back to original for IO[bytes] input, .json format, and header-only .eval.
    """
    # Fall back to original for bytes input
    if _is_bytes_input(log_file):
        return _fallback_to_original_sync(log_file, header_only, resolve_attachments, format)

    path = _resolve_path(log_file)
    fmt = _detect_format(path, format)

    if fmt == "eval":
        if header_only:
            # For header-only .eval reads, fall back to the original. The original
            # uses AsyncZipReader with targeted range reads (only reads the EOCD +
            # central directory + header.json entry). Our Rust zip crate opens the
            # full ZIP and parses the entire central directory, which is slower for
            # large files.
            return _fallback_to_original_sync(log_file, header_only, resolve_attachments, format)
        raw = read_eval_file(path, header_only=False)
        log = _build_eval_log_from_eval_file(raw, path, header_only=False)
    elif fmt == "json":
        if header_only:
            # Header-only .json reads: fall back to original (streaming parser is efficient)
            return _fallback_to_original_sync(log_file, header_only, resolve_attachments, format)
        # Full .json reads: use Rust JSON parser + Pydantic bypass for samples
        raw = read_json_file(path)
        log = _build_eval_log_from_json_file(raw, path, header_only=False)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    # Post-processing: resolve attachments
    if resolve_attachments and log.samples:
        from inspect_ai.log._condense import resolve_sample_attachments
        log.samples = [
            resolve_sample_attachments(sample, resolve_attachments)
            for sample in log.samples
        ]

    # Post-processing: populate sample_ids if missing
    if log.eval.dataset.sample_ids is None and log.samples is not None:
        sample_ids: dict[str | int, None] = {}
        for sample in log.samples:
            if sample.id not in sample_ids:
                sample_ids[sample.id] = None
        log.eval.dataset.sample_ids = list(sample_ids.keys())

    return log


async def _fast_read_eval_log_async_impl(
    log_file: str | Path | EvalLogInfo | IO[bytes],
    header_only: bool = False,
    resolve_attachments: bool | Literal["full", "core"] = False,
    format: Literal["eval", "json", "auto"] = "auto",
) -> EvalLog:
    """Async fast implementation of read_eval_log.

    The Rust functions do synchronous file I/O, so we run them in a thread
    to avoid blocking the event loop.
    """
    # Fall back to original for bytes input
    if _is_bytes_input(log_file):
        return await _originals["read_eval_log_async"](log_file, header_only, resolve_attachments, format)

    path = _resolve_path(log_file)
    fmt = _detect_format(path, format)

    # Fall back to original for header-only reads (both formats)
    if header_only:
        return await _originals["read_eval_log_async"](log_file, header_only, resolve_attachments, format)

    # Run synchronous Rust I/O in a thread (only .eval full reads)
    return await asyncio.to_thread(
        _fast_read_eval_log_impl, log_file, header_only, resolve_attachments, format
    )


def _fast_read_eval_log_headers_impl(
    log_files: list[str] | list[EvalLogInfo],
    progress: Any = None,
) -> list[EvalLog]:
    """Fast implementation of read_eval_log_headers using Rust."""
    from inspect_ai._util._async import run_coroutine
    return run_coroutine(_fast_read_eval_log_headers_async_impl(log_files, progress))


def _read_eval_header_rust(log_file: str | Path | EvalLogInfo) -> EvalLog:
    """Read a single .eval header using Rust (for batch parallelism)."""
    path = _resolve_path(log_file)
    raw = read_eval_file(path, header_only=True)
    return _build_eval_log_from_eval_file(raw, path, header_only=True)


async def _fast_read_eval_log_headers_async_impl(
    log_files: list[str] | list[Path] | list[EvalLogInfo],
    progress: Any = None,
) -> list[EvalLog]:
    """Async fast implementation of read_eval_log_headers.

    For .eval files, uses asyncio.to_thread with Rust for true thread parallelism.
    For .json files, falls back to original via the patched read_eval_log_async.
    """
    if progress:
        progress.before_reading_logs(len(log_files))

    read_fn = getattr(_file_module, "read_eval_log_async")

    async def _read(lf: str | Path | EvalLogInfo) -> EvalLog:
        path = _resolve_path(lf)
        fmt = _detect_format(path, "auto")

        if fmt == "eval":
            # Use Rust in a thread for true parallelism across files
            log = await asyncio.to_thread(_read_eval_header_rust, lf)
        else:
            # Fall back to original for .json
            log = await read_fn(lf, header_only=True)

        if progress:
            progress.after_read_log(
                lf.name if isinstance(lf, EvalLogInfo) else str(lf),
            )
        return log

    tasks = [_read(lf) for lf in log_files]
    return list(await asyncio.gather(*tasks))


def patch() -> None:
    """Replace inspect's log reading functions with Rust-accelerated implementations."""
    global _patched
    if _patched:
        return

    # Save originals
    for name in ["read_eval_log", "read_eval_log_async", "read_eval_log_headers", "read_eval_log_headers_async"]:
        _originals[name] = getattr(_file_module, name)

    # Create wrappers with proper attributes
    sync_read = functools.wraps(_originals["read_eval_log"])(_fast_read_eval_log_impl)
    sync_read._is_fast_loader_wrapper = True  # type: ignore[attr-defined]
    sync_read._original = _originals["read_eval_log"]  # type: ignore[attr-defined]
    setattr(_file_module, "read_eval_log", sync_read)

    async_read = functools.wraps(_originals["read_eval_log_async"])(_fast_read_eval_log_async_impl)
    async_read._is_fast_loader_wrapper = True  # type: ignore[attr-defined]
    async_read._original = _originals["read_eval_log_async"]  # type: ignore[attr-defined]
    setattr(_file_module, "read_eval_log_async", async_read)

    sync_headers = functools.wraps(_originals["read_eval_log_headers"])(_fast_read_eval_log_headers_impl)
    sync_headers._is_fast_loader_wrapper = True  # type: ignore[attr-defined]
    sync_headers._original = _originals["read_eval_log_headers"]  # type: ignore[attr-defined]
    setattr(_file_module, "read_eval_log_headers", sync_headers)

    async_headers = functools.wraps(_originals["read_eval_log_headers_async"])(_fast_read_eval_log_headers_async_impl)
    async_headers._is_fast_loader_wrapper = True  # type: ignore[attr-defined]
    async_headers._original = _originals["read_eval_log_headers_async"]  # type: ignore[attr-defined]
    setattr(_file_module, "read_eval_log_headers_async", async_headers)

    _patched = True


def unpatch() -> None:
    """Restore inspect's original log reading functions."""
    global _patched
    if not _patched:
        return

    for name, original in _originals.items():
        setattr(_file_module, name, original)

    _originals.clear()
    _patched = False


def is_patched() -> bool:
    """Check if patches are currently applied."""
    return _patched
