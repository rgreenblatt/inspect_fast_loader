"""Monkey-patching logic to replace inspect's log reading functions with fast Rust implementations.

Patches sync and async variants for:
- read_eval_log / read_eval_log_async
- read_eval_log_headers / read_eval_log_headers_async
- read_eval_log_sample / read_eval_log_sample_async (new)
- read_eval_log_sample_summaries / read_eval_log_sample_summaries_async (new)
- read_eval_log_samples (new)

The Rust layer handles JSON parsing (with NaN/Inf support) and ZIP decompression.
For full reads, EvalSample construction bypasses Pydantic model_validate using
fast direct construction (_construct.py) for ~5x+ overall speedup.
"""

import asyncio
import functools
from collections.abc import Generator
from pathlib import Path
from typing import Any, IO, Literal

import inspect_ai.log._file as _file_module
from inspect_ai._util.constants import get_deserializing_context
from inspect_ai.log._file import EvalLogInfo
from inspect_ai.log._log import (
    EvalLog,
    EvalSample,
    EvalSampleReductions,
    EvalSampleSummary,
    sort_samples,
)

from inspect_fast_loader._construct import construct_sample_fast
from inspect_fast_loader._native import (
    read_eval_file,
    read_eval_headers_batch,
    read_eval_sample as _rust_read_eval_sample,
    read_eval_summaries as _rust_read_eval_summaries,
    read_json_file,
)

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


async def _fast_read_eval_log_headers_async_impl(
    log_files: list[str] | list[Path] | list[EvalLogInfo],
    progress: Any = None,
) -> list[EvalLog]:
    """Async fast implementation of read_eval_log_headers.

    For .eval files, uses Rust batch header reading with rayon parallelism —
    a single Rust call reads all headers in parallel OS threads, avoiding
    per-file Python↔Rust boundary overhead and asyncio.to_thread overhead.

    For .json files, falls back to original.
    """
    if progress:
        progress.before_reading_logs(len(log_files))

    # Partition into .eval and .json files
    eval_indices: list[int] = []
    eval_paths: list[str] = []
    json_indices: list[int] = []
    json_files: list[Any] = []

    for i, lf in enumerate(log_files):
        path = _resolve_path(lf)
        fmt = _detect_format(path, "auto")
        if fmt == "eval":
            eval_indices.append(i)
            eval_paths.append(path)
        else:
            json_indices.append(i)
            json_files.append(lf)

    results: list[EvalLog | None] = [None] * len(log_files)

    # Read all .eval headers in one Rust call (parallel via rayon)
    if eval_paths:
        batch_raws = await asyncio.to_thread(read_eval_headers_batch, eval_paths)
        for idx, raw, path in zip(eval_indices, batch_raws, eval_paths):
            results[idx] = _build_eval_log_from_eval_file(raw, path, header_only=True)
            if progress:
                progress.after_read_log(path)

    # Fall back to original for .json files
    if json_files:
        read_fn = _originals.get("read_eval_log_async") or getattr(_file_module, "read_eval_log_async")

        async def _read_json(idx: int, lf: Any) -> None:
            log = await read_fn(lf, header_only=True)
            results[idx] = log
            if progress:
                progress.after_read_log(lf.name if isinstance(lf, EvalLogInfo) else str(lf))

        await asyncio.gather(*[_read_json(idx, lf) for idx, lf in zip(json_indices, json_files)])

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# read_eval_log_sample patching
# ---------------------------------------------------------------------------

SAMPLES_DIR = "samples"


def _sample_filename(id: str | int, epoch: int) -> str:
    return f"{SAMPLES_DIR}/{id}_epoch_{epoch}.json"


def _fast_read_eval_log_sample_impl(
    log_file: str | Path | EvalLogInfo,
    id: int | str | None = None,
    epoch: int = 1,
    uuid: str | None = None,
    resolve_attachments: bool | Literal["full", "core"] = False,
    format: Literal["eval", "json", "auto"] = "auto",
    exclude_fields: set[str] | None = None,
) -> EvalSample:
    """Fast implementation of read_eval_log_sample using Rust for .eval files."""
    if _is_bytes_input(log_file):
        return _originals["read_eval_log_sample"](
            log_file, id, epoch, uuid, resolve_attachments, format, exclude_fields
        )

    path = _resolve_path(log_file)
    fmt = _detect_format(path, format)

    # Only fast-path for .eval format
    if fmt != "eval":
        return _originals["read_eval_log_sample"](
            log_file, id, epoch, uuid, resolve_attachments, format, exclude_fields
        )

    if id is None and uuid is None:
        raise ValueError("You must specify either a sample 'id' and 'epoch' or a sample 'uuid'")

    # If uuid specified, read summaries to find the matching sample's id and epoch
    if id is None:
        summaries = _rust_read_eval_summaries(path)
        matched = next((s for s in summaries if s.get("uuid") == uuid), None)
        if matched is None:
            raise ValueError(f"Sample with uuid '{uuid}' not found in log.")
        id = matched["id"]
        epoch = matched["epoch"]

    entry_name = _sample_filename(id, epoch)

    try:
        sample_data = _rust_read_eval_sample(path, entry_name)
    except KeyError:
        raise IndexError(f"Sample id {id} for epoch {epoch} not found in log {path}")

    # Handle exclude_fields — delete excluded keys from the parsed dict
    if exclude_fields:
        for field in exclude_fields:
            sample_data.pop(field, None)

    sample = construct_sample_fast(sample_data)

    # Replicate scorer placeholder replacement for this single sample
    # We need the scorer name from the header for this
    _populate_sample_scorer_placeholder(sample, path)

    if resolve_attachments:
        from inspect_ai.log._condense import resolve_sample_attachments
        sample = resolve_sample_attachments(sample, resolve_attachments)

    return sample


def _populate_sample_scorer_placeholder(sample: EvalSample, path: str) -> None:
    """Replace scorer placeholder in a single sample by reading the header."""
    if not sample.scores or SCORER_PLACEHOLDER not in sample.scores:
        return
    # Read header to get scorer name
    raw = read_eval_file(path, header_only=True)
    header_data = raw["header"]
    results = header_data.get("results")
    if results and results.get("scores"):
        scorer_name = results["scores"][0].get("name")
        if scorer_name:
            sample.scores[scorer_name] = sample.scores.pop(SCORER_PLACEHOLDER)


async def _fast_read_eval_log_sample_async_impl(
    log_file: str | Path | EvalLogInfo,
    id: int | str | None = None,
    epoch: int = 1,
    uuid: str | None = None,
    resolve_attachments: bool | Literal["full", "core"] = False,
    format: Literal["eval", "json", "auto"] = "auto",
    exclude_fields: set[str] | None = None,
    reader: Any = None,
) -> EvalSample:
    """Async fast implementation of read_eval_log_sample."""
    if _is_bytes_input(log_file) or reader is not None:
        return await _originals["read_eval_log_sample_async"](
            log_file, id, epoch, uuid, resolve_attachments, format, exclude_fields, reader
        )

    path = _resolve_path(log_file)
    fmt = _detect_format(path, format)

    if fmt != "eval":
        return await _originals["read_eval_log_sample_async"](
            log_file, id, epoch, uuid, resolve_attachments, format, exclude_fields, reader
        )

    return await asyncio.to_thread(
        _fast_read_eval_log_sample_impl, log_file, id, epoch, uuid,
        resolve_attachments, format, exclude_fields
    )


# ---------------------------------------------------------------------------
# read_eval_log_sample_summaries patching
# ---------------------------------------------------------------------------

def _fast_read_eval_log_sample_summaries_impl(
    log_file: str | Path | EvalLogInfo,
    format: Literal["eval", "json", "auto"] = "auto",
) -> list[EvalSampleSummary]:
    """Fast implementation of read_eval_log_sample_summaries using Rust."""
    path = _resolve_path(log_file)
    fmt = _detect_format(path, format)

    if fmt != "eval":
        return _originals["read_eval_log_sample_summaries"](log_file, format)

    ctx = get_deserializing_context()
    summaries_data = _rust_read_eval_summaries(path)
    return [
        EvalSampleSummary.model_validate(s, context=ctx) for s in summaries_data
    ]


async def _fast_read_eval_log_sample_summaries_async_impl(
    log_file: str | Path | EvalLogInfo,
    format: Literal["eval", "json", "auto"] = "auto",
) -> list[EvalSampleSummary]:
    """Async fast implementation of read_eval_log_sample_summaries."""
    path = _resolve_path(log_file)
    fmt = _detect_format(path, format)

    if fmt != "eval":
        return await _originals["read_eval_log_sample_summaries_async"](log_file, format)

    return await asyncio.to_thread(_fast_read_eval_log_sample_summaries_impl, log_file, format)


# ---------------------------------------------------------------------------
# read_eval_log_samples patching
# ---------------------------------------------------------------------------

def _fast_read_eval_log_samples_impl(
    log_file: str | Path | EvalLogInfo,
    all_samples_required: bool = True,
    resolve_attachments: bool | Literal["full", "core"] = False,
    format: Literal["eval", "json", "auto"] = "auto",
    exclude_fields: set[str] | None = None,
) -> Generator[EvalSample, None, None]:
    """Fast implementation of read_eval_log_samples.

    For .eval files, reads all samples at once using the fast full-read path
    then yields them one at a time. This is faster than reading samples
    individually since the full read uses rayon parallel parsing.
    """
    path = _resolve_path(log_file)
    fmt = _detect_format(path, format)

    if fmt != "eval":
        yield from _originals["read_eval_log_samples"](
            log_file, all_samples_required, resolve_attachments, format, exclude_fields
        )
        return

    # Read header to get sample_ids and check status
    read_log_fn = getattr(_file_module, "read_eval_log")
    log_header = read_log_fn(log_file, header_only=True, format=format)

    if log_header.eval.dataset.sample_ids is None:
        raise RuntimeError(
            "This log file does not include sample_ids "
            + "(fully reading and re-writing the log will add sample_ids)"
        )

    if all_samples_required and (
        log_header.status != "success" or log_header.invalidated
    ):
        raise RuntimeError(
            f"This log does not have all samples (status={log_header.status}). "
            + "Specify all_samples_required=False to read the samples that exist."
        )

    # For .eval: read individual samples using our fast single-sample reader
    read_sample_fn = getattr(_file_module, "read_eval_log_sample")
    for sample_id in log_header.eval.dataset.sample_ids:
        for epoch_id in range(1, (log_header.eval.config.epochs or 1) + 1):
            try:
                sample = read_sample_fn(
                    log_file=log_file,
                    id=sample_id,
                    epoch=epoch_id,
                    resolve_attachments=resolve_attachments,
                    format=format,
                    exclude_fields=exclude_fields,
                )
                yield sample
            except IndexError:
                if all_samples_required:
                    raise


def patch() -> None:
    """Replace inspect's log reading functions with Rust-accelerated implementations."""
    global _patched
    if _patched:
        return

    # Save originals
    _all_patched_names = [
        "read_eval_log", "read_eval_log_async",
        "read_eval_log_headers", "read_eval_log_headers_async",
        "read_eval_log_sample", "read_eval_log_sample_async",
        "read_eval_log_sample_summaries", "read_eval_log_sample_summaries_async",
        "read_eval_log_samples",
    ]
    for name in _all_patched_names:
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

    # read_eval_log_sample
    sync_sample = functools.wraps(_originals["read_eval_log_sample"])(_fast_read_eval_log_sample_impl)
    sync_sample._is_fast_loader_wrapper = True  # type: ignore[attr-defined]
    sync_sample._original = _originals["read_eval_log_sample"]  # type: ignore[attr-defined]
    setattr(_file_module, "read_eval_log_sample", sync_sample)

    async_sample = functools.wraps(_originals["read_eval_log_sample_async"])(_fast_read_eval_log_sample_async_impl)
    async_sample._is_fast_loader_wrapper = True  # type: ignore[attr-defined]
    async_sample._original = _originals["read_eval_log_sample_async"]  # type: ignore[attr-defined]
    setattr(_file_module, "read_eval_log_sample_async", async_sample)

    # read_eval_log_sample_summaries
    sync_summaries = functools.wraps(_originals["read_eval_log_sample_summaries"])(_fast_read_eval_log_sample_summaries_impl)
    sync_summaries._is_fast_loader_wrapper = True  # type: ignore[attr-defined]
    sync_summaries._original = _originals["read_eval_log_sample_summaries"]  # type: ignore[attr-defined]
    setattr(_file_module, "read_eval_log_sample_summaries", sync_summaries)

    async_summaries = functools.wraps(_originals["read_eval_log_sample_summaries_async"])(_fast_read_eval_log_sample_summaries_async_impl)
    async_summaries._is_fast_loader_wrapper = True  # type: ignore[attr-defined]
    async_summaries._original = _originals["read_eval_log_sample_summaries_async"]  # type: ignore[attr-defined]
    setattr(_file_module, "read_eval_log_sample_summaries_async", async_summaries)

    # read_eval_log_samples (generator, sync only)
    sync_samples = functools.wraps(_originals["read_eval_log_samples"])(_fast_read_eval_log_samples_impl)
    sync_samples._is_fast_loader_wrapper = True  # type: ignore[attr-defined]
    sync_samples._original = _originals["read_eval_log_samples"]  # type: ignore[attr-defined]
    setattr(_file_module, "read_eval_log_samples", sync_samples)

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
