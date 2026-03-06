"""Monkey-patching logic to replace inspect's log reading functions with fast implementations.

Patches sync and async variants for:
- read_eval_log / read_eval_log_async
- read_eval_log_headers / read_eval_log_headers_async
- read_eval_log_sample / read_eval_log_sample_async
- read_eval_log_sample_summaries / read_eval_log_sample_summaries_async
- read_eval_log_samples

The Rust layer handles ZIP decompression (returning raw bytes).
JSON parsing uses Python's json.loads (which natively supports NaN/Inf).
EvalSample construction bypasses Pydantic model_validate using fast direct
construction (_construct.py) for ~5-10x overall speedup.
"""

import asyncio
import functools
import json
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
from inspect_fast_loader._zip import (
    read_eval_file,
    read_eval_headers_batch,
    read_eval_sample as _read_eval_sample,
    read_eval_summaries as _read_eval_summaries,
)

SCORER_PLACEHOLDER = "88F74D2C"
_SENTINEL = object()

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
    """Build an EvalLog from the parsed dict returned by read_eval_file.

    Uses construct_sample_fast to bypass Pydantic model_validate for samples.
    """
    header_data = raw["header"]
    has_header_json = raw["has_header_json"]
    ctx = get_deserializing_context()

    if has_header_json:
        header_data.pop("samples", None)
        log = EvalLog.model_validate(header_data, context=ctx)
        log.location = path
    else:
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
        _populate_scorer_placeholder(log)

    return log


def _populate_scorer_placeholder(log: EvalLog) -> None:
    """Replace scorer placeholder key in sample scores with actual scorer name."""
    if not log.samples or not log.results or not log.results.scores:
        return
    scorer_name = log.results.scores[0].name
    for sample in log.samples:
        if sample.scores and SCORER_PLACEHOLDER in sample.scores:
            sample.scores[scorer_name] = sample.scores.pop(SCORER_PLACEHOLDER)


def _build_eval_log_from_json_file(raw: dict, path: str) -> EvalLog:
    """Build an EvalLog from a parsed .json file dict.

    Uses construct_sample_fast to bypass Pydantic model_validate for samples.
    """
    ctx = get_deserializing_context()
    sample_dicts = raw.pop("samples", None)
    log = EvalLog.model_validate(raw, context=ctx)
    log.location = path

    if sample_dicts is not None:
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
    """Call the original read_eval_log_async directly, avoiding double-dispatch."""
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
    """Fast implementation of read_eval_log.

    For .eval: Rust ZIP decompression + json.loads + Pydantic bypass.
    For .json: json.loads + Pydantic bypass.
    Falls back to original for IO[bytes] input and header-only reads.
    """
    if _is_bytes_input(log_file):
        return _fallback_to_original_sync(log_file, header_only, resolve_attachments, format)

    path = _resolve_path(log_file)
    fmt = _detect_format(path, format)

    if header_only:
        return _fallback_to_original_sync(log_file, header_only, resolve_attachments, format)

    if fmt == "eval":
        raw = read_eval_file(path, header_only=False)
        log = _build_eval_log_from_eval_file(raw, path, header_only=False)
    elif fmt == "json":
        with open(path, "rb") as f:
            raw = json.loads(f.read())
        log = _build_eval_log_from_json_file(raw, path)
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
    """Async wrapper — runs the sync implementation in a thread."""
    if _is_bytes_input(log_file):
        return await _originals["read_eval_log_async"](log_file, header_only, resolve_attachments, format)

    if header_only:
        return await _originals["read_eval_log_async"](log_file, header_only, resolve_attachments, format)

    return await asyncio.to_thread(
        _fast_read_eval_log_impl, log_file, header_only, resolve_attachments, format
    )


def _fast_read_eval_log_headers_impl(
    log_files: list[str] | list[EvalLogInfo],
    progress: Any = None,
) -> list[EvalLog]:
    from inspect_ai._util._async import run_coroutine
    return run_coroutine(_fast_read_eval_log_headers_async_impl(log_files, progress))


async def _fast_read_eval_log_headers_async_impl(
    log_files: list[str] | list[Path] | list[EvalLogInfo],
    progress: Any = None,
) -> list[EvalLog]:
    """Batch header reading with rayon-parallel ZIP decompression for .eval files."""
    if progress:
        progress.before_reading_logs(len(log_files))

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

    if eval_paths:
        batch_raws = await asyncio.to_thread(read_eval_headers_batch, eval_paths)
        for idx, raw, path in zip(eval_indices, batch_raws, eval_paths):
            results[idx] = _build_eval_log_from_eval_file(raw, path, header_only=True)
            if progress:
                progress.after_read_log(path)

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
    _scorer_name: str | None | object = _SENTINEL,
) -> EvalSample:
    """Fast implementation of read_eval_log_sample for .eval files."""
    if _is_bytes_input(log_file):
        return _originals["read_eval_log_sample"](
            log_file, id, epoch, uuid, resolve_attachments, format, exclude_fields
        )

    path = _resolve_path(log_file)
    fmt = _detect_format(path, format)

    if fmt != "eval":
        return _originals["read_eval_log_sample"](
            log_file, id, epoch, uuid, resolve_attachments, format, exclude_fields
        )

    if id is None and uuid is None:
        raise ValueError("You must specify either a sample 'id' and 'epoch' or a sample 'uuid'")

    if id is None:
        summaries = _read_eval_summaries(path)
        matched = next((s for s in summaries if s.get("uuid") == uuid), None)
        if matched is None:
            raise ValueError(f"Sample with uuid '{uuid}' not found in log.")
        id = matched["id"]
        epoch = matched["epoch"]

    entry_name = _sample_filename(id, epoch)

    try:
        sample_data = _read_eval_sample(path, entry_name)
    except KeyError:
        raise IndexError(f"Sample id {id} for epoch {epoch} not found in log {path}")

    if exclude_fields:
        for field in exclude_fields:
            sample_data.pop(field, None)

    sample = construct_sample_fast(sample_data)

    # Replace scorer placeholder — use pre-computed name if provided, else read header
    scorer_name = _scorer_name if _scorer_name is not _SENTINEL else _get_scorer_name(path)
    _replace_scorer_placeholder(sample, scorer_name)

    if resolve_attachments:
        from inspect_ai.log._condense import resolve_sample_attachments
        sample = resolve_sample_attachments(sample, resolve_attachments)

    return sample


def _get_scorer_name(path: str) -> str | None:
    """Read the scorer name from the log header."""
    raw = read_eval_file(path, header_only=True)
    results = raw["header"].get("results")
    if results and results.get("scores"):
        return results["scores"][0].get("name")
    return None


def _replace_scorer_placeholder(sample: EvalSample, scorer_name: str | None) -> None:
    """Replace scorer placeholder in a sample's scores with the actual scorer name."""
    if scorer_name and sample.scores and SCORER_PLACEHOLDER in sample.scores:
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
    path = _resolve_path(log_file)
    fmt = _detect_format(path, format)

    if fmt != "eval":
        return _originals["read_eval_log_sample_summaries"](log_file, format)

    ctx = get_deserializing_context()
    summaries_data = _read_eval_summaries(path)
    return [
        EvalSampleSummary.model_validate(s, context=ctx) for s in summaries_data
    ]


async def _fast_read_eval_log_sample_summaries_async_impl(
    log_file: str | Path | EvalLogInfo,
    format: Literal["eval", "json", "auto"] = "auto",
) -> list[EvalSampleSummary]:
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
    path = _resolve_path(log_file)
    fmt = _detect_format(path, format)

    if fmt != "eval":
        yield from _originals["read_eval_log_samples"](
            log_file, all_samples_required, resolve_attachments, format, exclude_fields
        )
        return

    log_header = _fallback_to_original_sync(log_file, header_only=True, resolve_attachments=False, format=format)

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

    # Pre-compute scorer name to avoid re-reading the header for every sample
    scorer_name = None
    if log_header.results and log_header.results.scores:
        scorer_name = log_header.results.scores[0].name

    for sample_id in log_header.eval.dataset.sample_ids:
        for epoch_id in range(1, (log_header.eval.config.epochs or 1) + 1):
            try:
                sample = _fast_read_eval_log_sample_impl(
                    log_file=log_file,
                    id=sample_id,
                    epoch=epoch_id,
                    resolve_attachments=resolve_attachments,
                    format=format,
                    exclude_fields=exclude_fields,
                    _scorer_name=scorer_name,
                )
                yield sample
            except IndexError:
                if all_samples_required:
                    raise


def _apply_patch(name: str, fast_impl: Any) -> None:
    _originals[name] = getattr(_file_module, name)
    wrapper = functools.wraps(_originals[name])(fast_impl)
    wrapper._is_fast_loader_wrapper = True  # type: ignore[attr-defined]
    wrapper._original = _originals[name]  # type: ignore[attr-defined]
    setattr(_file_module, name, wrapper)


_PATCHES: list[tuple[str, Any]] = [
    ("read_eval_log", _fast_read_eval_log_impl),
    ("read_eval_log_async", _fast_read_eval_log_async_impl),
    ("read_eval_log_headers", _fast_read_eval_log_headers_impl),
    ("read_eval_log_headers_async", _fast_read_eval_log_headers_async_impl),
    ("read_eval_log_sample", _fast_read_eval_log_sample_impl),
    ("read_eval_log_sample_async", _fast_read_eval_log_sample_async_impl),
    ("read_eval_log_sample_summaries", _fast_read_eval_log_sample_summaries_impl),
    ("read_eval_log_sample_summaries_async", _fast_read_eval_log_sample_summaries_async_impl),
    ("read_eval_log_samples", _fast_read_eval_log_samples_impl),
]


def patch() -> None:
    """Replace inspect's log reading functions with fast implementations."""
    global _patched
    if _patched:
        return

    for name, fast_impl in _PATCHES:
        _apply_patch(name, fast_impl)

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
    return _patched
