"""ZIP reading functions with optional Rust acceleration.

Provides a unified interface for reading .eval ZIP files. Uses the Rust
native extension when available (~2x faster ZIP decompression, parallel
batch reading), falling back to Python's zipfile module otherwise.
"""

import json
import zipfile
from typing import Any

try:
    from inspect_fast_loader._native import (
        read_eval_file as _native_read_eval_file,
        read_eval_headers_batch as _native_read_eval_headers_batch,
        read_eval_sample as _native_read_eval_sample,
        read_eval_summaries as _native_read_eval_summaries,
    )
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False


# ---------------------------------------------------------------------------
# Pure Python fallback implementations
# ---------------------------------------------------------------------------

def _py_read_eval_file(path: str, header_only: bool = False) -> dict:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        has_header_json = "header.json" in names

        header_name = "header.json" if has_header_json else "_journal/start.json"
        header = json.loads(zf.read(header_name))

        reductions = None
        if "reductions.json" in names:
            reductions = json.loads(zf.read("reductions.json"))

        samples = None
        if not header_only:
            sample_names = [n for n in names if n.startswith("samples/") and n.endswith(".json")]
            samples = [json.loads(zf.read(n)) for n in sample_names]

    return {
        "header": header,
        "samples": samples,
        "reductions": reductions,
        "has_header_json": has_header_json,
    }


def _py_read_eval_headers_batch(paths: list[str]) -> list[dict]:
    return [_py_read_eval_file(p, header_only=True) for p in paths]


def _py_read_eval_sample(path: str, entry_name: str) -> dict:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read(entry_name))


def _py_read_eval_summaries(path: str) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()

        if "summaries.json" in names:
            return json.loads(zf.read("summaries.json"))

        journal_names = sorted(
            [n for n in names if n.startswith("_journal/summaries/") and n.endswith(".json")],
            key=lambda n: int(n.split("/")[-1].removesuffix(".json")) if n.split("/")[-1].removesuffix(".json").isdigit() else 0,
        )
        result: list[dict] = []
        for name in journal_names:
            result.extend(json.loads(zf.read(name)))
        return result


# ---------------------------------------------------------------------------
# Native-accelerated wrappers (parse bytes returned by Rust with json.loads)
# ---------------------------------------------------------------------------

def _native_read_eval_file_parsed(path: str, header_only: bool = False) -> dict:
    raw = _native_read_eval_file(path, header_only=header_only)
    return {
        "header": json.loads(raw["header"]),
        "samples": [json.loads(s) for s in raw["samples"]] if raw["samples"] is not None else None,
        "reductions": json.loads(raw["reductions"]) if raw["reductions"] is not None else None,
        "has_header_json": raw["has_header_json"],
    }


def _native_read_eval_headers_batch_parsed(paths: list[str]) -> list[dict]:
    raw_results = _native_read_eval_headers_batch(paths)
    return [
        {
            "header": json.loads(r["header"]),
            "samples": None,
            "reductions": json.loads(r["reductions"]) if r["reductions"] is not None else None,
            "has_header_json": r["has_header_json"],
        }
        for r in raw_results
    ]


def _native_read_eval_sample_parsed(path: str, entry_name: str) -> dict:
    return json.loads(_native_read_eval_sample(path, entry_name))


def _native_read_eval_summaries_parsed(path: str) -> list[dict]:
    raw = _native_read_eval_summaries(path)
    if isinstance(raw, bytes):
        return json.loads(raw)
    # Journal fallback returns a list of byte chunks
    result: list[dict] = []
    for chunk in raw:
        result.extend(json.loads(chunk))
    return result


# ---------------------------------------------------------------------------
# Public API — picks native or Python automatically
# ---------------------------------------------------------------------------

if HAS_NATIVE:
    read_eval_file = _native_read_eval_file_parsed
    read_eval_headers_batch = _native_read_eval_headers_batch_parsed
    read_eval_sample = _native_read_eval_sample_parsed
    read_eval_summaries = _native_read_eval_summaries_parsed
else:
    read_eval_file = _py_read_eval_file
    read_eval_headers_batch = _py_read_eval_headers_batch
    read_eval_sample = _py_read_eval_sample
    read_eval_summaries = _py_read_eval_summaries
