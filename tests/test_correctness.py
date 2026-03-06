"""Correctness tests comparing fast (patched) output vs original Python output.

For every test log file, reads with both the original Python implementation and
the patched implementation, then compares all fields.
"""

import asyncio
import math
import os

import pytest

from inspect_ai.log._file import read_eval_log, read_eval_log_headers
from inspect_fast_loader import patch, unpatch

from helpers import assert_logs_equal, TEST_LOG_DIR


@pytest.fixture(autouse=True)
def _ensure_unpatched():
    unpatch()
    yield
    unpatch()


def _read_original(path: str, header_only: bool = False):
    unpatch()
    return read_eval_log(path, header_only=header_only)


def _read_fast(path: str, header_only: bool = False):
    patch()
    result = read_eval_log(path, header_only=header_only)
    unpatch()
    return result


# ---- Full read tests ----

TEST_LOG_NAMES = [
    "test_10samples",
    "test_100samples",
    "test_1000samples",
    "test_multiepoch",
    "test_error",
    "test_cancelled",
    "test_nan_inf",
    "test_attachments",
    "test_empty",
]


@pytest.mark.parametrize("name", TEST_LOG_NAMES)
def test_correctness_full_read_json(name):
    path = str(TEST_LOG_DIR / f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"Test file not found: {path}")
    assert_logs_equal(_read_original(path), _read_fast(path))


@pytest.mark.parametrize("name", TEST_LOG_NAMES)
def test_correctness_full_read_eval(name):
    path = str(TEST_LOG_DIR / f"{name}.eval")
    if not os.path.exists(path):
        pytest.skip(f"Test file not found: {path}")
    assert_logs_equal(_read_original(path), _read_fast(path))


# ---- Header-only tests ----

HEADER_LOG_NAMES = [
    "test_10samples",
    "test_100samples",
    "test_multiepoch",
    "test_error",
    "test_cancelled",
    "test_nan_inf",
]


@pytest.mark.parametrize("name", HEADER_LOG_NAMES)
def test_correctness_header_only_json(name):
    path = str(TEST_LOG_DIR / f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"Test file not found: {path}")

    # inspect_ai's _read_header_streaming crashes on error/cancelled .json logs
    # when results=null (upstream bug). Since our fast path falls back to the
    # original for header-only .json, both will fail the same way.
    try:
        orig = _read_original(path, header_only=True)
    except TypeError:
        # Verify full-read works correctly for this log instead.
        assert_logs_equal(
            _read_original(path, header_only=False),
            _read_fast(path, header_only=False),
        )
        return

    fast = _read_fast(path, header_only=True)
    assert orig.samples is None
    assert fast.samples is None
    assert_logs_equal(orig, fast)


@pytest.mark.parametrize("name", HEADER_LOG_NAMES)
def test_correctness_header_only_eval(name):
    path = str(TEST_LOG_DIR / f"{name}.eval")
    if not os.path.exists(path):
        pytest.skip(f"Test file not found: {path}")

    orig = _read_original(path, header_only=True)
    fast = _read_fast(path, header_only=True)
    assert orig.samples is None
    assert fast.samples is None
    assert_logs_equal(orig, fast)


# ---- Batch header tests ----

def test_correctness_batch_headers():
    batch_files = sorted(TEST_LOG_DIR.glob("batch_*.eval"))[:10]
    if not batch_files:
        pytest.skip("No batch test files found")

    paths = [str(f) for f in batch_files]

    unpatch()
    orig_headers = read_eval_log_headers(paths)

    patch()
    fast_headers = read_eval_log_headers(paths)
    unpatch()

    assert len(orig_headers) == len(fast_headers)
    for orig, fast in zip(orig_headers, fast_headers):
        assert_logs_equal(orig, fast)


# ---- Sample ordering tests ----

def test_sample_ordering_preserved():
    """Samples should be in the same order (sorted by epoch, id)."""
    path = str(TEST_LOG_DIR / "test_multiepoch.eval")
    if not os.path.exists(path):
        pytest.skip("Multiepoch test file not found")

    orig = _read_original(path)
    fast = _read_fast(path)

    assert orig.samples is not None and fast.samples is not None
    assert len(orig.samples) == len(fast.samples)

    for i, (o, f) in enumerate(zip(orig.samples, fast.samples)):
        assert o.id == f.id, f"Sample {i}: id mismatch {o.id} vs {f.id}"
        assert o.epoch == f.epoch, f"Sample {i}: epoch mismatch {o.epoch} vs {f.epoch}"


# ---- NaN/Inf tests ----

def test_nan_inf_values_preserved():
    path = str(TEST_LOG_DIR / "test_nan_inf.json")
    if not os.path.exists(path):
        pytest.skip("NaN/Inf test file not found")
    assert_logs_equal(_read_original(path), _read_fast(path))


def test_json_loads_nan_inf():
    """Python json.loads handles NaN/Inf natively."""
    import json
    data = b'{"x": NaN, "y": Infinity, "z": -Infinity, "w": 42}'
    result = json.loads(data)
    assert math.isnan(result["x"])
    assert result["y"] == float("inf")
    assert result["z"] == float("-inf")
    assert result["w"] == 42


# ---- Rust native function tests ----

try:
    import inspect_fast_loader._native
    _has_native = True
except ImportError:
    _has_native = False


@pytest.mark.skipif(not _has_native, reason="Rust native extension not available")
def test_read_eval_file_basic():
    from inspect_fast_loader._native import read_eval_file
    import json

    path = str(TEST_LOG_DIR / "test_10samples.eval")
    result = read_eval_file(path)

    assert isinstance(result, dict)
    assert result["has_header_json"] is True
    assert isinstance(result["header"], bytes)
    header = json.loads(result["header"])
    assert isinstance(header, dict)
    assert isinstance(result["samples"], list)
    assert len(result["samples"]) == 10
    sample0 = json.loads(result["samples"][0])
    assert isinstance(sample0, dict)


@pytest.mark.skipif(not _has_native, reason="Rust native extension not available")
def test_read_eval_file_header_only():
    from inspect_fast_loader._native import read_eval_file

    path = str(TEST_LOG_DIR / "test_10samples.eval")
    result = read_eval_file(path, header_only=True)
    assert result["samples"] is None
    assert result["header"] is not None


@pytest.mark.skipif(not _has_native, reason="Rust native extension not available")
def test_read_eval_file_not_found():
    from inspect_fast_loader._native import read_eval_file

    with pytest.raises(FileNotFoundError):
        read_eval_file("/nonexistent/path.eval")


# ---- Async tests ----

def test_async_read_json():
    path = str(TEST_LOG_DIR / "test_10samples.json")

    patch()
    sync_log = read_eval_log(path)

    async def _read():
        from inspect_ai.log._file import read_eval_log_async
        return await read_eval_log_async(path)

    async_log = asyncio.run(_read())
    unpatch()
    assert_logs_equal(sync_log, async_log)


def test_async_read_eval():
    path = str(TEST_LOG_DIR / "test_10samples.eval")

    patch()
    sync_log = read_eval_log(path)

    async def _read():
        from inspect_ai.log._file import read_eval_log_async
        return await read_eval_log_async(path)

    async_log = asyncio.run(_read())
    unpatch()
    assert_logs_equal(sync_log, async_log)


# ---- Tests exercising patched functions via module attribute ----

def test_patched_sync_json_via_module():
    import inspect_ai.log._file as fm
    path = str(TEST_LOG_DIR / "test_10samples.json")

    patch()
    try:
        log = fm.read_eval_log(path)
        assert log.status == "success"
        assert log.samples is not None
        assert len(log.samples) == 10
    finally:
        unpatch()


def test_patched_sync_eval_header_only_via_module():
    import inspect_ai.log._file as fm
    path = str(TEST_LOG_DIR / "test_10samples.eval")

    patch()
    try:
        log = fm.read_eval_log(path, header_only=True)
        assert log.status == "success"
        assert log.samples is None
    finally:
        unpatch()


def test_patched_sync_eval_full_via_module():
    import inspect_ai.log._file as fm
    path = str(TEST_LOG_DIR / "test_10samples.eval")

    patch()
    try:
        log = fm.read_eval_log(path)
        assert log.status == "success"
        assert log.samples is not None
        assert len(log.samples) == 10
    finally:
        unpatch()


def test_patched_batch_headers_via_module():
    import inspect_ai.log._file as fm
    batch_files = sorted(TEST_LOG_DIR.glob("batch_*.eval"))[:5]
    if not batch_files:
        pytest.skip("No batch test files found")

    paths = [str(f) for f in batch_files]

    patch()
    try:
        headers = fm.read_eval_log_headers(paths)
        assert len(headers) == len(paths)
        for h in headers:
            assert h.samples is None
    finally:
        unpatch()


# ---- Location and fallback tests ----

def test_location_set_correctly_eval():
    path = str(TEST_LOG_DIR / "test_10samples.eval")
    orig = _read_original(path)
    fast = _read_fast(path)
    assert fast.location == path
    assert orig.location == fast.location


def test_location_set_correctly_json():
    path = str(TEST_LOG_DIR / "test_10samples.json")
    orig = _read_original(path)
    fast = _read_fast(path)
    assert fast.location == path
    assert orig.location == fast.location


def test_bytes_input_fallback_json():
    """IO[bytes] input falls back to original implementation."""
    patch()
    try:
        with open(TEST_LOG_DIR / "test_10samples.json", "rb") as f:
            log = read_eval_log(f)
            assert log.status == "success"
            assert log.samples is not None
            assert len(log.samples) == 10
    finally:
        unpatch()


def test_bytes_input_fallback_eval():
    """IO[bytes] input falls back to original implementation."""
    patch()
    try:
        with open(TEST_LOG_DIR / "test_10samples.eval", "rb") as f:
            log = read_eval_log(f)
            assert log.status == "success"
            assert log.samples is not None
            assert len(log.samples) == 10
    finally:
        unpatch()


def test_format_auto_detection():
    for ext in [".eval", ".json"]:
        fast = _read_fast(str(TEST_LOG_DIR / f"test_10samples{ext}"))
        assert fast.status == "success"
        assert fast.samples is not None
        assert len(fast.samples) == 10
