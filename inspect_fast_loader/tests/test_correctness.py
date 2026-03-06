"""Correctness tests comparing Rust-accelerated output vs Python original output.

For every test log file, reads with both the original Python implementation and
the Rust-accelerated (patched) implementation, then compares all fields.
"""

import asyncio
import math
import os
from pathlib import Path

import pytest

from inspect_ai.log._file import read_eval_log, read_eval_log_headers
from inspect_fast_loader import patch, unpatch

TEST_LOGS_DIR = Path(__file__).parent.parent.parent / "test_logs"


@pytest.fixture(autouse=True)
def _ensure_unpatched():
    """Ensure we start and end each test unpatched."""
    unpatch()
    yield
    unpatch()


def _read_original(path: str, header_only: bool = False):
    """Read using original Python implementation."""
    unpatch()
    return read_eval_log(path, header_only=header_only)


def _read_fast(path: str, header_only: bool = False):
    """Read using Rust-accelerated implementation."""
    patch()
    result = read_eval_log(path, header_only=header_only)
    unpatch()
    return result


def _approx_equal(a, b, rel_tol=1e-9, abs_tol=1e-12):
    """Compare floats with tolerance, handling NaN and Inf."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isinf(a) and math.isinf(b):
            return (a > 0 and b > 0) or (a < 0 and b < 0)
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    return False


def _deep_compare(orig, fast, path_str="root"):
    """Recursively compare two objects, collecting differences."""
    diffs = []

    if orig is None and fast is None:
        return diffs
    if orig is None or fast is None:
        diffs.append(f"{path_str}: one is None (orig={orig is None}, fast={fast is None})")
        return diffs

    if isinstance(orig, float) and isinstance(fast, float):
        if not _approx_equal(orig, fast):
            diffs.append(f"{path_str}: float mismatch orig={orig} fast={fast}")
        return diffs

    if type(orig) != type(fast):
        # Allow int/float comparison for close values
        if isinstance(orig, (int, float)) and isinstance(fast, (int, float)):
            if not _approx_equal(float(orig), float(fast)):
                diffs.append(f"{path_str}: numeric mismatch orig={orig} ({type(orig).__name__}) fast={fast} ({type(fast).__name__})")
        else:
            diffs.append(f"{path_str}: type mismatch orig={type(orig).__name__} fast={type(fast).__name__}")
        return diffs

    if isinstance(orig, dict):
        all_keys = set(orig.keys()) | set(fast.keys())
        for k in sorted(all_keys):
            if k not in orig:
                diffs.append(f"{path_str}.{k}: missing in original")
            elif k not in fast:
                diffs.append(f"{path_str}.{k}: missing in fast")
            else:
                diffs.extend(_deep_compare(orig[k], fast[k], f"{path_str}.{k}"))
        return diffs

    if isinstance(orig, (list, tuple)):
        if len(orig) != len(fast):
            diffs.append(f"{path_str}: length mismatch orig={len(orig)} fast={len(fast)}")
            return diffs
        for i in range(len(orig)):
            diffs.extend(_deep_compare(orig[i], fast[i], f"{path_str}[{i}]"))
        return diffs

    if isinstance(orig, str):
        if orig != fast:
            diffs.append(f"{path_str}: string mismatch orig={orig!r:.100} fast={fast!r:.100}")
        return diffs

    if isinstance(orig, (int, bool)):
        if orig != fast:
            diffs.append(f"{path_str}: value mismatch orig={orig} fast={fast}")
        return diffs

    # For Pydantic models and other complex objects, compare their dict representations
    if hasattr(orig, "model_dump"):
        return _deep_compare(orig.model_dump(), fast.model_dump(), path_str)

    if orig != fast:
        diffs.append(f"{path_str}: value mismatch orig={orig!r:.100} fast={fast!r:.100}")
    return diffs


def assert_logs_equal(orig, fast, ignore_location=True):
    """Compare two EvalLog objects field by field."""
    orig_dict = orig.model_dump()
    fast_dict = fast.model_dump()

    if ignore_location:
        orig_dict.pop("location", None)
        fast_dict.pop("location", None)

    diffs = _deep_compare(orig_dict, fast_dict)
    if diffs:
        msg = f"Found {len(diffs)} differences:\n" + "\n".join(f"  - {d}" for d in diffs[:30])
        if len(diffs) > 30:
            msg += f"\n  ... and {len(diffs) - 30} more"
        pytest.fail(msg)


# ---- Full read tests ----

@pytest.mark.parametrize("name", [
    "test_10samples",
    "test_100samples",
    "test_1000samples",
    "test_multiepoch",
    "test_error",
    "test_cancelled",
    "test_nan_inf",
    "test_attachments",
    "test_empty",
])
def test_correctness_full_read_json(name):
    """Full read of .json files: Rust-accelerated should match original."""
    path = str(TEST_LOGS_DIR / f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"Test file not found: {path}")

    orig = _read_original(path)
    fast = _read_fast(path)
    assert_logs_equal(orig, fast)


@pytest.mark.parametrize("name", [
    "test_10samples",
    "test_100samples",
    "test_1000samples",
    "test_multiepoch",
    "test_error",
    "test_cancelled",
    "test_nan_inf",
    "test_attachments",
    "test_empty",
])
def test_correctness_full_read_eval(name):
    """Full read of .eval files: Rust-accelerated should match original."""
    path = str(TEST_LOGS_DIR / f"{name}.eval")
    if not os.path.exists(path):
        pytest.skip(f"Test file not found: {path}")

    orig = _read_original(path)
    fast = _read_fast(path)
    assert_logs_equal(orig, fast)


# ---- Header-only tests ----

@pytest.mark.parametrize("name", [
    "test_10samples",
    "test_100samples",
    "test_multiepoch",
    "test_error",
    "test_cancelled",
    "test_nan_inf",
])
def test_correctness_header_only_json(name):
    """Header-only read of .json files should match."""
    path = str(TEST_LOGS_DIR / f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"Test file not found: {path}")

    # The original Python _read_header_streaming crashes on error/cancelled
    # .json logs when results=null (upstream bug). Since our fast path falls
    # back to the original for header-only .json, both will fail.
    try:
        orig = _read_original(path, header_only=True)
    except TypeError:
        # Known upstream bug: _read_header_streaming crashes when results=null.
        # Verify full-read works correctly for this log instead.
        orig_full = _read_original(path, header_only=False)
        fast_full = _read_fast(path, header_only=False)
        assert_logs_equal(orig_full, fast_full)
        return

    fast = _read_fast(path, header_only=True)

    assert orig.samples is None
    assert fast.samples is None
    assert_logs_equal(orig, fast)


@pytest.mark.parametrize("name", [
    "test_10samples",
    "test_100samples",
    "test_multiepoch",
    "test_error",
    "test_cancelled",
    "test_nan_inf",
])
def test_correctness_header_only_eval(name):
    """Header-only read of .eval files should match."""
    path = str(TEST_LOGS_DIR / f"{name}.eval")
    if not os.path.exists(path):
        pytest.skip(f"Test file not found: {path}")

    orig = _read_original(path, header_only=True)
    fast = _read_fast(path, header_only=True)

    assert orig.samples is None
    assert fast.samples is None
    assert_logs_equal(orig, fast)


# ---- Batch header tests ----

def test_correctness_batch_headers():
    """Batch header reading should match for multiple files."""
    batch_files = sorted(TEST_LOGS_DIR.glob("batch_*.eval"))[:10]
    if not batch_files:
        pytest.skip("No batch test files found")

    paths = [str(f) for f in batch_files]

    # Read with original
    unpatch()
    orig_headers = read_eval_log_headers(paths)

    # Read with fast
    patch()
    fast_headers = read_eval_log_headers(paths)
    unpatch()

    assert len(orig_headers) == len(fast_headers)
    for orig, fast in zip(orig_headers, fast_headers):
        assert_logs_equal(orig, fast)


# ---- Sample ordering tests ----

def test_sample_ordering_preserved():
    """Samples should be in the same order (sorted by epoch, id)."""
    path = str(TEST_LOGS_DIR / "test_multiepoch.eval")
    if not os.path.exists(path):
        pytest.skip("Multiepoch test file not found")

    orig = _read_original(path)
    fast = _read_fast(path)

    assert orig.samples is not None and fast.samples is not None
    assert len(orig.samples) == len(fast.samples)

    for i, (o, f) in enumerate(zip(orig.samples, fast.samples)):
        assert o.id == f.id, f"Sample {i}: id mismatch {o.id} vs {f.id}"
        assert o.epoch == f.epoch, f"Sample {i}: epoch mismatch {o.epoch} vs {f.epoch}"


# ---- NaN/Inf specific tests ----

def test_nan_inf_values_preserved():
    """NaN and Inf values in score metadata should be preserved."""
    path = str(TEST_LOGS_DIR / "test_nan_inf.json")
    if not os.path.exists(path):
        pytest.skip("NaN/Inf test file not found")

    orig = _read_original(path)
    fast = _read_fast(path)

    assert orig.samples is not None and fast.samples is not None
    assert_logs_equal(orig, fast)


# ---- Rust native function tests ----

def test_parse_json_bytes_nan_inf():
    """Rust parse_json_bytes should handle NaN/Inf."""
    from inspect_fast_loader._native import parse_json_bytes

    data = b'{"x": NaN, "y": Infinity, "z": -Infinity, "w": 42}'
    result = parse_json_bytes(data)

    assert math.isnan(result["x"])
    assert result["y"] == float("inf")
    assert result["z"] == float("-inf")
    assert result["w"] == 42


def test_parse_json_bytes_nan_in_string_not_replaced():
    """NaN/Inf inside JSON strings should not be replaced."""
    from inspect_fast_loader._native import parse_json_bytes

    data = b'{"msg": "NaN is not a number, Infinity is large"}'
    result = parse_json_bytes(data)
    assert result["msg"] == "NaN is not a number, Infinity is large"


def test_read_eval_file_basic():
    """Rust read_eval_file should return correct structure."""
    from inspect_fast_loader._native import read_eval_file

    path = str(TEST_LOGS_DIR / "test_10samples.eval")
    result = read_eval_file(path)

    assert isinstance(result, dict)
    assert result["has_header_json"] is True
    assert isinstance(result["header"], dict)
    assert isinstance(result["samples"], list)
    assert len(result["samples"]) == 10


def test_read_eval_file_header_only():
    """Rust read_eval_file with header_only should not read samples."""
    from inspect_fast_loader._native import read_eval_file

    path = str(TEST_LOGS_DIR / "test_10samples.eval")
    result = read_eval_file(path, header_only=True)

    assert result["samples"] is None
    assert result["header"] is not None


def test_read_json_file_basic():
    """Rust read_json_file should return a dict with expected keys."""
    from inspect_fast_loader._native import read_json_file

    path = str(TEST_LOGS_DIR / "test_10samples.json")
    result = read_json_file(path)

    assert isinstance(result, dict)
    assert "version" in result
    assert "samples" in result
    assert len(result["samples"]) == 10


def test_read_json_file_not_found():
    """Rust read_json_file should raise FileNotFoundError."""
    from inspect_fast_loader._native import read_json_file

    with pytest.raises(FileNotFoundError):
        read_json_file("/nonexistent/path.json")


def test_read_eval_file_not_found():
    """Rust read_eval_file should raise FileNotFoundError."""
    from inspect_fast_loader._native import read_eval_file

    with pytest.raises(FileNotFoundError):
        read_eval_file("/nonexistent/path.eval")


# ---- Async tests ----

def test_async_read_json():
    """Async read should work and match sync."""
    path = str(TEST_LOGS_DIR / "test_10samples.json")

    patch()
    sync_log = read_eval_log(path)

    async def _read():
        from inspect_ai.log._file import read_eval_log_async
        return await read_eval_log_async(path)

    async_log = asyncio.run(_read())
    unpatch()

    assert_logs_equal(sync_log, async_log)


def test_async_read_eval():
    """Async read should work and match sync."""
    path = str(TEST_LOGS_DIR / "test_10samples.eval")

    patch()
    sync_log = read_eval_log(path)

    async def _read():
        from inspect_ai.log._file import read_eval_log_async
        return await read_eval_log_async(path)

    async_log = asyncio.run(_read())
    unpatch()

    assert_logs_equal(sync_log, async_log)


# ---- Tests that exercise the patched function via module attribute ----
# These ensure the sync fallback path works when the function is called
# via the module (as real callers would), not via a pre-patching import.

def test_patched_sync_json_via_module():
    """Sync .json read via module attribute should work after patching."""
    import inspect_ai.log._file as fm
    path = str(TEST_LOGS_DIR / "test_10samples.json")

    patch()
    try:
        log = fm.read_eval_log(path)
        assert log.status == "success"
        assert log.samples is not None
        assert len(log.samples) == 10
    finally:
        unpatch()


def test_patched_sync_eval_header_only_via_module():
    """Sync header-only .eval read via module attribute should work."""
    import inspect_ai.log._file as fm
    path = str(TEST_LOGS_DIR / "test_10samples.eval")

    patch()
    try:
        log = fm.read_eval_log(path, header_only=True)
        assert log.status == "success"
        assert log.samples is None
    finally:
        unpatch()


def test_patched_sync_eval_full_via_module():
    """Sync full .eval read via module attribute should work."""
    import inspect_ai.log._file as fm
    path = str(TEST_LOGS_DIR / "test_10samples.eval")

    patch()
    try:
        log = fm.read_eval_log(path)
        assert log.status == "success"
        assert log.samples is not None
        assert len(log.samples) == 10
    finally:
        unpatch()


def test_patched_batch_headers_via_module():
    """Batch header reading via module attribute should work."""
    import inspect_ai.log._file as fm
    batch_files = sorted(TEST_LOGS_DIR.glob("batch_*.eval"))[:5]
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


# ---- Additional edge case tests (from Branch 2) ----

def test_location_set_correctly_eval():
    """Test that log.location is set correctly for .eval files."""
    path = str(TEST_LOGS_DIR / "test_10samples.eval")
    orig = _read_original(path)
    fast = _read_fast(path)
    assert fast.location == path
    assert orig.location == fast.location


def test_location_set_correctly_json():
    """Test that log.location is set correctly for .json files."""
    path = str(TEST_LOGS_DIR / "test_10samples.json")
    orig = _read_original(path)
    fast = _read_fast(path)
    assert fast.location == path
    assert orig.location == fast.location


def test_bytes_input_fallback_json():
    """Test that IO[bytes] input falls back to original for .json."""
    patch()
    try:
        with open(TEST_LOGS_DIR / "test_10samples.json", "rb") as f:
            log = read_eval_log(f)
            assert log.status == "success"
            assert log.samples is not None
            assert len(log.samples) == 10
    finally:
        unpatch()


def test_bytes_input_fallback_eval():
    """Test that IO[bytes] input falls back to original for .eval."""
    patch()
    try:
        with open(TEST_LOGS_DIR / "test_10samples.eval", "rb") as f:
            log = read_eval_log(f)
            assert log.status == "success"
            assert log.samples is not None
            assert len(log.samples) == 10
    finally:
        unpatch()


def test_format_auto_detection():
    """Test auto-detection works for both formats."""
    for ext in [".eval", ".json"]:
        fast = _read_fast(str(TEST_LOGS_DIR / f"test_10samples{ext}"))
        assert fast.status == "success"
        assert fast.samples is not None
        assert len(fast.samples) == 10
