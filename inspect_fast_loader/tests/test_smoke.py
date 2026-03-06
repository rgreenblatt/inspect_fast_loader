"""Smoke tests for inspect_fast_loader."""

import json
import zipfile
from io import BytesIO


def test_import():
    """Verify the Rust extension compiles and imports."""
    import inspect_fast_loader

    assert hasattr(inspect_fast_loader, "list_zip_entries")
    assert hasattr(inspect_fast_loader, "read_zip_member")
    assert hasattr(inspect_fast_loader, "read_eval_file")
    assert hasattr(inspect_fast_loader, "patch")
    assert hasattr(inspect_fast_loader, "unpatch")
    assert hasattr(inspect_fast_loader, "is_patched")


def test_list_zip_entries():
    """Verify ZIP entry listing works."""
    from inspect_fast_loader import list_zip_entries

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("file1.json", '{"a": 1}')
        zf.writestr("dir/file2.json", '{"b": 2}')
    zip_bytes = buf.getvalue()

    entries = list_zip_entries(zip_bytes)
    assert "file1.json" in entries
    assert "dir/file2.json" in entries
    assert len(entries) == 2


def test_read_zip_member():
    """Verify ZIP member reading works."""
    from inspect_fast_loader import read_zip_member

    content = b'{"hello": "world"}'
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("test.json", content)
    zip_bytes = buf.getvalue()

    result = read_zip_member(zip_bytes, "test.json")
    assert result == content


def test_read_zip_member_missing():
    """Verify missing ZIP member raises KeyError."""
    from inspect_fast_loader import read_zip_member

    import pytest

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("exists.json", "{}")
    zip_bytes = buf.getvalue()

    with pytest.raises(KeyError):
        read_zip_member(zip_bytes, "does_not_exist.json")


def test_patch_unpatch_sync():
    """Verify monkey-patching mechanism works for sync functions."""
    import inspect_ai.log._file as file_mod
    from inspect_fast_loader import is_patched, patch, unpatch

    original_read = file_mod.read_eval_log
    original_headers = file_mod.read_eval_log_headers

    assert not is_patched()

    # Patch
    patch()
    assert is_patched()
    assert file_mod.read_eval_log is not original_read
    assert file_mod.read_eval_log_headers is not original_headers

    # Unpatch
    unpatch()
    assert not is_patched()
    assert file_mod.read_eval_log is original_read
    assert file_mod.read_eval_log_headers is original_headers


def test_patch_unpatch_async():
    """Verify monkey-patching mechanism works for async functions."""
    import inspect_ai.log._file as file_mod
    from inspect_fast_loader import patch, unpatch

    original_read_async = file_mod.read_eval_log_async
    original_headers_async = file_mod.read_eval_log_headers_async

    # Patch
    patch()
    assert file_mod.read_eval_log_async is not original_read_async
    assert file_mod.read_eval_log_headers_async is not original_headers_async

    # Unpatch
    unpatch()
    assert file_mod.read_eval_log_async is original_read_async
    assert file_mod.read_eval_log_headers_async is original_headers_async


def test_patch_wrapper_attributes():
    """Verify patched functions have wrapper attributes for detection."""
    import inspect_ai.log._file as file_mod
    from inspect_fast_loader import patch, unpatch

    patch()
    try:
        assert getattr(file_mod.read_eval_log, "_is_fast_loader_wrapper", False)
        assert getattr(file_mod.read_eval_log_async, "_is_fast_loader_wrapper", False)
        assert getattr(file_mod.read_eval_log_headers, "_is_fast_loader_wrapper", False)
        assert getattr(file_mod.read_eval_log_headers_async, "_is_fast_loader_wrapper", False)

        assert hasattr(file_mod.read_eval_log, "_original")
        assert hasattr(file_mod.read_eval_log_async, "_original")
    finally:
        unpatch()


def test_patch_idempotent():
    """Verify patching twice doesn't break anything."""
    from inspect_fast_loader import patch, unpatch

    patch()
    patch()  # Should be a no-op
    unpatch()
    unpatch()  # Should be a no-op


# ---- Python fallback path tests ----
# These test the pure-Python implementations in _zip.py directly,
# which are normally unused when the Rust extension is available.

class TestPythonFallback:
    def test_py_read_eval_file(self):
        from inspect_fast_loader._zip import _py_read_eval_file, read_eval_file
        from helpers import TEST_LOG_DIR

        path = str(TEST_LOG_DIR / "test_10samples.eval")
        py_result = _py_read_eval_file(path)
        native_result = read_eval_file(path)

        assert py_result["has_header_json"] == native_result["has_header_json"]
        assert py_result["header"] == native_result["header"]
        assert len(py_result["samples"]) == len(native_result["samples"])

    def test_py_read_eval_file_header_only(self):
        from inspect_fast_loader._zip import _py_read_eval_file, read_eval_file
        from helpers import TEST_LOG_DIR

        path = str(TEST_LOG_DIR / "test_10samples.eval")
        py_result = _py_read_eval_file(path, header_only=True)
        native_result = read_eval_file(path, header_only=True)

        assert py_result["samples"] is None
        assert native_result["samples"] is None
        assert py_result["header"] == native_result["header"]

    def test_py_read_eval_sample(self):
        from inspect_fast_loader._zip import _py_read_eval_sample, read_eval_sample
        from helpers import TEST_LOG_DIR

        path = str(TEST_LOG_DIR / "test_10samples.eval")
        entry = "samples/1_epoch_1.json"
        py_result = _py_read_eval_sample(path, entry)
        native_result = read_eval_sample(path, entry)

        assert py_result == native_result
        assert py_result["id"] == 1

    def test_py_read_eval_summaries(self):
        from inspect_fast_loader._zip import _py_read_eval_summaries, read_eval_summaries
        from helpers import TEST_LOG_DIR

        path = str(TEST_LOG_DIR / "test_10samples.eval")
        py_result = _py_read_eval_summaries(path)
        native_result = read_eval_summaries(path)

        assert len(py_result) == len(native_result)
        for py_s, native_s in zip(py_result, native_result):
            assert py_s["id"] == native_s["id"]
            assert py_s["epoch"] == native_s["epoch"]

    def test_py_read_eval_headers_batch(self):
        from inspect_fast_loader._zip import _py_read_eval_headers_batch, read_eval_headers_batch
        from helpers import TEST_LOG_DIR

        paths = [str(f) for f in sorted(TEST_LOG_DIR.glob("batch_*.eval"))[:5]]
        py_results = _py_read_eval_headers_batch(paths)
        native_results = read_eval_headers_batch(paths)

        assert len(py_results) == len(native_results)
        for py_r, native_r in zip(py_results, native_results):
            assert py_r["header"] == native_r["header"]
            assert py_r["has_header_json"] == native_r["has_header_json"]
