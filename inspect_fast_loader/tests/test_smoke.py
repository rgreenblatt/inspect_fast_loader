"""Smoke tests for inspect_fast_loader."""

import json
import zipfile
from io import BytesIO


def test_import():
    """Verify the Rust extension compiles and imports."""
    import inspect_fast_loader

    assert hasattr(inspect_fast_loader, "parse_json_bytes")
    assert hasattr(inspect_fast_loader, "list_zip_entries")
    assert hasattr(inspect_fast_loader, "read_zip_member")
    assert hasattr(inspect_fast_loader, "patch")
    assert hasattr(inspect_fast_loader, "unpatch")


def test_parse_json_bytes():
    """Verify JSON parsing works correctly."""
    from inspect_fast_loader import parse_json_bytes

    data = b'{"key": "value", "num": 42, "arr": [1, 2, 3], "nested": {"a": true, "b": null}}'
    result = parse_json_bytes(data)

    assert result["key"] == "value"
    assert result["num"] == 42
    assert result["arr"] == [1, 2, 3]
    assert result["nested"]["a"] is True
    assert result["nested"]["b"] is None


def test_parse_json_bytes_empty_object():
    from inspect_fast_loader import parse_json_bytes

    result = parse_json_bytes(b"{}")
    assert result == {}


def test_parse_json_bytes_invalid():
    from inspect_fast_loader import parse_json_bytes

    import pytest

    with pytest.raises(ValueError):
        parse_json_bytes(b"not json")


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


def test_patch_unpatch():
    """Verify monkey-patching mechanism works."""
    import inspect_ai.log._file as file_mod
    from inspect_fast_loader import patch, unpatch

    original_read = file_mod.read_eval_log
    original_headers = file_mod.read_eval_log_headers

    # Patch
    patch()
    assert file_mod.read_eval_log is not original_read
    assert file_mod.read_eval_log_headers is not original_headers

    # Unpatch
    unpatch()
    assert file_mod.read_eval_log is original_read
    assert file_mod.read_eval_log_headers is original_headers


def test_patch_idempotent():
    """Verify patching twice doesn't break anything."""
    from inspect_fast_loader import patch, unpatch

    patch()
    patch()  # Should be a no-op
    unpatch()
    unpatch()  # Should be a no-op
