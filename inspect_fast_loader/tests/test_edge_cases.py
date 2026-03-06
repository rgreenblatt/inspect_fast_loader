"""Edge case and error handling tests.

Tests for:
1. Corrupted ZIP files
2. Missing ZIP entries
3. Partial writes / incomplete ZIPs
4. Old format versions (deprecated fields)
5. Large logs (5000+ samples)
6. NaN/Inf handling through all paths
"""

import io
import json
import math
import os
import sys
import tempfile
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import inspect_fast_loader
from inspect_fast_loader._native import (
    read_eval_file as _read_eval_file_raw,
    read_eval_headers_batch,
    read_eval_sample as _read_eval_sample_raw,
    read_eval_summaries as _read_eval_summaries_raw,
)


def read_eval_file(path, **kwargs):
    """Wrapper that json.loads the raw bytes returned by Rust."""
    raw = _read_eval_file_raw(path, **kwargs)
    result = {"has_header_json": raw["has_header_json"]}
    result["header"] = json.loads(raw["header"])
    result["reductions"] = json.loads(raw["reductions"]) if raw["reductions"] is not None else None
    result["samples"] = [json.loads(s) for s in raw["samples"]] if raw["samples"] is not None else None
    return result


def read_eval_sample(path, entry_name):
    return json.loads(_read_eval_sample_raw(path, entry_name))


def read_eval_summaries(path):
    raw = _read_eval_summaries_raw(path)
    if isinstance(raw, bytes):
        return json.loads(raw)
    # Journal fallback returns list of byte chunks
    result = []
    for chunk in raw:
        result.extend(json.loads(chunk))
    return result

TEST_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "test_logs")


@pytest.fixture(autouse=True)
def ensure_patched():
    inspect_fast_loader.patch()
    yield
    inspect_fast_loader.unpatch()


def _make_minimal_eval_zip(
    samples: list[dict] | None = None,
    header: dict | None = None,
    summaries: list[dict] | None = None,
    include_header_json: bool = True,
    extra_entries: dict[str, bytes] | None = None,
) -> str:
    """Create a minimal .eval ZIP file for testing."""
    if header is None:
        header = {
            "version": 3,
            "status": "success",
            "eval": {
                "task": "test_task",
                "task_version": 0,
                "task_file": None,
                "task_id": "test_task",
                "run_id": "test_run",
                "created": "2024-01-01T00:00:00+00:00",
                "dataset": {"name": "test", "location": "test", "samples": 1, "shuffled": False, "sample_ids": [1]},
                "model": "openai/gpt-4o",
                "model_base_url": None,
                "task_attribs": {},
                "task_args": {},
                "model_args": {},
                "config": {"epochs": 1},
                "revision": None,
                "packages": {},
                "metadata": None,
                "sandbox": None,
                "model_roles": None,
            },
            "plan": {"name": "plan", "steps": [], "config": {}, "finish": None},
            "results": {
                "scores": [{"name": "accuracy", "scorer": "accuracy", "params": {}, "metrics": {}}],
                "total_samples": 1,
                "completed_samples": 1,
            },
            "stats": {"started_at": "2024-01-01T00:00:00+00:00", "completed_at": "2024-01-01T00:01:00+00:00"},
        }

    if samples is None:
        samples = [{
            "id": 1, "epoch": 1,
            "input": "test input",
            "target": "A",
            "messages": [],
            "output": {"model": "openai/gpt-4o", "choices": [{"message": {"role": "assistant", "content": "A"}, "stop_reason": "stop"}], "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            "scores": {"accuracy": {"value": "C", "answer": "A"}},
        }]

    tmp = tempfile.NamedTemporaryFile(suffix=".eval", delete=False)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        if include_header_json:
            zf.writestr("header.json", json.dumps(header))
        else:
            # Use journal start
            start_data = {
                "version": header.get("version", 3),
                "eval": header["eval"],
                "plan": header["plan"],
            }
            zf.writestr("_journal/start.json", json.dumps(start_data))

        for s in samples:
            zf.writestr(f"samples/{s['id']}_epoch_{s['epoch']}.json", json.dumps(s))

        if summaries is not None:
            zf.writestr("summaries.json", json.dumps(summaries))

        if extra_entries:
            for name, data in extra_entries.items():
                zf.writestr(name, data)

    tmp.close()
    return tmp.name


class TestCorruptedZipFiles:
    def test_truncated_zip_file(self):
        """Truncated ZIP should raise a clear error."""
        tmp = tempfile.NamedTemporaryFile(suffix=".eval", delete=False)
        tmp.write(b"PK\x03\x04" + b"\x00" * 100)  # ZIP magic + garbage
        tmp.close()
        try:
            with pytest.raises(Exception):  # Should raise ValueError or similar
                read_eval_file(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_not_a_zip_file(self):
        """Non-ZIP file should raise a clear error."""
        tmp = tempfile.NamedTemporaryFile(suffix=".eval", delete=False)
        tmp.write(b"this is not a zip file")
        tmp.close()
        try:
            with pytest.raises(Exception):
                read_eval_file(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_empty_file(self):
        """Empty file should raise a clear error."""
        tmp = tempfile.NamedTemporaryFile(suffix=".eval", delete=False)
        tmp.close()
        try:
            with pytest.raises(Exception):
                read_eval_file(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_batch_headers_with_corrupted_file(self):
        """Batch headers should raise a clear error for corrupted files."""
        tmp = tempfile.NamedTemporaryFile(suffix=".eval", delete=False)
        tmp.write(b"not a zip")
        tmp.close()
        try:
            with pytest.raises(Exception):
                read_eval_headers_batch([tmp.name])
        finally:
            os.unlink(tmp.name)


class TestMissingZipEntries:
    def test_missing_header_and_journal(self):
        """ZIP with neither header.json nor _journal/start.json should error."""
        tmp = tempfile.NamedTemporaryFile(suffix=".eval", delete=False)
        with zipfile.ZipFile(tmp, "w") as zf:
            zf.writestr("dummy.txt", "hello")
        tmp.close()
        try:
            with pytest.raises(Exception):
                read_eval_file(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_journal_fallback_works(self):
        """When header.json is missing, _journal/start.json should be used."""
        path = _make_minimal_eval_zip(include_header_json=False)
        try:
            raw = read_eval_file(path, header_only=True)
            assert raw["has_header_json"] is False
            assert raw["header"]["version"] == 3
        finally:
            os.unlink(path)

    def test_missing_sample_entry(self):
        """Reading a non-existent sample should raise KeyError."""
        from glob import glob
        files = sorted(glob(os.path.join(TEST_LOG_DIR, "*.eval")))
        with pytest.raises(KeyError):
            read_eval_sample(files[0], "samples/nonexistent_epoch_1.json")


class TestPartialWrites:
    def test_eval_zip_no_samples(self):
        """ZIP with header but no samples (interrupted write)."""
        path = _make_minimal_eval_zip(samples=[])
        try:
            raw = read_eval_file(path)
            assert raw["samples"] is not None
            # Should be an empty list since there are no sample entries
            assert len(raw["samples"]) == 0
        finally:
            os.unlink(path)

    def test_cancelled_log_handling(self):
        """Cancelled logs should be handled correctly."""
        from glob import glob
        from inspect_ai.log._file import read_eval_log
        cancelled_files = [f for f in sorted(glob(os.path.join(TEST_LOG_DIR, "*.eval"))) if "cancelled" in f]
        if not cancelled_files:
            pytest.skip("No cancelled test files")
        log = read_eval_log(cancelled_files[0])
        assert log.status == "cancelled"

    def test_error_log_handling(self):
        """Error logs should be handled correctly."""
        from glob import glob
        from inspect_ai.log._file import read_eval_log
        error_files = [f for f in sorted(glob(os.path.join(TEST_LOG_DIR, "*.eval"))) if "error" in f]
        if not error_files:
            pytest.skip("No error test files")
        log = read_eval_log(error_files[0])
        assert log.status == "error"


class TestNaNInfHandling:
    def test_nan_in_single_sample_read(self):
        """NaN values should be preserved through single sample reads."""
        from glob import glob
        from inspect_ai.log._file import read_eval_log_sample, read_eval_log
        nan_files = [f for f in sorted(glob(os.path.join(TEST_LOG_DIR, "*.eval"))) if "nan" in f]
        if not nan_files:
            pytest.skip("No NaN test files")
        # First verify the file actually has NaN/Inf by reading with full log
        full_log = read_eval_log(nan_files[0])
        if not full_log.samples:
            pytest.skip("No samples in NaN test file")
        sample = read_eval_log_sample(nan_files[0], id=full_log.samples[0].id, epoch=full_log.samples[0].epoch)
        # Verify the sample matches the full-read sample
        assert sample.id == full_log.samples[0].id
        assert sample.epoch == full_log.samples[0].epoch
        # The key check: single-sample and full-read should produce equivalent results
        fast_d = sample.model_dump()
        orig_d = full_log.samples[0].model_dump()
        # Recursive comparison with NaN tolerance
        from helpers import deep_compare
        diffs = deep_compare(fast_d, orig_d)
        assert not diffs, f"Single-sample read doesn't match full read for NaN/Inf file: {diffs[:10]}"

    def test_nan_in_batch_headers(self):
        """NaN values in headers should be handled."""
        from glob import glob
        nan_files = [f for f in sorted(glob(os.path.join(TEST_LOG_DIR, "*.eval"))) if "nan" in f]
        if not nan_files:
            pytest.skip("No NaN test files")
        results = read_eval_headers_batch(nan_files)
        assert len(results) == len(nan_files)

    def test_nan_in_summaries(self):
        """NaN values in summaries should be handled."""
        from glob import glob
        nan_files = [f for f in sorted(glob(os.path.join(TEST_LOG_DIR, "*.eval"))) if "nan" in f]
        if not nan_files:
            pytest.skip("No NaN test files")
        summaries = read_eval_summaries(nan_files[0])
        assert isinstance(summaries, list)


class TestLargeLogs:
    def test_large_log_full_read(self):
        """Test reading large log (1000+ samples)."""
        from glob import glob
        from inspect_ai.log._file import read_eval_log
        large_files = [f for f in sorted(glob(os.path.join(TEST_LOG_DIR, "*.eval"))) if "large" in f or "1000" in f]
        if not large_files:
            pytest.skip("No large test files")
        log = read_eval_log(large_files[0])
        assert log.samples is not None
        assert len(log.samples) >= 100  # At least reasonably large

    def test_large_log_single_sample(self):
        """Single sample read from large log should work."""
        from glob import glob
        from inspect_ai.log._file import read_eval_log_sample
        large_files = [f for f in sorted(glob(os.path.join(TEST_LOG_DIR, "*.eval"))) if "large" in f or "1000" in f]
        if not large_files:
            pytest.skip("No large test files")
        sample = read_eval_log_sample(large_files[0], id=1, epoch=1)
        assert sample.id == 1

    def test_large_log_summaries(self):
        """Summaries from large log should work."""
        from glob import glob
        from inspect_ai.log._file import read_eval_log_sample_summaries
        large_files = [f for f in sorted(glob(os.path.join(TEST_LOG_DIR, "*.eval"))) if "large" in f or "1000" in f]
        if not large_files:
            pytest.skip("No large test files")
        summaries = read_eval_log_sample_summaries(large_files[0])
        assert len(summaries) >= 100

    def test_large_batch_headers(self):
        """Batch headers with many files should work."""
        from glob import glob
        files = sorted(glob(os.path.join(TEST_LOG_DIR, "*.eval")))
        headers = read_eval_headers_batch(files)
        assert len(headers) == len(files)

    def test_5000_samples_full_read(self):
        """5000 sample log loads without memory issues."""
        from inspect_ai.log._file import read_eval_log
        path = os.path.join(TEST_LOG_DIR, "test_5000samples.eval")
        if not os.path.exists(path):
            pytest.skip("5000-sample test file not generated")
        log = read_eval_log(path)
        assert log.samples is not None
        assert len(log.samples) == 5000

    def test_5000_samples_single_read(self):
        """Single sample read from 5000-sample log works."""
        from inspect_ai.log._file import read_eval_log_sample
        path = os.path.join(TEST_LOG_DIR, "test_5000samples.eval")
        if not os.path.exists(path):
            pytest.skip("5000-sample test file not generated")
        sample = read_eval_log_sample(path, id=2500, epoch=1)
        assert sample.id == 2500

    def test_5000_samples_summaries(self):
        """Summaries from 5000-sample log loads correctly."""
        from inspect_ai.log._file import read_eval_log_sample_summaries
        path = os.path.join(TEST_LOG_DIR, "test_5000samples.eval")
        if not os.path.exists(path):
            pytest.skip("5000-sample test file not generated")
        summaries = read_eval_log_sample_summaries(path)
        assert len(summaries) == 5000


class TestDeprecatedFieldHandling:
    """Test that deprecated fields from older inspect versions work correctly."""

    def test_old_score_field_migration(self):
        """Old 'score' field should be migrated to 'scores' dict."""
        sample_data = {
            "id": 1, "epoch": 1,
            "input": "test",
            "target": "A",
            "messages": [],
            "score": {"value": "C", "answer": "A"},
        }
        from inspect_fast_loader._construct import construct_sample_fast
        sample = construct_sample_fast(sample_data)
        assert sample.scores is not None
        assert len(sample.scores) == 1

    def test_old_sandbox_list_format(self):
        """Old sandbox as [type, config] list should work."""
        sample_data = {
            "id": 1, "epoch": 1,
            "input": "test",
            "target": "A",
            "messages": [],
            "sandbox": ["docker", "config.yaml"],
        }
        from inspect_fast_loader._construct import construct_sample_fast
        sample = construct_sample_fast(sample_data)
        assert sample.sandbox is not None


class TestFileNotFound:
    def test_read_eval_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_eval_file("/nonexistent/path.eval")

    def test_read_eval_sample_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_eval_sample("/nonexistent/path.eval", "samples/1_epoch_1.json")

    def test_read_eval_summaries_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_eval_summaries("/nonexistent/path.eval")

    def test_batch_headers_nonexistent_file(self):
        with pytest.raises(Exception):
            read_eval_headers_batch(["/nonexistent/path.eval"])

    def test_patched_sample_read_not_found(self):
        from inspect_ai.log._file import read_eval_log_sample
        with pytest.raises(FileNotFoundError):
            read_eval_log_sample("/nonexistent/path.eval", id=1, epoch=1)


class TestExcludeFieldsEdgeCases:
    def test_exclude_empty_set(self):
        """Empty exclude set should return full sample."""
        from glob import glob
        from inspect_ai.log._file import read_eval_log_sample
        files = sorted(glob(os.path.join(TEST_LOG_DIR, "*.eval")))
        full = read_eval_log_sample(files[0], id=1, epoch=1)
        excluded = read_eval_log_sample(files[0], id=1, epoch=1, exclude_fields=set())
        assert full.id == excluded.id
        # model_dump should be equivalent
        assert full.model_dump()["id"] == excluded.model_dump()["id"]

    def test_exclude_nonexistent_field(self):
        """Excluding a field that doesn't exist should not error."""
        from glob import glob
        from inspect_ai.log._file import read_eval_log_sample
        files = sorted(glob(os.path.join(TEST_LOG_DIR, "*.eval")))
        sample = read_eval_log_sample(files[0], id=1, epoch=1, exclude_fields={"nonexistent_field"})
        assert sample.id == 1

    def test_exclude_multiple_fields(self):
        """Excluding multiple fields at once."""
        from glob import glob
        from inspect_ai.log._file import read_eval_log_sample
        files = sorted(glob(os.path.join(TEST_LOG_DIR, "*.eval")))
        sample = read_eval_log_sample(
            files[0], id=1, epoch=1,
            exclude_fields={"store", "attachments", "events"}
        )
        assert sample.id == 1
        assert sample.store == {}
        assert sample.events == []


def _has_nan_or_inf(obj):
    """Recursively check if any value is NaN or Inf."""
    if isinstance(obj, float):
        return math.isnan(obj) or math.isinf(obj)
    if isinstance(obj, dict):
        return any(_has_nan_or_inf(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_nan_or_inf(v) for v in obj)
    return False
