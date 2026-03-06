"""Edge case and error handling tests.

Tests for corrupted ZIP files, missing entries, partial writes, deprecated
fields, large logs, NaN/Inf handling, exclude_fields edge cases, and
file-not-found errors.
"""

import os

import pytest

import inspect_fast_loader
from inspect_fast_loader._native import read_eval_headers_batch as _read_eval_headers_batch_raw

from helpers import (
    deep_compare,
    make_minimal_eval_zip,
    read_eval_file,
    read_eval_sample,
    read_eval_summaries,
    TEST_LOG_DIR,
)


@pytest.fixture(autouse=True)
def _ensure_patched():
    inspect_fast_loader.patch()
    yield
    inspect_fast_loader.unpatch()


class TestCorruptedZipFiles:
    def test_truncated_zip_file(self):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".eval", delete=False)
        tmp.write(b"PK\x03\x04" + b"\x00" * 100)
        tmp.close()
        try:
            with pytest.raises(Exception):
                read_eval_file(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_not_a_zip_file(self):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".eval", delete=False)
        tmp.write(b"this is not a zip file")
        tmp.close()
        try:
            with pytest.raises(Exception):
                read_eval_file(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_empty_file(self):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".eval", delete=False)
        tmp.close()
        try:
            with pytest.raises(Exception):
                read_eval_file(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_batch_headers_with_corrupted_file(self):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".eval", delete=False)
        tmp.write(b"not a zip")
        tmp.close()
        try:
            with pytest.raises(Exception):
                _read_eval_headers_batch_raw([tmp.name])
        finally:
            os.unlink(tmp.name)


class TestMissingZipEntries:
    def test_missing_header_and_journal(self):
        """ZIP with neither header.json nor _journal/start.json should error."""
        import tempfile
        import zipfile
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
        path = make_minimal_eval_zip(include_header_json=False)
        try:
            raw = read_eval_file(path, header_only=True)
            assert raw["has_header_json"] is False
            assert raw["header"]["version"] == 3
        finally:
            os.unlink(path)

    def test_missing_sample_entry(self):
        files = sorted(TEST_LOG_DIR.glob("*.eval"))
        with pytest.raises(KeyError):
            read_eval_sample(str(files[0]), "samples/nonexistent_epoch_1.json")


class TestPartialWrites:
    def test_eval_zip_no_samples(self):
        """ZIP with header but no samples (interrupted write)."""
        path = make_minimal_eval_zip(samples=[])
        try:
            raw = read_eval_file(path)
            assert raw["samples"] is not None
            assert len(raw["samples"]) == 0
        finally:
            os.unlink(path)

    def test_cancelled_log_handling(self):
        from inspect_ai.log._file import read_eval_log
        cancelled_files = [f for f in sorted(TEST_LOG_DIR.glob("*.eval")) if "cancelled" in str(f)]
        if not cancelled_files:
            pytest.skip("No cancelled test files")
        log = read_eval_log(str(cancelled_files[0]))
        assert log.status == "cancelled"

    def test_error_log_handling(self):
        from inspect_ai.log._file import read_eval_log
        error_files = [f for f in sorted(TEST_LOG_DIR.glob("*.eval")) if "error" in str(f)]
        if not error_files:
            pytest.skip("No error test files")
        log = read_eval_log(str(error_files[0]))
        assert log.status == "error"


class TestNaNInfHandling:
    def test_nan_in_single_sample_read(self):
        """NaN values should be preserved through single sample reads."""
        from inspect_ai.log._file import read_eval_log_sample, read_eval_log
        nan_files = [f for f in sorted(TEST_LOG_DIR.glob("*.eval")) if "nan" in str(f)]
        if not nan_files:
            pytest.skip("No NaN test files")
        full_log = read_eval_log(str(nan_files[0]))
        if not full_log.samples:
            pytest.skip("No samples in NaN test file")
        sample = read_eval_log_sample(str(nan_files[0]), id=full_log.samples[0].id, epoch=full_log.samples[0].epoch)
        assert sample.id == full_log.samples[0].id
        diffs = deep_compare(sample.model_dump(), full_log.samples[0].model_dump())
        assert not diffs, f"Single-sample read doesn't match full read for NaN/Inf file: {diffs[:10]}"

    def test_nan_in_batch_headers(self):
        nan_files = [str(f) for f in sorted(TEST_LOG_DIR.glob("*.eval")) if "nan" in str(f)]
        if not nan_files:
            pytest.skip("No NaN test files")
        results = _read_eval_headers_batch_raw(nan_files)
        assert len(results) == len(nan_files)

    def test_nan_in_summaries(self):
        nan_files = [f for f in sorted(TEST_LOG_DIR.glob("*.eval")) if "nan" in str(f)]
        if not nan_files:
            pytest.skip("No NaN test files")
        summaries = read_eval_summaries(str(nan_files[0]))
        assert isinstance(summaries, list)


class TestLargeLogs:
    def test_large_log_full_read(self):
        from inspect_ai.log._file import read_eval_log
        large_files = [f for f in sorted(TEST_LOG_DIR.glob("*.eval")) if "1000" in str(f)]
        if not large_files:
            pytest.skip("No large test files")
        log = read_eval_log(str(large_files[0]))
        assert log.samples is not None
        assert len(log.samples) >= 100

    def test_large_log_single_sample(self):
        from inspect_ai.log._file import read_eval_log_sample
        large_files = [f for f in sorted(TEST_LOG_DIR.glob("*.eval")) if "1000" in str(f)]
        if not large_files:
            pytest.skip("No large test files")
        sample = read_eval_log_sample(str(large_files[0]), id=1, epoch=1)
        assert sample.id == 1

    def test_large_log_summaries(self):
        from inspect_ai.log._file import read_eval_log_sample_summaries
        large_files = [f for f in sorted(TEST_LOG_DIR.glob("*.eval")) if "1000" in str(f)]
        if not large_files:
            pytest.skip("No large test files")
        summaries = read_eval_log_sample_summaries(str(large_files[0]))
        assert len(summaries) >= 100

    def test_large_batch_headers(self):
        files = [str(f) for f in sorted(TEST_LOG_DIR.glob("*.eval"))]
        results = _read_eval_headers_batch_raw(files)
        assert len(results) == len(files)

    def test_5000_samples_full_read(self):
        from inspect_ai.log._file import read_eval_log
        path = str(TEST_LOG_DIR / "test_5000samples.eval")
        if not os.path.exists(path):
            pytest.skip("5000-sample test file not generated")
        log = read_eval_log(path)
        assert log.samples is not None
        assert len(log.samples) == 5000

    def test_5000_samples_single_read(self):
        from inspect_ai.log._file import read_eval_log_sample
        path = str(TEST_LOG_DIR / "test_5000samples.eval")
        if not os.path.exists(path):
            pytest.skip("5000-sample test file not generated")
        sample = read_eval_log_sample(path, id=2500, epoch=1)
        assert sample.id == 2500

    def test_5000_samples_summaries(self):
        from inspect_ai.log._file import read_eval_log_sample_summaries
        path = str(TEST_LOG_DIR / "test_5000samples.eval")
        if not os.path.exists(path):
            pytest.skip("5000-sample test file not generated")
        summaries = read_eval_log_sample_summaries(path)
        assert len(summaries) == 5000


class TestDeprecatedFieldHandling:
    def test_old_score_field_migration(self):
        from inspect_fast_loader._construct import construct_sample_fast
        sample = construct_sample_fast({
            "id": 1, "epoch": 1,
            "input": "test", "target": "A",
            "messages": [],
            "score": {"value": "C", "answer": "A"},
        })
        assert sample.scores is not None
        assert len(sample.scores) == 1

    def test_old_sandbox_list_format(self):
        from inspect_fast_loader._construct import construct_sample_fast
        sample = construct_sample_fast({
            "id": 1, "epoch": 1,
            "input": "test", "target": "A",
            "messages": [],
            "sandbox": ["docker", "config.yaml"],
        })
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
            _read_eval_headers_batch_raw(["/nonexistent/path.eval"])

    def test_patched_sample_read_not_found(self):
        from inspect_ai.log._file import read_eval_log_sample
        with pytest.raises(FileNotFoundError):
            read_eval_log_sample("/nonexistent/path.eval", id=1, epoch=1)


class TestExcludeFieldsEdgeCases:
    def test_exclude_empty_set(self):
        from inspect_ai.log._file import read_eval_log_sample
        files = sorted(TEST_LOG_DIR.glob("*.eval"))
        full = read_eval_log_sample(str(files[0]), id=1, epoch=1)
        excluded = read_eval_log_sample(str(files[0]), id=1, epoch=1, exclude_fields=set())
        assert full.id == excluded.id
        assert full.model_dump()["id"] == excluded.model_dump()["id"]

    def test_exclude_nonexistent_field(self):
        from inspect_ai.log._file import read_eval_log_sample
        files = sorted(TEST_LOG_DIR.glob("*.eval"))
        sample = read_eval_log_sample(str(files[0]), id=1, epoch=1, exclude_fields={"nonexistent_field"})
        assert sample.id == 1

    def test_exclude_multiple_fields(self):
        from inspect_ai.log._file import read_eval_log_sample
        files = sorted(TEST_LOG_DIR.glob("*.eval"))
        sample = read_eval_log_sample(
            str(files[0]), id=1, epoch=1,
            exclude_fields={"store", "attachments", "events"},
        )
        assert sample.id == 1
        assert sample.store == {}
        assert sample.events == []
