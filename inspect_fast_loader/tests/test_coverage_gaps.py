"""Tests for coverage gaps identified in the existing test suite.

Covers resolve_attachments, sample_ids population, mixed batch headers,
explicit format parameter, EvalLogInfo/Path input types, streaming error
paths, helper error paths, reductions correctness, journal fallback,
async variants, and model_dump roundtrip.
"""

import asyncio
import json
import os

import pytest

import inspect_fast_loader
from inspect_ai.log._file import (
    EvalLogInfo,
    read_eval_log,
    read_eval_log_headers,
    read_eval_log_sample,
    read_eval_log_samples,
    read_eval_log_sample_summaries,
)
from inspect_fast_loader._patch import _detect_format, _resolve_path
from pathlib import Path

from helpers import assert_logs_equal, deep_compare, make_minimal_eval_zip, TEST_LOG_DIR


@pytest.fixture(autouse=True)
def _ensure_patched():
    inspect_fast_loader.patch()
    yield
    inspect_fast_loader.unpatch()


# ---- resolve_attachments ----

class TestResolveAttachments:
    def test_full_read_eval(self):
        path = str(TEST_LOG_DIR / "test_attachments.eval")
        if not os.path.exists(path):
            pytest.skip("Attachments test file not found")

        log_no_resolve = read_eval_log(path, resolve_attachments=False)
        log_with_resolve = read_eval_log(path, resolve_attachments=True)

        assert log_no_resolve.samples is not None
        assert log_with_resolve.samples is not None
        assert len(log_no_resolve.samples) == len(log_with_resolve.samples)

        inspect_fast_loader.unpatch()
        orig_resolved = read_eval_log(path, resolve_attachments=True)
        inspect_fast_loader.patch()

        assert_logs_equal(orig_resolved, log_with_resolve)

    def test_full_read_json(self):
        path = str(TEST_LOG_DIR / "test_attachments.json")
        if not os.path.exists(path):
            pytest.skip("Attachments test file not found")

        log_with_resolve = read_eval_log(path, resolve_attachments=True)

        inspect_fast_loader.unpatch()
        orig_resolved = read_eval_log(path, resolve_attachments=True)
        inspect_fast_loader.patch()

        assert_logs_equal(orig_resolved, log_with_resolve)

    def test_single_sample(self):
        path = str(TEST_LOG_DIR / "test_attachments.eval")
        if not os.path.exists(path):
            pytest.skip("Attachments test file not found")

        sample = read_eval_log_sample(path, id=1, epoch=1, resolve_attachments=True)

        inspect_fast_loader.unpatch()
        orig = read_eval_log_sample(path, id=1, epoch=1, resolve_attachments=True)
        inspect_fast_loader.patch()

        diffs = deep_compare(sample.model_dump(), orig.model_dump())
        assert not diffs, f"resolve_attachments single sample mismatch: {diffs[:10]}"

    def test_core_mode(self):
        path = str(TEST_LOG_DIR / "test_attachments.eval")
        if not os.path.exists(path):
            pytest.skip("Attachments test file not found")

        log = read_eval_log(path, resolve_attachments="core")

        inspect_fast_loader.unpatch()
        orig = read_eval_log(path, resolve_attachments="core")
        inspect_fast_loader.patch()

        assert_logs_equal(orig, log)

    def test_full_mode(self):
        path = str(TEST_LOG_DIR / "test_attachments.eval")
        if not os.path.exists(path):
            pytest.skip("Attachments test file not found")

        log = read_eval_log(path, resolve_attachments="full")

        inspect_fast_loader.unpatch()
        orig = read_eval_log(path, resolve_attachments="full")
        inspect_fast_loader.patch()

        assert_logs_equal(orig, log)


# ---- sample_ids population ----

class TestSampleIdsPopulation:
    def test_populated_when_missing(self):
        """When dataset.sample_ids is None, the fast path should populate it from samples."""
        path = make_minimal_eval_zip(sample_ids=None)
        try:
            log = read_eval_log(path)
            assert log.eval.dataset.sample_ids is not None
            assert 1 in log.eval.dataset.sample_ids
        finally:
            os.unlink(path)

    def test_not_overwritten_when_present(self):
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")

        log = read_eval_log(path)
        assert log.eval.dataset.sample_ids is not None

        inspect_fast_loader.unpatch()
        orig = read_eval_log(path)
        inspect_fast_loader.patch()

        assert log.eval.dataset.sample_ids == orig.eval.dataset.sample_ids


# ---- Mixed .eval and .json batch headers ----

class TestMixedBatchHeaders:
    def test_mixed_eval_and_json_batch(self):
        eval_files = sorted(TEST_LOG_DIR.glob("batch_*.eval"))[:3]
        json_files = sorted(TEST_LOG_DIR.glob("test_*samples.json"))[:2]
        if not eval_files or not json_files:
            pytest.skip("Need both .eval and .json test files")

        mixed = [str(f) for f in eval_files] + [str(f) for f in json_files]
        headers = read_eval_log_headers(mixed)

        assert len(headers) == len(mixed)
        for h in headers:
            assert h.samples is None
            assert h.status in ("success", "error", "cancelled", "started")

    def test_mixed_batch_matches_individual_reads(self):
        eval_files = sorted(TEST_LOG_DIR.glob("batch_*.eval"))[:2]
        json_files = sorted(TEST_LOG_DIR.glob("test_10samples.json"))
        if not eval_files or not json_files:
            pytest.skip("Need both .eval and .json test files")

        mixed = [str(f) for f in eval_files] + [str(f) for f in json_files]
        batch_headers = read_eval_log_headers(mixed)

        for path, batch_h in zip(mixed, batch_headers):
            single = read_eval_log(path, header_only=True)
            assert batch_h.eval.task == single.eval.task
            assert batch_h.status == single.status


# ---- Explicit format parameter ----

class TestExplicitFormat:
    def test_explicit_eval_format(self):
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")
        log = read_eval_log(path, format="eval")
        assert log.status == "success"
        assert len(log.samples) == 10

    def test_explicit_json_format(self):
        path = str(TEST_LOG_DIR / "test_10samples.json")
        if not os.path.exists(path):
            pytest.skip("Test file not found")
        log = read_eval_log(path, format="json")
        assert log.status == "success"
        assert len(log.samples) == 10

    def test_explicit_format_matches_auto(self):
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")
        assert_logs_equal(read_eval_log(path, format="auto"), read_eval_log(path, format="eval"))

    def test_explicit_format_sample_read(self):
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")
        sample = read_eval_log_sample(path, id=1, epoch=1, format="eval")
        assert sample.id == 1

    def test_explicit_format_summaries(self):
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")
        summaries = read_eval_log_sample_summaries(path, format="eval")
        assert len(summaries) > 0


# ---- EvalLogInfo and Path input types ----

class TestInputTypes:
    def test_path_object_input(self):
        path = TEST_LOG_DIR / "test_10samples.eval"
        if not path.exists():
            pytest.skip("Test file not found")
        log = read_eval_log(path)
        assert log.status == "success"
        assert log.samples is not None

    def test_path_object_sample_read(self):
        path = TEST_LOG_DIR / "test_10samples.eval"
        if not path.exists():
            pytest.skip("Test file not found")
        assert read_eval_log_sample(path, id=1, epoch=1).id == 1

    def test_path_object_summaries(self):
        path = TEST_LOG_DIR / "test_10samples.eval"
        if not path.exists():
            pytest.skip("Test file not found")
        assert len(read_eval_log_sample_summaries(path)) > 0

    def test_path_object_streaming(self):
        path = TEST_LOG_DIR / "test_10samples.eval"
        if not path.exists():
            pytest.skip("Test file not found")
        assert len(list(read_eval_log_samples(path))) == 10

    def _make_eval_log_info(self, path: str) -> EvalLogInfo:
        stat = os.stat(path)
        return EvalLogInfo(
            name=path, type="eval", size=stat.st_size, mtime=stat.st_mtime,
            task="test_task", task_id="test_task", suffix=".eval",
        )

    def test_eval_log_info_input(self):
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")
        log = read_eval_log(self._make_eval_log_info(path))
        assert log.status == "success"
        assert log.samples is not None

    def test_eval_log_info_sample_read(self):
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")
        assert read_eval_log_sample(self._make_eval_log_info(path), id=1, epoch=1).id == 1

    def test_eval_log_info_batch_headers(self):
        files = sorted(TEST_LOG_DIR.glob("batch_*.eval"))[:3]
        if not files:
            pytest.skip("No batch files found")
        infos = [self._make_eval_log_info(str(f)) for f in files]
        assert len(read_eval_log_headers(infos)) == len(infos)


# ---- Streaming error paths ----

class TestStreamingErrorPaths:
    def test_non_success_all_required_raises(self):
        cancelled_files = sorted(TEST_LOG_DIR.glob("test_cancelled.eval"))
        if not cancelled_files:
            pytest.skip("No cancelled test files")
        with pytest.raises(RuntimeError, match="does not have all samples"):
            list(read_eval_log_samples(str(cancelled_files[0]), all_samples_required=True))

    def test_non_success_not_required_works(self):
        error_files = sorted(TEST_LOG_DIR.glob("test_error.eval"))
        if not error_files:
            pytest.skip("No error test files")
        samples = list(read_eval_log_samples(str(error_files[0]), all_samples_required=False))
        assert isinstance(samples, list)

    def test_multiepoch(self):
        multiepoch_files = sorted(TEST_LOG_DIR.glob("test_multiepoch.eval"))
        if not multiepoch_files:
            pytest.skip("No multi-epoch test files")

        streamed = list(read_eval_log_samples(str(multiepoch_files[0])))
        full_log = read_eval_log(str(multiepoch_files[0]))

        assert len(streamed) == len(full_log.samples)
        epochs = {s.epoch for s in streamed}
        assert len(epochs) > 1

    def test_string_ids(self):
        string_files = sorted(TEST_LOG_DIR.glob("test_string_ids.eval"))
        if not string_files:
            pytest.skip("No string ID test files")
        samples = list(read_eval_log_samples(str(string_files[0])))
        assert len(samples) > 0
        for s in samples:
            assert isinstance(s.id, str)

    def test_with_resolve_attachments(self):
        path = str(TEST_LOG_DIR / "test_attachments.eval")
        if not os.path.exists(path):
            pytest.skip("Attachments test file not found")
        assert len(list(read_eval_log_samples(path, resolve_attachments=True))) > 0

    def test_with_exclude_and_resolve(self):
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")
        samples = list(read_eval_log_samples(path, exclude_fields={"store"}, resolve_attachments=False))
        assert len(samples) == 10
        for s in samples:
            assert s.store == {}


# ---- _detect_format / _resolve_path error paths ----

class TestHelperErrorPaths:
    def test_detect_format_unknown_extension(self):
        with pytest.raises(ValueError, match="Cannot detect format"):
            _detect_format("/some/file.txt", "auto")

    def test_detect_format_no_extension(self):
        with pytest.raises(ValueError, match="Cannot detect format"):
            _detect_format("/some/file", "auto")

    def test_detect_format_explicit_overrides_extension(self):
        assert _detect_format("/some/file.json", "eval") == "eval"
        assert _detect_format("/some/file.eval", "json") == "json"

    def test_resolve_path_string(self):
        assert _resolve_path("/some/path.eval") == "/some/path.eval"

    def test_resolve_path_pathlib(self):
        assert _resolve_path(Path("/some/path.eval")) == "/some/path.eval"

    def test_resolve_path_eval_log_info(self):
        info = EvalLogInfo(
            name="/some/path.eval", type="eval",
            size=0, mtime=0.0, task="t", task_id="t", suffix=".eval",
        )
        assert _resolve_path(info) == "/some/path.eval"

    def test_resolve_path_unexpected_type(self):
        with pytest.raises(TypeError, match="Unexpected log_file type"):
            _resolve_path(12345)


# ---- Reductions correctness ----

class TestReductionsCorrectness:
    def test_match_original(self):
        eval_files = sorted(TEST_LOG_DIR.glob("test_*.eval"))
        if not eval_files:
            pytest.skip("No test .eval files")

        for path in eval_files:
            fast = read_eval_log(str(path))

            inspect_fast_loader.unpatch()
            orig = read_eval_log(str(path))
            inspect_fast_loader.patch()

            if orig.reductions is None:
                assert fast.reductions is None, f"Reductions should be None for {path.name}"
            else:
                assert fast.reductions is not None, f"Reductions missing for {path.name}"
                orig_red = [r.model_dump() for r in orig.reductions]
                fast_red = [r.model_dump() for r in fast.reductions]
                diffs = deep_compare(orig_red, fast_red, f"reductions({path.name})")
                assert not diffs, f"Reductions mismatch for {path.name}: {diffs[:10]}"

    def test_in_custom_zip(self):
        reductions = [{
            "scorer": "accuracy",
            "samples": [{"value": "C", "answer": "A", "sample_id": 1}],
        }]
        path = make_minimal_eval_zip(
            reductions=reductions,
            summaries=[{"id": 1, "epoch": 1, "input": "test", "target": "A", "scores": {"accuracy": {"value": "C"}}}],
        )
        try:
            log = read_eval_log(path)
            assert log.reductions is not None
            assert len(log.reductions) == 1
        finally:
            os.unlink(path)


# ---- Journal fallback ----

class TestJournalFallback:
    def test_journal_start_fallback_full_read(self):
        path = make_minimal_eval_zip(include_header_json=False)
        try:
            log = read_eval_log(path)
            assert log.samples is not None
            assert len(log.samples) == 1
            assert log.samples[0].id == 1
        finally:
            os.unlink(path)


# ---- Async variants ----

class TestAsyncCoverageGaps:
    def test_full_read_with_resolve_attachments(self):
        path = str(TEST_LOG_DIR / "test_attachments.eval")
        if not os.path.exists(path):
            pytest.skip("Attachments test file not found")

        async def _read():
            from inspect_ai.log._file import read_eval_log_async
            return await read_eval_log_async(path, resolve_attachments=True)

        assert asyncio.run(_read()).samples is not None

    def test_sample_read_with_exclude_fields(self):
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")

        async def _read():
            from inspect_ai.log._file import read_eval_log_sample_async
            return await read_eval_log_sample_async(path, id=1, epoch=1, exclude_fields={"store"})

        sample = asyncio.run(_read())
        assert sample.id == 1
        assert sample.store == {}

    def test_json_full_read_matches_sync(self):
        path = str(TEST_LOG_DIR / "test_10samples.json")
        if not os.path.exists(path):
            pytest.skip("Test file not found")

        sync_log = read_eval_log(path)

        async def _read():
            from inspect_ai.log._file import read_eval_log_async
            return await read_eval_log_async(path)

        assert_logs_equal(sync_log, asyncio.run(_read()))

    def test_mixed_batch_headers(self):
        eval_files = sorted(TEST_LOG_DIR.glob("batch_*.eval"))[:2]
        json_files = sorted(TEST_LOG_DIR.glob("test_10samples.json"))
        if not eval_files or not json_files:
            pytest.skip("Need both .eval and .json test files")

        mixed = [str(f) for f in eval_files] + [str(f) for f in json_files]

        async def _read():
            from inspect_ai.log._file import read_eval_log_headers_async
            return await read_eval_log_headers_async(mixed)

        assert len(asyncio.run(_read())) == len(mixed)


# ---- model_dump roundtrip ----

class TestModelDumpRoundtrip:
    def test_full_read_eval(self):
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")
        d = read_eval_log(path).model_dump()
        json.dumps(d, allow_nan=True)

    def test_full_read_json(self):
        path = str(TEST_LOG_DIR / "test_10samples.json")
        if not os.path.exists(path):
            pytest.skip("Test file not found")
        d = read_eval_log(path).model_dump()
        json.dumps(d, allow_nan=True)

    def test_single_sample(self):
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")
        d = read_eval_log_sample(path, id=1, epoch=1).model_dump()
        json.dumps(d, allow_nan=True)


# ---- String IDs ----

class TestStringIds:
    def test_full_read_correctness(self):
        path = str(TEST_LOG_DIR / "test_string_ids.eval")
        if not os.path.exists(path):
            pytest.skip("String ID test file not found")

        fast = read_eval_log(path)

        inspect_fast_loader.unpatch()
        orig = read_eval_log(path)
        inspect_fast_loader.patch()

        assert_logs_equal(orig, fast)

    def test_json_correctness(self):
        path = str(TEST_LOG_DIR / "test_string_ids.json")
        if not os.path.exists(path):
            pytest.skip("String ID test file not found")

        fast = read_eval_log(path)

        inspect_fast_loader.unpatch()
        orig = read_eval_log(path)
        inspect_fast_loader.patch()

        assert_logs_equal(orig, fast)
