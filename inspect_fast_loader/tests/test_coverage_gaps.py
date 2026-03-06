"""Tests for coverage gaps identified in the existing test suite.

Covers:
1. resolve_attachments parameter through fast path
2. sample_ids population when missing from dataset
3. Mixed .eval and .json batch headers
4. Explicit format parameter (not "auto")
5. EvalLogInfo and Path input types
6. Streaming (read_eval_log_samples) error paths
7. _detect_format / _resolve_path error paths
8. Reductions content correctness
9. Journal fallback through full patched pipeline
10. Streaming for multi-epoch and non-success logs
11. resolve_attachments through single-sample reads
"""

import asyncio
import json
import os
import tempfile
import zipfile
from pathlib import Path

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

from helpers import assert_logs_equal, deep_compare

TEST_LOG_DIR = Path(__file__).parent.parent.parent / "test_logs"


@pytest.fixture(autouse=True)
def _ensure_patched():
    inspect_fast_loader.patch()
    yield
    inspect_fast_loader.unpatch()


def _make_minimal_eval_zip(
    samples: list[dict] | None = None,
    header: dict | None = None,
    summaries: list[dict] | None = None,
    include_header_json: bool = True,
    reductions: list[dict] | None = None,
    sample_ids: list | None = None,
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
                "dataset": {
                    "name": "test",
                    "location": "test",
                    "samples": 1,
                    "shuffled": False,
                    "sample_ids": sample_ids if sample_ids is not None else [1],
                },
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
            "stats": {
                "started_at": "2024-01-01T00:00:00+00:00",
                "completed_at": "2024-01-01T00:01:00+00:00",
            },
        }

    if samples is None:
        samples = [{
            "id": 1, "epoch": 1,
            "input": "test input",
            "target": "A",
            "messages": [],
            "output": {
                "model": "openai/gpt-4o",
                "choices": [{"message": {"role": "assistant", "content": "A"}, "stop_reason": "stop"}],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
            "scores": {"accuracy": {"value": "C", "answer": "A"}},
        }]

    tmp = tempfile.NamedTemporaryFile(suffix=".eval", delete=False)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        if include_header_json:
            zf.writestr("header.json", json.dumps(header))
        else:
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

        if reductions is not None:
            zf.writestr("reductions.json", json.dumps(reductions))

    tmp.close()
    return tmp.name


# ---- resolve_attachments through fast path ----

class TestResolveAttachments:
    def test_resolve_attachments_full_read_eval(self):
        """resolve_attachments should work through the fast .eval full read path."""
        path = str(TEST_LOG_DIR / "test_attachments.eval")
        if not os.path.exists(path):
            pytest.skip("Attachments test file not found")

        log_no_resolve = read_eval_log(path, resolve_attachments=False)
        log_with_resolve = read_eval_log(path, resolve_attachments=True)

        assert log_no_resolve.samples is not None
        assert log_with_resolve.samples is not None
        assert len(log_no_resolve.samples) == len(log_with_resolve.samples)

        # Both should have samples; compare with original
        inspect_fast_loader.unpatch()
        orig_resolved = read_eval_log(path, resolve_attachments=True)
        inspect_fast_loader.patch()

        assert_logs_equal(orig_resolved, log_with_resolve)

    def test_resolve_attachments_full_read_json(self):
        """resolve_attachments should work through the fast .json full read path."""
        path = str(TEST_LOG_DIR / "test_attachments.json")
        if not os.path.exists(path):
            pytest.skip("Attachments test file not found")

        log_with_resolve = read_eval_log(path, resolve_attachments=True)

        inspect_fast_loader.unpatch()
        orig_resolved = read_eval_log(path, resolve_attachments=True)
        inspect_fast_loader.patch()

        assert_logs_equal(orig_resolved, log_with_resolve)

    def test_resolve_attachments_single_sample(self):
        """resolve_attachments should work for single sample reads."""
        path = str(TEST_LOG_DIR / "test_attachments.eval")
        if not os.path.exists(path):
            pytest.skip("Attachments test file not found")

        sample = read_eval_log_sample(path, id=1, epoch=1, resolve_attachments=True)

        inspect_fast_loader.unpatch()
        orig = read_eval_log_sample(path, id=1, epoch=1, resolve_attachments=True)
        inspect_fast_loader.patch()

        diffs = deep_compare(sample.model_dump(), orig.model_dump())
        assert not diffs, f"resolve_attachments single sample mismatch: {diffs[:10]}"

    def test_resolve_attachments_core_mode(self):
        """resolve_attachments='core' mode should work."""
        path = str(TEST_LOG_DIR / "test_attachments.eval")
        if not os.path.exists(path):
            pytest.skip("Attachments test file not found")

        log = read_eval_log(path, resolve_attachments="core")

        inspect_fast_loader.unpatch()
        orig = read_eval_log(path, resolve_attachments="core")
        inspect_fast_loader.patch()

        assert_logs_equal(orig, log)

    def test_resolve_attachments_full_mode(self):
        """resolve_attachments='full' mode should work."""
        path = str(TEST_LOG_DIR / "test_attachments.eval")
        if not os.path.exists(path):
            pytest.skip("Attachments test file not found")

        log = read_eval_log(path, resolve_attachments="full")

        inspect_fast_loader.unpatch()
        orig = read_eval_log(path, resolve_attachments="full")
        inspect_fast_loader.patch()

        assert_logs_equal(orig, log)


# ---- sample_ids population when missing ----

class TestSampleIdsPopulation:
    def test_sample_ids_populated_when_missing(self):
        """When dataset.sample_ids is None, the fast path should populate it from samples."""
        path = _make_minimal_eval_zip(sample_ids=None)
        try:
            log = read_eval_log(path)
            assert log.eval.dataset.sample_ids is not None
            assert 1 in log.eval.dataset.sample_ids
        finally:
            os.unlink(path)

    def test_sample_ids_not_overwritten_when_present(self):
        """When dataset.sample_ids is already set, it should not be modified."""
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
        """Batch headers should handle a mix of .eval and .json files."""
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
        """Mixed batch results should match individual header reads."""
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
        """Passing format='eval' explicitly should work."""
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")

        log = read_eval_log(path, format="eval")
        assert log.status == "success"
        assert log.samples is not None
        assert len(log.samples) == 10

    def test_explicit_json_format(self):
        """Passing format='json' explicitly should work."""
        path = str(TEST_LOG_DIR / "test_10samples.json")
        if not os.path.exists(path):
            pytest.skip("Test file not found")

        log = read_eval_log(path, format="json")
        assert log.status == "success"
        assert log.samples is not None
        assert len(log.samples) == 10

    def test_explicit_format_matches_auto(self):
        """Explicit format should produce same result as auto-detection."""
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")

        auto = read_eval_log(path, format="auto")
        explicit = read_eval_log(path, format="eval")
        assert_logs_equal(auto, explicit)

    def test_explicit_format_sample_read(self):
        """Explicit format should work for single sample reads."""
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")

        sample = read_eval_log_sample(path, id=1, epoch=1, format="eval")
        assert sample.id == 1

    def test_explicit_format_summaries(self):
        """Explicit format should work for summaries."""
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")

        summaries = read_eval_log_sample_summaries(path, format="eval")
        assert len(summaries) > 0


# ---- EvalLogInfo and Path input types ----

class TestInputTypes:
    def test_path_object_input(self):
        """pathlib.Path input should work for full reads."""
        path = TEST_LOG_DIR / "test_10samples.eval"
        if not path.exists():
            pytest.skip("Test file not found")

        log = read_eval_log(path)
        assert log.status == "success"
        assert log.samples is not None

    def test_path_object_sample_read(self):
        """pathlib.Path input should work for single sample reads."""
        path = TEST_LOG_DIR / "test_10samples.eval"
        if not path.exists():
            pytest.skip("Test file not found")

        sample = read_eval_log_sample(path, id=1, epoch=1)
        assert sample.id == 1

    def test_path_object_summaries(self):
        """pathlib.Path input should work for summaries."""
        path = TEST_LOG_DIR / "test_10samples.eval"
        if not path.exists():
            pytest.skip("Test file not found")

        summaries = read_eval_log_sample_summaries(path)
        assert len(summaries) > 0

    def test_path_object_streaming(self):
        """pathlib.Path input should work for streaming samples."""
        path = TEST_LOG_DIR / "test_10samples.eval"
        if not path.exists():
            pytest.skip("Test file not found")

        samples = list(read_eval_log_samples(path))
        assert len(samples) == 10

    def _make_eval_log_info(self, path: str) -> EvalLogInfo:
        """Create an EvalLogInfo with all required fields."""
        stat = os.stat(path)
        return EvalLogInfo(
            name=path,
            type="eval",
            size=stat.st_size,
            mtime=stat.st_mtime,
            task="test_task",
            task_id="test_task",
            suffix=".eval",
        )

    def test_eval_log_info_input(self):
        """EvalLogInfo input should work for reads."""
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")

        info = self._make_eval_log_info(path)
        log = read_eval_log(info)
        assert log.status == "success"
        assert log.samples is not None

    def test_eval_log_info_sample_read(self):
        """EvalLogInfo input should work for single sample reads."""
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")

        info = self._make_eval_log_info(path)
        sample = read_eval_log_sample(info, id=1, epoch=1)
        assert sample.id == 1

    def test_eval_log_info_batch_headers(self):
        """EvalLogInfo list should work for batch headers."""
        files = sorted(TEST_LOG_DIR.glob("batch_*.eval"))[:3]
        if not files:
            pytest.skip("No batch files found")

        infos = [self._make_eval_log_info(str(f)) for f in files]
        headers = read_eval_log_headers(infos)
        assert len(headers) == len(infos)


# ---- Streaming error paths ----

class TestStreamingErrorPaths:
    def test_streaming_non_success_all_required_raises(self):
        """Streaming with all_samples_required=True on non-success log should raise."""
        cancelled_files = sorted(TEST_LOG_DIR.glob("test_cancelled.eval"))
        if not cancelled_files:
            pytest.skip("No cancelled test files")

        with pytest.raises(RuntimeError, match="does not have all samples"):
            list(read_eval_log_samples(str(cancelled_files[0]), all_samples_required=True))

    def test_streaming_non_success_not_required_works(self):
        """Streaming with all_samples_required=False on non-success log should yield available samples."""
        error_files = sorted(TEST_LOG_DIR.glob("test_error.eval"))
        if not error_files:
            pytest.skip("No error test files")

        # Should not raise, may yield partial samples
        samples = list(read_eval_log_samples(str(error_files[0]), all_samples_required=False))
        # We don't assert exact count since it's a partial log
        assert isinstance(samples, list)

    def test_streaming_multiepoch(self):
        """Streaming should yield all epoch samples in multi-epoch logs."""
        multiepoch_files = sorted(TEST_LOG_DIR.glob("test_multiepoch.eval"))
        if not multiepoch_files:
            pytest.skip("No multi-epoch test files")

        streamed = list(read_eval_log_samples(str(multiepoch_files[0])))
        full_log = read_eval_log(str(multiepoch_files[0]))

        assert len(streamed) == len(full_log.samples)

        # Check all epochs represented
        epochs = {s.epoch for s in streamed}
        assert len(epochs) > 1, "Multi-epoch log should have multiple epochs"

    def test_streaming_string_ids(self):
        """Streaming should work with string sample IDs."""
        string_files = sorted(TEST_LOG_DIR.glob("test_string_ids.eval"))
        if not string_files:
            pytest.skip("No string ID test files")

        samples = list(read_eval_log_samples(str(string_files[0])))
        assert len(samples) > 0
        for s in samples:
            assert isinstance(s.id, str)

    def test_streaming_with_resolve_attachments(self):
        """Streaming with resolve_attachments should work."""
        path = str(TEST_LOG_DIR / "test_attachments.eval")
        if not os.path.exists(path):
            pytest.skip("Attachments test file not found")

        samples = list(read_eval_log_samples(path, resolve_attachments=True))
        assert len(samples) > 0

    def test_streaming_with_exclude_and_resolve(self):
        """Streaming with both exclude_fields and resolve_attachments."""
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")

        samples = list(read_eval_log_samples(
            path, exclude_fields={"store"}, resolve_attachments=False
        ))
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
        result = _resolve_path(Path("/some/path.eval"))
        assert result == "/some/path.eval"

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
    def test_reductions_match_original(self):
        """Reductions should match between fast and original reads for all .eval files."""
        eval_files = sorted(TEST_LOG_DIR.glob("test_*.eval"))
        if not eval_files:
            pytest.skip("No test .eval files")

        for path in eval_files:
            fast = read_eval_log(str(path))

            inspect_fast_loader.unpatch()
            orig = read_eval_log(str(path))
            inspect_fast_loader.patch()

            # Compare reductions
            if orig.reductions is None:
                assert fast.reductions is None, f"Reductions should be None for {path.name}"
            else:
                assert fast.reductions is not None, f"Reductions missing for {path.name}"
                orig_red = [r.model_dump() for r in orig.reductions]
                fast_red = [r.model_dump() for r in fast.reductions]
                diffs = deep_compare(orig_red, fast_red, f"reductions({path.name})")
                assert not diffs, f"Reductions mismatch for {path.name}: {diffs[:10]}"

    def test_reductions_in_custom_zip(self):
        """Reductions in a custom .eval ZIP should be correctly read."""
        reductions = [
            {
                "scorer": "accuracy",
                "samples": [
                    {"value": "C", "answer": "A", "sample_id": 1},
                ],
            },
        ]
        path = _make_minimal_eval_zip(
            reductions=reductions,
            summaries=[{"id": 1, "epoch": 1, "input": "test", "target": "A", "scores": {"accuracy": {"value": "C"}}}],
        )
        try:
            log = read_eval_log(path)
            assert log.reductions is not None
            assert len(log.reductions) == 1
        finally:
            os.unlink(path)


# ---- Journal fallback through full patched pipeline ----

class TestJournalFallback:
    def test_journal_start_fallback_full_read(self):
        """Full read through patched pipeline with journal fallback (no header.json)."""
        path = _make_minimal_eval_zip(include_header_json=False)
        try:
            log = read_eval_log(path)
            assert log.samples is not None
            assert len(log.samples) == 1
            assert log.samples[0].id == 1
        finally:
            os.unlink(path)


# ---- Async variants of newly tested paths ----

class TestAsyncCoverageGaps:
    def test_async_full_read_with_resolve_attachments(self):
        """Async full read with resolve_attachments should work."""
        path = str(TEST_LOG_DIR / "test_attachments.eval")
        if not os.path.exists(path):
            pytest.skip("Attachments test file not found")

        async def _read():
            from inspect_ai.log._file import read_eval_log_async
            return await read_eval_log_async(path, resolve_attachments=True)

        log = asyncio.run(_read())
        assert log.samples is not None

    def test_async_sample_read_with_exclude_fields(self):
        """Async sample read with exclude_fields should work."""
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")

        async def _read():
            from inspect_ai.log._file import read_eval_log_sample_async
            return await read_eval_log_sample_async(path, id=1, epoch=1, exclude_fields={"store"})

        sample = asyncio.run(_read())
        assert sample.id == 1
        assert sample.store == {}

    def test_async_json_full_read_matches_sync(self):
        """Async .json full read should match sync read."""
        path = str(TEST_LOG_DIR / "test_10samples.json")
        if not os.path.exists(path):
            pytest.skip("Test file not found")

        sync_log = read_eval_log(path)

        async def _read():
            from inspect_ai.log._file import read_eval_log_async
            return await read_eval_log_async(path)

        async_log = asyncio.run(_read())
        assert_logs_equal(sync_log, async_log)

    def test_async_mixed_batch_headers(self):
        """Async batch headers with mixed formats should work."""
        eval_files = sorted(TEST_LOG_DIR.glob("batch_*.eval"))[:2]
        json_files = sorted(TEST_LOG_DIR.glob("test_10samples.json"))
        if not eval_files or not json_files:
            pytest.skip("Need both .eval and .json test files")

        mixed = [str(f) for f in eval_files] + [str(f) for f in json_files]

        async def _read():
            from inspect_ai.log._file import read_eval_log_headers_async
            return await read_eval_log_headers_async(mixed)

        headers = asyncio.run(_read())
        assert len(headers) == len(mixed)


# ---- model_dump roundtrip through all formats and paths ----

class TestModelDumpRoundtrip:
    def test_full_read_model_dump_json_serializable_eval(self):
        """model_dump from fast .eval read should be JSON-serializable."""
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")

        log = read_eval_log(path)
        d = log.model_dump()
        # Should not raise — all values must be JSON-serializable types
        json.dumps(d, default=str)

    def test_full_read_model_dump_json_serializable_json(self):
        """model_dump from fast .json read should be JSON-serializable."""
        path = str(TEST_LOG_DIR / "test_10samples.json")
        if not os.path.exists(path):
            pytest.skip("Test file not found")

        log = read_eval_log(path)
        d = log.model_dump()
        json.dumps(d, default=str)

    def test_single_sample_model_dump_json_serializable(self):
        """model_dump from single sample read should be JSON-serializable."""
        path = str(TEST_LOG_DIR / "test_10samples.eval")
        if not os.path.exists(path):
            pytest.skip("Test file not found")

        sample = read_eval_log_sample(path, id=1, epoch=1)
        d = sample.model_dump()
        json.dumps(d, default=str)


# ---- string_ids through full pipeline ----

class TestStringIds:
    def test_string_ids_full_read_correctness(self):
        """String ID logs should produce identical results to original."""
        path = str(TEST_LOG_DIR / "test_string_ids.eval")
        if not os.path.exists(path):
            pytest.skip("String ID test file not found")

        fast = read_eval_log(path)

        inspect_fast_loader.unpatch()
        orig = read_eval_log(path)
        inspect_fast_loader.patch()

        assert_logs_equal(orig, fast)

    def test_string_ids_json_correctness(self):
        """String ID .json logs should produce identical results to original."""
        path = str(TEST_LOG_DIR / "test_string_ids.json")
        if not os.path.exists(path):
            pytest.skip("String ID test file not found")

        fast = read_eval_log(path)

        inspect_fast_loader.unpatch()
        orig = read_eval_log(path)
        inspect_fast_loader.patch()

        assert_logs_equal(orig, fast)
