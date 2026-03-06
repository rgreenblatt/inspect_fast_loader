"""Correctness tests for patched functions:
- read_eval_log_sample / read_eval_log_sample_async
- read_eval_log_sample_summaries / read_eval_log_sample_summaries_async
- read_eval_log_samples (streaming)
- Batch headers (rayon-based read_eval_headers_batch)
"""

import asyncio

import pytest

import inspect_fast_loader

from helpers import (
    deep_compare,
    read_eval_file,
    read_eval_headers_batch,
    read_eval_sample,
    read_eval_summaries,
    TEST_LOG_DIR,
)


def _get_eval_files(pattern="*.eval"):
    return sorted(TEST_LOG_DIR.glob(pattern))


# ---- Rust native function tests ----

class TestRustBatchHeaders:
    def test_returns_correct_count(self):
        files = [str(f) for f in _get_eval_files()[:5]]
        results = read_eval_headers_batch(files)
        assert len(results) == len(files)

    def test_structure(self):
        files = [str(f) for f in _get_eval_files()[:3]]
        results = read_eval_headers_batch(files)
        for r in results:
            assert "header" in r
            assert r["samples"] is None
            assert "has_header_json" in r
            assert "reductions" in r

    def test_match_single_reads(self):
        """Batch results should match individual read_eval_file results."""
        files = [str(f) for f in _get_eval_files()[:10]]
        batch_results = read_eval_headers_batch(files)
        for path, batch_r in zip(files, batch_results):
            single_r = read_eval_file(path, header_only=True)
            assert batch_r["has_header_json"] == single_r["has_header_json"]
            assert not deep_compare(batch_r["header"], single_r["header"])


class TestRustReadEvalSample:
    def test_reads_sample(self):
        files = [str(f) for f in _get_eval_files()]
        sample = read_eval_sample(files[0], "samples/1_epoch_1.json")
        assert isinstance(sample, dict)
        assert sample["id"] == 1
        assert sample["epoch"] == 1

    def test_missing_sample_raises(self):
        files = [str(f) for f in _get_eval_files()]
        with pytest.raises(KeyError):
            read_eval_sample(files[0], "samples/99999_epoch_1.json")

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read_eval_sample("/nonexistent/file.eval", "samples/1_epoch_1.json")


class TestRustReadEvalSummaries:
    def test_reads_summaries(self):
        files = [str(f) for f in _get_eval_files()]
        summaries = read_eval_summaries(files[0])
        assert isinstance(summaries, list)
        assert len(summaries) > 0
        for s in summaries:
            assert "id" in s
            assert "epoch" in s

    def test_count_matches_samples(self):
        from inspect_ai.log._file import read_eval_log
        files = [str(f) for f in _get_eval_files()[:5]]
        for f in files:
            log = read_eval_log(f)
            if log.samples:
                summaries = read_eval_summaries(f)
                assert len(summaries) == len(log.samples), f"Summary count mismatch for {f}"


# ---- Patched function correctness tests ----

@pytest.fixture(autouse=True)
def _ensure_patched():
    inspect_fast_loader.patch()
    yield
    inspect_fast_loader.unpatch()


class TestPatchedReadEvalLogSample:
    def _compare_samples(self, fast_sample, orig_sample):
        diffs = deep_compare(fast_sample.model_dump(), orig_sample.model_dump())
        assert not diffs, f"Sample mismatch: {diffs[:10]}"

    def test_basic_read_by_id(self):
        from inspect_ai.log._file import read_eval_log_sample
        files = [str(f) for f in _get_eval_files()]
        sample = read_eval_log_sample(files[0], id=1, epoch=1)
        assert sample.id == 1
        assert sample.epoch == 1

    def test_matches_original(self):
        from inspect_ai.log._file import read_eval_log_sample
        files = [str(f) for f in _get_eval_files()[:5]]

        for f in files:
            fast = read_eval_log_sample(f, id=1, epoch=1)

            inspect_fast_loader.unpatch()
            from inspect_ai.log._file import read_eval_log_sample as orig_read
            orig = orig_read(f, id=1, epoch=1)
            inspect_fast_loader.patch()

            self._compare_samples(fast, orig)

    def test_exclude_fields(self):
        from inspect_ai.log._file import read_eval_log_sample
        files = [str(f) for f in _get_eval_files()]
        full_sample = read_eval_log_sample(files[0], id=1, epoch=1)
        excluded = read_eval_log_sample(files[0], id=1, epoch=1, exclude_fields={"store"})
        assert excluded.store == {}
        assert excluded.id == full_sample.id
        assert excluded.epoch == full_sample.epoch

    def test_exclude_fields_attachments(self):
        from inspect_ai.log._file import read_eval_log_sample
        attach_files = [str(f) for f in _get_eval_files() if "attach" in str(f)]
        if not attach_files:
            pytest.skip("No attachment test files")
        sample_with = read_eval_log_sample(attach_files[0], id=1, epoch=1)
        sample_without = read_eval_log_sample(
            attach_files[0], id=1, epoch=1, exclude_fields={"attachments"}
        )
        assert sample_without.attachments is None or sample_without.attachments == {}
        if sample_with.attachments:
            assert not sample_without.attachments
        assert sample_with.id == sample_without.id

    def test_missing_sample_raises_index_error(self):
        from inspect_ai.log._file import read_eval_log_sample
        files = [str(f) for f in _get_eval_files()]
        with pytest.raises(IndexError):
            read_eval_log_sample(files[0], id=99999, epoch=1)

    def test_uuid_lookup(self):
        from inspect_ai.log._file import read_eval_log_sample, read_eval_log_sample_summaries
        files = [str(f) for f in _get_eval_files()]
        summaries = read_eval_log_sample_summaries(files[0])
        if not summaries or not summaries[0].uuid:
            pytest.skip("No uuid in summaries")
        target_uuid = summaries[0].uuid
        sample = read_eval_log_sample(files[0], uuid=target_uuid)
        assert sample.id == summaries[0].id
        assert sample.epoch == summaries[0].epoch

    def test_multiepoch_sample(self):
        from inspect_ai.log._file import read_eval_log_sample
        multiepoch_files = [str(f) for f in _get_eval_files() if "multiepoch" in str(f)]
        if not multiepoch_files:
            pytest.skip("No multi-epoch test files")
        s1 = read_eval_log_sample(multiepoch_files[0], id=1, epoch=1)
        s2 = read_eval_log_sample(multiepoch_files[0], id=1, epoch=2)
        assert s1.epoch == 1
        assert s2.epoch == 2

    def test_string_sample_ids(self):
        from inspect_ai.log._file import read_eval_log_sample
        string_id_files = [str(f) for f in _get_eval_files() if "string" in str(f).lower()]
        if not string_id_files:
            pytest.skip("No string ID test files")
        sample = read_eval_log_sample(string_id_files[0], id="sample_A", epoch=1)
        assert sample.id == "sample_A"

    def test_scorer_placeholder_replaced(self):
        from inspect_ai.log._file import read_eval_log_sample
        files = [str(f) for f in _get_eval_files()[:5]]
        for f in files:
            sample = read_eval_log_sample(f, id=1, epoch=1)
            if sample.scores:
                assert "88F74D2C" not in sample.scores

    def test_json_format_fallback(self):
        from inspect_ai.log._file import read_eval_log_sample
        json_files = list(TEST_LOG_DIR.glob("*.json"))
        if not json_files:
            pytest.skip("No .json test files")
        sample = read_eval_log_sample(str(json_files[0]), id=1, epoch=1)
        assert sample.id == 1


class TestPatchedReadEvalLogSampleSummaries:
    def test_basic_summaries(self):
        from inspect_ai.log._file import read_eval_log_sample_summaries
        files = [str(f) for f in _get_eval_files()]
        summaries = read_eval_log_sample_summaries(files[0])
        assert isinstance(summaries, list)
        assert len(summaries) > 0

    def test_summary_fields(self):
        from inspect_ai.log._file import read_eval_log_sample_summaries
        files = [str(f) for f in _get_eval_files()]
        summaries = read_eval_log_sample_summaries(files[0])
        for s in summaries:
            assert hasattr(s, "id")
            assert hasattr(s, "epoch")
            assert hasattr(s, "input")
            assert hasattr(s, "target")
            assert hasattr(s, "scores")

    def test_matches_original(self):
        from inspect_ai.log._file import read_eval_log_sample_summaries
        files = [str(f) for f in _get_eval_files()[:5]]
        for f in files:
            fast_summaries = read_eval_log_sample_summaries(f)
            inspect_fast_loader.unpatch()
            from inspect_ai.log._file import read_eval_log_sample_summaries as orig_read
            orig_summaries = orig_read(f)
            inspect_fast_loader.patch()
            assert len(fast_summaries) == len(orig_summaries), f"Count mismatch for {f}"
            for fast_s, orig_s in zip(fast_summaries, orig_summaries):
                diffs = deep_compare(fast_s.model_dump(), orig_s.model_dump())
                assert not diffs, f"Summary mismatch for {f}: {diffs[:10]}"

    def test_json_format_fallback(self):
        from inspect_ai.log._file import read_eval_log_sample_summaries
        json_files = list(TEST_LOG_DIR.glob("*.json"))
        if not json_files:
            pytest.skip("No .json test files")
        summaries = read_eval_log_sample_summaries(str(json_files[0]))
        assert isinstance(summaries, list)


class TestPatchedReadEvalLogSamples:
    def test_basic_streaming(self):
        from inspect_ai.log._file import read_eval_log_samples
        files = [str(f) for f in _get_eval_files()]
        samples = list(read_eval_log_samples(files[0]))
        assert len(samples) > 0

    def test_matches_full_read(self):
        from inspect_ai.log._file import read_eval_log, read_eval_log_samples
        files = [str(f) for f in _get_eval_files()[:5]]
        for f in files:
            full_log = read_eval_log(f)
            if not full_log.samples:
                continue
            streaming = list(read_eval_log_samples(f))
            assert len(streaming) == len(full_log.samples), f"Count mismatch for {f}"
            for stream_s, full_s in zip(streaming, full_log.samples):
                assert stream_s.id == full_s.id
                assert stream_s.epoch == full_s.epoch

    def test_exclude_fields(self):
        from inspect_ai.log._file import read_eval_log_samples
        files = [str(f) for f in _get_eval_files()]
        samples = list(read_eval_log_samples(files[0], exclude_fields={"store"}))
        for s in samples:
            assert s.store == {}


class TestPatchedBatchHeaders:
    def test_batch_correctness(self):
        from inspect_ai.log._file import read_eval_log, read_eval_log_headers
        files = [str(f) for f in _get_eval_files()[:10]]
        batch = read_eval_log_headers(files)
        for f, header in zip(files, batch):
            single = read_eval_log(f, header_only=True)
            assert header.eval.task == single.eval.task
            assert header.status == single.status

    def test_batch_matches_original(self):
        from inspect_ai.log._file import read_eval_log_headers
        files = [str(f) for f in _get_eval_files()[:10]]
        fast_batch = read_eval_log_headers(files)
        inspect_fast_loader.unpatch()
        from inspect_ai.log._file import read_eval_log_headers as orig_read
        orig_batch = orig_read(files)
        inspect_fast_loader.patch()
        assert len(fast_batch) == len(orig_batch)
        for fast_h, orig_h in zip(fast_batch, orig_batch):
            assert fast_h.eval.task == orig_h.eval.task
            assert fast_h.status == orig_h.status


class TestAsyncVariants:
    def test_async_read_sample(self):
        from inspect_ai.log._file import read_eval_log_sample_async
        files = [str(f) for f in _get_eval_files()]
        sample = asyncio.run(read_eval_log_sample_async(files[0], id=1, epoch=1))
        assert sample.id == 1

    def test_async_read_summaries(self):
        from inspect_ai.log._file import read_eval_log_sample_summaries_async
        files = [str(f) for f in _get_eval_files()]
        summaries = asyncio.run(read_eval_log_sample_summaries_async(files[0]))
        assert len(summaries) > 0

    def test_async_batch_headers(self):
        from inspect_ai.log._file import read_eval_log_headers_async
        files = [str(f) for f in _get_eval_files()[:5]]
        headers = asyncio.run(read_eval_log_headers_async(files))
        assert len(headers) == len(files)
