"""Tests for the test log generator and log loading via inspect's Python API."""

import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from generate_test_logs import generate_all_test_logs, generate_log, write_eval_log, write_json_log
import random
import tempfile


@pytest.fixture(scope="module")
def test_logs_dir():
    """Generate test logs in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_all_test_logs(tmpdir, seed=42)
        yield tmpdir


def test_generator_produces_valid_eval_log(test_logs_dir):
    """Verify .eval logs are loadable by inspect."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(os.path.join(test_logs_dir, "test_10samples.eval"))
    assert log.status == "success"
    assert len(log.samples) == 10
    assert log.eval.task == "test_task"
    assert log.version == 2


def test_generator_produces_valid_json_log(test_logs_dir):
    """Verify .json logs are loadable by inspect."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(os.path.join(test_logs_dir, "test_10samples.json"))
    assert log.status == "success"
    assert len(log.samples) == 10
    assert log.eval.task == "test_task"
    assert log.version == 2


def test_eval_and_json_produce_same_data(test_logs_dir):
    """Verify .eval and .json formats produce equivalent data."""
    from inspect_ai.log import read_eval_log

    eval_log = read_eval_log(os.path.join(test_logs_dir, "test_10samples.eval"))
    json_log = read_eval_log(os.path.join(test_logs_dir, "test_10samples.json"))

    assert eval_log.status == json_log.status
    assert eval_log.eval.task == json_log.eval.task
    assert eval_log.eval.model == json_log.eval.model
    assert len(eval_log.samples) == len(json_log.samples)

    # Check sample ids and epochs match
    eval_ids = [(s.id, s.epoch) for s in eval_log.samples]
    json_ids = [(s.id, s.epoch) for s in json_log.samples]
    assert sorted(eval_ids) == sorted(json_ids)


def test_header_only_read_eval(test_logs_dir):
    """Verify header-only reads work for .eval format."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(os.path.join(test_logs_dir, "test_100samples.eval"), header_only=True)
    assert log.samples is None
    assert log.results is not None
    assert log.eval.task == "test_task"
    assert log.eval.config.epochs is not None


def test_header_only_read_json(test_logs_dir):
    """Verify header-only reads work for .json format."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(os.path.join(test_logs_dir, "test_100samples.json"), header_only=True)
    assert log.samples is None
    assert log.results is not None


def test_multiepoch_log(test_logs_dir):
    """Verify multi-epoch logs are correct."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(os.path.join(test_logs_dir, "test_multiepoch.eval"))
    assert log.eval.config.epochs == 3
    assert len(log.samples) == 60  # 20 samples * 3 epochs

    # Verify all epochs present
    epochs = {s.epoch for s in log.samples}
    assert epochs == {1, 2, 3}


def test_error_log(test_logs_dir):
    """Verify error status logs load correctly."""
    from inspect_ai.log import read_eval_log

    for fmt in ["eval", "json"]:
        log = read_eval_log(os.path.join(test_logs_dir, f"test_error.{fmt}"))
        assert log.status == "error"
        assert log.error is not None
        assert "something went wrong" in log.error.message


def test_cancelled_log(test_logs_dir):
    """Verify cancelled status logs load correctly."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(os.path.join(test_logs_dir, "test_cancelled.eval"))
    assert log.status == "cancelled"


def test_nan_inf_json_log(test_logs_dir):
    """Verify NaN/Inf values in JSON logs can be loaded."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(os.path.join(test_logs_dir, "test_nan_inf.json"))
    assert log.status == "success"
    assert len(log.samples) == 10


def test_nan_inf_eval_log(test_logs_dir):
    """Verify NaN/Inf values in .eval logs can be loaded."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(os.path.join(test_logs_dir, "test_nan_inf.eval"))
    assert log.status == "success"
    assert len(log.samples) == 10


def test_attachments_log(test_logs_dir):
    """Verify logs with attachments load correctly."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(os.path.join(test_logs_dir, "test_attachments.eval"))
    assert log.status == "success"
    assert len(log.samples) == 15

    samples_with_attachments = [s for s in log.samples if s.attachments]
    assert len(samples_with_attachments) > 0


def test_sample_summaries_eval(test_logs_dir):
    """Verify sample summaries can be read from .eval format."""
    from inspect_ai.log._file import read_eval_log_sample_summaries

    summaries = read_eval_log_sample_summaries(os.path.join(test_logs_dir, "test_10samples.eval"))
    assert len(summaries) == 10
    for summary in summaries:
        assert summary.id is not None
        assert summary.epoch >= 1


def test_batch_header_reading(test_logs_dir):
    """Verify batch header reading works."""
    from inspect_ai.log._file import read_eval_log_headers

    files = [os.path.join(test_logs_dir, f"batch_{i:03d}.eval") for i in range(50)]
    headers = read_eval_log_headers(files)
    assert len(headers) == 50
    for h in headers:
        assert h.status == "success"
        assert h.samples is None  # header_only


def test_large_log_loadable(test_logs_dir):
    """Verify 1000-sample logs load correctly."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(os.path.join(test_logs_dir, "test_1000samples.eval"))
    assert len(log.samples) == 1000

    log = read_eval_log(os.path.join(test_logs_dir, "test_1000samples.json"))
    assert len(log.samples) == 1000


def test_deterministic_generation():
    """Verify the generator is deterministic (same seed = same output)."""
    rng1 = random.Random(12345)
    log1 = generate_log(rng1, n_samples=5)

    rng2 = random.Random(12345)
    log2 = generate_log(rng2, n_samples=5)

    # Same seed should produce identical logs
    assert log1["eval"]["eval_id"] == log2["eval"]["eval_id"]
    assert len(log1["samples"]) == len(log2["samples"])
    for s1, s2 in zip(log1["samples"], log2["samples"]):
        assert s1["uuid"] == s2["uuid"]


def test_string_sample_ids(test_logs_dir):
    """Verify logs with string sample IDs load correctly."""
    from inspect_ai.log import read_eval_log

    for fmt in ["eval", "json"]:
        log = read_eval_log(os.path.join(test_logs_dir, f"test_string_ids.{fmt}"))
        assert log.status == "success"
        assert len(log.samples) == 5
        # Verify IDs are strings like "sample_A", "sample_B", etc.
        ids = {s.id for s in log.samples}
        assert all(isinstance(sid, str) for sid in ids)
        assert "sample_A" in ids


def test_monkey_patch_passthrough(test_logs_dir):
    """Verify monkey-patching works with real log files."""
    from inspect_ai.log import read_eval_log
    from inspect_fast_loader import patch, unpatch

    # Read before patching
    log1 = read_eval_log(os.path.join(test_logs_dir, "test_10samples.eval"))

    # Read with patching (passthrough)
    patch()
    try:
        log2 = read_eval_log(os.path.join(test_logs_dir, "test_10samples.eval"))

        assert log1.status == log2.status
        assert len(log1.samples) == len(log2.samples)
        assert log1.eval.task == log2.eval.task
    finally:
        unpatch()

    # Verify unpatching works
    log3 = read_eval_log(os.path.join(test_logs_dir, "test_10samples.eval"))
    assert log1.status == log3.status
