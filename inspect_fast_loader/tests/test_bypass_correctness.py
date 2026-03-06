"""Comprehensive correctness tests for the Pydantic bypass construction.

These tests verify that construct_sample_fast produces EvalSample instances
whose model_dump() output matches that of EvalSample.model_validate().
"""

import math
import os
from pathlib import Path

import pytest

from inspect_ai._util.constants import get_deserializing_context
from inspect_ai.log._file import read_eval_log
from inspect_ai.log._log import EvalLog, EvalSample
from inspect_ai.model._chat_message import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._model_output import ChatCompletionChoice, ModelOutput, ModelUsage
from inspect_ai.scorer._metric import Score
from inspect_fast_loader import patch, unpatch
from inspect_fast_loader._construct import construct_sample_fast, _fast_construct
from inspect_fast_loader._native import read_eval_file

TEST_LOGS_DIR = Path(__file__).parent.parent.parent / "test_logs"


@pytest.fixture(autouse=True)
def _ensure_unpatched():
    unpatch()
    yield
    unpatch()


def _approx_equal(a, b, rel_tol=1e-9, abs_tol=1e-12):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isinf(a) and math.isinf(b):
            return (a > 0 and b > 0) or (a < 0 and b < 0)
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    return False


def _deep_compare(orig, fast, path_str="root"):
    """Recursively compare two values, collecting differences."""
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
        if isinstance(orig, (int, float)) and isinstance(fast, (int, float)):
            if not _approx_equal(float(orig), float(fast)):
                diffs.append(f"{path_str}: numeric mismatch {orig} vs {fast}")
        else:
            diffs.append(f"{path_str}: type mismatch {type(orig).__name__} vs {type(fast).__name__}")
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
            diffs.append(f"{path_str}: length mismatch {len(orig)} vs {len(fast)}")
            return diffs
        for i in range(len(orig)):
            diffs.extend(_deep_compare(orig[i], fast[i], f"{path_str}[{i}]"))
        return diffs
    if isinstance(orig, str):
        if orig != fast:
            diffs.append(f"{path_str}: string mismatch")
        return diffs
    if isinstance(orig, (int, bool)):
        if orig != fast:
            diffs.append(f"{path_str}: value mismatch {orig} vs {fast}")
        return diffs
    if hasattr(orig, "model_dump"):
        return _deep_compare(orig.model_dump(), fast.model_dump(), path_str)
    if orig != fast:
        diffs.append(f"{path_str}: value mismatch")
    return diffs


# ---- Per-sample model_dump comparison across all test files ----

@pytest.mark.parametrize("name", [
    "test_10samples",
    "test_100samples",
    "test_1000samples",
    "test_multiepoch",
    "test_error",
    "test_nan_inf",
    "test_attachments",
])
def test_bypass_model_dump_matches_eval(name):
    """Verify construct_sample_fast produces identical model_dump output to model_validate."""
    path = str(TEST_LOGS_DIR / f"{name}.eval")
    if not os.path.exists(path):
        pytest.skip(f"Test file not found: {path}")

    ctx = get_deserializing_context()
    raw1 = read_eval_file(path)
    raw2 = read_eval_file(path)

    samples1 = raw1.get("samples") or []
    samples2 = raw2.get("samples") or []

    assert len(samples1) == len(samples2)
    total_diffs = []
    for i in range(len(samples1)):
        validated = EvalSample.model_validate(samples1[i], context=ctx)
        constructed = construct_sample_fast(samples2[i])

        v_dump = validated.model_dump()
        c_dump = constructed.model_dump()
        diffs = _deep_compare(v_dump, c_dump, f"sample[{i}]")
        total_diffs.extend(diffs)

    if total_diffs:
        msg = f"Found {len(total_diffs)} differences:\n" + "\n".join(f"  - {d}" for d in total_diffs[:20])
        pytest.fail(msg)


# ---- Full pipeline (end-to-end) correctness for .eval format ----

@pytest.mark.parametrize("name", [
    "test_10samples",
    "test_100samples",
    "test_multiepoch",
    "test_error",
    "test_nan_inf",
    "test_attachments",
])
def test_bypass_full_pipeline_eval(name):
    """Full pipeline with bypass should match original Python for .eval files."""
    path = str(TEST_LOGS_DIR / f"{name}.eval")
    if not os.path.exists(path):
        pytest.skip(f"Test file not found: {path}")

    unpatch()
    orig = read_eval_log(path)

    patch()
    fast = read_eval_log(path)
    unpatch()

    orig_dict = orig.model_dump()
    fast_dict = fast.model_dump()
    orig_dict.pop("location", None)
    fast_dict.pop("location", None)

    diffs = _deep_compare(orig_dict, fast_dict)
    if diffs:
        msg = f"Found {len(diffs)} differences:\n" + "\n".join(f"  - {d}" for d in diffs[:20])
        pytest.fail(msg)


# ---- Full pipeline correctness for .json format ----

@pytest.mark.parametrize("name", [
    "test_10samples",
    "test_100samples",
    "test_multiepoch",
    "test_nan_inf",
    "test_attachments",
])
def test_bypass_full_pipeline_json(name):
    """Full pipeline with bypass should match original Python for .json files."""
    path = str(TEST_LOGS_DIR / f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"Test file not found: {path}")

    unpatch()
    orig = read_eval_log(path)

    patch()
    fast = read_eval_log(path)
    unpatch()

    orig_dict = orig.model_dump()
    fast_dict = fast.model_dump()
    orig_dict.pop("location", None)
    fast_dict.pop("location", None)

    diffs = _deep_compare(orig_dict, fast_dict)
    if diffs:
        msg = f"Found {len(diffs)} differences:\n" + "\n".join(f"  - {d}" for d in diffs[:20])
        pytest.fail(msg)


# ---- Type verification for nested objects ----

def test_bypass_nested_types():
    """Verify that nested objects have correct Pydantic model types."""
    path = str(TEST_LOGS_DIR / "test_10samples.eval")
    if not os.path.exists(path):
        pytest.skip("Test file not found")

    raw = read_eval_file(path)
    sample = construct_sample_fast(raw["samples"][0])

    # Top level
    assert isinstance(sample, EvalSample)

    # Messages
    assert len(sample.messages) > 0
    for msg in sample.messages:
        assert isinstance(msg, (ChatMessageSystem, ChatMessageUser,
                                ChatMessageAssistant, ChatMessageTool)), \
            f"Expected ChatMessage type, got {type(msg).__name__}"
        assert msg.role in ("system", "user", "assistant", "tool")

    # Output
    assert isinstance(sample.output, ModelOutput)
    if sample.output.choices:
        for choice in sample.output.choices:
            assert isinstance(choice, ChatCompletionChoice)
    if sample.output.usage is not None:
        assert isinstance(sample.output.usage, ModelUsage)

    # Scores
    if sample.scores:
        for k, v in sample.scores.items():
            assert isinstance(v, Score), f"Expected Score, got {type(v).__name__}"

    # Model usage
    for k, v in sample.model_usage.items():
        assert isinstance(v, ModelUsage)

    # Events
    from inspect_ai.event._base import BaseEvent
    for event in sample.events:
        assert hasattr(event, "event"), f"Event missing 'event' field: {type(event).__name__}"
        assert hasattr(event, "timestamp"), f"Event missing 'timestamp' field"


# ---- NaN/Inf preservation ----

def test_bypass_nan_inf_preserved():
    """NaN/Inf test log should produce identical output through bypass vs original."""
    path = str(TEST_LOGS_DIR / "test_nan_inf.eval")
    if not os.path.exists(path):
        pytest.skip("NaN/Inf test file not found")

    # Compare bypass output with original to verify NaN/Inf handling is correct
    unpatch()
    orig = read_eval_log(path)

    patch()
    fast = read_eval_log(path)
    unpatch()

    assert orig.samples is not None
    assert fast.samples is not None
    assert len(orig.samples) == len(fast.samples)

    orig_dict = orig.model_dump()
    fast_dict = fast.model_dump()
    orig_dict.pop("location", None)
    fast_dict.pop("location", None)
    diffs = _deep_compare(orig_dict, fast_dict)
    assert not diffs, f"NaN/Inf log differences: {diffs[:10]}"


# ---- Scorer placeholder replacement ----

def test_bypass_scorer_placeholder_replaced():
    """The scorer placeholder '88F74D2C' should be replaced with actual scorer name."""
    path = str(TEST_LOGS_DIR / "test_10samples.eval")
    if not os.path.exists(path):
        pytest.skip("Test file not found")

    patch()
    log = read_eval_log(path)
    unpatch()

    assert log.samples is not None
    for sample in log.samples:
        if sample.scores:
            for key in sample.scores:
                assert key != "88F74D2C", \
                    "Scorer placeholder '88F74D2C' should have been replaced"


# ---- Edge case: empty/cancelled logs ----

def test_bypass_empty_log():
    """Empty log (no samples) should work correctly."""
    path = str(TEST_LOGS_DIR / "test_empty.eval")
    if not os.path.exists(path):
        pytest.skip("Empty test file not found")

    patch()
    log = read_eval_log(path)
    unpatch()

    # Empty log should have no samples or empty list
    assert log.samples is None or len(log.samples) == 0


def test_bypass_cancelled_log():
    """Cancelled log should work correctly."""
    path = str(TEST_LOGS_DIR / "test_cancelled.eval")
    if not os.path.exists(path):
        pytest.skip("Cancelled test file not found")

    patch()
    log = read_eval_log(path)
    unpatch()

    assert log.status in ("cancelled", "error")


# ---- Attribute access on nested models ----

def test_bypass_attribute_access():
    """Verify that attribute access works on bypassed nested models."""
    path = str(TEST_LOGS_DIR / "test_10samples.eval")
    if not os.path.exists(path):
        pytest.skip("Test file not found")

    raw = read_eval_file(path)
    sample = construct_sample_fast(raw["samples"][0])

    # Messages — attribute access
    for msg in sample.messages:
        assert isinstance(msg.role, str)
        assert msg.content is not None
        # source should be accessible (default None)
        _ = msg.source
        _ = msg.id
        _ = msg.metadata

    # Output — attribute access
    assert hasattr(sample.output, "completion")
    assert isinstance(sample.output.completion, str)
    if sample.output.choices:
        choice = sample.output.choices[0]
        assert hasattr(choice, "stop_reason")
        assert hasattr(choice.message, "role")

    # Scores — attribute access
    if sample.scores:
        for k, score in sample.scores.items():
            assert hasattr(score, "value")
            assert hasattr(score, "answer")
            assert hasattr(score, "explanation")
            assert hasattr(score, "history")

    # Events — attribute access
    for event in sample.events:
        assert hasattr(event, "event")
        assert hasattr(event, "timestamp")
