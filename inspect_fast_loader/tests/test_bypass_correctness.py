"""Comprehensive correctness tests for the Pydantic bypass construction.

These tests verify that construct_sample_fast produces EvalSample instances
whose model_dump() output matches that of EvalSample.model_validate().

Includes both end-to-end pipeline tests and isolated unit tests for individual
construction helpers (_construct_message, _construct_model_output, etc.).
"""

import copy
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
from inspect_fast_loader._construct import (
    construct_sample_fast,
    _fast_construct,
    _construct_message,
    _construct_model_output,
    _construct_score,
    _construct_model_usage,
    _construct_event,
)
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


# ---- Deprecated field migrations ----

def test_bypass_deprecated_score_field():
    """Test that deprecated 'score' field is migrated to 'scores' dict."""
    sample_data = {
        "id": 1,
        "epoch": 1,
        "input": "test input",
        "target": "test target",
        "score": {"value": 1, "answer": "yes", "explanation": None},
        "messages": [],
        "events": [],
    }
    ctx = get_deserializing_context()

    # model_validate handles the migration
    validated = EvalSample.model_validate(dict(sample_data), context=ctx)
    # construct_sample_fast should also handle it
    constructed = construct_sample_fast(dict(sample_data))

    # Both should have "88F74D2C" key (scorer placeholder)
    assert validated.scores is not None
    assert constructed.scores is not None
    assert "88F74D2C" in validated.scores
    assert "88F74D2C" in constructed.scores

    # Values should match
    v_dump = validated.model_dump()
    c_dump = constructed.model_dump()
    diffs = _deep_compare(v_dump, c_dump)
    assert not diffs, f"Deprecated score migration differences: {diffs[:10]}"


def test_bypass_deprecated_transcript_field():
    """Test that deprecated 'transcript' field is migrated to 'events' + 'attachments'."""
    sample_data = {
        "id": 1,
        "epoch": 1,
        "input": "test input",
        "target": "test target",
        "transcript": {
            "events": [
                {"event": "state", "timestamp": "2024-01-01T00:00:00+00:00",
                 "working_start": 0.0, "changes": []},
            ],
            "content": {"key1": "value1"},
        },
        "messages": [],
    }

    constructed = construct_sample_fast(dict(sample_data))

    # Transcript should be migrated to events and attachments
    assert len(constructed.events) == 1
    assert constructed.events[0].event == "state"
    assert constructed.attachments == {"key1": "value1"}
    # The 'transcript' key should not be in __dict__ (migrated away)
    assert "transcript" not in constructed.__dict__

    # Note: we don't compare model_dump with model_validate here because
    # the original migrate_deprecated constructs EvalEvents without context,
    # which generates random UUIDs for events — making comparison non-deterministic.


def test_bypass_sandbox_list_migration():
    """Test that sandbox list format is migrated to SandboxEnvironmentSpec."""
    sample_data = {
        "id": 1,
        "epoch": 1,
        "input": "test input",
        "target": "test target",
        "sandbox": ["docker", "config.yaml"],
        "messages": [],
        "events": [],
    }
    ctx = get_deserializing_context()

    validated = EvalSample.model_validate(dict(sample_data), context=ctx)
    constructed = construct_sample_fast(dict(sample_data))

    assert validated.sandbox is not None
    assert constructed.sandbox is not None
    assert validated.sandbox.type == "docker"
    assert constructed.sandbox.type == "docker"

    v_dump = validated.model_dump()
    c_dump = constructed.model_dump()
    diffs = _deep_compare(v_dump, c_dump)
    assert not diffs, f"Sandbox list migration differences: {diffs[:10]}"


# ---- Isolated unit tests for construction helpers ----
# (Inspired by Branch 1's test suite for additional coverage)


def test_construct_sample_fast_basic():
    """Isolated unit test: construct_sample_fast on every sample from 10-sample file."""
    raw = read_eval_file(str(TEST_LOGS_DIR / "test_10samples.eval"))
    ctx = get_deserializing_context()

    for sd in raw["samples"]:
        orig = EvalSample.model_validate(copy.deepcopy(sd), context=ctx)
        fast = construct_sample_fast(copy.deepcopy(sd))
        diffs = _deep_compare(orig.model_dump(), fast.model_dump())
        assert not diffs, f"Differences: {diffs[:10]}"


def test_construct_message_with_content_list():
    """Test message construction with content as a list of ContentText."""
    msg = _construct_message({
        "role": "user",
        "content": [{"type": "text", "text": "hello world"}],
    })
    assert isinstance(msg.content, list)
    assert len(msg.content) == 1
    assert hasattr(msg.content[0], "text")
    assert msg.content[0].text == "hello world"


def test_construct_assistant_with_tool_calls():
    """Test assistant message with tool_calls constructs ToolCall objects."""
    msg = _construct_message({
        "role": "assistant",
        "content": "I'll use a tool",
        "tool_calls": [
            {"id": "tc1", "function": "search", "arguments": {"q": "test"}},
        ],
    })
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "tc1"
    assert msg.tool_calls[0].function == "search"
    assert msg.tool_calls[0].arguments == {"q": "test"}


def test_construct_model_output_sets_completion():
    """Test that ModelOutput.completion is auto-populated from choices when empty."""
    output = _construct_model_output({
        "model": "test-model",
        "choices": [
            {"message": {"role": "assistant", "content": "hello world"}, "stop_reason": "stop"},
        ],
    })
    assert output.completion == "hello world"


def test_construct_model_output_preserves_completion():
    """Test that an existing completion is NOT overwritten by set_completion logic."""
    output = _construct_model_output({
        "model": "test-model",
        "completion": "existing completion",
        "choices": [
            {"message": {"role": "assistant", "content": "different text"}, "stop_reason": "stop"},
        ],
    })
    assert output.completion == "existing completion"


def test_construct_score_basic():
    """Test isolated Score construction with all fields."""
    score = _construct_score({"value": "C", "answer": "C", "explanation": "correct"})
    assert isinstance(score, Score)
    assert score.value == "C"
    assert score.answer == "C"
    assert score.explanation == "correct"
    assert score.history == []  # default
    assert score.metadata is None  # default


def test_construct_model_usage_basic():
    """Test isolated ModelUsage construction."""
    usage = _construct_model_usage({
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    })
    assert isinstance(usage, ModelUsage)
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150
    assert usage.reasoning_tokens is None  # default


def test_construct_event_from_real_data():
    """Test Event construction matches model_validate for real events from test data."""
    from pydantic import TypeAdapter
    from inspect_ai.log._log import Event

    raw = read_eval_file(str(TEST_LOGS_DIR / "test_10samples.eval"))
    ctx = get_deserializing_context()
    event_adapter = TypeAdapter(Event)

    for sample_data in raw["samples"]:
        for event_data in sample_data.get("events", []):
            orig = event_adapter.validate_python(copy.deepcopy(event_data), context=ctx)
            fast = _construct_event(copy.deepcopy(event_data))
            diffs = _deep_compare(orig.model_dump(), fast.model_dump())
            assert not diffs, f"Event '{event_data.get('event')}' differences: {diffs[:5]}"


def test_bypass_sample_field_types():
    """Comprehensive check that all EvalSample fields have correct Python types."""
    raw = read_eval_file(str(TEST_LOGS_DIR / "test_10samples.eval"))
    s = construct_sample_fast(raw["samples"][0])

    assert isinstance(s.id, (int, str))
    assert isinstance(s.epoch, int)
    assert isinstance(s.input, (str, list))
    assert isinstance(s.target, (str, list))
    assert isinstance(s.messages, list)
    assert isinstance(s.events, list)
    assert isinstance(s.metadata, dict)
    assert isinstance(s.store, dict)
    assert isinstance(s.model_usage, dict)
    assert isinstance(s.attachments, dict)
    if s.scores is not None:
        assert isinstance(s.scores, dict)
    if s.output is not None:
        assert isinstance(s.output, ModelOutput)


def test_bypass_model_dump_roundtrip():
    """Test that bypass objects can be serialized back with model_dump for every sample."""
    raw = read_eval_file(str(TEST_LOGS_DIR / "test_10samples.eval"))

    for sd in raw["samples"]:
        s = construct_sample_fast(sd)
        d = s.model_dump()
        assert isinstance(d, dict)
        assert "id" in d
        assert "epoch" in d
        assert "input" in d
        assert "messages" in d
        assert "events" in d


def test_construct_message_all_roles():
    """Test that all four message role types are correctly constructed."""
    for role in ["system", "user", "assistant", "tool"]:
        msg = _construct_message({"role": role, "content": f"hello from {role}"})
        assert msg.role == role
        assert msg.content == f"hello from {role}"
