"""Correctness tests for the Pydantic bypass construction.

Verifies that construct_sample_fast produces EvalSample instances whose
model_dump() output matches that of EvalSample.model_validate().
"""

import copy
import os

import pytest

from inspect_ai._util.constants import get_deserializing_context
from inspect_ai.log._file import read_eval_log
from inspect_ai.log._log import EvalSample
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
    _construct_message,
    _construct_model_output,
    _construct_score,
    _construct_model_usage,
    _construct_event,
)

from helpers import deep_compare, read_eval_file, TEST_LOG_DIR


@pytest.fixture(autouse=True)
def _ensure_unpatched():
    unpatch()
    yield
    unpatch()


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
    path = str(TEST_LOG_DIR / f"{name}.eval")
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
        diffs = deep_compare(v_dump, c_dump, f"sample[{i}]")
        total_diffs.extend(diffs)

    if total_diffs:
        msg = f"Found {len(total_diffs)} differences:\n" + "\n".join(f"  - {d}" for d in total_diffs[:20])
        pytest.fail(msg)


# ---- Full pipeline (end-to-end) correctness ----

@pytest.mark.parametrize("name", [
    "test_10samples",
    "test_100samples",
    "test_multiepoch",
    "test_error",
    "test_nan_inf",
    "test_attachments",
])
@pytest.mark.parametrize("ext", [".eval", ".json"])
def test_bypass_full_pipeline(name, ext):
    """Full pipeline with bypass should match original Python."""
    path = str(TEST_LOG_DIR / f"{name}{ext}")
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

    diffs = deep_compare(orig_dict, fast_dict)
    if diffs:
        msg = f"Found {len(diffs)} differences:\n" + "\n".join(f"  - {d}" for d in diffs[:20])
        pytest.fail(msg)


# ---- Type verification for nested objects ----

def test_bypass_nested_types():
    path = str(TEST_LOG_DIR / "test_10samples.eval")
    if not os.path.exists(path):
        pytest.skip("Test file not found")

    raw = read_eval_file(path)
    sample = construct_sample_fast(raw["samples"][0])

    assert isinstance(sample, EvalSample)

    # Messages
    assert len(sample.messages) > 0
    for msg in sample.messages:
        assert isinstance(msg, (ChatMessageSystem, ChatMessageUser,
                                ChatMessageAssistant, ChatMessageTool))
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
            assert isinstance(v, Score)

    # Model usage
    for k, v in sample.model_usage.items():
        assert isinstance(v, ModelUsage)

    # Events
    for event in sample.events:
        assert hasattr(event, "event")
        assert hasattr(event, "timestamp")


# ---- NaN/Inf preservation ----

def test_bypass_nan_inf_preserved():
    path = str(TEST_LOG_DIR / "test_nan_inf.eval")
    if not os.path.exists(path):
        pytest.skip("NaN/Inf test file not found")

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
    diffs = deep_compare(orig_dict, fast_dict)
    assert not diffs, f"NaN/Inf log differences: {diffs[:10]}"


# ---- Scorer placeholder replacement ----

def test_bypass_scorer_placeholder_replaced():
    """The scorer placeholder '88F74D2C' should be replaced with actual scorer name."""
    path = str(TEST_LOG_DIR / "test_10samples.eval")
    if not os.path.exists(path):
        pytest.skip("Test file not found")

    patch()
    log = read_eval_log(path)
    unpatch()

    assert log.samples is not None
    for sample in log.samples:
        if sample.scores:
            for key in sample.scores:
                assert key != "88F74D2C"


# ---- Edge case: empty/cancelled logs ----

def test_bypass_empty_log():
    path = str(TEST_LOG_DIR / "test_empty.eval")
    if not os.path.exists(path):
        pytest.skip("Empty test file not found")

    patch()
    log = read_eval_log(path)
    unpatch()

    assert log.samples is None or len(log.samples) == 0


def test_bypass_cancelled_log():
    path = str(TEST_LOG_DIR / "test_cancelled.eval")
    if not os.path.exists(path):
        pytest.skip("Cancelled test file not found")

    patch()
    log = read_eval_log(path)
    unpatch()

    assert log.status in ("cancelled", "error")


# ---- Attribute access on bypassed nested models ----

def test_bypass_attribute_access():
    path = str(TEST_LOG_DIR / "test_10samples.eval")
    if not os.path.exists(path):
        pytest.skip("Test file not found")

    raw = read_eval_file(path)
    sample = construct_sample_fast(raw["samples"][0])

    for msg in sample.messages:
        assert isinstance(msg.role, str)
        assert msg.content is not None
        _ = msg.source
        _ = msg.id
        _ = msg.metadata

    assert hasattr(sample.output, "completion")
    assert isinstance(sample.output.completion, str)
    if sample.output.choices:
        choice = sample.output.choices[0]
        assert hasattr(choice, "stop_reason")
        assert hasattr(choice.message, "role")

    if sample.scores:
        for k, score in sample.scores.items():
            assert hasattr(score, "value")
            assert hasattr(score, "answer")
            assert hasattr(score, "explanation")
            assert hasattr(score, "history")

    for event in sample.events:
        assert hasattr(event, "event")
        assert hasattr(event, "timestamp")


# ---- Deprecated field migrations ----

def test_bypass_deprecated_score_field():
    """Deprecated 'score' field should be migrated to 'scores' dict."""
    sample_data = {
        "id": 1, "epoch": 1,
        "input": "test input", "target": "test target",
        "score": {"value": 1, "answer": "yes", "explanation": None},
        "messages": [], "events": [],
    }
    ctx = get_deserializing_context()

    validated = EvalSample.model_validate(dict(sample_data), context=ctx)
    constructed = construct_sample_fast(dict(sample_data))

    assert validated.scores is not None
    assert constructed.scores is not None
    assert "88F74D2C" in validated.scores
    assert "88F74D2C" in constructed.scores

    diffs = deep_compare(validated.model_dump(), constructed.model_dump())
    assert not diffs, f"Deprecated score migration differences: {diffs[:10]}"


def test_bypass_deprecated_transcript_field():
    """Deprecated 'transcript' field should be migrated to 'events' + 'attachments'."""
    sample_data = {
        "id": 1, "epoch": 1,
        "input": "test input", "target": "test target",
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

    assert len(constructed.events) == 1
    assert constructed.events[0].event == "state"
    assert constructed.attachments == {"key1": "value1"}
    assert "transcript" not in constructed.__dict__


def test_bypass_sandbox_list_migration():
    """Sandbox list format [type, config] should be migrated to SandboxEnvironmentSpec."""
    sample_data = {
        "id": 1, "epoch": 1,
        "input": "test input", "target": "test target",
        "sandbox": ["docker", "config.yaml"],
        "messages": [], "events": [],
    }
    ctx = get_deserializing_context()

    validated = EvalSample.model_validate(dict(sample_data), context=ctx)
    constructed = construct_sample_fast(dict(sample_data))

    assert validated.sandbox is not None
    assert constructed.sandbox is not None
    assert validated.sandbox.type == "docker"
    assert constructed.sandbox.type == "docker"

    diffs = deep_compare(validated.model_dump(), constructed.model_dump())
    assert not diffs, f"Sandbox list migration differences: {diffs[:10]}"


# ---- Isolated unit tests for construction helpers ----

def test_construct_sample_fast_basic():
    raw = read_eval_file(str(TEST_LOG_DIR / "test_10samples.eval"))
    ctx = get_deserializing_context()

    for sd in raw["samples"]:
        orig = EvalSample.model_validate(copy.deepcopy(sd), context=ctx)
        fast = construct_sample_fast(copy.deepcopy(sd))
        diffs = deep_compare(orig.model_dump(), fast.model_dump())
        assert not diffs, f"Differences: {diffs[:10]}"


def test_construct_message_with_content_list():
    msg = _construct_message({
        "role": "user",
        "content": [{"type": "text", "text": "hello world"}],
    })
    assert isinstance(msg.content, list)
    assert len(msg.content) == 1
    assert msg.content[0].text == "hello world"


def test_construct_assistant_with_tool_calls():
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
    """ModelOutput.completion is auto-populated from choices when empty."""
    output = _construct_model_output({
        "model": "test-model",
        "choices": [
            {"message": {"role": "assistant", "content": "hello world"}, "stop_reason": "stop"},
        ],
    })
    assert output.completion == "hello world"


def test_construct_model_output_preserves_completion():
    """Existing completion is NOT overwritten by set_completion logic."""
    output = _construct_model_output({
        "model": "test-model",
        "completion": "existing completion",
        "choices": [
            {"message": {"role": "assistant", "content": "different text"}, "stop_reason": "stop"},
        ],
    })
    assert output.completion == "existing completion"


def test_construct_score_basic():
    score = _construct_score({"value": "C", "answer": "C", "explanation": "correct"})
    assert isinstance(score, Score)
    assert score.value == "C"
    assert score.answer == "C"
    assert score.explanation == "correct"
    assert score.history == []
    assert score.metadata is None


def test_construct_model_usage_basic():
    usage = _construct_model_usage({
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    })
    assert isinstance(usage, ModelUsage)
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150
    assert usage.reasoning_tokens is None


def test_construct_event_from_real_data():
    """Event construction should match model_validate for real events from test data."""
    from pydantic import TypeAdapter
    from inspect_ai.log._log import Event

    raw = read_eval_file(str(TEST_LOG_DIR / "test_10samples.eval"))
    ctx = get_deserializing_context()
    event_adapter = TypeAdapter(Event)

    for sample_data in raw["samples"]:
        for event_data in sample_data.get("events", []):
            orig = event_adapter.validate_python(copy.deepcopy(event_data), context=ctx)
            fast = _construct_event(copy.deepcopy(event_data))
            diffs = deep_compare(orig.model_dump(), fast.model_dump())
            assert not diffs, f"Event '{event_data.get('event')}' differences: {diffs[:5]}"


def test_bypass_sample_field_types():
    raw = read_eval_file(str(TEST_LOG_DIR / "test_10samples.eval"))
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
    raw = read_eval_file(str(TEST_LOG_DIR / "test_10samples.eval"))
    for sd in raw["samples"]:
        d = construct_sample_fast(sd).model_dump()
        assert isinstance(d, dict)
        assert "id" in d
        assert "epoch" in d
        assert "input" in d
        assert "messages" in d
        assert "events" in d


def test_construct_message_all_roles():
    for role in ["system", "user", "assistant", "tool"]:
        msg = _construct_message({"role": role, "content": f"hello from {role}"})
        assert msg.role == role
        assert msg.content == f"hello from {role}"


def test_construct_event_with_completed_timestamp():
    """Events with 'completed' field should correctly convert to datetime."""
    from datetime import datetime

    for event_type in ["tool", "model", "sandbox", "subtask"]:
        data = {
            "event": event_type,
            "timestamp": "2024-01-01T00:00:00+00:00",
            "completed": "2024-01-01T00:00:05+00:00",
            "working_start": 0.0,
        }
        if event_type == "tool":
            data["function"] = "test_fn"
            data["arguments"] = {}
        elif event_type == "model":
            data["model"] = "test-model"
            data["input"] = []
            data["output"] = {"model": "m", "choices": []}

        event = _construct_event(data)
        assert isinstance(event.timestamp, datetime)
        assert isinstance(event.completed, datetime)
        d = event.model_dump()
        assert d["completed"] is not None


def test_construct_tool_event_with_error_and_view():
    from inspect_ai.tool._tool_call import ToolCallContent, ToolCallError

    data = {
        "event": "tool",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "function": "test_fn",
        "arguments": {},
        "result": "output",
        "error": {"type": "timeout", "message": "Tool timed out"},
        "view": {"title": "Output", "format": "text", "content": "tool output"},
        "working_start": 0.0,
    }

    event = _construct_event(data)
    assert isinstance(event.error, ToolCallError)
    assert event.error.type == "timeout"
    assert isinstance(event.view, ToolCallContent)
    assert event.view.title == "Output"
    d = event.model_dump()
    assert d["error"]["type"] == "timeout"
    assert d["view"]["title"] == "Output"


def test_construct_approval_event():
    from inspect_ai.tool._tool_call import ToolCall, ToolCallView

    data = {
        "event": "approval",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "call": {"id": "tc1", "function": "bash", "arguments": {"cmd": "ls"}, "type": "function"},
        "view": {
            "context": {"title": "Ctx", "format": "text", "content": "context"},
            "call": {"title": "Call", "format": "markdown", "content": "call info"},
        },
        "modified": {"id": "tc2", "function": "bash", "arguments": {"cmd": "pwd"}, "type": "function"},
        "working_start": 0.0,
    }

    event = _construct_event(data)
    assert isinstance(event.call, ToolCall)
    assert event.call.function == "bash"
    assert isinstance(event.view, ToolCallView)
    assert isinstance(event.modified, ToolCall)
    d = event.model_dump()
    assert d["call"]["function"] == "bash"
    assert d["view"]["context"]["title"] == "Ctx"


def test_construct_error_event():
    from inspect_ai._util.error import EvalError

    data = {
        "event": "error",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "error": {"message": "something failed", "traceback": "tb", "traceback_ansi": "tb_ansi"},
        "working_start": 0.0,
    }
    event = _construct_event(data)
    assert isinstance(event.error, EvalError)
    assert event.error.message == "something failed"
    d = event.model_dump()
    assert d["error"]["message"] == "something failed"


def test_construct_logger_event_with_level_migration():
    """Deprecated log level 'tools' should be migrated to 'trace'."""
    data = {
        "event": "logger",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "message": {
            "level": "tools",
            "message": "log message",
            "created": 1000.0,
            "filename": "test.py",
            "module": "test",
            "lineno": 42,
        },
        "working_start": 0.0,
    }
    event = _construct_event(data)
    assert event.message.level == "trace"
    d = event.model_dump()
    assert d["message"]["level"] == "trace"


def test_construct_score_edit_event():
    data = {
        "event": "score_edit",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "score_name": "accuracy",
        "edit": {"value": 1, "answer": "yes", "explanation": "correct", "metadata": "UNCHANGED"},
        "working_start": 0.0,
    }
    event = _construct_event(data)
    from inspect_ai.scorer._metric import ScoreEdit
    assert isinstance(event.edit, ScoreEdit)
    assert event.edit.value == 1
    d = event.model_dump()
    assert d["edit"]["value"] == 1


def test_construct_score_event_with_usage():
    data = {
        "event": "score",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "score": {"value": 1},
        "model_usage": {"gpt-4": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}},
        "role_usage": {"solver": {"input_tokens": 80, "output_tokens": 40, "total_tokens": 120}},
        "working_start": 0.0,
    }
    event = _construct_event(data)
    assert isinstance(event.model_usage["gpt-4"], ModelUsage)
    assert isinstance(event.role_usage["solver"], ModelUsage)
    d = event.model_dump()
    assert d["model_usage"]["gpt-4"]["input_tokens"] == 100


def test_construct_model_event_with_tools_and_call():
    from inspect_ai.tool._tool_info import ToolInfo
    from inspect_ai.model._model_call import ModelCall
    from inspect_ai.tool._tool_choice import ToolFunction

    data = {
        "event": "model",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "model": "test",
        "input": [],
        "output": {"model": "m", "choices": []},
        "tools": [{"name": "bash", "description": "run bash", "parameters": {"type": "object", "properties": {}}}],
        "call": {"request": {"model": "test"}, "response": None},
        "tool_choice": {"name": "bash"},
        "working_start": 0.0,
    }
    event = _construct_event(data)
    assert isinstance(event.tools[0], ToolInfo)
    assert isinstance(event.call, ModelCall)
    assert isinstance(event.tool_choice, ToolFunction)
    d = event.model_dump()
    assert d["tools"][0]["name"] == "bash"


def test_construct_subtask_event_input_migration():
    """Non-dict SubtaskEvent input should be migrated to empty dict."""
    data = {
        "event": "subtask",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "name": "my_subtask",
        "input": ["not", "a", "dict"],
        "working_start": 0.0,
    }
    event = _construct_event(data)
    assert event.input == {}
    d = event.model_dump()
    assert d["input"] == {}


def test_construct_tool_message_with_deprecated_tool_error():
    """ChatMessageTool with deprecated 'tool_error' should migrate to 'error'."""
    from inspect_ai.tool._tool_call import ToolCallError

    msg = _construct_message({
        "role": "tool",
        "content": "tool output",
        "tool_error": "something went wrong",
    })
    assert msg.role == "tool"
    assert isinstance(msg.error, ToolCallError)
    assert msg.error.type == "unknown"
    assert msg.error.message == "something went wrong"
    d = msg.model_dump()
    assert d["error"]["type"] == "unknown"


def test_construct_tool_call_with_none_type():
    """ToolCall with type=None should be migrated to 'function'."""
    from inspect_fast_loader._construct import _construct_tool_call

    tc = _construct_tool_call({"id": "tc1", "function": "bash", "arguments": {}, "type": None})
    assert tc.type == "function"

    tc2 = _construct_tool_call({"id": "tc2", "function": "bash", "arguments": {}})
    assert tc2.type == "function"

    tc3 = _construct_tool_call({"id": "tc3", "function": "bash", "arguments": {}, "type": "function"})
    assert tc3.type == "function"
