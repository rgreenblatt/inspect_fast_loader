"""Fast EvalSample construction bypassing Pydantic model_validate.

Constructs EvalSample and nested Pydantic model instances without running
validators, achieving ~10-20x speedup over model_validate while maintaining
correct model_dump() output.

The key techniques:
1. Use _fast_construct() which sets __dict__ directly, skipping model_post_init
   (critical for ChatMessage types which generate UUIDs in model_post_init)
2. Apply the same migrations that model_validate would apply (migrate_deprecated, etc.)
3. Construct all nested Pydantic model instances so model_dump() produces correct output
4. Convert timestamp strings to datetime objects for events (required by serializer)

FRAGILITY WARNING:
    This module hard-codes knowledge of inspect_ai's Pydantic model types, their
    fields, defaults, validators, and migrations. It WILL break if inspect_ai
    adds/removes/renames fields, adds new model_validators, changes migration
    logic, or adds new event/content/message types.

    The correctness test suite (test_bypass_correctness.py) is the primary safety
    net — it compares model_dump() output of bypassed construction vs standard
    model_validate() for all test logs. Run these tests after any inspect_ai
    version upgrade.

    Tested against inspect_ai version: 0.3.188.

    To update after an inspect_ai upgrade:
    1. Run the test suite — failing tests reveal what changed.
    2. Check new model_validators on EvalSample, ModelOutput, ChatCompletionChoice.
    3. Check for new Event types (add to _EVENT_CLS dict).
    4. Check for new Content types (add to _CONTENT_CLS dict).
    5. Check for new/changed field defaults on EvalSample and nested models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from inspect_ai._util.error import EvalError
from inspect_ai._util.json import JsonChange
from inspect_ai.dataset._dataset import Sample
from inspect_ai.event._approval import ApprovalEvent
from inspect_ai.event._compaction import CompactionEvent
from inspect_ai.event._error import ErrorEvent
from inspect_ai.event._info import InfoEvent
from inspect_ai.event._input import InputEvent
from inspect_ai.event._logger import LoggerEvent, LoggingMessage
from inspect_ai.event._model import ModelEvent
from inspect_ai.event._sample_init import SampleInitEvent
from inspect_ai.event._sample_limit import SampleLimitEvent
from inspect_ai.event._sandbox import SandboxEvent
from inspect_ai.event._score import ScoreEvent
from inspect_ai.event._score_edit import ScoreEditEvent
from inspect_ai.event._span import SpanBeginEvent, SpanEndEvent
from inspect_ai.event._state import StateEvent
from inspect_ai.event._step import StepEvent
from inspect_ai.event._store import StoreEvent
from inspect_ai.event._subtask import SubtaskEvent
from inspect_ai.event._tool import ToolEvent
from inspect_ai.log._edit import ProvenanceData
from inspect_ai.log._log import EvalSample, EvalSampleLimit
from inspect_ai.model._chat_message import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model_call import ModelCall
from inspect_ai.model._model_output import ChatCompletionChoice, ModelOutput, ModelUsage
from inspect_ai.scorer._metric import Score, ScoreEdit
from inspect_ai.tool._tool_call import ToolCall, ToolCallContent, ToolCallError, ToolCallView
from inspect_ai.tool._tool_choice import ToolFunction
from inspect_ai.tool._tool_info import ToolInfo
from inspect_ai.tool._tool_params import ToolParams

from inspect_ai._util.content import (
    ContentAudio,
    ContentData,
    ContentDocument,
    ContentImage,
    ContentReasoning,
    ContentText,
    ContentToolUse,
    ContentVideo,
)

_ROLE_CLS: dict[str, type[BaseModel]] = {
    "system": ChatMessageSystem,
    "user": ChatMessageUser,
    "assistant": ChatMessageAssistant,
    "tool": ChatMessageTool,
}

_CONTENT_CLS: dict[str, type[BaseModel]] = {
    "text": ContentText,
    "reasoning": ContentReasoning,
    "image": ContentImage,
    "audio": ContentAudio,
    "video": ContentVideo,
    "data": ContentData,
    "tool_use": ContentToolUse,
    "document": ContentDocument,
}

_EVENT_CLS: dict[str, type[BaseModel]] = {
    "sample_init": SampleInitEvent,
    "sample_limit": SampleLimitEvent,
    "sandbox": SandboxEvent,
    "state": StateEvent,
    "store": StoreEvent,
    "model": ModelEvent,
    "tool": ToolEvent,
    "approval": ApprovalEvent,
    "compaction": CompactionEvent,
    "input": InputEvent,
    "score": ScoreEvent,
    "score_edit": ScoreEditEvent,
    "error": ErrorEvent,
    "logger": LoggerEvent,
    "info": InfoEvent,
    "span_begin": SpanBeginEvent,
    "span_end": SpanEndEvent,
    "step": StepEvent,
    "subtask": SubtaskEvent,
}

# Pre-computed per-class construction info, split for fast lookup.
# _static_defaults: dict of {field_name: default_value} for fields with non-callable defaults
# _factory_fields: list of (field_name, factory_fn) for fields with default_factory
# _alias_map: dict of {alias: field_name} for aliased fields
_CLS_CACHE: dict[type, tuple[dict[str, Any], list[tuple[str, Any]], dict[str, str]]] = {}


def _get_cls_info(cls: type) -> tuple[dict[str, Any], list[tuple[str, Any]], dict[str, str]]:
    """Get cached construction info for a Pydantic model class.

    Returns (static_defaults, factory_fields, alias_map) where:
    - static_defaults: {field_name: default} for non-required fields with static defaults
    - factory_fields: [(field_name, factory)] for non-required fields with default_factory
    - alias_map: {alias: field_name} for aliased fields
    """
    cached = _CLS_CACHE.get(cls)
    if cached is not None:
        return cached

    static_defaults: dict[str, Any] = {}
    factory_fields: list[tuple[str, Any]] = []
    alias_map: dict[str, str] = {}

    for name, field in cls.model_fields.items():
        if not field.is_required():
            if field.default_factory is not None:
                factory_fields.append((name, field.default_factory))
            else:
                static_defaults[name] = field.default
        if field.alias and field.alias != name:
            alias_map[field.alias] = name
        if field.validation_alias and isinstance(field.validation_alias, str) and field.validation_alias != name:
            alias_map[field.validation_alias] = name

    result = (static_defaults, factory_fields, alias_map)
    _CLS_CACHE[cls] = result
    return result


def _fast_construct(cls: type, data: dict) -> Any:
    """Construct a Pydantic model instance without calling validators or model_post_init.

    This sets __dict__ directly, bypassing:
    - model_validators (mode="before", "after", "wrap")
    - field validators
    - model_post_init (e.g. ChatMessage UUID generation)

    Fields not present in data get their defaults.
    Handles field aliases (e.g. JSON "from" -> Python "from_").

    WARNING: Mutates and consumes ``data`` — the dict is used directly as the
    object's ``__dict__`` and must not be reused by the caller. This is safe
    because each dict from the Rust JSON parser is used exactly once.
    """
    static_defaults, factory_fields, alias_map = _get_cls_info(cls)

    # Remap aliased keys in data
    if alias_map:
        for alias, field_name in alias_map.items():
            if alias in data and field_name not in data:
                data[field_name] = data.pop(alias)

    # Fill missing static defaults in one pass
    for name, default in static_defaults.items():
        if name not in data:
            data[name] = default

    # Fill missing factory defaults
    for name, factory in factory_fields:
        if name not in data:
            data[name] = factory()

    obj = cls.__new__(cls)
    object.__setattr__(obj, "__dict__", data)
    object.__setattr__(obj, "__pydantic_fields_set__", set())
    object.__setattr__(obj, "__pydantic_extra__", None)
    object.__setattr__(obj, "__pydantic_private__", None)
    return obj


def _construct_content_item(item: dict) -> Any:
    """Construct a Content item (ContentText, ContentImage, etc.) from a dict."""
    content_type = item.get("type", "text")
    cls = _CONTENT_CLS.get(content_type)
    if cls is not None:
        return _fast_construct(cls, item)
    return item


def _construct_content(content: Any) -> Any:
    """Handle message content — either a string or list of content dicts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [
            _construct_content_item(item) if isinstance(item, dict) else item
            for item in content
        ]
    return content


def _construct_message(data: dict) -> Any:
    """Construct a ChatMessage instance from a dict."""
    role = data.get("role", "user")
    cls = _ROLE_CLS.get(role)
    if cls is None:
        return data

    # Replicate ChatMessageTool.convert_tool_error_to_error (model_validator mode="before")
    # Migrates deprecated "tool_error" field to "error" as ToolCallError
    if role == "tool" and "tool_error" in data:
        tool_error = data.pop("tool_error")
        if tool_error:
            data["error"] = ToolCallError(type="unknown", message=tool_error)

    # Process content if it's a list of content objects
    if isinstance(data.get("content"), list):
        data["content"] = _construct_content(data["content"])

    # Construct tool_calls (pydantic_dataclass, not BaseModel)
    if "tool_calls" in data and data["tool_calls"] is not None:
        data["tool_calls"] = [
            _construct_tool_call(tc) if isinstance(tc, dict) else tc
            for tc in data["tool_calls"]
        ]

    return _fast_construct(cls, data)


def _construct_model_usage(data: dict) -> ModelUsage:
    return _fast_construct(ModelUsage, data)


def _construct_score(data: dict) -> Score:
    return _fast_construct(Score, data)


def _construct_choice(data: dict) -> ChatCompletionChoice:
    # Migrate stop_reason: "length" -> "max_tokens"
    if data.get("stop_reason") == "length":
        data["stop_reason"] = "max_tokens"

    # Construct the message inside the choice
    if "message" in data and isinstance(data["message"], dict):
        data["message"] = _construct_message(data["message"])

    return _fast_construct(ChatCompletionChoice, data)


def _construct_model_output(data: dict) -> ModelOutput:
    # Construct choices
    if "choices" in data:
        data["choices"] = [
            _construct_choice(c) if isinstance(c, dict) else c
            for c in data["choices"]
        ]

    # Construct usage
    if "usage" in data and isinstance(data["usage"], dict):
        data["usage"] = _construct_model_usage(data["usage"])

    output = _fast_construct(ModelOutput, data)

    # Replicate ModelOutput.set_completion validator (mode="after")
    if not output.completion and output.choices:
        output.completion = output.choices[0].message.text if output.choices else ""

    return output


def _construct_json_change(data: dict) -> JsonChange:
    return _fast_construct(JsonChange, data)


def _construct_tool_call(data: dict) -> ToolCall:
    """Construct a ToolCall (pydantic_dataclass) bypassing Pydantic validation.

    ToolCall is a @pydantic_dataclass, so its __init__ runs validation.
    Direct attribute assignment is ~5x faster.
    """
    tc = object.__new__(ToolCall)
    tc.id = data.get("id", "")
    tc.function = data.get("function", "")
    tc.arguments = data.get("arguments", {})
    tc.parse_error = data.get("parse_error")
    tc.view = data.get("view")
    # Replicate ToolCall.migrate_type (field_validator): None → "function"
    tc.type = data.get("type") or "function"
    return tc


def _construct_tool_call_error(data: dict) -> ToolCallError:
    """Construct a ToolCallError (plain dataclass) from a dict."""
    return ToolCallError(type=data["type"], message=data["message"])


def _construct_tool_call_content(data: dict) -> ToolCallContent:
    """Construct a ToolCallContent (BaseModel) from a dict."""
    return _fast_construct(ToolCallContent, data)


def _construct_tool_call_view(data: dict) -> ToolCallView:
    """Construct a ToolCallView (BaseModel) from a dict."""
    if "context" in data and isinstance(data["context"], dict):
        data["context"] = _construct_tool_call_content(data["context"])
    if "call" in data and isinstance(data["call"], dict):
        data["call"] = _construct_tool_call_content(data["call"])
    return _fast_construct(ToolCallView, data)


def _construct_logging_message(data: dict) -> Any:
    """Construct a LoggingMessage with the convert_log_levels migration."""
    # Replicate LoggingMessage.convert_log_levels (model_validator mode="before")
    level = data.get("level")
    if level in ("tools", "sandbox"):
        data["level"] = "trace"
    return _fast_construct(LoggingMessage, data)


def _construct_tool_info(data: dict) -> ToolInfo:
    """Construct a ToolInfo (BaseModel) from a dict."""
    if "parameters" in data and isinstance(data["parameters"], dict):
        data["parameters"] = _fast_construct(ToolParams, data["parameters"])
    return _fast_construct(ToolInfo, data)


def _construct_model_call(data: dict) -> ModelCall:
    return _fast_construct(ModelCall, data)


def _construct_score_edit(data: dict) -> ScoreEdit:
    """Construct a ScoreEdit (BaseModel) from a dict."""
    if "provenance" in data and isinstance(data["provenance"], dict):
        data["provenance"] = _fast_construct(ProvenanceData, data["provenance"])
    return _fast_construct(ScoreEdit, data)


def _parse_timestamp(ts: Any) -> Any:
    """Convert timestamp string to datetime if needed.

    Event timestamps are UtcDatetime (pydantic AwareDatetime) which requires
    a datetime object. The serializer also expects datetime (not str).
    """
    if isinstance(ts, str):
        return datetime.fromisoformat(ts)
    return ts


def _construct_event(data: dict) -> Any:
    """Construct an Event from a dict.

    Events are a discriminated union of 19 types. We use the 'event' field
    to determine the correct type and construct it.
    """
    event_type = data.get("event")
    if event_type is None:
        return data

    cls = _EVENT_CLS.get(event_type)
    if cls is None:
        return data

    # Convert timestamp strings to datetime (required by serializer)
    if "timestamp" in data:
        data["timestamp"] = _parse_timestamp(data["timestamp"])
    if "completed" in data and data["completed"] is not None:
        data["completed"] = _parse_timestamp(data["completed"])

    # Construct nested objects within specific event types
    if event_type in ("state", "store"):
        if "changes" in data and isinstance(data["changes"], list):
            data["changes"] = [
                _construct_json_change(c) if isinstance(c, dict) else c
                for c in data["changes"]
            ]
    elif event_type == "model":
        if "input" in data and isinstance(data["input"], list):
            data["input"] = [
                _construct_message(m) if isinstance(m, dict) else m
                for m in data["input"]
            ]
        if "output" in data and isinstance(data["output"], dict):
            data["output"] = _construct_model_output(data["output"])
        if "config" in data and isinstance(data["config"], dict):
            # Replicate GenerateConfig.migrate_reasoning (model_validator mode="before")
            rh = data["config"].get("reasoning_history")
            if rh is True:
                data["config"]["reasoning_history"] = "all"
            elif rh is False:
                data["config"]["reasoning_history"] = "none"
            data["config"] = _fast_construct(GenerateConfig, data["config"])
        if "tools" in data and isinstance(data["tools"], list):
            data["tools"] = [
                _construct_tool_info(t) if isinstance(t, dict) else t
                for t in data["tools"]
            ]
        if "call" in data and isinstance(data["call"], dict):
            data["call"] = _construct_model_call(data["call"])
        # tool_choice can be a string ("auto"/"any"/"none") or a ToolFunction dict
        if isinstance(data.get("tool_choice"), dict):
            data["tool_choice"] = ToolFunction(name=data["tool_choice"]["name"])
    elif event_type == "tool":
        if "error" in data and isinstance(data["error"], dict):
            data["error"] = _construct_tool_call_error(data["error"])
        if "view" in data and isinstance(data["view"], dict):
            data["view"] = _construct_tool_call_content(data["view"])
        # result can be a content type dict or list of content type dicts
        result = data.get("result")
        if isinstance(result, dict) and "type" in result:
            data["result"] = _construct_content_item(result)
        elif isinstance(result, list):
            data["result"] = [
                _construct_content_item(item) if isinstance(item, dict) else item
                for item in result
            ]
    elif event_type == "approval":
        if "call" in data and isinstance(data["call"], dict):
            data["call"] = _construct_tool_call(data["call"])
        if "modified" in data and isinstance(data["modified"], dict):
            data["modified"] = _construct_tool_call(data["modified"])
        if "view" in data and isinstance(data["view"], dict):
            data["view"] = _construct_tool_call_view(data["view"])
    elif event_type == "score":
        if "score" in data and isinstance(data["score"], dict):
            data["score"] = _construct_score(data["score"])
        if data.get("model_usage"):
            data["model_usage"] = {
                k: _construct_model_usage(v) if isinstance(v, dict) else v
                for k, v in data["model_usage"].items()
            }
        if data.get("role_usage"):
            data["role_usage"] = {
                k: _construct_model_usage(v) if isinstance(v, dict) else v
                for k, v in data["role_usage"].items()
            }
    elif event_type == "score_edit":
        if "edit" in data and isinstance(data["edit"], dict):
            data["edit"] = _construct_score_edit(data["edit"])
    elif event_type == "error":
        if "error" in data and isinstance(data["error"], dict):
            data["error"] = _construct_eval_error(data["error"])
    elif event_type == "logger":
        if "message" in data and isinstance(data["message"], dict):
            data["message"] = _construct_logging_message(data["message"])
    elif event_type == "sample_init":
        if "sample" in data and isinstance(data["sample"], dict):
            sample_data = data["sample"]
            # Sample.input can be a list of ChatMessage dicts
            if isinstance(sample_data.get("input"), list):
                sample_data["input"] = [
                    _construct_message(m) if isinstance(m, dict) else m
                    for m in sample_data["input"]
                ]
            data["sample"] = _fast_construct(Sample, sample_data)
    elif event_type == "subtask":
        # Replicate SubtaskEvent.validate_input (field_validator mode="before"):
        # converts non-dict input to {} for backward compatibility
        if "input" in data and not isinstance(data["input"], dict):
            data["input"] = {}

    return _fast_construct(cls, data)


def _construct_eval_error(data: dict) -> EvalError:
    return _fast_construct(EvalError, data)


def _construct_eval_sample_limit(data: dict) -> EvalSampleLimit:
    return _fast_construct(EvalSampleLimit, data)


def construct_sample_fast(data: dict) -> EvalSample:
    """Construct an EvalSample bypassing Pydantic model_validate.

    Replicates the data transformations from:
    - EvalSample.migrate_deprecated (score -> scores, transcript -> events+attachments)
    - migrate_values (sandbox list -> spec)
    - ModelOutput.set_completion (auto-populate completion from choices)
    - ChatCompletionChoice.migrate_stop_reason ("length" -> "max_tokens")
    - ToolCall.migrate_type (None -> "function")

    Constructs all nested Pydantic model instances using _fast_construct
    which skips validators and model_post_init.

    WARNING: Mutates and consumes ``data`` — the dict (and all nested dicts)
    become the ``__dict__`` of the constructed Pydantic objects. Callers must
    not reuse the dict after calling this function.
    """
    # === Apply EvalSample.migrate_deprecated ===

    # 1. Convert old "score" field to "scores" dict with placeholder key
    if "score" in data:
        data["scores"] = {"88F74D2C": data.pop("score")}

    # 2. Convert old "transcript" to "events" + "attachments"
    if "transcript" in data:
        transcript = data.pop("transcript")
        data["events"] = transcript.get("events", [])
        data["attachments"] = transcript.get("content", {})

    # 3. Apply migrate_values (sandbox spec migration)
    sandbox = data.get("sandbox")
    if isinstance(sandbox, list):
        from inspect_ai.util._sandbox.environment import SandboxEnvironmentSpec
        data["sandbox"] = SandboxEnvironmentSpec(type=sandbox[0], config=sandbox[1])
    elif isinstance(sandbox, dict):
        from inspect_ai.util._sandbox.environment import SandboxEnvironmentSpec
        data["sandbox"] = _fast_construct(SandboxEnvironmentSpec, sandbox)

    # === Construct nested Pydantic model instances ===

    # Messages
    if "messages" in data:
        data["messages"] = [
            _construct_message(m) if isinstance(m, dict) else m
            for m in data["messages"]
        ]

    # Input (when it's a list of messages)
    if isinstance(data.get("input"), list):
        data["input"] = [
            _construct_message(m) if isinstance(m, dict) else m
            for m in data["input"]
        ]

    # Output
    if isinstance(data.get("output"), dict):
        data["output"] = _construct_model_output(data["output"])

    # Scores
    scores = data.get("scores")
    if scores:
        data["scores"] = {
            k: _construct_score(v) if isinstance(v, dict) else v
            for k, v in scores.items()
        }

    # Events
    if "events" in data:
        data["events"] = [
            _construct_event(e) if isinstance(e, dict) else e
            for e in data["events"]
        ]

    # Model usage
    if data.get("model_usage"):
        data["model_usage"] = {
            k: _construct_model_usage(v) if isinstance(v, dict) else v
            for k, v in data["model_usage"].items()
        }

    # Role usage
    if data.get("role_usage"):
        data["role_usage"] = {
            k: _construct_model_usage(v) if isinstance(v, dict) else v
            for k, v in data["role_usage"].items()
        }

    # Error
    if isinstance(data.get("error"), dict):
        data["error"] = _construct_eval_error(data["error"])

    # Error retries
    if data.get("error_retries"):
        data["error_retries"] = [
            _construct_eval_error(e) if isinstance(e, dict) else e
            for e in data["error_retries"]
        ]

    # Limit
    if isinstance(data.get("limit"), dict):
        data["limit"] = _construct_eval_sample_limit(data["limit"])

    return _fast_construct(EvalSample, data)
