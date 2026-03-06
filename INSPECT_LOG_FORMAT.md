# Inspect AI Log Format Reference

This document serves as the definitive reference for implementing a Rust native extension that replaces inspect's log reading functions with high-performance implementations.

## Table of Contents

1. [Data Model](#data-model)
2. [.eval ZIP Format](#eval-zip-format)
3. [.json Format](#json-format)
4. [Code Paths for Reading Logs](#code-paths-for-reading-logs)
5. [Header-Only Optimization Paths](#header-only-optimization-paths)
6. [Eval Set Merging/Combining Logic](#eval-set-mergingcombining-logic)
7. [Async/Sync Duality](#asyncsync-duality)
8. [Model Validators and Data Transformations](#model-validators-and-data-transformations)
9. [Deserialization Context Mechanism](#deserialization-context-mechanism)
10. [NaN/Inf in JSON](#naninf-in-json)
11. [Complex Type Hierarchies](#complex-type-hierarchies)
12. [Format Auto-Detection](#format-auto-detection)
13. [model_validate vs model_construct](#model_validate-vs-model_construct)

---

## Data Model

### EvalLog (top-level)

The root model for an evaluation log. **Field order matters for the .json format** (see warning in source).

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `version` | `int` | `2` | Log file format version |
| `status` | `EvalStatus` | `"started"` | `"started" \| "success" \| "cancelled" \| "error"` |
| `eval` | `EvalSpec` | *required* | Eval identity and configuration |
| `plan` | `EvalPlan` | `EvalPlan()` | Eval plan (solvers and config) |
| `results` | `EvalResults \| None` | `None` | Scoring results |
| `stats` | `EvalStats` | `EvalStats()` | Runtime/usage statistics |
| `error` | `EvalError \| None` | `None` | Error that halted eval |
| `invalidated` | `bool` | `False` | Whether any samples were invalidated |
| `samples` | `list[EvalSample] \| None` | `None` | Samples processed by eval |
| `reductions` | `list[EvalSampleReductions] \| None` | `None` | Reduced sample values |
| `location` | `str` | `""` | **Excluded from serialization** (`exclude=True`) |
| `etag` | `str \| None` | `None` | **Excluded from serialization** (`exclude=True`) |

Has two model_validators: `resolve_sample_reductions` (mode="before") and `populate_scorer_name_for_samples` (mode="after"). See [Model Validators](#model-validators-and-data-transformations).

### EvalSpec

Eval target and configuration.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `eval_set_id` | `str \| None` | `None` | Globally unique eval set id |
| `eval_id` | `str` | `""` (then generated) | Globally unique eval id |
| `run_id` | `str` | `""` | Unique run id |
| `created` | `UtcDatetimeStr` | *required* | ISO 8601 datetime string |
| `task` | `str` | *required* | Task name |
| `task_id` | `str` | `""` | Unique task id |
| `task_version` | `int \| str` | `0` | Task version |
| `task_file` | `str \| None` | `None` | Task source file |
| `task_display_name` | `str \| None` | `None` | Task display name |
| `task_registry_name` | `str \| None` | `None` | Task registry name |
| `task_attribs` | `dict[str, Any]` | `{}` | Attributes of @task decorator |
| `task_args` | `dict[str, Any]` | `{}` | Arguments for invoking task |
| `task_args_passed` | `dict[str, Any]` | `{}` | Explicitly passed task args |
| `solver` | `str \| None` | `None` | Solver name |
| `solver_args` | `dict[str, Any] \| None` | `None` | Solver invocation args |
| `solver_args_passed` | `dict[str, Any] \| None` | `None` | Explicitly passed solver args |
| `tags` | `list[str] \| None` | `None` | Tags for eval run |
| `dataset` | `EvalDataset` | *required* | Dataset used for eval |
| `sandbox` | `SandboxEnvironmentSpec \| None` | `None` | Sandbox environment spec |
| `model` | `str` | *required* | Model used for eval |
| `model_generate_config` | `GenerateConfig` | `GenerateConfig()` | Generation config for model |
| `model_base_url` | `str \| None` | `None` | Model base URL override |
| `model_args` | `dict[str, Any]` | `{}` | Model-specific args |
| `model_roles` | `dict[str, ModelConfig] \| None` | `None` | Model roles |
| `config` | `EvalConfig` | *required* | Configuration values |
| `revision` | `EvalRevision \| None` | `None` | Source revision |
| `packages` | `dict[str, str]` | `{}` | Package versions |
| `metadata` | `dict[str, Any] \| None` | `None` | Additional metadata |
| `scorers` | `list[EvalScorer] \| None` | `None` | Scorers and args |
| `metrics` | complex type | `None` | Metrics and args |

Has `model_post_init` that generates `eval_id` differently during deserialization (uses hash instead of random UUID). Has `read_sandbox_spec` model_validator that calls `migrate_values`.

### EvalConfig

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `limit` | `int \| tuple[int, int] \| None` | `None` | Sample limit |
| `sample_id` | `str \| int \| list[...] \| None` | `None` | Specific sample(s) |
| `sample_shuffle` | `bool \| int \| None` | `None` | Shuffle order |
| `epochs` | `int \| None` | `None` | Number of epochs |
| `epochs_reducer` | `list[str] \| None` | `None` | Reducers for scores |
| `approval` | `ApprovalPolicyConfig \| None` | `None` | Approval policy |
| `fail_on_error` | `bool \| float \| None` | `None` | Fail on sample errors |
| `continue_on_fail` | `bool \| None` | `None` | Continue on fail |
| `retry_on_error` | `int \| None` | `None` | Retry count |
| `message_limit` | `int \| None` | `None` | Max messages per sample |
| `token_limit` | `int \| None` | `None` | Max tokens per sample |
| `time_limit` | `int \| None` | `None` | Max clock time per sample |
| `working_limit` | `int \| None` | `None` | Max working time per sample |
| `cost_limit` | `float \| None` | `None` | Max cost per sample |
| `max_samples` | `int \| None` | `None` | Max parallel samples |
| `max_tasks` | `int \| None` | `None` | Max parallel tasks |
| `max_subprocesses` | `int \| None` | `None` | Max concurrent subprocesses |
| `max_sandboxes` | `int \| None` | `None` | Max concurrent sandboxes |
| `sandbox_cleanup` | `bool \| None` | `None` | Cleanup sandboxes |
| `log_samples` | `bool \| None` | `None` | Log detailed sample info |
| `log_realtime` | `bool \| None` | `None` | Log events in realtime |
| `log_images` | `bool \| None` | `None` | Log base64 images |
| `log_model_api` | `bool \| None` | `None` | Log raw model API |
| `log_buffer` | `int \| None` | `None` | Sample buffer before write |
| `log_shared` | `int \| None` | `None` | Sync interval (seconds) |
| `score_display` | `bool \| None` | `None` | Display scoring metrics |

Has `convert_max_messages_to_message_limit` model_validator (mode="before") that migrates deprecated `max_messages` to `message_limit`.

### EvalPlan

| Field | Type | Default |
|-------|------|---------|
| `name` | `str` | `"plan"` |
| `steps` | `list[EvalPlanStep]` | `[]` |
| `finish` | `EvalPlanStep \| None` | `None` |
| `config` | `GenerateConfig` | `GenerateConfig()` |

### EvalPlanStep

| Field | Type | Default |
|-------|------|---------|
| `solver` | `str` | *required* |
| `params` | `dict[str, Any]` | `{}` |
| `params_passed` | `dict[str, Any]` | `{}` |

Has `read_params` model_validator: if `params_passed` missing, sets it to value of `params`.

### EvalResults

| Field | Type | Default |
|-------|------|---------|
| `total_samples` | `int` | `0` |
| `completed_samples` | `int` | `0` |
| `early_stopping` | `EarlyStoppingSummary \| None` | `None` |
| `scores` | `list[EvalScore]` | `[]` |
| `metadata` | `dict[str, Any] \| None` | `None` |
| `_sample_reductions` | `PrivateAttr(list[EvalSampleReductions] \| None)` | `None` |

Has `convert_scorer_to_scorers` model_validator (mode="before"): migrates old `scorer`/`metrics` format to `scores` list.

### EvalScore

| Field | Type | Default |
|-------|------|---------|
| `name` | `str` | *required* |
| `scorer` | `str` | *required* |
| `reducer` | `str \| None` | `None` |
| `scored_samples` | `int \| None` | `None` |
| `unscored_samples` | `int \| None` | `None` |
| `params` | `dict[str, Any]` | `{}` |
| `metrics` | `dict[str, EvalMetric]` | `{}` |
| `metadata` | `dict[str, Any] \| None` | `None` |

### EvalMetric

| Field | Type | Default |
|-------|------|---------|
| `name` | `str` | *required* |
| `value` | `int \| float` | *required* |
| `params` | `dict[str, Any]` | `{}` |
| `metadata` | `dict[str, Any] \| None` | `None` |

### EvalStats

| Field | Type | Default |
|-------|------|---------|
| `started_at` | `UtcDatetimeStr \| Literal[""]` | `""` |
| `completed_at` | `UtcDatetimeStr \| Literal[""]` | `""` |
| `model_usage` | `dict[str, ModelUsage]` | `{}` |
| `role_usage` | `dict[str, ModelUsage]` | `{}` |

### EvalDataset

| Field | Type | Default |
|-------|------|---------|
| `name` | `str \| None` | `None` |
| `location` | `str \| None` | `None` |
| `samples` | `int \| None` | `None` |
| `sample_ids` | `list[str] \| list[int] \| list[str \| int] \| None` | `None` |
| `shuffled` | `bool \| None` | `None` |

### EvalSample

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `id` | `int \| str` | *required* | Sample id |
| `epoch` | `int` | *required* | Epoch number |
| `input` | `str \| list[ChatMessage]` | *required* | Sample input |
| `choices` | `list[str] \| None` | `None` | |
| `target` | `str \| list[str]` | *required* | Target value(s) |
| `sandbox` | `SandboxEnvironmentSpec \| None` | `None` | |
| `files` | `list[str] \| None` | `None` | |
| `setup` | `str \| None` | `None` | |
| `messages` | `list[ChatMessage]` | `[]` | Chat history |
| `output` | `ModelOutput` | `ModelOutput()` | Model output |
| `scores` | `dict[str, Score] \| None` | `None` | |
| `metadata` | `dict[str, Any]` | `{}` | |
| `store` | `dict[str, Any]` | `{}` | State at end |
| `events` | `list[Event]` | `[]` | Event list |
| `timelines` | `list[Timeline] \| None` | `None` | Custom timelines |
| `model_usage` | `dict[str, ModelUsage]` | `{}` | Token usage |
| `role_usage` | `dict[str, ModelUsage]` | `{}` | Token usage by role |
| `started_at` | `UtcDatetimeStr \| None` | `None` | |
| `completed_at` | `UtcDatetimeStr \| None` | `None` | |
| `total_time` | `float \| None` | `None` | |
| `working_time` | `float \| None` | `None` | |
| `uuid` | `str \| None` | `None` | |
| `invalidation` | `ProvenanceData \| None` | `None` | |
| `error` | `EvalError \| None` | `None` | |
| `error_retries` | `list[EvalError] \| None` | `None` | |
| `attachments` | `dict[str, str]` | `{}` | Attachment references |
| `limit` | `EvalSampleLimit \| None` | `None` | Limit that halted sample |

Has `migrate_deprecated` model_validator (mode="before"): converts old `score` → `scores`, old `transcript` → `events`+`attachments`, and calls `migrate_values`.

### EvalSampleSummary

| Field | Type | Default |
|-------|------|---------|
| `id` | `int \| str` | *required* |
| `epoch` | `int` | *required* |
| `input` | `str \| list[ChatMessage]` | *required* |
| `choices` | `list[str] \| None` | `None` |
| `target` | `str \| list[str]` | *required* |
| `metadata` | `dict[str, Any]` | `{}` |
| `scores` | `dict[str, Score] \| None` | `None` |
| `model_usage` | `dict[str, ModelUsage]` | `{}` |
| `role_usage` | `dict[str, ModelUsage]` | `{}` |
| `started_at` | `UtcDatetimeStr \| None` | `None` |
| `completed_at` | `UtcDatetimeStr \| None` | `None` |
| `total_time` | `float \| None` | `None` |
| `working_time` | `float \| None` | `None` |
| `uuid` | `str \| None` | `None` |
| `error` | `str \| None` | `None` |
| `limit` | `str \| None` | `None` |
| `retries` | `int \| None` | `None` |
| `completed` | `bool` | `False` |
| `message_count` | `int \| None` | `None` |

Has `thin_data` model_validator (mode="after"): truncates input, target, metadata, and score fields.

### EvalSampleLimit

| Field | Type |
|-------|------|
| `type` | `Literal["context", "time", "working", "message", "token", "cost", "operator", "custom"]` |
| `limit` | `float` |

### EvalSampleReductions

| Field | Type |
|-------|------|
| `scorer` | `str` |
| `reducer` | `str \| None` |
| `samples` | `list[EvalSampleScore]` |

### EvalSampleScore (extends Score)

| Field | Type | Default |
|-------|------|---------|
| `sample_id` | `str \| int \| None` | `None` |
| *(plus all Score fields)* | | |

### Score

| Field | Type | Default |
|-------|------|---------|
| `value` | `Value` | *required* |
| `answer` | `str \| None` | `None` |
| `explanation` | `str \| None` | `None` |
| `metadata` | `dict[str, Any] \| None` | `None` |
| `history` | `list[ScoreEdit]` | `[]` |

Where `Value = str | int | float | bool | Sequence[str|int|float|bool] | Mapping[str, str|int|float|bool|None]`.

### EvalError

| Field | Type |
|-------|------|
| `message` | `str` |
| `traceback` | `str` |
| `traceback_ansi` | `str` |

### ModelUsage

| Field | Type | Default |
|-------|------|---------|
| `input_tokens` | `int` | `0` |
| `output_tokens` | `int` | `0` |
| `total_tokens` | `int` | `0` |
| `input_tokens_cache_write` | `int \| None` | `None` |
| `input_tokens_cache_read` | `int \| None` | `None` |
| `reasoning_tokens` | `int \| None` | `None` |
| `total_cost` | `float \| None` | `None` |

### ModelOutput

| Field | Type | Default |
|-------|------|---------|
| `model` | `str` | `""` |
| `choices` | `list[ChatCompletionChoice]` | `[]` |
| `completion` | `str` | `""` |
| `usage` | `ModelUsage \| None` | `None` |
| `time` | `float \| None` | `None` |
| `metadata` | `dict[str, Any] \| None` | `None` |
| `error` | `str \| None` | `None` |

Has `set_completion` model_validator (mode="after"): auto-populates `completion` from first choice if empty.

### ChatCompletionChoice

| Field | Type | Default |
|-------|------|---------|
| `message` | `ChatMessageAssistant` | *required* |
| `stop_reason` | `StopReason` | `"unknown"` |
| `logprobs` | `Logprobs \| None` | `None` |

Has `migrate_stop_reason` model_validator: converts `"length"` → `"max_tokens"`.

`StopReason = Literal["stop", "max_tokens", "model_length", "tool_calls", "content_filter", "unknown"]`

### EvalRevision

| Field | Type |
|-------|------|
| `type` | `Literal["git"]` |
| `origin` | `str` |
| `commit` | `str` |
| `dirty` | `bool \| None` |

### EvalScorer

| Field | Type | Default |
|-------|------|---------|
| `name` | `str` | *required* |
| `options` | `dict[str, Any] \| None` | `None` |
| `metrics` | complex | `None` |
| `metadata` | `dict[str, Any] \| None` | `None` |

### EvalMetricDefinition

| Field | Type | Default |
|-------|------|---------|
| `name` | `str` | *required* |
| `options` | `dict[str, Any] \| None` | `None` |

### SandboxEnvironmentSpec (frozen model)

| Field | Type | Default |
|-------|------|---------|
| `type` | `str` | *required* |
| `config` | `Any` | `None` |

Has `load_config_model` model_validator: for dict config values, attempts to deserialize sandbox-specific config.

### GenerateConfig

| Field | Type | Default |
|-------|------|---------|
| `max_retries` | `int \| None` | `None` |
| `timeout` | `int \| None` | `None` |
| `attempt_timeout` | `int \| None` | `None` |
| `max_connections` | `int \| None` | `None` |
| `system_message` | `str \| None` | `None` |
| `max_tokens` | `int \| None` | `None` |
| `top_p` | `float \| None` | `None` |
| `temperature` | `float \| None` | `None` |
| `stop_seqs` | `list[str] \| None` | `None` |
| `best_of` | `int \| None` | `None` |
| `frequency_penalty` | `float \| None` | `None` |
| `presence_penalty` | `float \| None` | `None` |
| `logit_bias` | `dict[int, float] \| None` | `None` |
| `seed` | `int \| None` | `None` |
| `top_k` | `int \| None` | `None` |
| `num_choices` | `int \| None` | `None` |
| `logprobs` | `bool \| None` | `None` |
| `top_logprobs` | `int \| None` | `None` |
| `parallel_tool_calls` | `bool \| None` | `None` |
| `internal_tools` | `bool \| None` | `None` |
| `max_tool_output` | `int \| None` | `None` |
| `cache_prompt` | `Literal["auto"] \| bool \| None` | `None` |
| `verbosity` | `Literal["low", "medium", "high"] \| None` | `None` |
| `effort` | `Literal["low", "medium", "high", "max"] \| None` | `None` |
| `reasoning_effort` | `Literal["none", "minimal", "low", "medium", "high", "xhigh"] \| None` | `None` |
| `reasoning_tokens` | `int \| None` | `None` |
| `reasoning_summary` | `Literal["none", "concise", "detailed", "auto"] \| None` | `None` |
| `reasoning_history` | `Literal["none", "all", "last", "auto"] \| None` | `None` |
| `response_schema` | `ResponseSchema \| None` | `None` |
| `extra_headers` | `dict[str, str] \| None` | `None` |
| `extra_body` | `dict[str, Any] \| None` | `None` |
| `cache` | `bool \| CachePolicy \| None` | `None` |
| `batch` | `bool \| int \| BatchConfig \| None` | `None` |

Has `migrate_reasoning` model_validator: converts `reasoning_history` from `True`→`"all"`, `False`→`"none"`.

### EarlyStoppingSummary

| Field | Type | Default |
|-------|------|---------|
| `stopped` | `bool` | *required* |
| `reason` | `str \| None` | `None` |
| `completed_epochs` | `int` | *required* |

---

## .eval ZIP Format

The `.eval` format is a ZIP archive (standard ZIP, identifiable by magic bytes `PK\x03\x04`). Internal structure:

```
archive.eval (ZIP)
├── header.json                        # Complete EvalLog header (written at eval finish)
├── _journal/
│   ├── start.json                     # LogStart: {version, eval (EvalSpec), plan (EvalPlan)}
│   └── summaries/
│       ├── 1.json                     # Batch of EvalSampleSummary objects (list)
│       ├── 2.json                     # Another batch
│       └── ...
├── summaries.json                     # Consolidated list of all EvalSampleSummary objects
├── samples/
│   ├── {id}_epoch_{epoch}.json        # Individual EvalSample data
│   ├── {id}_epoch_{epoch}.json
│   └── ...
└── reductions.json                    # Optional: list of EvalSampleReductions
```

### Key files in the ZIP:

1. **`header.json`**: Written at eval finish. Contains a complete `EvalLog` object (without samples). Used for fast header-only reads. If present, this is preferred over `_journal/start.json`.

2. **`_journal/start.json`**: Written at eval start. Contains `LogStart` model:
   ```json
   {"version": 2, "eval": {...EvalSpec...}, "plan": {...EvalPlan...}}
   ```
   Used as fallback when `header.json` doesn't exist (e.g., eval still running).

3. **`_journal/summaries/N.json`**: Incrementally written batches of sample summaries. Each file contains a JSON array of `EvalSampleSummary` objects. `N` is a 1-based counter.

4. **`summaries.json`**: Written at eval finish. Consolidated list of all summaries. Preferred over reading individual journal summaries.

5. **`samples/{id}_epoch_{epoch}.json`**: Individual sample data. Each file is a JSON object representing one `EvalSample`. The filename encodes the sample id and epoch.

6. **`reductions.json`**: Optional. JSON array of `EvalSampleReductions` objects.

### Internal models used only for .eval format:

**`LogStart`** (in `_recorders/eval.py`):
| Field | Type |
|-------|------|
| `version` | `int` |
| `eval` | `EvalSpec` |
| `plan` | `EvalPlan` |

**`LogResults`** (in `_recorders/eval.py`, used internally only):
| Field | Type |
|-------|------|
| `status` | `EvalStatus` |
| `stats` | `EvalStats` |
| `results` | `EvalResults \| None` |
| `error` | `EvalError \| None` |

---

## .json Format

A single monolithic JSON file containing the entire `EvalLog` structure. Fields appear in this order (important for streaming header reads):

```json
{
  "version": 2,
  "status": "success",
  "eval": {...},
  "plan": {...},
  "results": {...},
  "stats": {...},
  "error": null,
  "invalidated": false,
  "samples": [...],
  "reductions": [...]
}
```

The field ordering enables streaming header reads with `ijson`: the parser can stop once it reaches `"samples"` (or `"error"` if present after `"stats"`), avoiding parsing the potentially huge samples array.

Full reads use `pydantic_core.from_json()` which handles NaN/Inf values. See [NaN/Inf in JSON](#naninf-in-json).

---

## Code Paths for Reading Logs

### Full Log Read

```
read_eval_log(log_file, header_only=False)
  → read_eval_log_async(log_file, header_only=False)
    → recorder_type_for_location(log_file)  # or recorder_type_for_bytes()

    For .eval format (EvalRecorder.read_log):
      → AsyncZipReader(async_fs, location)
      → reader.entries()  # reads central directory
      → _read_log(reader, entries, location, header_only=False)
        → _read_header_async(reader, entry_names, location)
          → If "header.json" in entries:
              Read header.json → EvalLog.model_validate(data, context=get_deserializing_context())
          → Else:
              Read _journal/start.json → LogStart.model_validate(...)
              Construct minimal EvalLog from LogStart
        → If "reductions.json" in entries:
            Read and parse EvalSampleReductions
        → For each "samples/*.json" entry:
            Read → EvalSample.model_validate(data, context=get_deserializing_context())
        → sort_samples(samples)

    For .json format (JSONRecorder.read_log):
      → pydantic_core.from_json(file_content)  # raw parsing
      → _parse_json_log(raw_data, header_only=False)
        → EvalLog.model_validate(raw_data, context=get_deserializing_context())
        → _validate_version(log.version)
        → log.version = LOG_SCHEMA_VERSION

    → resolve_sample_attachments() if requested
    → populate dataset.sample_ids if missing
```

### Header-Only Read

```
read_eval_log(log_file, header_only=True)
  → read_eval_log_async(log_file, header_only=True)

    For .eval format:
      → Same as full read but skips all samples/*.json entries
      → Still reads header.json or _journal/start.json
      → Still reads reductions.json if present

    For .json format:
      → Fast path: _read_header_streaming(location)
        → Uses ijson to parse fields one by one
        → First pass: get version, detect results/error presence
        → Second pass: parse eval, plan, results, stats, error
        → STOPS before parsing "samples"
      → Falls back to full parse if ijson encounters NaN/Inf
      → If full parse needed: _parse_json_log(raw_data, header_only=True)
        → Full EvalLog.model_validate() then sets samples=None
        → Also nullifies reductions
```

### Batch Header Read

```
read_eval_log_headers(log_files)
  → read_eval_log_headers_async(log_files)
    → tg_collect([partial(_read, lf) for lf in log_files])
      → Concurrent reads of all headers using async task group
      → Each calls read_eval_log_async(lf, header_only=True)
```

### Single Sample Read

```
read_eval_log_sample(log_file, id, epoch)
  → read_eval_log_sample_async(log_file, id, epoch)

    For .eval format (EvalRecorder.read_log_sample):
      → AsyncZipReader
      → If uuid specified: read summaries to find matching id/epoch
      → If exclude_fields:
          Use ijson.kvitems_async to stream parse, skipping excluded fields
          Falls back to json.loads on NaN/Inf errors
      → Else: json.loads(reader.read_member_fully("samples/{id}_epoch_{epoch}.json"))
      → EvalSample.model_validate(data, context=get_deserializing_context())

    For .json format (FileRecorder.read_log_sample - base class):
      → _log_file_maybe_cached(location)  # caches last read log
      → Find matching sample by id/epoch or uuid in log.samples
```

### Sample Summaries Read

```
read_eval_log_sample_summaries(log_file)
  → read_eval_log_sample_summaries_async(log_file)

    For .eval format (EvalRecorder.read_log_sample_summaries):
      → _read_all_summaries_async(reader)
        → If "summaries.json" exists: parse it directly
        → Else: iterate _journal/summaries/N.json files
      → Each summary parsed with EvalSampleSummary.model_validate(...)

    For .json format (FileRecorder.read_log_sample_summaries - base class):
      → Read full log (cached), then call sample.summary() for each
```

### Incremental Sample Read (Generator)

```
read_eval_log_samples(log_file)
  → read_eval_log(log_file, header_only=True)  # get header
  → For each sample_id × epoch:
      read_eval_log_sample(log_file, id=sample_id, epoch=epoch_id)
      yield sample
```

---

## Header-Only Optimization Paths

### .eval Format
- Reads only `header.json` (single ZIP entry) — skips all `samples/*.json` entries
- If `header.json` absent (eval in progress), falls back to `_journal/start.json`
- Still reads `reductions.json` if present

### .json Format — Streaming Header
`_read_header_streaming(log_file)` uses `ijson` for streaming JSON parsing:

1. **First pass** (`ijson.parse`): Scans for `version`, detects presence of `results` and `error` sections. Breaks at `samples` field.
2. **Second pass** (`ijson.kvitems`): Parses actual field values — `status`, `invalidated`, `eval`, `plan`, `results`, `stats`, `error`. Breaks after `stats` (or `error` if present). Uses `EvalSpec(**v)` etc. (equivalent to `model_validate` since these are dicts).
3. **Fallback**: If ijson fails on NaN/Inf, falls through to full `pydantic_core.from_json()` parse.

Key observation: The streaming approach avoids parsing the `samples` array entirely, which can be hundreds of megabytes.

---

## Eval Set Merging/Combining Logic

`list_all_eval_logs()` in `evalset.py`:
1. `list_eval_logs(log_dir)` — lists all `.eval`/`.json` files in directory
2. `read_eval_log_headers(log_files)` — batch reads all headers concurrently
3. `task_identifier(log_header)` — computes task identity string
4. Returns list of `Log(info, header, task_identifier)` tuples

`list_latest_eval_logs()`:
- Groups logs by task identifier
- Finds the latest completed log for each task
- Optionally cleans up older logs

The batch header read (`read_eval_log_headers`) is a primary performance bottleneck when there are many log files. It uses `tg_collect` for concurrent async reads.

---

## Async/Sync Duality

Inspect uses an async-first architecture with sync wrappers:

### Pattern
```python
def read_eval_log(log_file, header_only=False, ...):
    """Sync wrapper."""
    return run_coroutine(read_eval_log_async(log_file, header_only, ...))

async def read_eval_log_async(log_file, header_only=False, ...):
    """Actual async implementation."""
    ...
```

### `run_coroutine` (from `inspect_ai._util._async`)
- Runs an async coroutine synchronously
- Handles the case where it's called from different async backends

### `tg_collect` (from `inspect_ai._util._async`)
- Collects results from multiple async callables concurrently
- Used for batch operations like `read_eval_log_headers_async`
- Takes a list of `Callable[[], Awaitable[T]]` and returns `list[T]`

### Key sync → async mappings:
| Sync Function | Async Implementation |
|--------------|---------------------|
| `read_eval_log()` | `read_eval_log_async()` |
| `read_eval_log_headers()` | `read_eval_log_headers_async()` |
| `read_eval_log_sample()` | `read_eval_log_sample_async()` |
| `read_eval_log_sample_summaries()` | `read_eval_log_sample_summaries_async()` |
| `write_eval_log()` | `write_eval_log_async()` |
| `list_eval_logs()` | (sync only, no async variant) |

---

## Model Validators and Data Transformations

**Critical for Rust implementation**: These validators perform data transformations during `model_validate()`, not just validation. Bypassing `model_validate()` later requires replicating ALL of these.

### EvalSample.migrate_deprecated (mode="before")
```python
@model_validator(mode="before")
def migrate_deprecated(cls, values):
    # 1. Convert old "score" field to "scores" dict
    if "score" in values:
        values["scores"] = {SCORER_PLACEHOLDER: values["score"]}
        del values["score"]

    # 2. Convert old "transcript" to "events" + "attachments"
    if "transcript" in values:
        eval_events = EvalEvents(**values["transcript"])
        values["events"] = eval_events.events
        values["attachments"] = eval_events.content
        del values["transcript"]

    # 3. Call migrate_values (sandbox spec, missing args)
    return migrate_values(values)
```
`SCORER_PLACEHOLDER = "88F74D2C"` — temporary key replaced later by `EvalLog.populate_scorer_name_for_samples`.

### EvalSampleSummary.thin_data (mode="after")
```python
@model_validator(mode="after")
def thin_data(self):
    self.input = thin_input(self.input)      # truncate/simplify
    self.target = thin_target(self.target)    # truncate
    self.metadata = thin_metadata(self.metadata)  # filter/truncate
    # thin score explanations and metadata
    if self.scores is not None:
        self.scores = {
            key: Score(
                value=score.value,
                answer=thin_text(score.answer),
                explanation=thin_text(score.explanation),
                metadata=thin_metadata(score.metadata),
            )
            for key, score in self.scores.items()
        }
    return self
```

Thinning functions (from `_util.py`):
- `thin_input`: For ChatMessage lists, replaces non-text content with placeholders like `"(Image)"`, truncates text to 5120 chars. For strings, truncates to 5120 chars.
- `thin_target`: Truncates strings to 5120 chars.
- `thin_metadata`: Keeps primitives and dates; truncates strings to 1024 chars; removes complex values >1024 bytes.
- `thin_text`: Uses `textwrap.shorten()` to 1024 chars.

### EvalConfig.convert_max_messages_to_message_limit (mode="before")
```python
@model_validator(mode="before")
def convert_max_messages_to_message_limit(cls, values):
    if max_messages := values.get("max_messages"):
        values["message_limit"] = max_messages
    return values
```

### EvalSpec.read_sandbox_spec (mode="before")
```python
@model_validator(mode="before")
def read_sandbox_spec(cls, values):
    return migrate_values(values)
```

### EvalPlanStep.read_params (mode="before")
```python
@model_validator(mode="before")
def read_params(cls, values):
    if "params_passed" not in values:
        values["params_passed"] = values.get("params", {})
    return values
```

### EvalResults.convert_scorer_to_scorers (mode="before")
```python
@model_validator(mode="before")
def convert_scorer_to_scorers(cls, values):
    if "scorer" in values:
        metrics = values.pop("metrics", None)
        score = values["scorer"]
        if metrics:
            score["metrics"] = metrics
        score["scorer"] = score["name"]
        values["scores"] = [score]
        del values["scorer"]
    return values
```

### EvalLog.resolve_sample_reductions (mode="before")
```python
@model_validator(mode="before")
def resolve_sample_reductions(cls, values):
    has_reductions = "reductions" in values
    has_results = values.get("results") is not None
    has_sample_reductions = has_results and "sample_reductions" in values["results"]

    if has_sample_reductions and not has_reductions:
        values["reductions"] = values["results"]["sample_reductions"]
    elif has_reductions and has_results and not has_sample_reductions:
        values["results"]["sample_reductions"] = values["reductions"]
    return values
```

### EvalLog.populate_scorer_name_for_samples (mode="after")
```python
@model_validator(mode="after")
def populate_scorer_name_for_samples(self):
    if self.samples and self.results and self.results.scores:
        scorer_name = self.results.scores[0].name
        for sample in self.samples:
            if sample.scores and SCORER_PLACEHOLDER in sample.scores:
                sample.scores[scorer_name] = sample.scores[SCORER_PLACEHOLDER]
                del sample.scores[SCORER_PLACEHOLDER]
    return self
```
This replaces the temporary `"88F74D2C"` key with the actual scorer name from results.

### migrate_values (shared helper)
```python
def migrate_values(values):
    # 1. Convert list-format sandbox specs to SandboxEnvironmentSpec
    if "sandbox" in values:
        sandbox = values.get("sandbox")
        if isinstance(sandbox, list):
            values["sandbox"] = SandboxEnvironmentSpec(type=sandbox[0], config=sandbox[1])

    # 2. Default missing task_args_passed
    if "task_args_passed" not in values:
        values["task_args_passed"] = values.get("task_args", {})

    # 3. Default missing solver_args_passed
    if "solver_args_passed" not in values:
        values["solver_args_passed"] = values.get("solver_args", {})

    return values
```

### ChatMessageBase._wrap (mode="wrap")
A wrap-mode validator on `ChatMessageBase` (the base for all ChatMessage types). Implements deserialization caching:
- During deserialization, uses a `MESSAGE_CACHE` dict from the context
- Caches parsed messages by a content-based key to deduplicate repeated messages
- This is important because messages are often repeated in event transcripts

### ChatMessageTool.convert_tool_error_to_error (mode="before")
Migrates old `tool_error` field to `error` field as `ToolCallError`.

### ModelOutput.set_completion (mode="after")
Auto-populates `completion` from the first choice's message text if empty.

### ChatCompletionChoice.migrate_stop_reason (mode="before")
Converts `"length"` → `"max_tokens"` for stop_reason.

### GenerateConfig.migrate_reasoning (mode="before")
Converts `reasoning_history` from `True`→`"all"`, `False`→`"none"`.

### SubtaskEvent.validate_input (field_validator, mode="before")
Converts non-dict input to empty dict.

### ContentDocument.set_name_and_mime_type (mode="before")
Auto-sets filename and mime_type from document path/data URI.

### SandboxEnvironmentSpec.load_config_model (mode="before")
For dict config values, attempts to deserialize sandbox-specific config using registered handlers.

---

## Deserialization Context Mechanism

`get_deserializing_context()` from `_util/constants.py`:
```python
DESERIALIZING = "deserializing"
MESSAGE_CACHE = "message_cache"

def get_deserializing_context():
    return {DESERIALIZING: True, MESSAGE_CACHE: {}}
```

This context dict is passed to `model_validate()` calls:
```python
EvalLog.model_validate(data, context=get_deserializing_context())
```

The context is accessed via `__context` parameter in `model_post_init`:
```python
# In EvalSpec.model_post_init:
is_deserializing = isinstance(__context, dict) and __context.get(DESERIALIZING, False)
if self.eval_id == "":
    if is_deserializing:
        # Use deterministic hash for eval_id (stable across reads)
        self.eval_id = base57_id_hash(self.run_id + self.task_id + self.created)
    else:
        # Use random UUID for new evals
        self.eval_id = uuid()
```

The `MESSAGE_CACHE` is used by `ChatMessageBase._wrap` for deduplication during deserialization.

**Important**: The `DESERIALIZING` flag affects:
1. `EvalSpec.model_post_init` — eval_id generation
2. `BaseEvent.model_post_init` — skips uuid/span_id generation during deserialization
3. `ChatMessageBase._wrap` — enables message deduplication cache

---

## NaN/Inf in JSON

Python's `json` module outputs `NaN` and `Infinity` as literal tokens, which are **not valid JSON** per the spec. This happens commonly in evaluation metrics and scores.

### Current handling:
1. **`pydantic_core.from_json()`**: Handles NaN/Inf natively (used for full .json reads)
2. **`json.loads()`** / Python `json`: Handles NaN/Inf natively (used for .eval sample reads)
3. **`ijson`**: Does **NOT** handle NaN/Inf — raises `IncompleteJSONError` or `UnexpectedSymbol`
   - Detected by `is_ijson_nan_inf_error(ex)` helper
   - Fallback: retry with `json.loads()` or `pydantic_core.from_json()`

### Rust implementation implications:
- `serde_json` by default **rejects** NaN/Inf
- Options:
  - Use custom deserializer for float fields
  - Pre-process JSON to replace `NaN`/`Infinity`/`-Infinity` with `null` or sentinel values
  - Use `serde_json`'s `float_roundtrip` feature + custom handling
  - Consider the `serde_json` `arbitrary_precision` feature or a custom JSON parser

The Rust implementation **must** handle these values since they appear in real inspect logs.

---

## Complex Type Hierarchies

### Event (Discriminated Union)

`Event` is a TypeAlias for a Union of 19 event types, discriminated by the `event` literal field:

```python
Event = Union[
    SampleInitEvent,    # event="sample_init"
    SampleLimitEvent,   # event="sample_limit"
    SandboxEvent,       # event="sandbox"
    StateEvent,         # event="state"
    StoreEvent,         # event="store"
    ModelEvent,         # event="model"
    ToolEvent,          # event="tool"
    ApprovalEvent,      # event="approval"
    CompactionEvent,    # event="compaction"
    InputEvent,         # event="input"
    ScoreEvent,         # event="score"
    ScoreEditEvent,     # event="score_edit"
    ErrorEvent,         # event="error"
    LoggerEvent,        # event="logger"
    InfoEvent,          # event="info"
    SpanBeginEvent,     # event="span_begin"
    SpanEndEvent,       # event="span_end"
    StepEvent,          # event="step"
    SubtaskEvent,       # event="subtask"
]
```

All inherit from `BaseEvent`:
| Field | Type | Default |
|-------|------|---------|
| `uuid` | `str \| None` | `None` |
| `span_id` | `str \| None` | `None` |
| `timestamp` | `UtcDatetime` | auto-generated |
| `working_start` | `float` | auto-generated |
| `metadata` | `dict[str, Any] \| None` | `None` |
| `pending` | `bool \| None` | `None` |

Key event type fields:

**ModelEvent**: `model`, `role`, `input` (list[ChatMessage]), `tools` (list[ToolInfo]), `tool_choice`, `config` (GenerateConfig), `output` (ModelOutput), `retries`, `error`, `traceback`, `traceback_ansi`, `cache`, `call` (ModelCall), `completed`, `working_time`

**ToolEvent**: `type`, `id`, `function`, `arguments` (dict), `view`, `result`, `truncated`, `error`, `events` (list), `completed`, `working_time`, `agent`, `agent_span_id`, `failed`, `message_id`

**SampleInitEvent**: `sample` (Sample), `state` (JsonValue)

**StateEvent**: `changes` (list[JsonChange])

**StoreEvent**: `changes` (list[JsonChange])

**SubtaskEvent**: `name`, `type`, `input` (dict), `result`, `events` (list), `completed`, `working_time`

**InfoEvent**: `source`, `data` (JsonValue)

### ChatMessage (Discriminated Union)

```python
ChatMessage = Union[
    ChatMessageSystem,     # role="system"
    ChatMessageUser,       # role="user"
    ChatMessageAssistant,  # role="assistant"
    ChatMessageTool,       # role="tool"
]
```

All inherit from `ChatMessageBase`:
| Field | Type | Default |
|-------|------|---------|
| `id` | `str \| None` | `None` |
| `content` | `str \| list[Content]` | *required* |
| `source` | `Literal["input", "generate"] \| None` | `None` |
| `metadata` | `dict[str, Any] \| None` | `None` |

Additional fields per subtype:
- **ChatMessageSystem**: (none)
- **ChatMessageUser**: `tool_call_id: list[str] | None`
- **ChatMessageAssistant**: `tool_calls: list[ToolCall] | None`, `model: str | None`
- **ChatMessageTool**: `tool_call_id: str | None`, `function: str | None`, `error: ToolCallError | None`

### Content (Union Type)

```python
Content = Union[
    ContentText,       # type="text"
    ContentReasoning,  # type="reasoning"
    ContentImage,      # type="image"
    ContentAudio,      # type="audio"
    ContentVideo,      # type="video"
    ContentData,       # type="data"
    ContentToolUse,    # type="tool_use"
    ContentDocument,   # type="document"
]
```

All inherit from `ContentBase`: `internal: JsonValue | None = None`

Key content type fields:
- **ContentText**: `text`, `refusal`, `citations`
- **ContentReasoning**: `reasoning`, `summary`, `signature`, `redacted`
- **ContentToolUse**: `tool_type`, `id`, `name`, `context`, `arguments`, `result`, `error`
- **ContentImage**: `image`, `detail`
- **ContentAudio**: `audio`, `format`
- **ContentVideo**: `video`, `format`
- **ContentDocument**: `document`, `filename`, `mime_type`
- **ContentData**: `data` (dict)

---

## Format Auto-Detection

### By bytes (`recorder_type_for_bytes`):
```python
def recorder_type_for_bytes(log_bytes):
    first_bytes = log_bytes.read(4)
    log_bytes.seek(0)
    # ZIP magic: b"PK\x03\x04" → EvalRecorder (.eval)
    # JSON start: b"{"        → JSONRecorder (.json)
```

### By location (`recorder_type_for_location`):
```python
def recorder_type_for_location(location):
    # Checks file extension:
    # .eval → EvalRecorder
    # .json → JSONRecorder
    # Uses handles_location() on each registered recorder
```

### By format string (`recorder_type_for_format`):
```python
_recorders = {"eval": EvalRecorder, "json": JSONRecorder}

def recorder_type_for_format(format):
    return _recorders[format]
```

---

## model_validate vs model_construct

### `model_validate()` (full validation)
- Runs ALL model_validators (both `mode="before"` and `mode="after"`)
- Runs field validators
- Performs type coercion
- Runs `model_post_init`
- **Used everywhere in deserialization**: All `EvalLog.model_validate(data, context=get_deserializing_context())` calls

### `model_construct()` (skip all validation)
- Creates model instance directly from provided values
- **Skips ALL validators and transformations**
- No type coercion
- No `model_post_init`
- **Not currently used in log reading**

### Implications for Rust optimization:
If the Rust implementation bypasses `model_validate()` (e.g., by constructing Python objects directly or using `model_construct()`), it **must replicate** all the transformations listed in [Model Validators](#model-validators-and-data-transformations):
1. `EvalSample.migrate_deprecated` — score/transcript migration
2. `EvalSampleSummary.thin_data` — data truncation
3. `EvalConfig.convert_max_messages_to_message_limit` — field rename
4. `EvalSpec.read_sandbox_spec` + `migrate_values` — sandbox/args migration
5. `EvalPlanStep.read_params` — default params_passed
6. `EvalResults.convert_scorer_to_scorers` — scorer migration
7. `EvalLog.resolve_sample_reductions` — reductions sync
8. `EvalLog.populate_scorer_name_for_samples` — scorer name replacement
9. `EvalSpec.model_post_init` — eval_id generation (hash vs UUID)
10. `ChatMessageBase._wrap` — message deduplication cache
11. `ChatMessageTool.convert_tool_error_to_error` — tool_error migration
12. `ModelOutput.set_completion` — completion auto-population
13. `ChatCompletionChoice.migrate_stop_reason` — stop_reason rename
14. `GenerateConfig.migrate_reasoning` — reasoning_history conversion
15. `SubtaskEvent.validate_input` — input normalization
16. `ContentDocument.set_name_and_mime_type` — auto-set fields
17. `SandboxEnvironmentSpec.load_config_model` — config deserialization

A hybrid approach may be optimal: use Rust for fast JSON parsing and ZIP extraction, then pass the parsed dicts through Pydantic `model_validate()` on the Python side for correctness. Later optimization can selectively bypass Pydantic for specific models where all transformations have been replicated in Rust.

---

## Condensation / Attachment System

The condensation system (`_condense.py`) reduces storage by deduplicating content:

1. **Deduplication**: Large content strings (>100 chars in events, images in messages) are replaced with `attachment://{hash}` references. The original content is stored in `EvalSample.attachments`.
2. **Hash function**: Uses `mm3_hash` (MurmurHash3) for content hashing.
3. **Resolution**: `resolve_sample_attachments()` replaces `attachment://*` references back with content.
4. Also handles legacy `tc://` protocol prefix (converted to `attachment://`).
5. Base64 images can be stripped entirely (replaced with `"<base64-data-removed>"`).

The `walk_*` functions traverse the entire sample structure (messages, events, content) applying content transformations.

---

## Additional Constants

```python
LOG_SCHEMA_VERSION = 2
SCORER_PLACEHOLDER = "88F74D2C"
DESERIALIZING = "deserializing"
MESSAGE_CACHE = "message_cache"
BASE_64_DATA_REMOVED = "<base64-data-removed>"
ATTACHMENT_PROTOCOL = "attachment://"

# ZIP internal paths
JOURNAL_DIR = "_journal"
SUMMARY_DIR = "summaries"
SAMPLES_DIR = "samples"
START_JSON = "start.json"
RESULTS_JSON = "results.json"
REDUCTIONS_JSON = "reductions.json"
SUMMARIES_JSON = "summaries.json"
HEADER_JSON = "header.json"

# Sample filename pattern
# samples/{id}_epoch_{epoch}.json
```
