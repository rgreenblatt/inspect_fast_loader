"""Generate realistic inspect log files in both .eval and .json formats.

Procedurally generates logs WITHOUT running any AI, producing structurally
valid data loadable by inspect's existing Python code.
"""

import argparse
import json
import os
import random
import zipfile
from io import BytesIO
from typing import Any

# --------------------------------------------------------------------------- #
# Lorem-ipsum-style text generation
# --------------------------------------------------------------------------- #

WORDS = (
    "the quick brown fox jumps over lazy dog lorem ipsum dolor sit amet "
    "consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore "
    "et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation "
    "ullamco laboris nisi aliquip ex ea commodo consequat duis aute irure "
    "dolor in reprehenderit voluptate velit esse cillum fugiat nulla pariatur"
).split()


def lorem(rng: random.Random, n_words: int = 20) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def lorem_paragraph(rng: random.Random, n_sentences: int = 3) -> str:
    return ". ".join(lorem(rng, rng.randint(8, 25)) for _ in range(n_sentences)) + "."


# --------------------------------------------------------------------------- #
# Timestamp helpers
# --------------------------------------------------------------------------- #

def make_timestamp(rng: random.Random, base_year: int = 2024) -> str:
    """Generate a deterministic ISO-8601 UTC timestamp."""
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return f"{base_year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}+00:00"


# --------------------------------------------------------------------------- #
# Data generation helpers
# --------------------------------------------------------------------------- #

def make_model_usage(rng: random.Random) -> dict:
    inp = rng.randint(100, 5000)
    out = rng.randint(50, 2000)
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": inp + out,
    }


def make_content_text(rng: random.Random) -> dict:
    return {"type": "text", "text": lorem(rng, rng.randint(5, 50))}


def make_content_image() -> dict:
    return {"type": "image", "image": "data:image/png;base64,iVBORw0KGgo=", "detail": "auto"}


def make_content_reasoning(rng: random.Random) -> dict:
    return {
        "type": "reasoning",
        "reasoning": lorem(rng, rng.randint(10, 40)),
        "redacted": False,
    }


def make_content_tool_use(rng: random.Random) -> dict:
    return {
        "type": "tool_use",
        "tool_type": "code_execution",
        "id": f"tooluse_{rng.randint(1000, 9999)}",
        "name": rng.choice(["python", "bash", "web_search"]),
        "arguments": json.dumps({"code": lorem(rng, 10)}),
        "result": lorem(rng, 15),
    }


def make_chat_message_system(rng: random.Random) -> dict:
    return {"role": "system", "content": lorem(rng, rng.randint(10, 30))}


def make_chat_message_user(rng: random.Random, use_content_list: bool = False) -> dict:
    if use_content_list:
        content: Any = [make_content_text(rng)]
        if rng.random() < 0.3:
            content.append(make_content_image())
    else:
        content = lorem(rng, rng.randint(10, 50))
    return {"role": "user", "content": content}


def make_chat_message_assistant(rng: random.Random, include_tool_calls: bool = False) -> dict:
    msg: dict[str, Any] = {"role": "assistant", "content": lorem(rng, rng.randint(15, 80))}
    if include_tool_calls:
        msg["tool_calls"] = [
            {
                "id": f"call_{rng.randint(10000, 99999)}",
                "function": rng.choice(["web_search", "bash", "python"]),
                "arguments": {"query": lorem(rng, 5)},
                "type": "function",
            }
        ]
    return msg


def make_chat_message_tool(rng: random.Random) -> dict:
    return {
        "role": "tool",
        "content": lorem(rng, rng.randint(5, 30)),
        "tool_call_id": f"call_{rng.randint(10000, 99999)}",
        "function": rng.choice(["web_search", "bash", "python"]),
    }


def make_messages(rng: random.Random, n_turns: int = 5) -> list[dict]:
    """Generate a realistic conversation history."""
    messages = [make_chat_message_system(rng)]
    for _ in range(n_turns):
        use_content_list = rng.random() < 0.2
        messages.append(make_chat_message_user(rng, use_content_list))
        include_tool_calls = rng.random() < 0.3
        messages.append(make_chat_message_assistant(rng, include_tool_calls))
        if include_tool_calls:
            messages.append(make_chat_message_tool(rng))
    return messages


def make_score(rng: random.Random) -> dict:
    value = rng.choice([0, 0.5, 1, "C", "I"])
    return {
        "value": value,
        "answer": lorem(rng, 3) if rng.random() < 0.5 else None,
        "explanation": lorem(rng, 10) if rng.random() < 0.5 else None,
    }


def make_event_model(rng: random.Random) -> dict:
    return {
        "event": "model",
        "timestamp": make_timestamp(rng),
        "working_start": rng.random() * 10,
        "model": "openai/gpt-4o",
        "input": [make_chat_message_user(rng)],
        "tools": [],
        "tool_choice": "auto",
        "config": {},
        "output": {
            "model": "openai/gpt-4o",
            "choices": [
                {
                    "message": make_chat_message_assistant(rng),
                    "stop_reason": "stop",
                }
            ],
            "usage": make_model_usage(rng),
        },
    }


def make_event_tool(rng: random.Random) -> dict:
    return {
        "event": "tool",
        "timestamp": make_timestamp(rng),
        "working_start": rng.random() * 10,
        "type": "function",
        "id": f"call_{rng.randint(10000, 99999)}",
        "function": rng.choice(["web_search", "bash", "python"]),
        "arguments": {"query": lorem(rng, 5)},
        "result": lorem(rng, 15),
        "events": [],
    }


def make_event_state(rng: random.Random) -> dict:
    return {
        "event": "state",
        "timestamp": make_timestamp(rng),
        "working_start": rng.random() * 10,
        "changes": [
            {"op": "replace", "path": "/messages", "value": lorem(rng, 5), "from": None}
        ],
    }


def make_event_store(rng: random.Random) -> dict:
    return {
        "event": "store",
        "timestamp": make_timestamp(rng),
        "working_start": rng.random() * 10,
        "changes": [
            {"op": "add", "path": "/counter", "value": rng.randint(1, 100), "from": None}
        ],
    }


def make_event_info(rng: random.Random) -> dict:
    return {
        "event": "info",
        "timestamp": make_timestamp(rng),
        "working_start": rng.random() * 10,
        "data": {"message": lorem(rng, 10)},
    }


def make_events(rng: random.Random, n_events: int = 4) -> list[dict]:
    """Generate a mix of event types."""
    event_makers = [make_event_model, make_event_tool, make_event_state, make_event_store, make_event_info]
    events = []
    for _ in range(n_events):
        maker = rng.choice(event_makers)
        events.append(maker(rng))
    return events


# --------------------------------------------------------------------------- #
# Sample generation
# --------------------------------------------------------------------------- #

def make_sample(
    rng: random.Random,
    sample_id: int | str,
    epoch: int,
    scorer_name: str = "accuracy",
    include_attachments: bool = False,
    include_nan: bool = False,
) -> dict:
    """Generate a single EvalSample dict."""
    started = make_timestamp(rng)
    completed = make_timestamp(rng)
    n_turns = rng.randint(2, 8)
    messages = make_messages(rng, n_turns)

    sample: dict[str, Any] = {
        "id": sample_id,
        "epoch": epoch,
        "input": lorem(rng, rng.randint(10, 30)),
        "target": rng.choice(["A", "B", "C", "D"]),
        "messages": messages,
        "output": {
            "model": "openai/gpt-4o",
            "choices": [
                {
                    "message": make_chat_message_assistant(rng),
                    "stop_reason": "stop",
                }
            ],
            "usage": make_model_usage(rng),
        },
        "scores": {scorer_name: make_score(rng)},
        "metadata": {"difficulty": rng.choice(["easy", "medium", "hard"]), "category": lorem(rng, 2)},
        "store": {},
        "events": make_events(rng, rng.randint(2, 6)),
        "model_usage": {"openai/gpt-4o": make_model_usage(rng)},
        "started_at": started,
        "completed_at": completed,
        "total_time": rng.uniform(0.5, 30.0),
        "working_time": rng.uniform(0.1, 15.0),
        "uuid": f"sample-uuid-{sample_id}-{epoch}",
    }

    if include_attachments:
        attachment_key = f"attach_{rng.randint(1000, 9999)}"
        sample["attachments"] = {attachment_key: lorem_paragraph(rng, 5)}

    if include_nan:
        # Add NaN/Inf values in score metadata
        sample["scores"][scorer_name]["metadata"] = {
            "nan_value": float("nan"),
            "inf_value": float("inf"),
            "neg_inf_value": float("-inf"),
        }

    return sample


# --------------------------------------------------------------------------- #
# Full log generation
# --------------------------------------------------------------------------- #

def make_eval_spec(rng: random.Random, task_name: str = "test_task") -> dict:
    created = make_timestamp(rng)
    task_id = f"taskid_{rng.randint(100000, 999999)}"
    run_id = f"runid_{rng.randint(100000, 999999)}"
    return {
        "eval_id": f"evalid_{rng.randint(100000, 999999)}",
        "run_id": run_id,
        "created": created,
        "task": task_name,
        "task_id": task_id,
        "task_version": 1,
        "task_file": "tasks/test.py",
        "task_args": {},
        "task_args_passed": {},
        "dataset": {
            "name": "test_dataset",
            "location": "data/test.jsonl",
            "samples": 100,
        },
        "model": "openai/gpt-4o",
        "model_args": {},
        "model_generate_config": {},
        "config": {"epochs": 1, "log_samples": True},
        "packages": {"inspect_ai": "0.3.80"},
    }


def make_eval_plan() -> dict:
    return {
        "name": "plan",
        "steps": [
            {"solver": "chain_of_thought", "params": {}, "params_passed": {}},
            {"solver": "generate", "params": {"model": "openai/gpt-4o"}, "params_passed": {}},
        ],
        "config": {},
    }


def make_eval_results(
    rng: random.Random,
    total_samples: int,
    scorer_name: str = "accuracy",
) -> dict:
    accuracy_value = rng.uniform(0.3, 0.95)
    return {
        "total_samples": total_samples,
        "completed_samples": total_samples,
        "scores": [
            {
                "name": scorer_name,
                "scorer": scorer_name,
                "params": {},
                "metrics": {
                    "accuracy": {"name": "accuracy", "value": accuracy_value},
                    "stderr": {"name": "stderr", "value": rng.uniform(0.01, 0.1)},
                },
            }
        ],
    }


def make_eval_stats(rng: random.Random) -> dict:
    return {
        "started_at": make_timestamp(rng),
        "completed_at": make_timestamp(rng),
        "model_usage": {"openai/gpt-4o": make_model_usage(rng)},
    }


def generate_log(
    rng: random.Random,
    n_samples: int = 10,
    n_epochs: int = 1,
    status: str = "success",
    scorer_name: str = "accuracy",
    include_nan: bool = False,
    include_attachments: bool = False,
    task_name: str = "test_task",
) -> dict:
    """Generate a complete EvalLog dict."""
    eval_spec = make_eval_spec(rng, task_name)
    eval_spec["config"]["epochs"] = n_epochs
    eval_spec["dataset"]["samples"] = n_samples

    total_samples = n_samples * n_epochs
    samples = []
    sample_ids = list(range(1, n_samples + 1))

    for epoch in range(1, n_epochs + 1):
        for sid in sample_ids:
            samples.append(
                make_sample(
                    rng,
                    sid,
                    epoch,
                    scorer_name=scorer_name,
                    include_attachments=include_attachments and rng.random() < 0.3,
                    include_nan=include_nan and rng.random() < 0.2,
                )
            )

    eval_spec["dataset"]["sample_ids"] = sample_ids

    log: dict[str, Any] = {
        "version": 2,
        "status": status,
        "eval": eval_spec,
        "plan": make_eval_plan(),
        "results": make_eval_results(rng, total_samples, scorer_name) if status == "success" else None,
        "stats": make_eval_stats(rng),
    }

    if status == "error":
        log["error"] = {
            "message": "Test error: something went wrong",
            "traceback": "Traceback (most recent call last):\n  File ...\nRuntimeError: test error",
            "traceback_ansi": "Traceback...",
        }

    log["invalidated"] = False
    log["samples"] = samples if status != "cancelled" else None

    return log


# --------------------------------------------------------------------------- #
# File writing
# --------------------------------------------------------------------------- #

def custom_json_dumps(obj: Any) -> str:
    """JSON serializer that handles NaN/Inf like Python's json module."""
    return json.dumps(obj, allow_nan=True)


def write_json_log(log_dict: dict, output_path: str) -> None:
    """Write a .json format log file."""
    with open(output_path, "w") as f:
        f.write(custom_json_dumps(log_dict))


def write_eval_log(log_dict: dict, output_path: str) -> None:
    """Write an .eval format log (ZIP archive)."""
    eval_spec = log_dict["eval"]
    plan = log_dict["plan"]
    samples = log_dict.get("samples") or []
    status = log_dict["status"]
    stats = log_dict["stats"]
    results = log_dict.get("results")
    error = log_dict.get("error")

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # _journal/start.json
        start = {"version": log_dict["version"], "eval": eval_spec, "plan": plan}
        zf.writestr("_journal/start.json", custom_json_dumps(start))

        # Write samples
        summaries = []
        for sample in samples:
            filename = f"samples/{sample['id']}_epoch_{sample['epoch']}.json"
            zf.writestr(filename, custom_json_dumps(sample))

            # Build summary
            summaries.append({
                "id": sample["id"],
                "epoch": sample["epoch"],
                "input": sample["input"],
                "target": sample["target"],
                "metadata": sample.get("metadata", {}),
                "scores": sample.get("scores"),
                "model_usage": sample.get("model_usage", {}),
                "started_at": sample.get("started_at"),
                "completed_at": sample.get("completed_at"),
                "total_time": sample.get("total_time"),
                "working_time": sample.get("working_time"),
                "uuid": sample.get("uuid"),
                "completed": True,
                "message_count": len(sample.get("messages", [])),
            })

        # _journal/summaries/1.json
        if summaries:
            zf.writestr("_journal/summaries/1.json", custom_json_dumps(summaries))

        # summaries.json (consolidated)
        zf.writestr("summaries.json", custom_json_dumps(summaries))

        # header.json (complete EvalLog header without samples)
        header = {
            "version": log_dict["version"],
            "status": status,
            "eval": eval_spec,
            "plan": plan,
            "results": results,
            "stats": stats,
            "error": error,
            "invalidated": log_dict.get("invalidated", False),
        }
        zf.writestr("header.json", custom_json_dumps(header))

    with open(output_path, "wb") as f:
        f.write(buf.getvalue())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def generate_all_test_logs(output_dir: str = "test_logs", seed: int = 42) -> dict[str, str]:
    """Generate all test log files and return a mapping of description to path."""
    os.makedirs(output_dir, exist_ok=True)
    generated: dict[str, str] = {}

    # Standard logs with varying sample counts
    for n_samples in [10, 100, 1000]:
        rng = random.Random(seed + n_samples)
        log = generate_log(rng, n_samples=n_samples)

        json_path = os.path.join(output_dir, f"test_{n_samples}samples.json")
        eval_path = os.path.join(output_dir, f"test_{n_samples}samples.eval")
        write_json_log(log, json_path)
        write_eval_log(log, eval_path)
        generated[f"{n_samples}_samples_json"] = json_path
        generated[f"{n_samples}_samples_eval"] = eval_path

    # Multi-epoch log
    rng = random.Random(seed + 7777)
    log = generate_log(rng, n_samples=20, n_epochs=3)
    json_path = os.path.join(output_dir, "test_multiepoch.json")
    eval_path = os.path.join(output_dir, "test_multiepoch.eval")
    write_json_log(log, json_path)
    write_eval_log(log, eval_path)
    generated["multiepoch_json"] = json_path
    generated["multiepoch_eval"] = eval_path

    # Error status log
    rng = random.Random(seed + 8888)
    log = generate_log(rng, n_samples=5, status="error")
    json_path = os.path.join(output_dir, "test_error.json")
    eval_path = os.path.join(output_dir, "test_error.eval")
    write_json_log(log, json_path)
    write_eval_log(log, eval_path)
    generated["error_json"] = json_path
    generated["error_eval"] = eval_path

    # Cancelled status log (no samples)
    rng = random.Random(seed + 9999)
    log = generate_log(rng, n_samples=5, status="cancelled")
    json_path = os.path.join(output_dir, "test_cancelled.json")
    eval_path = os.path.join(output_dir, "test_cancelled.eval")
    write_json_log(log, json_path)
    write_eval_log(log, eval_path)
    generated["cancelled_json"] = json_path
    generated["cancelled_eval"] = eval_path

    # Log with NaN/Inf values
    rng = random.Random(seed + 1111)
    log = generate_log(rng, n_samples=10, include_nan=True)
    json_path = os.path.join(output_dir, "test_nan_inf.json")
    eval_path = os.path.join(output_dir, "test_nan_inf.eval")
    write_json_log(log, json_path)
    write_eval_log(log, eval_path)
    generated["nan_inf_json"] = json_path
    generated["nan_inf_eval"] = eval_path

    # Log with attachments
    rng = random.Random(seed + 2222)
    log = generate_log(rng, n_samples=15, include_attachments=True)
    json_path = os.path.join(output_dir, "test_attachments.json")
    eval_path = os.path.join(output_dir, "test_attachments.eval")
    write_json_log(log, json_path)
    write_eval_log(log, eval_path)
    generated["attachments_json"] = json_path
    generated["attachments_eval"] = eval_path

    # Empty samples list
    rng = random.Random(seed + 3333)
    log = generate_log(rng, n_samples=0)
    # Need to fixup: n_samples=0 produces empty samples, results.total_samples=0
    json_path = os.path.join(output_dir, "test_empty.json")
    eval_path = os.path.join(output_dir, "test_empty.eval")
    write_json_log(log, json_path)
    write_eval_log(log, eval_path)
    generated["empty_json"] = json_path
    generated["empty_eval"] = eval_path

    # Batch of logs for header reading tests
    for i in range(50):
        rng = random.Random(seed + 50000 + i)
        log = generate_log(rng, n_samples=5, task_name=f"batch_task_{i % 5}")
        eval_path = os.path.join(output_dir, f"batch_{i:03d}.eval")
        write_eval_log(log, eval_path)
        generated[f"batch_{i:03d}_eval"] = eval_path

    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test inspect log files")
    parser.add_argument("--output-dir", default="test_logs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generated = generate_all_test_logs(args.output_dir, args.seed)
    print(f"Generated {len(generated)} test log files in {args.output_dir}/")
    for desc, path in sorted(generated.items()):
        size = os.path.getsize(path)
        print(f"  {desc}: {path} ({size:,} bytes)")
