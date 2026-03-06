"""Test inspect_fast_loader against real inspect evaluation logs using a real model."""

import os
import sys
import time
import tempfile

ANTHROPIC_API_KEY = open(os.path.expanduser("~/.anthropic_api_key")).read().strip()
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

LOG_DIR = os.path.join(tempfile.mkdtemp(), "real_logs")
os.environ["INSPECT_LOG_DIR"] = LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)

from inspect_ai import Task, eval
from inspect_ai.dataset import example_dataset, Sample, MemoryDataset
from inspect_ai.scorer import match, includes
from inspect_ai.solver import generate, system_message, chain_of_thought
from inspect_ai.log import read_eval_log, EvalLog
from inspect_ai.log._file import read_eval_log_headers
import inspect_fast_loader

MODEL = "anthropic/claude-haiku-4-5-20251001"


def compare_logs(original: EvalLog, fast: EvalLog, label: str) -> bool:
    """Deep compare two EvalLog objects for equivalence."""
    errors = []

    if original.status != fast.status:
        errors.append(f"status: {original.status} vs {fast.status}")
    if original.eval.task != fast.eval.task:
        errors.append(f"task: {original.eval.task} vs {fast.eval.task}")
    if original.eval.model != fast.eval.model:
        errors.append(f"model: {original.eval.model} vs {fast.eval.model}")

    # Compare results
    if original.results and fast.results:
        if original.results.scores != fast.results.scores:
            errors.append("scores differ")
    elif (original.results is None) != (fast.results is None):
        errors.append("results: one is None")

    # Compare samples
    orig_samples = original.samples or []
    fast_samples = fast.samples or []
    if len(orig_samples) != len(fast_samples):
        errors.append(f"sample count: {len(orig_samples)} vs {len(fast_samples)}")
    else:
        for i, (os_, fs) in enumerate(zip(orig_samples, fast_samples)):
            if os_.id != fs.id:
                errors.append(f"sample[{i}].id: {os_.id} vs {fs.id}")
            if os_.epoch != fs.epoch:
                errors.append(f"sample[{i}].epoch: {os_.epoch} vs {fs.epoch}")
            if len(os_.input) != len(fs.input):
                errors.append(f"sample[{i}].input length: {len(os_.input)} vs {len(fs.input)}")
            if len(os_.messages) != len(fs.messages):
                errors.append(f"sample[{i}].messages length: {len(os_.messages)} vs {len(fs.messages)}")
            else:
                for j, (om, fm) in enumerate(zip(os_.messages, fs.messages)):
                    if om.role != fm.role:
                        errors.append(f"sample[{i}].messages[{j}].role: {om.role} vs {fm.role}")
                    if om.content != fm.content:
                        errors.append(f"sample[{i}].messages[{j}].content differs")
            if os_.scores != fs.scores:
                errors.append(f"sample[{i}].scores differ")
            if os_.model_usage != fs.model_usage:
                errors.append(f"sample[{i}].model_usage differs")

    if errors:
        print(f"  FAIL [{label}]: {len(errors)} differences found:")
        for e in errors[:10]:
            print(f"    - {e}")
        return False
    else:
        print(f"  PASS [{label}]: logs match")
        return True


def run_eval_and_test(task: Task, task_name: str, limit: int | None = None) -> tuple[bool, str]:
    """Run an eval, then compare standard vs fast log reading."""
    print(f"\n{'='*60}")
    print(f"Running eval: {task_name} (model={MODEL}, limit={limit})")
    print(f"{'='*60}")

    results = eval(task, model=MODEL, log_dir=LOG_DIR, limit=limit)
    log_file = results[0].location
    assert log_file is not None
    print(f"  Log file: {os.path.basename(log_file)}")
    print(f"  Status: {results[0].status}")
    n_samples = len(results[0].samples or [])
    print(f"  Samples: {n_samples}")

    # Read with standard loader
    print("\n  Reading with standard loader...")
    t0 = time.perf_counter()
    standard_log = read_eval_log(log_file)
    t_standard = time.perf_counter() - t0
    print(f"  Standard read: {t_standard*1000:.1f}ms")

    # Patch and read with fast loader
    inspect_fast_loader.patch()
    print("  Reading with fast loader...")
    t0 = time.perf_counter()
    fast_log = read_eval_log(log_file)
    t_fast = time.perf_counter() - t0
    print(f"  Fast read: {t_fast*1000:.1f}ms")
    inspect_fast_loader.unpatch()

    ok = compare_logs(standard_log, fast_log, task_name)

    if t_fast > 0:
        speedup = t_standard / t_fast
        print(f"  Speedup: {speedup:.1f}x")

    return ok, log_file


def main():
    print(f"Log directory: {LOG_DIR}")
    print(f"Model: {MODEL}")
    print(f"inspect_ai version: {__import__('inspect_ai').__version__}")

    all_ok = True
    log_files = []

    # --- Test 1: Theory of mind with match scorer (5 samples) ---
    task1 = Task(
        dataset=example_dataset("theory_of_mind"),
        solver=[generate()],
        scorer=match(),
    )
    ok, lf = run_eval_and_test(task1, "theory_of_mind/match", limit=5)
    all_ok &= ok
    log_files.append(lf)

    # --- Test 2: Theory of mind with chain_of_thought (3 samples) ---
    task2 = Task(
        dataset=example_dataset("theory_of_mind"),
        solver=[
            system_message("You are a helpful assistant. Think step by step."),
            chain_of_thought(),
            generate(),
        ],
        scorer=match(),
    )
    ok, lf = run_eval_and_test(task2, "theory_of_mind/cot", limit=3)
    all_ok &= ok
    log_files.append(lf)

    # --- Test 3: Custom dataset with metadata (5 samples) ---
    custom_samples = [
        Sample(
            input="What is 2+2?",
            target="4",
            id="math_0",
            metadata={"difficulty": "easy", "topic": "arithmetic"},
        ),
        Sample(
            input="What is the capital of France?",
            target="Paris",
            id="geo_1",
            metadata={"difficulty": "easy", "topic": "geography"},
        ),
        Sample(
            input=[
                {"role": "system", "content": "You are a math tutor. Answer with just the number."},
                {"role": "user", "content": "What is 7*8?"},
            ],
            target="56",
            id="math_2",
            metadata={"difficulty": "medium", "topic": "arithmetic"},
        ),
        Sample(
            input="Name a primary color.",
            target=["red", "blue", "yellow"],
            id="misc_3",
        ),
        Sample(
            input="What programming language uses .py files?",
            target="Python",
            id="cs_4",
            metadata={"topic": "cs"},
        ),
    ]
    task3 = Task(
        dataset=MemoryDataset(custom_samples, name="custom_mixed"),
        solver=[generate()],
        scorer=includes(),
    )
    ok, lf = run_eval_and_test(task3, "custom_mixed/includes")
    all_ok &= ok
    log_files.append(lf)

    # --- Test 4: Multiple epochs (2 samples, 2 epochs) ---
    task4 = Task(
        dataset=example_dataset("theory_of_mind"),
        solver=[generate()],
        scorer=match(),
        epochs=2,
    )
    ok, lf = run_eval_and_test(task4, "theory_of_mind/2_epochs", limit=2)
    all_ok &= ok
    log_files.append(lf)

    # --- Test 5: Batch header reading ---
    print(f"\n{'='*60}")
    print(f"Testing batch header reading ({len(log_files)} files)")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    standard_headers = read_eval_log_headers(log_files)
    t_standard = time.perf_counter() - t0
    print(f"  Standard: {t_standard*1000:.1f}ms")

    inspect_fast_loader.patch()
    t0 = time.perf_counter()
    fast_headers = read_eval_log_headers(log_files)
    t_fast = time.perf_counter() - t0
    print(f"  Fast: {t_fast*1000:.1f}ms")
    inspect_fast_loader.unpatch()

    assert len(standard_headers) == len(fast_headers)
    header_ok = True
    for i, (sh, fh) in enumerate(zip(standard_headers, fast_headers)):
        if sh.status != fh.status or sh.eval.task != fh.eval.task:
            print(f"  FAIL: header[{i}] mismatch")
            header_ok = False
    if header_ok:
        print(f"  PASS: all headers match")
    all_ok &= header_ok

    # --- Test 6: header_only reads ---
    print(f"\n{'='*60}")
    print("Testing header_only reads")
    print(f"{'='*60}")
    for lf in log_files:
        standard = read_eval_log(lf, header_only=True)
        inspect_fast_loader.patch()
        fast = read_eval_log(lf, header_only=True)
        inspect_fast_loader.unpatch()

        if standard.eval.task != fast.eval.task or standard.status != fast.status:
            print(f"  FAIL: {os.path.basename(lf)}")
            all_ok = False
        else:
            print(f"  PASS: {os.path.basename(lf)}")

    # --- Summary ---
    print(f"\n{'='*60}")
    if all_ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print(f"{'='*60}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
