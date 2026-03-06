import atexit
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
from collections import defaultdict


def _get_git_commit() -> str | None:
    """Get current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# Pricing per million tokens (input, output) - updated Jan 2026
# For Anthropic extended thinking: output_tokens includes thinking tokens, billed at output rate
# Note, will be a bit off on long context (>1 million tokens) pricing for Opus, but probably doesn't matter that much
MODEL_PRICING_PER_MILLION = {
    # Anthropic models
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5-20251101": (5.0, 25.0),
    "claude-opus-4-1-20250805": (15.0, 75.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-3-7-sonnet-20250219": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-3-5-haiku-20241022": (0.80, 4.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
    # OpenAI models
    "davinci-002": (2.0, 2.0),
    "gpt-3.5-turbo-0125": (0.50, 1.50),
    "gpt-4-0314": (30.0, 60.0),
    "gpt-4-0613": (30.0, 60.0),
    "gpt-4-0125-preview": (10.0, 30.0),
    "gpt-4-turbo-2024-04-09": (10.0, 30.0),
    "gpt-4-1106-preview": (10.0, 30.0),
    "gpt-4o-2024-05-13": (2.50, 10.0),
    "gpt-4o-2024-08-06": (2.50, 10.0),
    "gpt-4.1-2025-04-14": (2.0, 8.0),
    "gpt-5-2025-08-07": (1.25, 10.0),
    "gpt-5.1-2025-11-13": (1.25, 10.0),
    "gpt-5.2-2025-12-11": (1.75, 14.0),
    # OpenRouter models (DeepSeek)
    "deepseek/deepseek-chat-v3-0324": (0.24, 0.38),
    "deepseek/deepseek-v3.2": (0.24, 0.38),
    # OpenRouter models (Qwen)
    "qwen/qwen3-235b-a22b": (0.18, 0.54),
    "qwen/qwen3-235b-a22b-2507": (0.18, 0.54),
    "qwen/qwen3-coder": (0.20, 0.60),
    "qwen/qwen3-32b": (0.08, 0.24),
    # OpenRouter models (Moonshot)
    "moonshotai/kimi-k2": (0.40, 2.0),
    # OpenRouter models (Google Gemini)
    "google/gemini-2.5-pro": (1.25, 10.0),
    "google/gemini-2.5-pro-preview": (1.25, 10.0),
    "google/gemini-3-pro-preview": (2.0, 12.0),
}

# Modal pricing (per second) — as of Feb 2026, https://modal.com/pricing
MODAL_CPU_PER_CORE_PER_SEC = {
    "sandbox": 0.00003942,
    "standard": 0.0000131,
}
MODAL_MEM_PER_GIB_PER_SEC = {
    "sandbox": 0.00000672,
    "standard": 0.00000222,
}
MODAL_GPU_PER_SEC = {
    "B200": 0.001736,
    "H200": 0.001261,
    "H100": 0.001097,
    "A100-80GB": 0.000694,
    "A100": 0.000583,  # 40GB
    "L40S": 0.000542,
    "A10G": 0.000306,
    "L4": 0.000222,
    "T4": 0.000164,
}


class ModalTimer:
    """Yielded by track_modal context managers. Set .elapsed to override wall-clock timing."""
    elapsed: float | None = None


class CostTracker:
    def __init__(self, cost_file: Path, run_description: str | None = None, **save_kwargs):
        self.cost_file = cost_file
        self.cost_file.parent.mkdir(parents=True, exist_ok=True)
        self.run_description = run_description
        self.save_kwargs = save_kwargs
        self.run_cost = 0.0
        self.start_time = datetime.now(timezone.utc)
        self.git_commit = _get_git_commit()
        self.warned = set()
        self.run_cost_by_model = defaultdict(float)
        self.run_cost_by_model_input_output = defaultdict(lambda: defaultdict(float))
        self.modal_compute_cost = 0.0
        self.prior_cumulative_cost = self._load_cumulative_cost()
        atexit.register(self._save_on_exit)

    def _load_cumulative_cost(self) -> float:
        """Read total_cost.jsonl and sum all run_cost values."""
        if not self.cost_file.exists():
            self.cost_file.touch()
            return 0.0
        cumulative = 0.0
        with open(self.cost_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse line in cost file: {line}")
                        continue
                    cumulative += entry.get("run_cost", 0.0)
        return cumulative

    def add_llm_api_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Add cost for an API call. Returns cost for this call."""
        if model not in MODEL_PRICING_PER_MILLION and model not in self.warned:
            print(f"Warning: Model {model} not in pricing list. Using default Sonnet pricing.")
            self.warned.add(model)
        input_price, output_price = MODEL_PRICING_PER_MILLION.get(model, (3.0, 15.0))  # Default to Sonnet pricing
        input_cost = input_tokens * input_price / 1_000_000
        output_cost = output_tokens * output_price / 1_000_000
        cost = input_cost + output_cost
        self.run_cost_by_model[model] += cost
        self.run_cost_by_model_input_output[model]["input"] += input_cost
        self.run_cost_by_model_input_output[model]["output"] += output_cost
        self.add_cost(cost)
        return cost

    def add_modal_gpu_cost(
        self, wall_seconds: float, gpu: str, gpu_count: int = 1, cpu: float = 0, memory_mib: int = 0, is_sandbox: bool = True,
    ) -> float:
        """Add cost for a Modal GPU run. CPU/memory default to 0 since GPU cost dominates."""
        return self.add_modal_cost(wall_seconds, cpu=cpu, memory_mib=memory_mib, is_sandbox=is_sandbox, gpu=gpu, gpu_count=gpu_count)

    def add_modal_cost(
        self, wall_seconds: float, cpu: float, memory_mib: int, is_sandbox: bool = True, gpu: str | None = None, gpu_count: int = 1,
    ) -> float:
        """Add cost for a Modal run. Returns the computed cost.

        gpu: GPU type string matching MODAL_GPU_PER_SEC keys (e.g. "H100", "H200", "B200").
        gpu_count: number of GPUs (default 1, ignored if gpu is None).
        """
        tier = "sandbox" if is_sandbox else "standard"
        cost = (
            cpu * wall_seconds * MODAL_CPU_PER_CORE_PER_SEC[tier]
            + (memory_mib / 1024) * wall_seconds * MODAL_MEM_PER_GIB_PER_SEC[tier]
        )
        if gpu is not None:
            assert gpu.upper() in MODAL_GPU_PER_SEC, f"Unknown GPU type {gpu!r}. Known: {list(MODAL_GPU_PER_SEC)}"
            cost += gpu_count * wall_seconds * MODAL_GPU_PER_SEC[gpu.upper()]
        self.modal_compute_cost += cost
        self.add_cost(cost)
        return cost

    @contextmanager
    def track_modal_gpu(self, gpu: str, gpu_count: int = 1, cpu: float = 0, memory_mib: int = 0, is_sandbox: bool = True):
        """Context manager for Modal GPU runs. CPU/memory default to 0 since GPU cost dominates."""
        with self.track_modal(cpu=cpu, memory_mib=memory_mib, is_sandbox=is_sandbox, gpu=gpu, gpu_count=gpu_count) as t:
            yield t

    @contextmanager
    def track_modal(self, cpu: float, memory_mib: int, is_sandbox: bool = True, gpu: str | None = None, gpu_count: int = 1):
        """Context manager that times a block and records Modal compute cost.

        Yields a ModalTimer. Set t.elapsed = <seconds> from inside the container
        for accurate timing (excludes scheduling wait). Falls back to wall-clock if not set.
        """
        t = ModalTimer()
        start = time.monotonic()
        try:
            yield t
        finally:
            elapsed = t.elapsed if t.elapsed is not None else (time.monotonic() - start)
            self.add_modal_cost(elapsed, cpu=cpu, memory_mib=memory_mib, is_sandbox=is_sandbox, gpu=gpu, gpu_count=gpu_count)

    def add_cost(self, cost: float):
        # could add hook here if desired
        self.run_cost += cost

    def total_cost(self) -> float:
        return self._load_cumulative_cost() + self.run_cost

    def get_token_usage_budget(self, default: float = 5000.0) -> float:
        budget_file = self.cost_file.parent / ".token_usage_budget"
        if not budget_file.exists():
            return default
        try:
            return float(budget_file.read_text().strip())
        except ValueError:
            print(f"Warning: .token_usage_budget exists but couldn't parse as float. Using default {default}.")
            return default

    def get_api_usage_budget(self, default: float = 5000.0) -> float:
        budget_file = self.cost_file.parent / ".api_usage_budget"
        if not budget_file.exists():
            return default
        try:
            return float(budget_file.read_text().strip())
        except ValueError:
            print(f"Warning: .api_usage_budget exists but couldn't parse as float. Using default {default}.")
            return default

    def is_over_budget(self) -> bool:
        budget = self.get_api_usage_budget()
        total = self.total_cost()
        return total > budget


    def _save_on_exit(self):
        """Append run summary to cost file on exit."""
        if self.run_cost == 0:
            return  # Don't log runs with no API calls
        entry = {
            "start_timestamp": self.start_time.isoformat(),
            "end_timestamp": datetime.now(timezone.utc).isoformat(),
            "script": sys.argv[0],
            "full_args": sys.argv,
            "git_commit": self.git_commit,
            "description": self.run_description,
            **self.save_kwargs,
            "run_cost": self.run_cost,
            "modal_compute_cost": self.modal_compute_cost,
            "run_cost_by_model": self.run_cost_by_model,
            "run_cost_by_model_input_output": self.run_cost_by_model_input_output,
        }
        with open(self.cost_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
