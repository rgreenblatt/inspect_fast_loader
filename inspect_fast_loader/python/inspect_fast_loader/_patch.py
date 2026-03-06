"""Monkey-patching logic to replace inspect's log reading functions with fast implementations.

Currently contains skeleton/passthrough wrappers. The actual Rust-accelerated
implementations will be added in later segments.

Patches both sync and async variants:
- read_eval_log / read_eval_log_async
- read_eval_log_headers / read_eval_log_headers_async
"""

import functools
from typing import Any

import inspect_ai.log._file as _file_module

# Store original functions so we can restore them
_originals: dict[str, Any] = {}
_patched = False

# Names of all functions we patch (sync and async)
_PATCHED_SYNC = ["read_eval_log", "read_eval_log_headers"]
_PATCHED_ASYNC = ["read_eval_log_async", "read_eval_log_headers_async"]
_PATCHED_ALL = _PATCHED_SYNC + _PATCHED_ASYNC


def _wrap_sync(original: Any) -> Any:
    """Create a sync passthrough wrapper."""

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs)

    wrapper._is_fast_loader_wrapper = True  # type: ignore[attr-defined]
    wrapper._original = original  # type: ignore[attr-defined]
    return wrapper


def _wrap_async(original: Any) -> Any:
    """Create an async passthrough wrapper."""

    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await original(*args, **kwargs)

    wrapper._is_fast_loader_wrapper = True  # type: ignore[attr-defined]
    wrapper._original = original  # type: ignore[attr-defined]
    return wrapper


def patch() -> None:
    """Replace inspect's log reading functions with fast implementations.

    Patches both sync and async variants. Currently wraps the original
    functions (passthrough). In future segments, these will be replaced
    with Rust-accelerated implementations.
    """
    global _patched
    if _patched:
        return

    # Save originals and install wrappers
    for name in _PATCHED_SYNC:
        original = getattr(_file_module, name)
        _originals[name] = original
        setattr(_file_module, name, _wrap_sync(original))

    for name in _PATCHED_ASYNC:
        original = getattr(_file_module, name)
        _originals[name] = original
        setattr(_file_module, name, _wrap_async(original))

    _patched = True


def unpatch() -> None:
    """Restore inspect's original log reading functions."""
    global _patched
    if not _patched:
        return

    for name, original in _originals.items():
        setattr(_file_module, name, original)

    _originals.clear()
    _patched = False


def is_patched() -> bool:
    """Check if patches are currently applied."""
    return _patched
