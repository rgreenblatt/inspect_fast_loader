"""Monkey-patching logic to replace inspect's log reading functions with fast implementations.

Currently contains skeleton/passthrough wrappers. The actual Rust-accelerated
implementations will be added in later segments.
"""

import inspect_ai.log._file as _file_module

# Store original functions so we can restore them
_originals: dict[str, object] = {}
_patched = False


def patch() -> None:
    """Replace inspect's log reading functions with fast implementations.

    Currently wraps the original functions (passthrough). In future segments,
    these will be replaced with Rust-accelerated implementations.
    """
    global _patched
    if _patched:
        return

    # Save originals
    _originals["read_eval_log"] = _file_module.read_eval_log
    _originals["read_eval_log_headers"] = _file_module.read_eval_log_headers

    # Install passthrough wrappers (for now)
    original_read = _originals["read_eval_log"]
    original_headers = _originals["read_eval_log_headers"]

    def fast_read_eval_log(*args, **kwargs):
        # Passthrough wrapper — will be replaced with Rust implementation
        return original_read(*args, **kwargs)

    def fast_read_eval_log_headers(*args, **kwargs):
        # Passthrough wrapper — will be replaced with Rust implementation
        return original_headers(*args, **kwargs)

    _file_module.read_eval_log = fast_read_eval_log
    _file_module.read_eval_log_headers = fast_read_eval_log_headers

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
