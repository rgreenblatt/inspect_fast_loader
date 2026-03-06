"""inspect_fast_loader: High-performance Rust-based log loading for inspect_ai.

Usage::

    import inspect_fast_loader

    inspect_fast_loader.patch()   # monkey-patches inspect_ai.log._file functions
    # ... use inspect_ai as normal — log reads are now Rust-accelerated ...
    inspect_fast_loader.unpatch() # restore originals (optional)
"""

import warnings

from inspect_fast_loader._native import (
    list_zip_entries,
    parse_json_bytes,
    read_eval_file,
    read_eval_headers_batch,
    read_eval_sample,
    read_eval_summaries,
    read_json_file,
    read_zip_member,
)
from inspect_fast_loader._patch import is_patched, patch, unpatch

# Warn if the installed inspect_ai version differs from what we tested against.
# The Pydantic bypass in _construct.py hard-codes model field knowledge.
_TESTED_INSPECT_VERSION = "0.3.188"
try:
    import inspect_ai as _inspect_ai
    _installed = getattr(_inspect_ai, "__version__", None)
    if _installed and _installed != _TESTED_INSPECT_VERSION:
        warnings.warn(
            f"inspect_fast_loader was tested against inspect_ai {_TESTED_INSPECT_VERSION}, "
            f"but {_installed} is installed. The Pydantic bypass may produce incorrect results "
            f"if inspect_ai's model types have changed. Run the test suite to verify correctness.",
            UserWarning,
            stacklevel=2,
        )
    del _inspect_ai, _installed
except ImportError:
    pass

__all__ = [
    "parse_json_bytes",
    "read_json_file",
    "read_eval_file",
    "read_eval_headers_batch",
    "read_eval_sample",
    "read_eval_summaries",
    "list_zip_entries",
    "read_zip_member",
    "patch",
    "unpatch",
    "is_patched",
]
