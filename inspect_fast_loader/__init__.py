"""inspect_fast_loader: High-performance log loading for inspect_ai.

The speedup comes primarily from bypassing Pydantic model_validate when
constructing EvalSample objects (pure Python). An optional Rust native
extension provides faster ZIP decompression (~2x) for .eval files.

Usage::

    import inspect_fast_loader

    inspect_fast_loader.patch()   # monkey-patches inspect_ai.log._file functions
    # ... use inspect_ai as normal — log reads are now accelerated ...
    inspect_fast_loader.unpatch() # restore originals (optional)
"""

import warnings

from inspect_fast_loader._patch import is_patched, patch, unpatch
from inspect_fast_loader._zip import HAS_NATIVE

# Re-export ZIP reading functions
from inspect_fast_loader._zip import (
    read_eval_file,
    read_eval_headers_batch,
    read_eval_sample,
    read_eval_summaries,
)

# Re-export low-level ZIP utilities if native extension is available
try:
    from inspect_fast_loader._native import list_zip_entries, read_zip_member
except ImportError:
    pass

# Warn if the installed inspect_ai version differs from what we tested against.
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
    "read_eval_file",
    "read_eval_headers_batch",
    "read_eval_sample",
    "read_eval_summaries",
    "patch",
    "unpatch",
    "is_patched",
    "HAS_NATIVE",
]
