"""inspect_fast_loader: High-performance Rust-based log loading for inspect_ai."""

from inspect_fast_loader._native import (
    list_zip_entries,
    parse_json_bytes,
    read_eval_file,
    read_json_file,
    read_zip_member,
)
from inspect_fast_loader._patch import is_patched, patch, unpatch

__all__ = [
    "parse_json_bytes",
    "read_json_file",
    "read_eval_file",
    "list_zip_entries",
    "read_zip_member",
    "patch",
    "unpatch",
    "is_patched",
]
