"""Shared test utilities for deep comparison of log data structures."""

import math


def approx_equal(a, b, rel_tol=1e-9, abs_tol=1e-12):
    """Compare floats with tolerance, handling NaN and Inf."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isinf(a) and math.isinf(b):
            return (a > 0 and b > 0) or (a < 0 and b < 0)
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    return False


def deep_compare(orig, fast, path_str="root"):
    """Recursively compare two values, collecting differences as a list of strings.

    Returns an empty list if values match, or a list of human-readable
    difference descriptions otherwise.
    """
    diffs = []

    if orig is None and fast is None:
        return diffs
    if orig is None or fast is None:
        diffs.append(f"{path_str}: one is None (orig={orig is None}, fast={fast is None})")
        return diffs

    if isinstance(orig, float) and isinstance(fast, float):
        if not approx_equal(orig, fast):
            diffs.append(f"{path_str}: float mismatch orig={orig} fast={fast}")
        return diffs

    if type(orig) != type(fast):
        if isinstance(orig, (int, float)) and isinstance(fast, (int, float)):
            if not approx_equal(float(orig), float(fast)):
                diffs.append(
                    f"{path_str}: numeric mismatch orig={orig} ({type(orig).__name__}) "
                    f"fast={fast} ({type(fast).__name__})"
                )
        else:
            diffs.append(f"{path_str}: type mismatch {type(orig).__name__} vs {type(fast).__name__}")
        return diffs

    if isinstance(orig, dict):
        all_keys = set(orig.keys()) | set(fast.keys())
        for k in sorted(all_keys):
            if k not in orig:
                diffs.append(f"{path_str}.{k}: missing in original")
            elif k not in fast:
                diffs.append(f"{path_str}.{k}: missing in fast")
            else:
                diffs.extend(deep_compare(orig[k], fast[k], f"{path_str}.{k}"))
        return diffs

    if isinstance(orig, (list, tuple)):
        if len(orig) != len(fast):
            diffs.append(f"{path_str}: length mismatch orig={len(orig)} fast={len(fast)}")
            return diffs
        for i in range(len(orig)):
            diffs.extend(deep_compare(orig[i], fast[i], f"{path_str}[{i}]"))
        return diffs

    if isinstance(orig, str):
        if orig != fast:
            diffs.append(f"{path_str}: string mismatch orig={orig!r:.100} fast={fast!r:.100}")
        return diffs

    if isinstance(orig, (int, bool)):
        if orig != fast:
            diffs.append(f"{path_str}: value mismatch orig={orig} fast={fast}")
        return diffs

    # For Pydantic models and other complex objects, compare their dict representations
    if hasattr(orig, "model_dump"):
        return deep_compare(orig.model_dump(), fast.model_dump(), path_str)

    if orig != fast:
        diffs.append(f"{path_str}: value mismatch orig={orig!r:.100} fast={fast!r:.100}")
    return diffs


def assert_logs_equal(orig, fast, ignore_location=True):
    """Compare two EvalLog objects field by field, raising AssertionError on mismatch."""
    orig_dict = orig.model_dump()
    fast_dict = fast.model_dump()

    if ignore_location:
        orig_dict.pop("location", None)
        fast_dict.pop("location", None)

    diffs = deep_compare(orig_dict, fast_dict)
    if diffs:
        msg = f"Found {len(diffs)} differences:\n" + "\n".join(f"  - {d}" for d in diffs[:30])
        if len(diffs) > 30:
            msg += f"\n  ... and {len(diffs) - 30} more"
        raise AssertionError(msg)
