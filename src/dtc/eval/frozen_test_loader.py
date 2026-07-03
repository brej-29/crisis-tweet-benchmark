"""The ONLY sanctioned way to read a frozen test split.

Standing rule (docs/PLAN.md): frozen test sets are read by evaluation code
paths only, never by training/selection code. This module enforces that at
call time by inspecting the caller's module name and file path, and raises
`FrozenTestAccessError` if the caller is not an allowed evaluation
entrypoint. `tests/test_frozen_test_guard.py` additionally statically scans
the training/data-preparation source tree to make sure nothing there even
attempts to import this module.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd

# Callers are allowed if either:
#   - their module name is `dtc.eval` or a submodule of it, or
#   - their source file lives directly under a `scripts/` directory and its
#     filename starts with "evaluate" (e.g. scripts/evaluate_lstm.py), which
#     covers direct `python scripts/evaluate_xxx.py` invocation (__name__ == "__main__").
_ALLOWED_MODULE_PREFIXES = ("dtc.eval",)


class FrozenTestAccessError(RuntimeError):
    """Raised when non-evaluation code attempts to read a frozen test split."""


def _is_allowed(caller_module: str, caller_file: str) -> bool:
    if caller_module == "dtc.eval" or caller_module.startswith("dtc.eval."):
        return True
    caller_path = Path(caller_file)
    if caller_path.parent.name == "scripts" and caller_path.name.startswith("evaluate"):
        return True
    return False


def _resolve_caller(caller_module: str | None, caller_file: str | None) -> tuple[str, str]:
    if caller_module is not None and caller_file is not None:
        return caller_module, caller_file
    frame_info = inspect.stack()[2]  # skip this function's frame and load_frozen_test's frame
    module_name = frame_info.frame.f_globals.get("__name__", "")
    file_name = frame_info.filename
    return module_name, file_name


def load_frozen_test(
    csv_path: str | Path,
    *,
    _caller_module: str | None = None,
    _caller_file: str | None = None,
) -> pd.DataFrame:
    """Load a frozen test split CSV.

    Raises FrozenTestAccessError unless called from `dtc.eval.*` or a
    `scripts/evaluate*.py` entrypoint.
    """
    caller_module, caller_file = _resolve_caller(_caller_module, _caller_file)
    if not _is_allowed(caller_module, caller_file):
        raise FrozenTestAccessError(
            f"load_frozen_test() was called from module '{caller_module}' ({caller_file}), "
            "which is not an allowed evaluation entrypoint. Frozen test splits may only be "
            "read from dtc.eval.* modules or scripts/evaluate_*.py entrypoints."
        )
    return pd.read_csv(csv_path)
