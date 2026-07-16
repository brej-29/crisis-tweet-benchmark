"""Append-only results ledger.

Every run appends exactly one JSON line to results/ledger.jsonl. There is
deliberately no "update" or "delete" function in this module -- the only
way to affect the ledger's content is to append to it, which is the
enforcement mechanism for the plan's "ledger lines are append-only (no
rewrite of history)" standing rule.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path


def generate_run_id() -> str:
    return uuid.uuid4().hex


def get_git_commit_hash(repo_root: str | Path) -> str:
    """Returns the full commit hash of HEAD, or 'unknown' if not resolvable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


_LEDGER_RELPATH = "results/ledger.jsonl"


def get_git_dirty_paths(repo_root: str | Path) -> list[str]:
    """Returns the list of dirty paths from `git status --porcelain`, forward-slash
    normalized. Empty list if the tree is clean or git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    paths = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ")[-1]
        paths.append(path.strip('"').replace("\\", "/"))
    return paths


def is_git_dirty(repo_root: str | Path) -> bool:
    """True if there are uncommitted changes other than the ledger file itself.

    The ledger dirties itself mid-invocation (appending a line to
    results/ledger.jsonl makes the tree "dirty" by the time the next run in the
    same script invocation checks) -- see docs/DECISIONS.md and
    PHASE0_REPORT.md sec. 5. Excluding that one path keeps this flag meaningful
    as "did source/config change", not "did the ledger get appended to."
    """
    paths = get_git_dirty_paths(repo_root)
    return any(p != _LEDGER_RELPATH for p in paths)


def append_run_record(ledger_path: str | Path, record: dict) -> None:
    """Append one JSON line to the ledger. Creates the file/parents if needed."""
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True)
    with open(ledger_path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def _backfill_dataset_fields(record: dict) -> dict:
    """Read-time-only backfill of Phase 2's `train_dataset`/`eval_dataset`
    fields on records written before they existed. Phase-1 records all
    trained AND evaluated on the record's own `dataset` (kaggle), so both
    default to it. This never touches the file -- the ledger is append-only,
    and old lines are never rewritten (docs/DECISIONS.md).
    """
    default = record.get("dataset") or "kaggle"
    if "train_dataset" not in record:
        record["train_dataset"] = default
    if "eval_dataset" not in record:
        record["eval_dataset"] = default
    return record


def read_ledger(ledger_path: str | Path) -> list[dict]:
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return []
    records = []
    with open(ledger_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(_backfill_dataset_fields(json.loads(line)))
    return records