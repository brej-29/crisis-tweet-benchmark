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


def is_git_dirty(repo_root: str | Path) -> bool:
    """True if there are uncommitted changes in the working tree."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def append_run_record(ledger_path: str | Path, record: dict) -> None:
    """Append one JSON line to the ledger. Creates the file/parents if needed."""
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True)
    with open(ledger_path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def read_ledger(ledger_path: str | Path) -> list[dict]:
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return []
    records = []
    with open(ledger_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records