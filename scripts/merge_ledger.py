"""Merges a remote ledger.jsonl (e.g. downloaded after a Colab run) into the
local results/ledger.jsonl: appends only new, well-formed lines, rejecting
duplicates (by run_id, already-present lines are left untouched -- ledger
append-only discipline) and schema violations (missing required keys).

Usage:
    uv run python scripts/merge_ledger.py --remote /path/to/remote_ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_KEYS = {
    "run_id",
    "timestamp_utc",
    "git_commit",
    "git_dirty",
    "git_dirty_paths",
    "model_name",
    "dataset",
    "split",
    "seed",
    "config",
    "config_id",
    "protocol",
    "phase",
    "stage",
    "smoke",
    "train_fraction",
    "dataset_manifest_path",
    "dataset_split_hashes",
    "metrics",
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_record(record: dict) -> list[str]:
    """Returns schema-violation messages for `record` (empty list if valid)."""
    missing = REQUIRED_KEYS - set(record.keys())
    if missing:
        return [f"missing required keys: {sorted(missing)}"]
    return []


def merge(local_path: Path, remote_path: Path) -> dict:
    local_records = _read_jsonl(local_path)
    remote_records = _read_jsonl(remote_path)
    seen_run_ids = {r["run_id"] for r in local_records if "run_id" in r}

    accepted = []
    rejected_duplicate = []
    rejected_invalid = []

    for record in remote_records:
        errors = validate_record(record)
        if errors:
            rejected_invalid.append({"record": record, "errors": errors})
            continue
        if record["run_id"] in seen_run_ids:
            rejected_duplicate.append(record["run_id"])
            continue
        accepted.append(record)
        seen_run_ids.add(record["run_id"])

    if accepted:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "a", encoding="utf-8", newline="\n") as f:
            for record in accepted:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    return {
        "accepted": len(accepted),
        "rejected_duplicate": len(rejected_duplicate),
        "rejected_invalid": len(rejected_invalid),
        "rejected_invalid_details": rejected_invalid,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", type=Path, required=True)
    parser.add_argument("--local", type=Path, default=REPO_ROOT / "results" / "ledger.jsonl")
    args = parser.parse_args()
    result = merge(args.local, args.remote)
    print(
        f"Accepted: {result['accepted']}, duplicates skipped: {result['rejected_duplicate']}, "
        f"invalid rejected: {result['rejected_invalid']}"
    )
    for detail in result["rejected_invalid_details"]:
        print(f"  INVALID: {detail['errors']} -- record keys: {sorted(detail['record'].keys())}")
