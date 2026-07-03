"""Tests for scripts/merge_ledger.py: dedup by run_id, schema validation,
and append-only behavior against the local ledger."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_merge_ledger_module():
    spec = importlib.util.spec_from_file_location("merge_ledger", REPO_ROOT / "scripts" / "merge_ledger.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["merge_ledger"] = module
    spec.loader.exec_module(module)
    return module


def _valid_record(run_id: str) -> dict:
    return {k: None for k in [
        "run_id", "timestamp_utc", "git_commit", "git_dirty", "git_dirty_paths",
        "model_name", "dataset", "split", "seed", "config", "config_id",
        "protocol", "phase", "stage", "smoke", "train_fraction",
        "dataset_manifest_path", "dataset_split_hashes", "metrics",
    ]} | {"run_id": run_id}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_validate_record_flags_missing_keys():
    module = _load_merge_ledger_module()
    errors = module.validate_record({"run_id": "abc"})
    assert errors  # non-empty: missing keys reported


def test_validate_record_accepts_well_formed_record():
    module = _load_merge_ledger_module()
    assert module.validate_record(_valid_record("abc")) == []


def test_merge_appends_new_valid_records(tmp_path):
    module = _load_merge_ledger_module()
    local_path = tmp_path / "local.jsonl"
    remote_path = tmp_path / "remote.jsonl"
    _write_jsonl(local_path, [_valid_record("existing-1")])
    _write_jsonl(remote_path, [_valid_record("new-1"), _valid_record("new-2")])

    result = module.merge(local_path, remote_path)
    assert result["accepted"] == 2
    assert result["rejected_duplicate"] == 0
    assert result["rejected_invalid"] == 0

    from dtc.harness.ledger import read_ledger

    merged = read_ledger(local_path)
    assert {r["run_id"] for r in merged} == {"existing-1", "new-1", "new-2"}


def test_merge_rejects_duplicate_run_ids(tmp_path):
    module = _load_merge_ledger_module()
    local_path = tmp_path / "local.jsonl"
    remote_path = tmp_path / "remote.jsonl"
    _write_jsonl(local_path, [_valid_record("dup-1")])
    _write_jsonl(remote_path, [_valid_record("dup-1"), _valid_record("new-1")])

    result = module.merge(local_path, remote_path)
    assert result["accepted"] == 1
    assert result["rejected_duplicate"] == 1

    from dtc.harness.ledger import read_ledger

    merged = read_ledger(local_path)
    assert len(merged) == 2  # original dup-1 kept once, new-1 appended


def test_merge_rejects_schema_violations(tmp_path):
    module = _load_merge_ledger_module()
    local_path = tmp_path / "local.jsonl"
    remote_path = tmp_path / "remote.jsonl"
    _write_jsonl(local_path, [])
    _write_jsonl(remote_path, [{"run_id": "broken"}, _valid_record("good-1")])

    result = module.merge(local_path, remote_path)
    assert result["accepted"] == 1
    assert result["rejected_invalid"] == 1

    from dtc.harness.ledger import read_ledger

    merged = read_ledger(local_path)
    assert {r["run_id"] for r in merged} == {"good-1"}


def test_merge_does_not_touch_local_file_when_remote_has_nothing_new(tmp_path):
    module = _load_merge_ledger_module()
    local_path = tmp_path / "local.jsonl"
    remote_path = tmp_path / "remote.jsonl"
    _write_jsonl(local_path, [_valid_record("only-1")])
    _write_jsonl(remote_path, [_valid_record("only-1")])

    before = local_path.read_text(encoding="utf-8")
    module.merge(local_path, remote_path)
    after = local_path.read_text(encoding="utf-8")
    assert before == after
