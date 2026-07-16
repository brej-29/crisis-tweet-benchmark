"""Tests for the append-only results ledger and run-record construction."""

from __future__ import annotations

import json

import pandas as pd

from dtc.harness import ledger as ledger_module
from dtc.harness.config import compute_config_id
from dtc.harness.ledger import (
    append_run_record,
    generate_run_id,
    get_git_commit_hash,
    read_ledger,
)
from dtc.harness.run import build_run_record, save_predictions

REPO_ROOT_FOR_GIT_TEST = __file__  # any path inside the repo works for `git rev-parse`


def test_generate_run_id_is_unique():
    ids = {generate_run_id() for _ in range(100)}
    assert len(ids) == 100


def test_get_git_commit_hash_returns_full_sha(tmp_path):
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    sha = get_git_commit_hash(repo_root)
    assert sha != "unknown"
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_append_run_record_writes_well_formed_jsonl(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    record = {"run_id": "abc123", "model_name": "dummy", "metrics": {"accuracy": 0.5}}
    append_run_record(ledger_path, record)

    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed == record


def test_ledger_is_append_only_not_overwritten(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    append_run_record(ledger_path, {"run_id": "first"})
    append_run_record(ledger_path, {"run_id": "second"})

    records = read_ledger(ledger_path)
    assert len(records) == 2
    assert records[0]["run_id"] == "first"
    assert records[1]["run_id"] == "second"


def test_append_run_record_never_truncates_existing_lines(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    for i in range(5):
        append_run_record(ledger_path, {"run_id": f"run-{i}"})
    records = read_ledger(ledger_path)
    assert [r["run_id"] for r in records] == [f"run-{i}" for i in range(5)]


def test_build_run_record_has_required_keys(tmp_path):
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    manifest = {
        "splits": {
            "train": {"sha256": "aaa"},
            "val": {"sha256": "bbb"},
            "test": {"sha256": "ccc"},
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    record = build_run_record(
        run_id="test-run-1",
        repo_root=repo_root,
        model_name="majority_class",
        dataset="kaggle_nlp_getting_started",
        split="val",
        seed=42,
        config={"strategy": "most_frequent"},
        metrics={"accuracy": 0.5},
        dataset_manifest_path=manifest_path,
    )

    required_keys = {
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
    assert required_keys <= set(record.keys())
    assert record["dataset_split_hashes"] == {"train": "aaa", "val": "bbb", "test": "ccc"}
    assert record["seed"] == 42
    # defaults, since this call site didn't pass protocol/phase/smoke/train_fraction/config_id
    assert record["protocol"] is None
    assert record["phase"] == "phase0"
    assert record["smoke"] is False
    assert record["train_fraction"] == 1.0
    assert record["config_id"] == compute_config_id({"strategy": "most_frequent"})
    assert isinstance(record["git_dirty_paths"], list)
    assert record["stage"] is None


def test_build_run_record_honors_explicit_protocol_phase_smoke_and_config_id(tmp_path):
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"splits": {"train": {"sha256": "a"}, "val": {"sha256": "b"}, "test": {"sha256": "c"}}}),
        encoding="utf-8",
    )
    record = build_run_record(
        run_id="test-run-2",
        repo_root=repo_root,
        model_name="bilstm",
        dataset="kaggle_nlp_getting_started",
        split="test",
        seed=1,
        config={"lr": 0.001},
        metrics={"accuracy": 0.9},
        dataset_manifest_path=manifest_path,
        protocol="B",
        phase="phase1",
        stage="tuning",
        smoke=True,
        train_fraction=0.05,
        config_id="deadbeef",
    )
    assert record["protocol"] == "B"
    assert record["phase"] == "phase1"
    assert record["stage"] == "tuning"
    assert record["smoke"] is True
    assert record["train_fraction"] == 0.05
    assert record["config_id"] == "deadbeef"


def test_build_run_record_without_manifest_uses_explicit_split_hashes():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    record = build_run_record(
        run_id="protocol-a-run-1",
        repo_root=repo_root,
        model_name="lstm",
        dataset="kaggle",
        split="protocol_a_eval",
        seed=42,
        config={"lr": 0.001},
        metrics={"accuracy": 0.7},
        dataset_split_hashes={"protocol_a_train": "aaa", "protocol_a_eval": "bbb"},
    )
    assert record["dataset_manifest_path"] is None
    assert record["dataset_split_hashes"] == {"protocol_a_train": "aaa", "protocol_a_eval": "bbb"}


def test_build_run_record_with_neither_manifest_nor_hashes_leaves_hashes_none():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    record = build_run_record(
        run_id="no-manifest-run",
        repo_root=repo_root,
        model_name="lstm",
        dataset="kaggle",
        split="val",
        seed=0,
        config={},
        metrics={"accuracy": 0.5},
    )
    assert record["dataset_manifest_path"] is None
    assert record["dataset_split_hashes"] is None


def test_build_run_record_includes_dataset_provenance_fields_when_provided():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    record = build_run_record(
        run_id="cross-eval-run",
        repo_root=repo_root,
        model_name="tfidf_mnb",
        dataset="crisislex",
        split="test",
        seed=0,
        config={"alpha": 1.0},
        metrics={"accuracy": 0.5},
        train_dataset="crisislex",
        eval_dataset="kaggle",
        training_id="tid-123",
    )
    assert record["dataset"] == "crisislex"  # stays = the training dataset
    assert record["train_dataset"] == "crisislex"
    assert record["eval_dataset"] == "kaggle"
    assert record["training_id"] == "tid-123"


def test_build_run_record_omits_dataset_provenance_fields_by_default():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    record = build_run_record(
        run_id="legacy-style-run",
        repo_root=repo_root,
        model_name="tfidf_mnb",
        dataset="kaggle",
        split="val",
        seed=0,
        config={},
        metrics={"accuracy": 0.5},
    )
    # backward compat: old call sites emit records byte-identical in shape
    assert "train_dataset" not in record
    assert "eval_dataset" not in record
    assert "training_id" not in record


def test_read_ledger_backfills_train_and_eval_dataset_in_memory_only(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    append_run_record(ledger_path, {"run_id": "old-1", "dataset": "kaggle"})
    append_run_record(ledger_path, {"run_id": "old-2"})  # no dataset field at all
    append_run_record(
        ledger_path,
        {"run_id": "new-1", "dataset": "crisislex", "train_dataset": "crisislex", "eval_dataset": "kaggle"},
    )
    before_bytes = ledger_path.read_bytes()

    records = read_ledger(ledger_path)
    by_id = {r["run_id"]: r for r in records}
    assert by_id["old-1"]["train_dataset"] == "kaggle"
    assert by_id["old-1"]["eval_dataset"] == "kaggle"
    assert by_id["old-2"]["train_dataset"] == "kaggle"  # no dataset -> default "kaggle"
    assert by_id["old-2"]["eval_dataset"] == "kaggle"
    # records that already carry the fields keep their values
    assert by_id["new-1"]["train_dataset"] == "crisislex"
    assert by_id["new-1"]["eval_dataset"] == "kaggle"
    # backfill is read-time only: the file bytes are untouched
    assert ledger_path.read_bytes() == before_bytes


def test_compute_config_id_is_deterministic_and_key_order_independent():
    c1 = {"a": 1, "b": 2}
    c2 = {"b": 2, "a": 1}
    assert compute_config_id(c1) == compute_config_id(c2)
    assert compute_config_id({"a": 1, "b": 3}) != compute_config_id(c1)


def test_is_git_dirty_ignores_ledger_only_changes(monkeypatch):
    monkeypatch.setattr(ledger_module, "get_git_dirty_paths", lambda repo_root: ["results/ledger.jsonl"])
    assert ledger_module.is_git_dirty("unused") is False


def test_is_git_dirty_true_when_non_ledger_paths_dirty(monkeypatch):
    monkeypatch.setattr(
        ledger_module,
        "get_git_dirty_paths",
        lambda repo_root: ["results/ledger.jsonl", "src/dtc/models/lstm.py"],
    )
    assert ledger_module.is_git_dirty("unused") is True


def test_is_git_dirty_false_when_tree_clean(monkeypatch):
    monkeypatch.setattr(ledger_module, "get_git_dirty_paths", lambda repo_root: [])
    assert ledger_module.is_git_dirty("unused") is False


def test_save_predictions_writes_expected_columns(tmp_path):
    path = save_predictions(
        run_id="run-xyz",
        results_dir=tmp_path,
        ids=[1, 2, 3],
        texts=["a", "b", "c"],
        y_true=[0, 1, 0],
        y_pred=[0, 1, 1],
        y_prob=[0.1, 0.9, 0.6],
    )
    df = pd.read_csv(path)
    assert list(df.columns) == ["id", "text_sha256", "y_true", "y_pred", "y_prob"]
    assert len(df) == 3
    # text hashes should be deterministic sha256 hex digests, not raw text
    assert df["text_sha256"].iloc[0] != "a"
    assert len(df["text_sha256"].iloc[0]) == 64