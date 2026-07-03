"""Tests for scripts/run_matrix.py: pure enumeration logic against synthetic
experiment definitions, plus a fully self-contained (tmp_path) end-to-end
resumability test that doesn't depend on the real Kaggle data or Task 8's
smoke-produced final configs -- those are exercised for real in Task 8's
smoke verification.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_run_matrix_module():
    spec = importlib.util.spec_from_file_location("run_matrix", REPO_ROOT / "scripts" / "run_matrix.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_matrix"] = module
    spec.loader.exec_module(module)
    return module


SYNTHETIC_EXPERIMENTS = {
    "e1": {
        "stage": "E1",
        "protocol": "B",
        "dataset": "kaggle",
        "models": ["tfidf_mnb", "lstm"],
        "config_source": "final",
        "seeds": [0, 1],
        "train_fraction": 1.0,
    },
    "e3": {
        "stage": "E3",
        "protocol": "B",
        "dataset": "kaggle",
        "models": ["tfidf_mnb"],
        "config_source": "final",
        "seeds": [0],
        "train_fractions": [0.1, 1.0],
    },
}


def test_build_run_specs_enumerates_all_axes():
    module = _load_run_matrix_module()
    specs = module.build_run_specs(SYNTHETIC_EXPERIMENTS)
    # e1: 2 models x 2 seeds x 1 fraction = 4; e3: 1 model x 1 seed x 2 fractions = 2
    assert len(specs) == 6
    e1_specs = [s for s in specs if s["experiment_key"] == "e1"]
    assert len(e1_specs) == 4
    e3_specs = [s for s in specs if s["experiment_key"] == "e3"]
    assert {s["train_fraction"] for s in e3_specs} == {0.1, 1.0}


def test_build_run_specs_respects_only_filter():
    module = _load_run_matrix_module()
    specs = module.build_run_specs(SYNTHETIC_EXPERIMENTS, only=["e3"])
    assert {s["experiment_key"] for s in specs} == {"e3"}


def test_build_run_specs_respects_models_filter():
    module = _load_run_matrix_module()
    specs = module.build_run_specs(SYNTHETIC_EXPERIMENTS, models_filter=["lstm"])
    assert {s["model_name"] for s in specs} == {"lstm"}


def test_skip_key_and_already_ledgered_keys_round_trip():
    module = _load_run_matrix_module()
    spec = {
        "stage": "E1",
        "protocol": "B",
        "model_name": "lstm",
        "seed": 0,
        "train_fraction": 1.0,
    }
    key = module._skip_key(spec, config_id="abc123", smoke=False)
    ledger_records = [
        {
            "stage": "E1",
            "protocol": "B",
            "model_name": "lstm",
            "config_id": "abc123",
            "seed": 0,
            "train_fraction": 1.0,
            "smoke": False,
        }
    ]
    ledgered_keys = module._already_ledgered_keys(ledger_records)
    assert key in ledgered_keys


def test_skip_key_distinguishes_smoke_from_real_runs():
    module = _load_run_matrix_module()
    spec = {"stage": "E1", "protocol": "B", "model_name": "lstm", "seed": 0, "train_fraction": 1.0}
    real_key = module._skip_key(spec, config_id="abc123", smoke=False)
    smoke_key = module._skip_key(spec, config_id="abc123", smoke=True)
    assert real_key != smoke_key


def test_real_experiments_yaml_parses_with_expected_shape():
    module = _load_run_matrix_module()
    experiments = module.load_experiments_config()
    assert set(experiments.keys()) == {"e1", "e2", "e3"}
    assert experiments["e1"]["seeds"] == [0, 1, 2, 3, 4]
    assert experiments["e2"]["seeds"] == [42]
    assert experiments["e2"]["protocol"] == "A"
    assert len(experiments["e1"]["models"]) == 9
    assert len(experiments["e3"]["models"]) == 5
    assert experiments["e3"]["train_fractions"] == [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]


def _build_tiny_kaggle_fixture(tmp_path: Path, n: int = 40) -> None:
    dataset_dir = tmp_path / "data" / "kaggle"
    dataset_dir.mkdir(parents=True)
    rows = []
    for i in range(n):
        label = i % 2
        text = "fire flood disaster" if label == 1 else "nice sunny day today"
        rows.append({"id": i, "text": text, "target": label})
    df = pd.DataFrame(rows)
    train_df, val_df, test_df = df.iloc[:24], df.iloc[24:32], df.iloc[32:]
    train_df.to_csv(dataset_dir / "train.csv", index=False)
    val_df.to_csv(dataset_dir / "val.csv", index=False)
    test_df.to_csv(dataset_dir / "test.csv", index=False)
    import json

    manifest = {
        "splits": {
            "train": {"sha256": "t"},
            "val": {"sha256": "v"},
            "test": {"sha256": "s"},
        }
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_end_to_end_dry_run_and_resume_skip_against_tiny_fixture(tmp_path):
    module = _load_run_matrix_module()
    _build_tiny_kaggle_fixture(tmp_path)

    configs_dir = tmp_path / "configs" / "final"
    configs_dir.mkdir(parents=True)
    (configs_dir / "tfidf_mnb.yaml").write_text(yaml.safe_dump({"alpha": 1.0}), encoding="utf-8")

    experiments = {
        "e1": {
            "stage": "E1",
            "protocol": "B",
            "dataset": "kaggle",
            "models": ["tfidf_mnb"],
            "config_source": "final",
            "seeds": [0],
            "train_fraction": 1.0,
        }
    }
    experiments_path = tmp_path / "experiments.yaml"
    experiments_path.write_text(yaml.safe_dump(experiments), encoding="utf-8")

    # dry-run: one pending run
    dry_results = module.main(experiments_config_path=experiments_path, repo_root=tmp_path, dry_run=True)
    assert len(dry_results) == 1
    assert dry_results[0]["would_run"] is True

    # real run: executes and ledgers exactly one run
    first_results = module.main(experiments_config_path=experiments_path, repo_root=tmp_path, dry_run=False)
    assert len(first_results) == 1
    assert "run_id" in first_results[0]

    from dtc.harness.ledger import read_ledger

    ledger_records = read_ledger(tmp_path / "results" / "ledger.jsonl")
    assert len(ledger_records) == 1
    assert ledger_records[0]["stage"] == "E1"
    assert ledger_records[0]["protocol"] == "B"
    assert ledger_records[0]["split"] == "test"

    # second invocation: same spec is now ledgered -> all-skips
    second_results = module.main(experiments_config_path=experiments_path, repo_root=tmp_path, dry_run=False)
    assert len(second_results) == 1
    assert second_results[0]["skipped"] is True

    ledger_records_after = read_ledger(tmp_path / "results" / "ledger.jsonl")
    assert len(ledger_records_after) == 1  # unchanged: no new line appended
