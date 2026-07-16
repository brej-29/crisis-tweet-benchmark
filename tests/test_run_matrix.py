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


def test_build_run_specs_defaults_eval_datasets_to_single_train_dataset():
    module = _load_run_matrix_module()
    specs = module.build_run_specs(SYNTHETIC_EXPERIMENTS)
    assert all(s["train_dataset"] == "kaggle" for s in specs)
    assert all(s["eval_datasets"] == ["kaggle"] for s in specs)


def test_build_run_specs_keeps_one_spec_per_training_for_multi_eval():
    module = _load_run_matrix_module()
    experiments = {
        "e4": {
            "stage": "E4",
            "protocol": "B",
            "dataset": "crisislex",
            "eval_datasets": ["crisislex", "kaggle"],
            "models": ["tfidf_mnb", "lstm"],
            "config_source": "final",
            "seeds": [0, 1],
            "train_fraction": 1.0,
        }
    }
    specs = module.build_run_specs(experiments)
    # one spec per TRAINING run (model x seed), not per eval dataset
    assert len(specs) == 4
    assert all(s["train_dataset"] == "crisislex" for s in specs)
    assert all(s["eval_datasets"] == ["crisislex", "kaggle"] for s in specs)


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
        "train_dataset": "kaggle",
    }
    key = module._skip_key(spec, config_id="abc123", smoke=False, eval_dataset="kaggle")
    ledger_records = [
        {
            "stage": "E1",
            "protocol": "B",
            "model_name": "lstm",
            "config_id": "abc123",
            "seed": 0,
            "train_fraction": 1.0,
            "smoke": False,
            "train_dataset": "kaggle",
            "eval_dataset": "kaggle",
        }
    ]
    ledgered_keys = module._already_ledgered_keys(ledger_records)
    assert key in ledgered_keys


def test_skip_key_matches_old_ledger_records_via_read_time_backfill(tmp_path):
    # Phase-1 ledger lines lack train_dataset/eval_dataset; read_ledger
    # backfills them to "kaggle", so the widened skip-key still hits.
    module = _load_run_matrix_module()
    from dtc.harness.ledger import append_run_record, read_ledger

    ledger_path = tmp_path / "ledger.jsonl"
    append_run_record(
        ledger_path,
        {
            "stage": "E1",
            "protocol": "B",
            "model_name": "lstm",
            "config_id": "abc123",
            "seed": 0,
            "train_fraction": 1.0,
            "smoke": False,
        },
    )
    ledgered_keys = module._already_ledgered_keys(read_ledger(ledger_path))
    spec = {
        "stage": "E1",
        "protocol": "B",
        "model_name": "lstm",
        "seed": 0,
        "train_fraction": 1.0,
        "train_dataset": "kaggle",
    }
    key = module._skip_key(spec, config_id="abc123", smoke=False, eval_dataset="kaggle")
    assert key in ledgered_keys


def test_skip_key_distinguishes_smoke_from_real_runs():
    module = _load_run_matrix_module()
    spec = {
        "stage": "E1",
        "protocol": "B",
        "model_name": "lstm",
        "seed": 0,
        "train_fraction": 1.0,
        "train_dataset": "kaggle",
    }
    real_key = module._skip_key(spec, config_id="abc123", smoke=False, eval_dataset="kaggle")
    smoke_key = module._skip_key(spec, config_id="abc123", smoke=True, eval_dataset="kaggle")
    assert real_key != smoke_key


def test_skip_key_distinguishes_eval_datasets():
    module = _load_run_matrix_module()
    spec = {
        "stage": "E5",
        "protocol": "B",
        "model_name": "lstm",
        "seed": 0,
        "train_fraction": 1.0,
        "train_dataset": "kaggle",
    }
    key_a = module._skip_key(spec, config_id="abc123", smoke=False, eval_dataset="kaggle")
    key_b = module._skip_key(spec, config_id="abc123", smoke=False, eval_dataset="crisislex")
    assert key_a != key_b


def test_real_experiments_yaml_parses_with_expected_shape():
    module = _load_run_matrix_module()
    experiments = module.load_experiments_config()
    assert set(experiments.keys()) == {"tuning", "e1", "e2", "e3"}
    assert experiments["e1"]["seeds"] == [0, 1, 2, 3, 4]
    assert experiments["e2"]["seeds"] == [42]
    assert experiments["e2"]["protocol"] == "A"
    assert len(experiments["e1"]["models"]) == 9
    assert len(experiments["e3"]["models"]) == 5
    assert experiments["e3"]["train_fractions"] == [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
    assert experiments["tuning"]["config_source"] == "tuning"
    assert len(experiments["tuning"]["models"]) == 9


def test_build_run_specs_expands_tuning_grid_into_one_spec_per_entry():
    module = _load_run_matrix_module()
    experiments = module.load_experiments_config()
    specs = module.build_run_specs(experiments, only=["tuning"], models_filter=["tfidf_mnb"])
    # configs/tuning/tfidf_mnb.yaml has 4 grid entries, single seed 0
    assert len(specs) == 4
    assert all(s["stage"] == "tuning" for s in specs)
    assert {s["grid_index"] for s in specs} == {0, 1, 2, 3}
    assert all(isinstance(s["grid_config"], dict) for s in specs)


def test_dry_run_enumerates_full_matrix_including_tuning():
    module = _load_run_matrix_module()
    experiments = module.load_experiments_config()
    specs = module.build_run_specs(experiments)
    stages = {s["stage"] for s in specs}
    assert stages == {"tuning", "E1", "E2", "E3"}


def _build_tiny_dataset_fixture(tmp_path: Path, dataset: str, id_col: str, label_col: str, n: int = 40) -> None:
    dataset_dir = tmp_path / "data" / dataset
    dataset_dir.mkdir(parents=True)
    rows = []
    for i in range(n):
        label = i % 2
        text = "fire flood disaster" if label == 1 else "nice sunny day today"
        rows.append({id_col: i, "text": text, label_col: label})
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


def _build_tiny_kaggle_fixture(tmp_path: Path, n: int = 40) -> None:
    _build_tiny_dataset_fixture(tmp_path, "kaggle", id_col="id", label_col="target", n=n)


def _build_tiny_crisislex_fixture(tmp_path: Path, n: int = 40) -> None:
    _build_tiny_dataset_fixture(tmp_path, "crisislex", id_col="tweet_id", label_col="label", n=n)


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
    # single-eval records still carry the provenance fields, populated to
    # the one dataset (harmless, consistent with dual-eval records)
    assert ledger_records_after[0]["train_dataset"] == "kaggle"
    assert ledger_records_after[0]["eval_dataset"] == "kaggle"
    assert ledger_records_after[0]["training_id"]


def _setup_dual_eval_experiment(tmp_path: Path) -> Path:
    """Tiny kaggle + crisislex fixtures and an E4-shaped experiments.yaml:
    train on crisislex, evaluate on both frozen tests."""
    _build_tiny_kaggle_fixture(tmp_path)
    _build_tiny_crisislex_fixture(tmp_path)
    configs_dir = tmp_path / "configs" / "final"
    configs_dir.mkdir(parents=True)
    (configs_dir / "tfidf_mnb.yaml").write_text(yaml.safe_dump({"alpha": 1.0}), encoding="utf-8")
    experiments = {
        "e4": {
            "stage": "E4",
            "protocol": "B",
            "dataset": "crisislex",
            "eval_datasets": ["crisislex", "kaggle"],
            "models": ["tfidf_mnb"],
            "config_source": "final",
            "seeds": [0],
            "train_fraction": 1.0,
        }
    }
    experiments_path = tmp_path / "experiments.yaml"
    experiments_path.write_text(yaml.safe_dump(experiments), encoding="utf-8")
    return experiments_path


def test_dual_eval_training_emits_two_records_sharing_training_id(tmp_path):
    module = _load_run_matrix_module()
    experiments_path = _setup_dual_eval_experiment(tmp_path)

    # dry-run: ONE spec (one training), both eval records pending
    dry_results = module.main(experiments_config_path=experiments_path, repo_root=tmp_path, dry_run=True)
    assert len(dry_results) == 1
    assert dry_results[0]["would_run"] is True
    assert dry_results[0]["pending_eval_datasets"] == ["crisislex", "kaggle"]

    results = module.main(experiments_config_path=experiments_path, repo_root=tmp_path, dry_run=False)
    assert len(results) == 1

    from dtc.harness.ledger import read_ledger

    records = read_ledger(tmp_path / "results" / "ledger.jsonl")
    assert len(records) == 2
    assert [r["eval_dataset"] for r in records] == ["crisislex", "kaggle"]
    assert all(r["train_dataset"] == "crisislex" for r in records)
    assert all(r["dataset"] == "crisislex" for r in records)  # `dataset` stays = training dataset
    assert all(r["split"] == "test" for r in records)
    # same training, so same training_id -- but each eval is its own run
    assert len({r["training_id"] for r in records}) == 1
    assert len({r["run_id"] for r in records}) == 2
    for r in records:
        assert (tmp_path / "results" / "runs" / r["run_id"] / "predictions.csv").exists()

    # fully-ledgered dual-eval training: second invocation skips, appends nothing
    second_results = module.main(experiments_config_path=experiments_path, repo_root=tmp_path, dry_run=False)
    assert second_results[0]["skipped"] is True
    assert len(read_ledger(tmp_path / "results" / "ledger.jsonl")) == 2


def test_partial_dual_eval_training_retrains_and_fills_only_missing_record(tmp_path):
    module = _load_run_matrix_module()
    experiments_path = _setup_dual_eval_experiment(tmp_path)

    from dtc.harness.ledger import append_run_record, read_ledger

    # simulate a crash between the two evals: only the crisislex eval record landed
    ledger_path = tmp_path / "results" / "ledger.jsonl"
    append_run_record(
        ledger_path,
        {
            "run_id": "pre-existing",
            "stage": "E4",
            "protocol": "B",
            "model_name": "tfidf_mnb",
            "config_id": module.compute_config_id({"alpha": 1.0}),
            "seed": 0,
            "train_fraction": 1.0,
            "smoke": False,
            "train_dataset": "crisislex",
            "eval_dataset": "crisislex",
            "training_id": "tid-pre",
        },
    )

    dry_results = module.main(experiments_config_path=experiments_path, repo_root=tmp_path, dry_run=True)
    assert dry_results[0]["would_run"] is True
    assert dry_results[0]["pending_eval_datasets"] == ["kaggle"]

    module.main(experiments_config_path=experiments_path, repo_root=tmp_path, dry_run=False)
    records = read_ledger(ledger_path)
    assert len(records) == 2  # only the missing eval record was appended
    new = [r for r in records if r["run_id"] != "pre-existing"]
    assert len(new) == 1
    assert new[0]["eval_dataset"] == "kaggle"
    assert new[0]["train_dataset"] == "crisislex"
    # retrained in a fresh run: its training_id differs from the crashed one
    assert new[0]["training_id"] != "tid-pre"
