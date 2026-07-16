"""Tests for scripts/make_tables.py against synthetic ledger fixtures:
T1-T4 aggregation correctness, smoke exclusion/inclusion, idempotency, and
the dirty-source-path refusal.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_make_tables_module():
    spec = importlib.util.spec_from_file_location("make_tables", REPO_ROOT / "scripts" / "make_tables.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_tables"] = module
    spec.loader.exec_module(module)
    return module


def _record(**overrides) -> dict:
    base = {
        "run_id": "r",
        "model_name": "lstm",
        "stage": "E1",
        "protocol": "B",
        "split": "test",
        "seed": 0,
        "train_fraction": 1.0,
        "smoke": False,
        "config_id": "cfg-a",
        "git_dirty_paths": [],
        "metrics": {"accuracy": 0.8, "macro_f1": 0.75, "positive_f1": 0.7, "weighted_f1_legacy": 0.78},
    }
    base.update(overrides)
    return base


def _write_ledger(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_build_t1_computes_mean_and_std_per_model():
    module = _load_make_tables_module()
    records = [
        _record(run_id="a", model_name="lstm", seed=0, metrics={"accuracy": 0.8, "macro_f1": 0.7, "positive_f1": 0.6, "weighted_f1_legacy": 0.75}),
        _record(run_id="b", model_name="lstm", seed=1, metrics={"accuracy": 0.9, "macro_f1": 0.8, "positive_f1": 0.7, "weighted_f1_legacy": 0.85}),
    ]
    rows = module.build_t1_protocol_b_main(records)
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "lstm"
    assert row["n_seeds"] == 2
    assert row["accuracy_mean"] == pytest.approx(0.85)
    assert row["accuracy_std"] == pytest.approx(0.0707106781, rel=1e-3)


def test_build_t1_excludes_non_e1_or_non_test_split_records():
    module = _load_make_tables_module()
    records = [
        _record(run_id="a", stage="E1", split="test"),
        _record(run_id="b", stage="E1", split="val"),  # wrong split, excluded
        _record(run_id="c", stage="tuning", split="test"),  # wrong stage, excluded
        _record(run_id="d", stage="E1", protocol="A", split="test"),  # wrong protocol, excluded
    ]
    rows = module.build_t1_protocol_b_main(records)
    assert len(rows) == 1
    assert rows[0]["n_seeds"] == 1


def test_build_t1_raises_on_mixed_config_ids_for_non_smoke_e1_records():
    module = _load_make_tables_module()
    records = [
        _record(run_id="a", model_name="lstm", seed=0, config_id="cfg-a"),
        _record(run_id="b", model_name="lstm", seed=1, config_id="cfg-b"),  # different config, same model
    ]
    with pytest.raises(module.MixedConfigError) as excinfo:
        module.build_t1_protocol_b_main(records)
    assert "lstm" in str(excinfo.value)
    assert "cfg-a" in str(excinfo.value)
    assert "cfg-b" in str(excinfo.value)


def test_build_t1_succeeds_with_single_config_id_per_model():
    module = _load_make_tables_module()
    records = [
        _record(run_id="a", model_name="lstm", seed=0, config_id="cfg-a"),
        _record(run_id="b", model_name="lstm", seed=1, config_id="cfg-a"),
        _record(run_id="c", model_name="gru", seed=0, config_id="cfg-x"),
    ]
    rows = module.build_t1_protocol_b_main(records)  # should not raise
    by_model = {r["model"]: r for r in rows}
    assert by_model["lstm"]["n_seeds"] == 2
    assert by_model["gru"]["n_seeds"] == 1


def test_build_t1_ignores_mixed_config_ids_among_smoke_records():
    module = _load_make_tables_module()
    records = [
        _record(run_id="a", model_name="lstm", seed=0, config_id="cfg-a", smoke=True),
        _record(run_id="b", model_name="lstm", seed=1, config_id="cfg-b", smoke=True),  # differs, but smoke-exempt
    ]
    rows = module.build_t1_protocol_b_main(records)  # should not raise
    assert rows[0]["n_seeds"] == 2


def test_build_t2_ranks_and_computes_rank_delta():
    module = _load_make_tables_module()
    records = [
        _record(run_id="b1", model_name="lstm", stage="E1", protocol="B", metrics={"accuracy": 0, "macro_f1": 0.9, "positive_f1": 0, "weighted_f1_legacy": 0}),
        _record(run_id="b2", model_name="gru", stage="E1", protocol="B", metrics={"accuracy": 0, "macro_f1": 0.7, "positive_f1": 0, "weighted_f1_legacy": 0}),
        _record(run_id="a1", model_name="lstm", stage="E2", protocol="A", split="protocol_a_eval", metrics={"accuracy": 0, "macro_f1": 0, "positive_f1": 0, "weighted_f1_legacy": 0.6}),
        _record(run_id="a2", model_name="gru", stage="E2", protocol="A", split="protocol_a_eval", metrics={"accuracy": 0, "macro_f1": 0, "positive_f1": 0, "weighted_f1_legacy": 0.8}),
    ]
    rows = module.build_t2_protocol_comparison(records)
    by_model = {r["model"]: r for r in rows}
    # Protocol B: lstm (0.9) ranked 1, gru (0.7) ranked 2
    assert by_model["lstm"]["protocol_b_rank"] == 1
    assert by_model["gru"]["protocol_b_rank"] == 2
    # Protocol A: gru (0.8) ranked 1, lstm (0.6) ranked 2 -- rankings flip
    assert by_model["gru"]["protocol_a_rank"] == 1
    assert by_model["lstm"]["protocol_a_rank"] == 2
    assert by_model["lstm"]["rank_delta"] == 1  # 2 - 1
    assert by_model["gru"]["rank_delta"] == -1  # 1 - 2


def test_build_t3_computes_min_max_std():
    module = _load_make_tables_module()
    records = [
        _record(run_id="a", model_name="conv1d", metrics={"accuracy": 0, "macro_f1": 0.6, "positive_f1": 0, "weighted_f1_legacy": 0}),
        _record(run_id="b", model_name="conv1d", metrics={"accuracy": 0, "macro_f1": 0.8, "positive_f1": 0, "weighted_f1_legacy": 0}),
    ]
    rows = module.build_t3_seed_variance(records)
    assert len(rows) == 1
    assert rows[0]["min_macro_f1"] == pytest.approx(0.6)
    assert rows[0]["max_macro_f1"] == pytest.approx(0.8)
    assert rows[0]["std_macro_f1"] == pytest.approx(0.1414213562, rel=1e-3)


def test_build_t4_groups_by_model_and_fraction():
    module = _load_make_tables_module()
    records = [
        _record(run_id="a", stage="E3", model_name="bilstm", train_fraction=0.1, metrics={"accuracy": 0, "macro_f1": 0.5, "positive_f1": 0, "weighted_f1_legacy": 0}),
        _record(run_id="b", stage="E3", model_name="bilstm", train_fraction=0.1, metrics={"accuracy": 0, "macro_f1": 0.6, "positive_f1": 0, "weighted_f1_legacy": 0}),
        _record(run_id="c", stage="E3", model_name="bilstm", train_fraction=1.0, metrics={"accuracy": 0, "macro_f1": 0.9, "positive_f1": 0, "weighted_f1_legacy": 0}),
    ]
    rows = module.build_t4_data_efficiency(records)
    assert len(rows) == 2
    frac_01 = next(r for r in rows if r["train_fraction"] == 0.1)
    assert frac_01["n_seeds"] == 2
    assert frac_01["macro_f1_mean"] == pytest.approx(0.55)


def test_build_t4_raises_on_mixed_config_ids_for_same_model_fraction():
    module = _load_make_tables_module()
    records = [
        _record(run_id="a", stage="E3", model_name="bilstm", train_fraction=0.1, config_id="cfg-a"),
        _record(run_id="b", stage="E3", model_name="bilstm", train_fraction=0.1, config_id="cfg-b"),
    ]
    with pytest.raises(module.MixedConfigError) as excinfo:
        module.build_t4_data_efficiency(records)
    assert "bilstm" in str(excinfo.value)
    assert "cfg-a" in str(excinfo.value)
    assert "cfg-b" in str(excinfo.value)


def test_build_t4_allows_same_model_different_fraction_different_config_id():
    """Mixed config_ids across DIFFERENT fractions for the same model are
    fine -- the guard is per (model, fraction), not per model alone."""
    module = _load_make_tables_module()
    records = [
        _record(run_id="a", stage="E3", model_name="bilstm", train_fraction=0.1, config_id="cfg-a"),
        _record(run_id="b", stage="E3", model_name="bilstm", train_fraction=0.5, config_id="cfg-b"),
    ]
    rows = module.build_t4_data_efficiency(records)  # should not raise
    assert len(rows) == 2


def test_build_t4_ignores_mixed_config_ids_among_smoke_records():
    module = _load_make_tables_module()
    records = [
        _record(run_id="a", stage="E3", model_name="bilstm", train_fraction=0.1, config_id="cfg-a", smoke=True),
        _record(run_id="b", stage="E3", model_name="bilstm", train_fraction=0.1, config_id="cfg-b", smoke=True),
    ]
    rows = module.build_t4_data_efficiency(records)  # should not raise
    assert rows[0]["n_seeds"] == 2


def test_resolve_final_config_ids_reads_yaml_and_injects_use_frozen_cache_dir(tmp_path):
    module = _load_make_tables_module()
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "lstm.yaml").write_text("lr: 0.001\nhidden_size: 64\n", encoding="utf-8")
    (final_dir / "use_frozen.yaml").write_text("hidden_size: 64\n", encoding="utf-8")
    (final_dir / "README.md").write_text("not a config", encoding="utf-8")  # must be ignored (not *.yaml)

    result = module.resolve_final_config_ids(final_dir, dataset="kaggle", repo_root=tmp_path)

    assert set(result.keys()) == {"lstm", "use_frozen"}
    assert result["lstm"] == module.compute_config_id({"lr": 0.001, "hidden_size": 64})
    expected_use_frozen_config = {
        "hidden_size": 64,
        "use_cache_dir": str(tmp_path / "data" / "kaggle" / "use_embeddings"),
    }
    assert result["use_frozen"] == module.compute_config_id(expected_use_frozen_config)


def test_filter_to_final_config_ids_drops_superseded_keeps_current_and_unknown():
    module = _load_make_tables_module()
    final_config_ids = {"lstm": "cfg-new"}
    records = [
        _record(run_id="old", model_name="lstm", config_id="cfg-old"),  # superseded, dropped
        _record(run_id="new", model_name="lstm", config_id="cfg-new"),  # current, kept
        _record(run_id="other", model_name="gru", config_id="cfg-anything"),  # no known-current config, kept
    ]
    kept = module.filter_to_final_config_ids(records, final_config_ids)
    assert {r["run_id"] for r in kept} == {"new", "other"}


def test_main_only_config_ids_from_recovers_t1_from_a_mixed_ledger(tmp_path):
    """End-to-end: a ledger with an old (superseded) config_id and a new
    (current) one for the same model would normally make T1 refuse;
    --only-config-ids-from filters to just the current one instead."""
    module = _load_make_tables_module()

    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "lstm.yaml").write_text("lr: 0.0005\nhidden_size: 64\n", encoding="utf-8")
    new_config_id = module.compute_config_id({"lr": 0.0005, "hidden_size": 64})
    old_config_id = module.compute_config_id({"lr": 0.001, "hidden_size": 32})

    ledger_path = tmp_path / "ledger.jsonl"
    records = [
        _record(run_id="old-1", model_name="lstm", seed=0, config_id=old_config_id,
                metrics={"accuracy": 0.5, "macro_f1": 0.4, "positive_f1": 0.3, "weighted_f1_legacy": 0.4}),
        _record(run_id="new-1", model_name="lstm", seed=0, config_id=new_config_id,
                metrics={"accuracy": 0.9, "macro_f1": 0.85, "positive_f1": 0.8, "weighted_f1_legacy": 0.85}),
        _record(run_id="new-2", model_name="lstm", seed=1, config_id=new_config_id,
                metrics={"accuracy": 0.88, "macro_f1": 0.83, "positive_f1": 0.78, "weighted_f1_legacy": 0.83}),
    ]
    with open(ledger_path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Without the override: refuses (mixed config_ids for lstm)
    with pytest.raises(module.MixedConfigError):
        module.main(ledger_path, tmp_path / "tables_refused")

    # With the override: succeeds, using only the 2 current-config runs
    written = module.main(ledger_path, tmp_path / "tables_ok", only_config_ids_from=final_dir)
    t1_content = written["T1"].read_text(encoding="utf-8")
    assert "old-1" not in t1_content  # sanity: run_id isn't rendered, but n_seeds should reflect only 2 runs
    assert "| lstm | 2 |" in t1_content


def test_check_no_dirty_source_runs_raises_for_source_paths():
    module = _load_make_tables_module()
    records = [_record(run_id="a", git_dirty_paths=["src/dtc/models/lstm.py"])]
    with pytest.raises(module.DirtyLedgerError):
        module.check_no_dirty_source_runs(records)


def test_check_no_dirty_source_runs_allows_data_and_results_paths():
    module = _load_make_tables_module()
    records = [_record(run_id="a", git_dirty_paths=["results/ledger.jsonl", "data/kaggle/manifest.json"])]
    module.check_no_dirty_source_runs(records)  # should not raise


def test_main_excludes_smoke_by_default_and_includes_with_flag(tmp_path):
    module = _load_make_tables_module()
    ledger_path = tmp_path / "ledger.jsonl"
    _write_ledger(
        ledger_path,
        [
            _record(run_id="real-1", smoke=False),
            _record(run_id="smoke-1", smoke=True, model_name="gru"),
        ],
    )
    output_dir = tmp_path / "tables"

    written = module.main(ledger_path, output_dir, include_smoke=False)
    t1_content = written["T1"].read_text(encoding="utf-8")
    assert "gru" not in t1_content
    assert "lstm" in t1_content
    assert "SMOKE" not in t1_content

    written_smoke = module.main(ledger_path, output_dir, include_smoke=True)
    t1_smoke_content = written_smoke["T1"].read_text(encoding="utf-8")
    assert "gru" in t1_smoke_content
    assert "SMOKE" in t1_smoke_content


def test_main_is_idempotent(tmp_path):
    module = _load_make_tables_module()
    ledger_path = tmp_path / "ledger.jsonl"
    _write_ledger(ledger_path, [_record(run_id="real-1")])
    output_dir = tmp_path / "tables"

    written_1 = module.main(ledger_path, output_dir)
    content_1 = {name: p.read_text(encoding="utf-8") for name, p in written_1.items()}
    written_2 = module.main(ledger_path, output_dir)
    content_2 = {name: p.read_text(encoding="utf-8") for name, p in written_2.items()}
    assert content_1 == content_2


def test_main_refuses_when_referenced_run_has_dirty_source_paths(tmp_path):
    module = _load_make_tables_module()
    ledger_path = tmp_path / "ledger.jsonl"
    _write_ledger(ledger_path, [_record(run_id="dirty-1", git_dirty_paths=["src/dtc/models/lstm.py"])])
    output_dir = tmp_path / "tables"
    with pytest.raises(module.DirtyLedgerError):
        module.main(ledger_path, output_dir)


# --- T5: cross-dataset matrix -----------------------------------------------


def test_build_t5_computes_cross_dataset_matrix_and_deltas():
    module = _load_make_tables_module()
    cell_values = {
        ("kaggle", "kaggle"): [0.80, 0.90],
        ("kaggle", "crisislex"): [0.60, 0.70],
        ("crisislex", "kaggle"): [0.50, 0.60],
        ("crisislex", "crisislex"): [0.75, 0.85],
    }
    records = []
    for model_name in ("lstm", "gru"):
        for (train_ds, eval_ds), values in cell_values.items():
            for seed, value in enumerate(values):
                records.append(
                    _record(
                        run_id=f"{model_name}-{train_ds}-{eval_ds}-{seed}",
                        model_name=model_name,
                        stage="E4" if train_ds == "crisislex" else "E5",
                        train_dataset=train_ds,
                        eval_dataset=eval_ds,
                        seed=seed,
                        metrics={"accuracy": 0, "macro_f1": value, "positive_f1": 0, "weighted_f1_legacy": 0},
                    )
                )

    rows = module.build_t5_cross_dataset_matrix(records)
    assert len(rows) == 2
    row = {r["model"]: r for r in rows}["lstm"]
    assert row["train_kaggle_eval_kaggle_mean"] == pytest.approx(0.85)
    assert row["train_kaggle_eval_kaggle_std"] == pytest.approx(0.0707106781, rel=1e-3)
    assert row["train_kaggle_eval_crisislex_mean"] == pytest.approx(0.65)
    assert row["train_crisislex_eval_kaggle_mean"] == pytest.approx(0.55)
    assert row["train_crisislex_eval_crisislex_mean"] == pytest.approx(0.80)
    # ood_delta_eval_kaggle = mean(train=crisislex, eval=kaggle) - mean(train=kaggle, eval=kaggle) = 0.55 - 0.85
    assert row["ood_delta_eval_kaggle"] == pytest.approx(-0.30)
    # ood_delta_eval_crisislex = mean(train=kaggle, eval=crisislex) - mean(train=crisislex, eval=crisislex) = 0.65 - 0.80
    assert row["ood_delta_eval_crisislex"] == pytest.approx(-0.15)


def test_build_t5_excludes_non_e4_e5_or_non_test_split_records():
    module = _load_make_tables_module()
    records = [
        _record(run_id="a", stage="E5", split="test", train_dataset="kaggle", eval_dataset="kaggle",
                metrics={"accuracy": 0, "macro_f1": 0.7, "positive_f1": 0, "weighted_f1_legacy": 0}),
        _record(run_id="b", stage="E5", split="val", train_dataset="kaggle", eval_dataset="kaggle"),  # wrong split
        _record(run_id="c", stage="E1", split="test", train_dataset="kaggle", eval_dataset="kaggle"),  # wrong stage
    ]
    rows = module.build_t5_cross_dataset_matrix(records)
    assert len(rows) == 1
    assert rows[0]["train_kaggle_eval_kaggle_mean"] == pytest.approx(0.7)


def test_build_t5_leaves_missing_cells_and_deltas_as_none():
    module = _load_make_tables_module()
    records = [
        _record(run_id="a", model_name="lstm", stage="E5", split="test", train_dataset="kaggle",
                eval_dataset="kaggle", seed=0,
                metrics={"accuracy": 0, "macro_f1": 0.7, "positive_f1": 0, "weighted_f1_legacy": 0}),
    ]
    rows = module.build_t5_cross_dataset_matrix(records)
    assert len(rows) == 1
    row = rows[0]
    assert row["train_kaggle_eval_kaggle_mean"] == pytest.approx(0.7)
    assert row["train_kaggle_eval_crisislex_mean"] is None
    assert row["train_crisislex_eval_kaggle_mean"] is None
    assert row["train_crisislex_eval_crisislex_mean"] is None
    assert row["ood_delta_eval_kaggle"] is None
    assert row["ood_delta_eval_crisislex"] is None


def test_build_t5_raises_on_mixed_config_ids_within_model_train_dataset_group():
    module = _load_make_tables_module()
    records = [
        _record(run_id="a", model_name="lstm", stage="E5", split="test", train_dataset="kaggle",
                eval_dataset="kaggle", seed=0, config_id="cfg-a"),
        _record(run_id="b", model_name="lstm", stage="E5", split="test", train_dataset="kaggle",
                eval_dataset="crisislex", seed=0, config_id="cfg-b"),
    ]
    with pytest.raises(module.MixedConfigError) as excinfo:
        module.build_t5_cross_dataset_matrix(records)
    assert "lstm" in str(excinfo.value)
    assert "cfg-a" in str(excinfo.value)
    assert "cfg-b" in str(excinfo.value)


def test_build_t5_allows_different_config_ids_across_different_train_datasets():
    """A model may legitimately have different config_ids for its kaggle-
    trained vs. crisislex-trained records (use_frozen's use_cache_dir
    injection) -- the guard is per (model, train_dataset), not per model."""
    module = _load_make_tables_module()
    records = [
        _record(run_id="a", model_name="use_frozen", stage="E5", split="test", train_dataset="kaggle",
                eval_dataset="kaggle", seed=0, config_id="cfg-kaggle"),
        _record(run_id="b", model_name="use_frozen", stage="E4", split="test", train_dataset="crisislex",
                eval_dataset="crisislex", seed=0, config_id="cfg-crisislex"),
    ]
    rows = module.build_t5_cross_dataset_matrix(records)  # should not raise
    assert len(rows) == 1


def test_build_t5_ignores_mixed_config_ids_among_smoke_records():
    module = _load_make_tables_module()
    records = [
        _record(run_id="a", model_name="lstm", stage="E5", split="test", train_dataset="kaggle",
                eval_dataset="kaggle", seed=0, config_id="cfg-a", smoke=True),
        _record(run_id="b", model_name="lstm", stage="E5", split="test", train_dataset="kaggle",
                eval_dataset="crisislex", seed=0, config_id="cfg-b", smoke=True),
    ]
    rows = module.build_t5_cross_dataset_matrix(records)  # should not raise
    assert len(rows) == 1


# --- T6: per-event breakdown ------------------------------------------------


def _write_predictions_csv(run_dir: Path, event_rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {"id": i, "text_sha256": "x", "y_true": r["y_true"], "y_pred": r["y_pred"], "y_prob": 0.5, "event": r["event"]}
            for i, r in enumerate(event_rows)
        ]
    )
    df.to_csv(run_dir / "predictions.csv", index=False)


def test_build_t6_computes_per_event_macro_f1_mean_std(tmp_path):
    module = _load_make_tables_module()
    results_dir = tmp_path / "results"

    per_seed_events = {
        0: {
            "eventA": {"y_true": [1, 0, 1, 0], "y_pred": [1, 0, 0, 0]},
            "eventB": {"y_true": [1, 1, 0], "y_pred": [1, 0, 0]},
        },
        1: {
            "eventA": {"y_true": [1, 0, 1, 1], "y_pred": [1, 1, 1, 0]},
            "eventB": {"y_true": [0, 0, 1], "y_pred": [0, 0, 1]},
        },
    }
    records = []
    for seed, events in per_seed_events.items():
        run_id = f"run-seed{seed}"
        event_rows = [
            {"event": event, "y_true": yt, "y_pred": yp}
            for event, data in events.items()
            for yt, yp in zip(data["y_true"], data["y_pred"])
        ]
        _write_predictions_csv(results_dir / "runs" / run_id, event_rows)
        records.append(
            _record(
                run_id=run_id, model_name="lstm", stage="E4", split="test",
                train_dataset="crisislex", eval_dataset="crisislex", seed=seed,
            )
        )

    rows = module.build_t6_per_event(records, results_dir)
    by_event = {r["event"]: r for r in rows}
    assert set(by_event) == {"eventA", "eventB"}
    for event in ("eventA", "eventB"):
        expected = [
            f1_score(per_seed_events[s][event]["y_true"], per_seed_events[s][event]["y_pred"], average="macro")
            for s in (0, 1)
        ]
        assert by_event[event]["macro_f1_mean"] == pytest.approx(statistics.mean(expected))
        assert by_event[event]["macro_f1_std"] == pytest.approx(statistics.stdev(expected))
        assert by_event[event]["n_seeds"] == 2
        assert by_event[event]["model"] == "lstm"
        assert by_event[event]["train_dataset"] == "crisislex"


def test_build_t6_groups_separately_by_train_dataset(tmp_path):
    """Same model/event evaluated from an E4 (train=crisislex) and an E5
    (train=kaggle) record must land in two different T6 rows, not pooled."""
    module = _load_make_tables_module()
    results_dir = tmp_path / "results"
    _write_predictions_csv(results_dir / "runs" / "e4-run", [{"event": "eventA", "y_true": 1, "y_pred": 1}])
    _write_predictions_csv(results_dir / "runs" / "e5-run", [{"event": "eventA", "y_true": 1, "y_pred": 0}])
    records = [
        _record(run_id="e4-run", model_name="lstm", stage="E4", split="test",
                train_dataset="crisislex", eval_dataset="crisislex", seed=0),
        _record(run_id="e5-run", model_name="lstm", stage="E5", split="test",
                train_dataset="kaggle", eval_dataset="crisislex", seed=0),
    ]
    rows = module.build_t6_per_event(records, results_dir)
    by_train_dataset = {r["train_dataset"]: r for r in rows}
    assert set(by_train_dataset) == {"crisislex", "kaggle"}
    assert by_train_dataset["crisislex"]["macro_f1_mean"] == pytest.approx(1.0)
    assert by_train_dataset["kaggle"]["macro_f1_mean"] == pytest.approx(0.0)


def test_build_t6_raises_on_missing_predictions_file(tmp_path):
    module = _load_make_tables_module()
    results_dir = tmp_path / "results"
    records = [
        _record(run_id="missing-run", model_name="lstm", stage="E4", split="test",
                train_dataset="crisislex", eval_dataset="crisislex", seed=0),
    ]
    with pytest.raises(module.MissingPredictionsError) as excinfo:
        module.build_t6_per_event(records, results_dir)
    assert "missing-run" in str(excinfo.value)


def test_build_t6_raises_when_predictions_csv_has_no_event_column(tmp_path):
    module = _load_make_tables_module()
    results_dir = tmp_path / "results"
    run_dir = results_dir / "runs" / "no-event-run"
    run_dir.mkdir(parents=True)
    pd.DataFrame({"id": [0, 1], "y_true": [1, 0], "y_pred": [1, 0]}).to_csv(run_dir / "predictions.csv", index=False)
    records = [
        _record(run_id="no-event-run", model_name="lstm", stage="E4", split="test",
                train_dataset="crisislex", eval_dataset="crisislex", seed=0),
    ]
    with pytest.raises(module.MissingPredictionsError):
        module.build_t6_per_event(records, results_dir)


def test_build_t6_raises_on_mixed_config_ids_within_model_train_dataset_group(tmp_path):
    module = _load_make_tables_module()
    results_dir = tmp_path / "results"
    _write_predictions_csv(results_dir / "runs" / "a", [{"event": "eventA", "y_true": 1, "y_pred": 1}])
    _write_predictions_csv(results_dir / "runs" / "b", [{"event": "eventA", "y_true": 1, "y_pred": 1}])
    records = [
        _record(run_id="a", model_name="lstm", stage="E4", split="test", train_dataset="crisislex",
                eval_dataset="crisislex", seed=0, config_id="cfg-a"),
        _record(run_id="b", model_name="lstm", stage="E4", split="test", train_dataset="crisislex",
                eval_dataset="crisislex", seed=1, config_id="cfg-b"),
    ]
    with pytest.raises(module.MixedConfigError):
        module.build_t6_per_event(records, results_dir)


# --- T7: reproducibility check ----------------------------------------------


def test_build_t7_computes_delta_and_handles_missing_side():
    module = _load_make_tables_module()
    records = [
        _record(run_id="e1-lstm-0", model_name="lstm", stage="E1", protocol="B", split="test", seed=0,
                metrics={"accuracy": 0, "macro_f1": 0.70, "positive_f1": 0, "weighted_f1_legacy": 0}),
        _record(run_id="e1-lstm-1", model_name="lstm", stage="E1", protocol="B", split="test", seed=1,
                metrics={"accuracy": 0, "macro_f1": 0.72, "positive_f1": 0, "weighted_f1_legacy": 0}),
        _record(run_id="e5-lstm-0", model_name="lstm", stage="E5", split="test", train_dataset="kaggle",
                eval_dataset="kaggle", seed=0,
                metrics={"accuracy": 0, "macro_f1": 0.705, "positive_f1": 0, "weighted_f1_legacy": 0}),
        # seed 1 has no E5 kaggle-eval record -- one-sided
        _record(run_id="e1-gru-0", model_name="gru", stage="E1", protocol="B", split="test", seed=0,
                metrics={"accuracy": 0, "macro_f1": 0.60, "positive_f1": 0, "weighted_f1_legacy": 0}),
        # gru has no E5 records at all -- entirely one-sided
        # a crisislex-eval E5 record for lstm/seed0 must NOT be treated as the kaggle reproducibility check
        _record(run_id="e5-lstm-0-crisislex", model_name="lstm", stage="E5", split="test",
                train_dataset="kaggle", eval_dataset="crisislex", seed=0,
                metrics={"accuracy": 0, "macro_f1": 0.999, "positive_f1": 0, "weighted_f1_legacy": 0}),
    ]
    rows = module.build_t7_reproducibility(records)
    by_model_seed = {(r["model"], r["seed"]): r for r in rows if not r["is_summary"]}

    lstm_seed0 = by_model_seed[("lstm", 0)]
    assert lstm_seed0["e1_macro_f1"] == pytest.approx(0.70)
    assert lstm_seed0["e5_macro_f1"] == pytest.approx(0.705)
    assert lstm_seed0["delta"] == pytest.approx(0.005)

    lstm_seed1 = by_model_seed[("lstm", 1)]
    assert lstm_seed1["e1_macro_f1"] == pytest.approx(0.72)
    assert lstm_seed1["e5_macro_f1"] is None
    assert lstm_seed1["delta"] is None

    gru_seed0 = by_model_seed[("gru", 0)]
    assert gru_seed0["e1_macro_f1"] == pytest.approx(0.60)
    assert gru_seed0["e5_macro_f1"] is None
    assert gru_seed0["delta"] is None

    summaries = {r["model"]: r for r in rows if r["is_summary"]}
    assert summaries["lstm"]["delta"] == pytest.approx(0.005)  # only seed0 had both sides
    assert summaries["gru"]["delta"] is None  # no seed had both sides


def test_build_t7_raises_on_mixed_config_ids_within_same_model_stage():
    module = _load_make_tables_module()
    records = [
        _record(run_id="e1-a", model_name="lstm", stage="E1", protocol="B", split="test", seed=0, config_id="cfg-a"),
        _record(run_id="e1-b", model_name="lstm", stage="E1", protocol="B", split="test", seed=1, config_id="cfg-b"),
    ]
    with pytest.raises(module.MixedConfigError) as excinfo:
        module.build_t7_reproducibility(records)
    assert "lstm" in str(excinfo.value)


def test_build_t7_allows_different_config_ids_between_e1_and_e5():
    """E1 and E5 are compared AS RUNS (the reproducibility check itself),
    not required to share a config_id -- the guard checks E1 and E5 for
    internal drift separately (grouped by (model, stage))."""
    module = _load_make_tables_module()
    records = [
        _record(run_id="e1-a", model_name="lstm", stage="E1", protocol="B", split="test", seed=0, config_id="cfg-e1"),
        _record(run_id="e5-a", model_name="lstm", stage="E5", split="test", train_dataset="kaggle",
                eval_dataset="kaggle", seed=0, config_id="cfg-e5"),
    ]
    rows = module.build_t7_reproducibility(records)  # should not raise
    assert len(rows) == 2  # 1 seed row + 1 summary row


def test_render_t7_shows_missing_side_as_em_dash():
    module = _load_make_tables_module()
    records = [
        _record(run_id="e1-lstm-0", model_name="lstm", stage="E1", protocol="B", split="test", seed=0,
                metrics={"accuracy": 0, "macro_f1": 0.70, "positive_f1": 0, "weighted_f1_legacy": 0}),
    ]
    rows = module.build_t7_reproducibility(records)
    content = module.render_t7(rows, None)
    assert "—" in content
    assert "delta = macro_f1(E5" in content


# --- Config-uniqueness guard extension: per-train_dataset use_frozen resolution ---


def test_resolve_final_config_ids_for_datasets_resolves_use_frozen_per_dataset(tmp_path):
    module = _load_make_tables_module()
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "use_frozen.yaml").write_text("hidden_size: 64\n", encoding="utf-8")
    (final_dir / "lstm.yaml").write_text("lr: 0.001\n", encoding="utf-8")

    kaggle_config_id = module.compute_config_id(
        {"hidden_size": 64, "use_cache_dir": str(tmp_path / "data" / "kaggle" / "use_embeddings")}
    )
    crisislex_config_id = module.compute_config_id(
        {"hidden_size": 64, "use_cache_dir": str(tmp_path / "data" / "crisislex" / "use_embeddings")}
    )
    assert kaggle_config_id != crisislex_config_id  # sanity: the injected dir genuinely changes the hash

    result = module.resolve_final_config_ids_for_datasets(final_dir, ["kaggle", "crisislex"], repo_root=tmp_path)

    assert result[("use_frozen", "kaggle")] == kaggle_config_id
    assert result[("use_frozen", "crisislex")] == crisislex_config_id
    # lstm's config doesn't depend on dataset -- same config_id under both keys
    lstm_config_id = module.compute_config_id({"lr": 0.001})
    assert result[("lstm", "kaggle")] == lstm_config_id
    assert result[("lstm", "crisislex")] == lstm_config_id


def test_filter_to_final_config_ids_by_train_dataset_keeps_e4_use_frozen_records(tmp_path):
    """The bug this guards against: resolving use_frozen's expected
    config_id from a single dataset="kaggle" call would never match E4's
    crisislex-trained use_frozen records (different injected use_cache_dir),
    so --only-config-ids-from would silently drop ALL of them. Per-
    train_dataset resolution keeps them."""
    module = _load_make_tables_module()
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "use_frozen.yaml").write_text("hidden_size: 64\n", encoding="utf-8")

    kaggle_config_id = module.compute_config_id(
        {"hidden_size": 64, "use_cache_dir": str(tmp_path / "data" / "kaggle" / "use_embeddings")}
    )
    crisislex_config_id = module.compute_config_id(
        {"hidden_size": 64, "use_cache_dir": str(tmp_path / "data" / "crisislex" / "use_embeddings")}
    )

    final_config_ids = module.resolve_final_config_ids_for_datasets(
        final_dir, ["kaggle", "crisislex"], repo_root=tmp_path
    )
    records = [
        _record(run_id="e5-kaggle", model_name="use_frozen", stage="E5", split="test", train_dataset="kaggle",
                eval_dataset="kaggle", seed=0, config_id=kaggle_config_id),
        _record(run_id="e4-crisislex", model_name="use_frozen", stage="E4", split="test", train_dataset="crisislex",
                eval_dataset="crisislex", seed=0, config_id=crisislex_config_id),
        _record(run_id="e4-stale", model_name="use_frozen", stage="E4", split="test", train_dataset="crisislex",
                eval_dataset="crisislex", seed=1, config_id="stale-superseded-id"),
    ]

    kept = module.filter_to_final_config_ids_by_train_dataset(records, final_config_ids)
    assert {r["run_id"] for r in kept} == {"e5-kaggle", "e4-crisislex"}

    # Sanity: the OLD single-dataset resolution would have wrongly dropped
    # the current e4-crisislex record too, since its config_id never
    # matches a "kaggle"-resolved expectation.
    old_style_final_config_ids = module.resolve_final_config_ids(final_dir, dataset="kaggle", repo_root=tmp_path)
    old_kept = module.filter_to_final_config_ids(records, old_style_final_config_ids)
    assert "e4-crisislex" not in {r["run_id"] for r in old_kept}


def test_main_only_config_ids_from_keeps_e4_use_frozen_records_end_to_end(tmp_path):
    """End-to-end via main(): a ledger with a current-config E4 use_frozen
    record and a superseded one must keep the current one and filter the
    superseded one, using the per-train_dataset resolution wired into
    main()'s --only-config-ids-from path."""
    module = _load_make_tables_module()
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "use_frozen.yaml").write_text("hidden_size: 64\n", encoding="utf-8")
    # main() doesn't take a repo_root override (matching the pre-existing
    # resolve_final_config_ids/T1-T4 behavior) -- it always resolves the
    # injected use_cache_dir against the real module.REPO_ROOT, so the
    # "current" config_id here must be computed against that same root,
    # not tmp_path, to actually match what main() resolves internally.
    current_crisislex_config_id = module.compute_config_id(
        {"hidden_size": 64, "use_cache_dir": str(module.REPO_ROOT / "data" / "crisislex" / "use_embeddings")}
    )

    results_dir = tmp_path / "results"
    _write_predictions_csv(
        results_dir / "runs" / "e4-current", [{"event": "eventA", "y_true": 1, "y_pred": 1}]
    )
    ledger_path = results_dir / "ledger.jsonl"
    records = [
        _record(run_id="e4-current", model_name="use_frozen", stage="E4", split="test", train_dataset="crisislex",
                eval_dataset="crisislex", seed=0, config_id=current_crisislex_config_id,
                metrics={"accuracy": 0, "macro_f1": 0.8, "positive_f1": 0, "weighted_f1_legacy": 0}),
        _record(run_id="e4-stale", model_name="use_frozen", stage="E4", split="test", train_dataset="crisislex",
                eval_dataset="crisislex", seed=1, config_id="stale-id",
                metrics={"accuracy": 0, "macro_f1": 0.1, "positive_f1": 0, "weighted_f1_legacy": 0}),
    ]
    _write_ledger(ledger_path, records)

    written = module.main(ledger_path, tmp_path / "tables", only_config_ids_from=final_dir)
    t6_content = written["T6"].read_text(encoding="utf-8")

    # Independently recompute the expected T6 content from just the
    # current-config record (what --only-config-ids-from is supposed to
    # reduce `records` to) and compare byte-for-byte against what main()
    # actually wrote -- proves main()'s per-train_dataset filtering kept
    # e4-current and dropped e4-stale, not merely that SOME row exists.
    expected_rows = module.build_t6_per_event([records[0]], results_dir)
    assert expected_rows[0]["n_seeds"] == 1
    expected_content = module.render_t6(expected_rows, None)
    assert t6_content == expected_content


# --- Wiring: T5/T6/T7 are wired into main(), and legacy records don't break them ---


def test_main_writes_t5_t6_t7_files(tmp_path):
    module = _load_make_tables_module()
    ledger_path = tmp_path / "ledger.jsonl"
    _write_ledger(ledger_path, [_record(run_id="real-1")])
    written = module.main(ledger_path, tmp_path / "tables")
    assert set(written) == {"T1", "T2", "T3", "T4", "T5", "T6", "T7"}
    assert written["T5"].name == "T5_cross_dataset.md"
    assert written["T6"].name == "T6_per_event.md"
    assert written["T7"].name == "T7_reproducibility.md"
    for name in ("T5", "T6", "T7"):
        assert written[name].exists()


def test_main_excludes_smoke_e5_records_from_t5_by_default(tmp_path):
    module = _load_make_tables_module()
    ledger_path = tmp_path / "ledger.jsonl"
    _write_ledger(
        ledger_path,
        [
            _record(run_id="smoke-e5", model_name="lstm", stage="E5", split="test", smoke=True,
                     train_dataset="kaggle", eval_dataset="kaggle", seed=0,
                     metrics={"accuracy": 0, "macro_f1": 0.99, "positive_f1": 0, "weighted_f1_legacy": 0}),
        ],
    )
    written = module.main(ledger_path, tmp_path / "tables")
    t5_content = written["T5"].read_text(encoding="utf-8")
    assert "lstm" not in t5_content


def test_main_handles_legacy_record_without_train_eval_dataset_fields(tmp_path):
    """A pre-Phase-2 ledger line has no train_dataset/eval_dataset keys at
    all; dtc.harness.ledger.read_ledger backfills them (to the record's own
    `dataset`, kaggle) at READ time. main() must not KeyError building any
    of T1-T7 against a ledger containing such a line. T5/T6 (E4/E5-only)
    simply don't include it (its stage is E1); T7 legitimately DOES include
    it (E1 is one of its two inputs) with the E5 side rendered as "-"."""
    module = _load_make_tables_module()
    ledger_path = tmp_path / "ledger.jsonl"
    legacy_record = _record(
        run_id="legacy-1", model_name="lstm", stage="E1", protocol="B", split="test", seed=0,
        metrics={"accuracy": 0, "macro_f1": 0.75, "positive_f1": 0, "weighted_f1_legacy": 0},
    )
    assert "train_dataset" not in legacy_record
    assert "eval_dataset" not in legacy_record
    _write_ledger(ledger_path, [legacy_record])

    written = module.main(ledger_path, tmp_path / "tables")  # should not raise
    t1_content = written["T1"].read_text(encoding="utf-8")
    assert "lstm" in t1_content
    for name in ("T5", "T6"):
        content = written[name].read_text(encoding="utf-8")
        assert f"T{name[1]}" in content  # title rendered
        assert "lstm" not in content  # legacy E1 record isn't E4/E5, so excluded

    t7_content = written["T7"].read_text(encoding="utf-8")
    assert "lstm" in t7_content  # E1 is a real T7 input, backfill or not
    assert "0.7500" in t7_content  # its e1_macro_f1
    assert "—" in t7_content  # e5_macro_f1 side is missing
