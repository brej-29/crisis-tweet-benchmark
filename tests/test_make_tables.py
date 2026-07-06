"""Tests for scripts/make_tables.py against synthetic ledger fixtures:
T1-T4 aggregation correctness, smoke exclusion/inclusion, idempotency, and
the dirty-source-path refusal.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

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
