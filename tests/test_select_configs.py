"""Tests for scripts/select_configs.py against a synthetic ledger fixture."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_select_configs_module():
    spec = importlib.util.spec_from_file_location("select_configs", REPO_ROOT / "scripts" / "select_configs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["select_configs"] = module
    spec.loader.exec_module(module)
    return module


def _record(model_name, macro_f1, config, stage="tuning"):
    return {
        "model_name": model_name,
        "stage": stage,
        "config": config,
        "metrics": {"macro_f1": macro_f1},
    }


def test_select_best_configs_picks_highest_macro_f1_per_model():
    module = _load_select_configs_module()
    records = [
        _record("lstm", 0.70, {"lr": 0.001, "hidden_size": 32}),
        _record("lstm", 0.85, {"lr": 0.001, "hidden_size": 64}),
        _record("lstm", 0.60, {"lr": 0.0005, "hidden_size": 128}),
        _record("gru", 0.50, {"lr": 0.001, "hidden_size": 32}),
        _record("gru", 0.55, {"lr": 0.0005, "hidden_size": 64}),
    ]
    winners = module.select_best_configs(records)
    assert winners["lstm"] == {"lr": 0.001, "hidden_size": 64}
    assert winners["gru"] == {"lr": 0.0005, "hidden_size": 64}


def test_select_best_configs_ignores_non_tuning_stage_records():
    module = _load_select_configs_module()
    records = [
        _record("lstm", 0.99, {"lr": 0.1, "hidden_size": 999}, stage="final"),
        _record("lstm", 0.70, {"lr": 0.001, "hidden_size": 32}, stage="tuning"),
        _record("lstm", 0.30, {"lr": 0.001, "hidden_size": 8}, stage=None),
    ]
    winners = module.select_best_configs(records)
    assert winners["lstm"] == {"lr": 0.001, "hidden_size": 32}


def test_select_best_configs_respects_models_filter():
    module = _load_select_configs_module()
    records = [
        _record("lstm", 0.70, {"lr": 0.001}),
        _record("gru", 0.80, {"lr": 0.001}),
    ]
    winners = module.select_best_configs(records, models=["lstm"])
    assert set(winners.keys()) == {"lstm"}


def test_write_final_configs_and_end_to_end_main(tmp_path):
    module = _load_select_configs_module()
    ledger_path = tmp_path / "ledger.jsonl"
    import json

    records = [
        _record("lstm", 0.70, {"lr": 0.001, "hidden_size": 32}),
        _record("lstm", 0.85, {"lr": 0.001, "hidden_size": 64}),
    ]
    with open(ledger_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    output_dir = tmp_path / "final"
    written = module.main(ledger_path, output_dir)
    assert set(written.keys()) == {"lstm"}
    written_config = yaml.safe_load(written["lstm"].read_text(encoding="utf-8"))
    assert written_config == {"lr": 0.001, "hidden_size": 64}


def test_main_with_no_tuning_records_writes_nothing(tmp_path):
    module = _load_select_configs_module()
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text("", encoding="utf-8")
    output_dir = tmp_path / "final"
    written = module.main(ledger_path, output_dir)
    assert written == {}
