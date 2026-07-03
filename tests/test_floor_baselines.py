"""Tests for floor baselines (majority-class, stratified-random) and the
end-to-end scripts/run_floor_baselines.py entrypoint against the real
Kaggle val split.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from dtc.harness.ledger import read_ledger
from dtc.models.floor_baselines import fit_majority_class, fit_stratified_random, predict

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_VAL_CSV = REPO_ROOT / "data" / "kaggle" / "val.csv"


def test_majority_class_always_predicts_the_majority_label():
    y_train = np.array([0, 0, 0, 1, 1])
    clf = fit_majority_class(y_train, seed=42)
    y_pred, y_prob = predict(clf, n=10)
    assert set(y_pred.tolist()) == {0}
    assert y_prob is not None
    assert len(y_pred) == 10


def test_stratified_random_is_seeded_deterministically():
    y_train = np.array([0, 0, 0, 1, 1, 1, 1])
    clf1 = fit_stratified_random(y_train, seed=7)
    clf2 = fit_stratified_random(y_train, seed=7)
    y_pred1, _ = predict(clf1, n=200)
    y_pred2, _ = predict(clf2, n=200)
    np.testing.assert_array_equal(y_pred1, y_pred2)


def test_stratified_random_roughly_matches_train_distribution():
    rng_labels = np.array([0] * 700 + [1] * 300)
    clf = fit_stratified_random(rng_labels, seed=1)
    y_pred, _ = predict(clf, n=5000)
    positive_rate = y_pred.mean()
    assert abs(positive_rate - 0.3) < 0.03


pytestmark_real_data = pytest.mark.skipif(
    not KAGGLE_VAL_CSV.exists(), reason="Kaggle val.csv not present locally (run scripts/prepare_kaggle.py first)"
)


def _load_run_floor_baselines_module():
    spec = importlib.util.spec_from_file_location(
        "run_floor_baselines", REPO_ROOT / "scripts" / "run_floor_baselines.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_floor_baselines"] = module
    spec.loader.exec_module(module)
    return module


@pytestmark_real_data
def test_run_floor_baselines_end_to_end_ledgers_two_runs(tmp_path, monkeypatch):
    module = _load_run_floor_baselines_module()

    # Redirect the script's ledger/results output to a temp dir so this test
    # doesn't append extra lines to the committed results/ledger.jsonl.
    monkeypatch.setattr(module, "REPO_ROOT", REPO_ROOT)
    tmp_ledger = tmp_path / "ledger.jsonl"
    tmp_results = tmp_path

    import dtc.harness.run as run_module

    original_log = run_module.log_evaluation_run

    def patched_log(**kwargs):
        kwargs["ledger_path"] = tmp_ledger
        kwargs["results_dir"] = tmp_results
        return original_log(**kwargs)

    monkeypatch.setattr(module, "log_evaluation_run", patched_log)

    records = module.main(seed=42)

    assert len(records) == 2
    model_names = {r["model_name"] for r in records}
    assert model_names == {"majority_class", "stratified_random"}
    for r in records:
        assert r["dataset"] == "kaggle_nlp_getting_started"
        assert r["split"] == "val"
        assert "accuracy" in r["metrics"]

    ledgered = read_ledger(tmp_ledger)
    assert len(ledgered) == 2