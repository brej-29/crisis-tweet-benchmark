"""Tests for dtc.eval.run_evaluation, the only code path allowed to read
the frozen test split for evaluation (docs/PLAN.md standing rule)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtc.eval.run_evaluation import evaluate_model_on_frozen_test, load_frozen_test_standardized


def test_load_frozen_test_standardized_renames_kaggle_columns(tmp_path):
    dataset_dir = tmp_path / "data" / "kaggle"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "text": ["a", "b"], "target": [0, 1]}).to_csv(dataset_dir / "test.csv", index=False)
    df = load_frozen_test_standardized(tmp_path, "kaggle")
    assert list(df.columns) == ["id", "text", "label"]
    assert df["label"].tolist() == [0, 1]


def test_load_frozen_test_standardized_renames_crisislex_columns(tmp_path):
    dataset_dir = tmp_path / "data" / "crisislex"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame({"tweet_id": ["1", "2"], "text": ["a", "b"], "label": [0, 1]}).to_csv(
        dataset_dir / "test.csv", index=False
    )
    df = load_frozen_test_standardized(tmp_path, "crisislex")
    assert list(df.columns) == ["id", "text", "label"]


class _FakeModel:
    def predict_proba(self, texts):
        return np.array([0.9 if "flood" in t else 0.1 for t in texts])


def test_evaluate_model_on_frozen_test_returns_expected_fields(tmp_path):
    dataset_dir = tmp_path / "data" / "kaggle"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame(
        {"id": [1, 2], "text": ["flood warning", "nice day"], "target": [1, 0]}
    ).to_csv(dataset_dir / "test.csv", index=False)
    result = evaluate_model_on_frozen_test(_FakeModel(), tmp_path, "kaggle")
    assert list(result["y_true"]) == [1, 0]
    assert list(result["y_pred"]) == [1, 0]
    assert len(result["ids"]) == 2
    assert len(result["texts"]) == 2
