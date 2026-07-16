"""The only code path allowed to read the frozen test split for evaluation
(docs/PLAN.md standing rule / Hard Rule 1). scripts/run_matrix.py calls
into this module rather than reading data/<dataset>/test.csv directly, so
`load_frozen_test`'s caller-allowlist (dtc.eval.*) is satisfied by
construction, not by convention.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dtc.eval.frozen_test_loader import load_frozen_test

DATASET_LABEL_COLUMNS = {"kaggle": "target", "crisislex": "label"}
DATASET_ID_COLUMNS = {"kaggle": "id", "crisislex": "tweet_id"}


def load_frozen_test_standardized(repo_root: str | Path, dataset: str) -> pd.DataFrame:
    path = Path(repo_root) / "data" / dataset / "test.csv"
    df = load_frozen_test(path)
    label_col = DATASET_LABEL_COLUMNS[dataset]
    id_col = DATASET_ID_COLUMNS[dataset]
    return df.rename(columns={label_col: "label", id_col: "id"})[["id", "text", "label"]]


def evaluate_model_on_frozen_test(model, repo_root: str | Path, dataset: str) -> dict:
    """Runs `model.predict_proba` on the frozen test split and returns the
    fields dtc.harness.run.log_evaluation_run needs (ids/texts/y_true/
    y_pred/y_prob). Evaluated ONCE per (run, dataset) -- callers must not
    call this more than once per trained model instance for a given
    experiment run and eval dataset (cross-dataset E4/E5 call it once per
    frozen test, which is one ledgered eval record each).
    """
    test_df = load_frozen_test_standardized(repo_root, dataset)
    y_prob = model.predict_proba(test_df["text"])
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "ids": test_df["id"],
        "texts": test_df["text"],
        "y_true": test_df["label"].to_numpy(),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
