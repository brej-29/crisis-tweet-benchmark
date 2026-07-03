"""Thin, dataset-agnostic loaders that standardize train/val split CSVs to
the model interface's expected ("id", "text", "label") columns.

Deliberately does NOT handle the frozen test split -- that is
`dtc.eval.run_evaluation`'s job, so the frozen-test-set guard's
caller-allowlist (dtc.eval.*) is satisfied by construction rather than by
convention. This module is safe for the run driver and training code to
import directly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASET_LABEL_COLUMNS = {"kaggle": "target", "crisislex": "label"}
DATASET_ID_COLUMNS = {"kaggle": "id", "crisislex": "tweet_id"}


def load_split_standardized(repo_root: str | Path, dataset: str, split: str) -> pd.DataFrame:
    """Loads data/<dataset>/<split>.csv and renames columns to id/text/label.

    `split` must be "train" or "val" -- use dtc.eval.run_evaluation for
    "test" (the frozen split).
    """
    if split == "test":
        raise ValueError(
            "load_split_standardized() does not serve the frozen test split; "
            "use dtc.eval.run_evaluation.load_frozen_test_standardized() instead."
        )
    path = Path(repo_root) / "data" / dataset / f"{split}.csv"
    df = pd.read_csv(path)
    label_col = DATASET_LABEL_COLUMNS[dataset]
    id_col = DATASET_ID_COLUMNS[dataset]
    return df.rename(columns={label_col: "label", id_col: "id"})[["id", "text", "label"]]
