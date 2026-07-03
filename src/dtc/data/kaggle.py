"""Kaggle "NLP Getting Started" dataset pipeline: dedup policy + stratified split.

Dedup policy and split logic are shared with dtc.data.crisislex via
dtc.data.common (see docs/DATASETS.md and docs/PLAN.md 0.2/0.3) so that
both datasets go through literally the same code path, not two copies.

This module contains only pure/testable pipeline logic. I/O orchestration
(reading config, writing files) lives in scripts/prepare_kaggle.py.
"""

from __future__ import annotations

import pandas as pd

from dtc.data.common import (
    SplitRatios,
    class_balance,
    count_exact_duplicate_rows,
    dataframe_csv_bytes,
    resolve_duplicates,
    sha256_bytes,
    sha256_file,
    stratified_split,
)

__all__ = [
    "SplitRatios",
    "class_balance",
    "count_exact_duplicate_rows",
    "dataframe_csv_bytes",
    "resolve_duplicates",
    "sha256_bytes",
    "sha256_file",
    "stratified_split",
    "build_manifest",
]


def build_manifest(
    *,
    raw_csv_path: str,
    raw_sha256: str,
    raw_row_count: int,
    dropped_conflicting_count: int,
    dropped_conflicting_group_count: int,
    exact_duplicate_row_count: int,
    deduped_row_count: int,
    seed: int,
    ratios: SplitRatios,
    splits: dict[str, pd.DataFrame],
    split_hashes: dict[str, str],
    label_col: str,
) -> dict:
    return {
        "dataset": "kaggle_nlp_getting_started",
        "raw_csv_path": raw_csv_path,
        "raw_sha256": raw_sha256,
        "raw_row_count": raw_row_count,
        "exact_duplicate_row_count": exact_duplicate_row_count,
        "dropped_conflicting_row_count": dropped_conflicting_count,
        "dropped_conflicting_group_count": dropped_conflicting_group_count,
        "deduped_row_count": deduped_row_count,
        "seed": seed,
        "split_ratios": {"train": ratios.train, "val": ratios.val, "test": ratios.test},
        "splits": {
            name: {
                "row_count": len(split_df),
                "sha256": split_hashes[name],
                "class_balance": class_balance(split_df, label_col),
            }
            for name, split_df in splits.items()
        },
    }