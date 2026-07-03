"""Kaggle "NLP Getting Started" dataset pipeline: dedup policy + stratified split.

Dedup policy (see docs/DATASETS.md and docs/PLAN.md 0.2):
  - Exact-text duplicate rows with a single, consistent label -> keep the
    first occurrence only.
  - Exact-text duplicate rows with conflicting labels across the group ->
    drop the ENTIRE group (no principled adjudication available; removal is
    the conservative choice and is reported, not hidden).

This module contains only pure/testable pipeline logic. I/O orchestration
(reading config, writing files) lives in scripts/prepare_kaggle.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitRatios:
    train: float
    val: float
    test: float

    def __post_init__(self):
        total = self.train + self.val + self.test
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"split ratios must sum to 1.0, got {total}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def dataframe_csv_bytes(df: pd.DataFrame) -> bytes:
    """Canonical CSV serialization used for hashing (fixed line terminator)."""
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


def count_exact_duplicate_rows(df: pd.DataFrame, text_col: str) -> int:
    """Rows sharing text with at least one earlier row (first occurrence excluded)."""
    return int(df[text_col].duplicated(keep="first").sum())


def resolve_duplicates(
    df: pd.DataFrame, text_col: str, label_col: str, id_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the exact-text duplicate policy.

    Returns (deduped_df, dropped_conflicting_df). `dropped_conflicting_df`
    contains every row (not just the extras) belonging to a conflicting-label
    group, for full audit-trail logging.
    """
    label_nunique = df.groupby(text_col)[label_col].transform("nunique")
    conflicting_mask = label_nunique > 1

    dropped_conflicting = df.loc[conflicting_mask, [id_col, text_col, label_col]].copy()
    non_conflicting = df.loc[~conflicting_mask].copy()

    deduped = non_conflicting.drop_duplicates(subset=[text_col], keep="first").reset_index(drop=True)
    dropped_conflicting = dropped_conflicting.reset_index(drop=True)

    return deduped, dropped_conflicting


def stratified_split(
    df: pd.DataFrame, label_col: str, ratios: SplitRatios, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """80/10/10-style stratified split via two sequential stratified splits."""
    train_df, temp_df = train_test_split(
        df,
        test_size=(ratios.val + ratios.test),
        stratify=df[label_col],
        random_state=seed,
    )
    relative_test_size = ratios.test / (ratios.val + ratios.test)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df[label_col],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def class_balance(df: pd.DataFrame, label_col: str) -> dict:
    counts = df[label_col].value_counts().to_dict()
    total = len(df)
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "positive_rate": float(df[label_col].mean()),
        "total": total,
    }


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
