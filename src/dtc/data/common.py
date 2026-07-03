"""Dataset-agnostic pipeline primitives shared by every dataset pipeline.

Both dtc.data.kaggle and dtc.data.crisislex build on these so that "same
dedup + stratified split + frozen-test discipline" (docs/PLAN.md 0.3) is
literally one code path, not two independently-maintained copies.
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

    - Exact-text duplicate rows with a single, consistent label -> keep the
      first occurrence only.
    - Exact-text duplicate rows with conflicting labels across the group ->
      drop the ENTIRE group (no principled adjudication available; removal
      is the conservative choice and is reported, not hidden).

    Returns (deduped_df, dropped_conflicting_df). `dropped_conflicting_df`
    contains every row (not just the extras) belonging to a conflicting-label
    group, for full audit-trail logging. Both retain all original columns of
    `df` intersected with [id_col, text_col, label_col] for the dropped log,
    and all original columns for the deduped frame.
    """
    label_nunique = df.groupby(text_col)[label_col].transform("nunique")
    conflicting_mask = label_nunique > 1

    log_cols = [c for c in (id_col, text_col, label_col) if c in df.columns]
    dropped_conflicting = df.loc[conflicting_mask, log_cols].copy()
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