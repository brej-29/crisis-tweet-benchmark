"""CrisisLex T6 dataset pipeline: load 6 per-event files, map labels, dedup,
stratified split.

Dedup policy and split logic are shared with dtc.data.kaggle via
dtc.data.common (see docs/DATASETS.md and docs/PLAN.md 0.2/0.3) so both
datasets go through literally the same code path, not two copies.

Source file format (verified against the actual distributed files, see
docs/DATASETS.md): one CSV per event, header `tweet id, tweet, label`,
comma-separated with quoted fields (tweet text may contain embedded commas
and newlines). The `tweet id` field values are wrapped in literal single
quotes (e.g. "'325208201740029952'") as part of the CSV cell content, not
CSV quoting -- these are stripped on load.

This module contains only pure/testable pipeline logic. I/O orchestration
(reading config, writing files) lives in scripts/prepare_crisislex.py.
"""

from __future__ import annotations

import pandas as pd

from dtc.data.common import (
    SplitRatios,  # noqa: F401 (re-exported for callers/tests)
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
    "EVENTS",
    "LABEL_MAP",
    "class_balance",
    "count_exact_duplicate_rows",
    "dataframe_csv_bytes",
    "load_event_csv",
    "load_and_combine_events",
    "map_labels",
    "resolve_duplicates",
    "sha256_bytes",
    "sha256_file",
    "stratified_split",
    "build_manifest",
]

EVENTS = [
    "2012_Sandy_Hurricane",
    "2013_Alberta_Floods",
    "2013_Boston_Bombings",
    "2013_Oklahoma_Tornado",
    "2013_Queensland_Floods",
    "2013_West_Texas_Explosion",
]

# on-topic (related to the crisis) -> 1 ; off-topic (unrelated) -> 0
# Documented mismatch vs. the Kaggle dataset's "real disaster" semantics:
# see docs/DATASETS.md.
LABEL_MAP = {"on-topic": 1, "off-topic": 0}


def load_event_csv(path) -> pd.DataFrame:
    """Load a single CrisisLex T6 event file into a clean DataFrame.

    Columns returned: event (filled in by the caller), tweet_id, text, label_raw.
    """
    df = pd.read_csv(path, skipinitialspace=True)
    df = df.rename(columns={"tweet id": "tweet_id", "tweet": "text", "label": "label_raw"})
    df["tweet_id"] = df["tweet_id"].astype(str).str.strip("'")
    return df[["tweet_id", "text", "label_raw"]]


def load_and_combine_events(raw_dir, events: list[str] = EVENTS) -> pd.DataFrame:
    """Load all 6 event files and concatenate, preserving an `event` column."""
    frames = []
    for event in events:
        path = raw_dir / f"{event}-ontopic_offtopic.csv"
        df = load_event_csv(path)
        df.insert(0, "event", event)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def map_labels(df: pd.DataFrame, label_raw_col: str = "label_raw") -> pd.DataFrame:
    """Map on-topic/off-topic strings to 1/0. Raises on any unmapped value."""
    unknown = set(df[label_raw_col].unique()) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"Unmapped CrisisLex label values encountered: {unknown}")
    df = df.copy()
    df["label"] = df[label_raw_col].map(LABEL_MAP)
    return df


def per_event_breakdown(df: pd.DataFrame, label_col: str = "label") -> dict:
    breakdown = {}
    for event, group in df.groupby("event"):
        breakdown[event] = class_balance(group, label_col)
    return breakdown


def build_manifest(
    *,
    raw_dir: str,
    raw_file_hashes: dict[str, str],
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
        "dataset": "crisislex_t6",
        "raw_dir": raw_dir,
        "raw_file_sha256": raw_file_hashes,
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
                "per_event_class_balance": per_event_breakdown(split_df, label_col),
            }
            for name, split_df in splits.items()
        },
    }