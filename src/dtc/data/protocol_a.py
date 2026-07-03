"""Protocol A: the replicated-flawed evaluation protocol (docs/PLAN.md 1.3).

Deliberately isolated from Protocol B (Hard Rule 7): reads the RAW Kaggle
csv directly (no dedup policy applied), does a 90/10 NON-stratified split
with a documented seed, and truncates at the MEAN token length rather than
Protocol B's 95th percentile. These are the flaws being *replicated* for
the contrast experiment, not bugs -- do not "fix" anything in this module.

This module must never import the frozen-test-set guard module from
dtc.eval (Protocol A's 10% holdout is a different split of a different,
non-deduped dataframe -- not the Protocol B frozen test split), and no
Protocol B code path (dtc.data.kaggle, dtc.data.common, dtc.models.*,
dtc.train.*) may import this module. Both directions are enforced by
tests/test_protocol_isolation.py.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from dtc.data.text import whitespace_token_count

PROTOCOL_A_SEED = 42
PROTOCOL_A_TEST_SIZE = 0.10


def load_raw_kaggle(raw_csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(raw_csv_path)


def protocol_a_split(
    df: pd.DataFrame, seed: int = PROTOCOL_A_SEED, test_size: float = PROTOCOL_A_TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """90/10, NON-stratified split (the flaw under study, not corrected here)."""
    train_df, eval_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def mean_token_length(texts: Sequence[str]) -> int:
    """Truncate-at-MEAN policy: the original audited project's approach,
    replicated here on purpose so Protocol A vs. B isolates this exact
    confound (docs/PLAN.md 1.1/1.3)."""
    counts = [whitespace_token_count(t) for t in texts]
    return max(1, int(math.ceil(sum(counts) / len(counts))))
