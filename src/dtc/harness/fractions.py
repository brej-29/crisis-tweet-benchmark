"""Seeded, nested stratified train-fraction subsampling for the
data-efficiency experiment (E3, docs/PLAN.md 1.4).

"Nested" means the 1% subset is literally a subset of the 5% subset, which
is a subset of the 10% subset, and so on -- achieved by drawing one
per-label shuffle order per seed and taking prefixes of increasing length,
rather than re-sampling independently at each fraction (which would not
nest).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def nested_fraction_indices(labels: np.ndarray, fractions: list[float], seed: int) -> dict[float, np.ndarray]:
    """Per fraction, the row-index array of that fraction's stratified
    subsample. For a fixed seed, the indices for a smaller fraction are
    always a subset of the indices for any larger fraction.
    """
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    per_label_shuffled = {}
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        per_label_shuffled[label] = shuffled

    result = {}
    for fraction in fractions:
        chosen = []
        for shuffled in per_label_shuffled.values():
            n = max(1, int(round(len(shuffled) * fraction)))
            chosen.append(shuffled[:n])
        result[fraction] = np.sort(np.concatenate(chosen))
    return result


def subsample_train_df(
    train_df: pd.DataFrame, label_col: str, fraction: float, fractions: list[float], seed: int
) -> pd.DataFrame:
    """The `fraction` subsample of `train_df`, nested (for the given seed)
    within every larger fraction in `fractions`. `fractions` must include
    `fraction` itself and should be the experiment's full fraction list, so
    repeated calls at different fractions (same seed) are mutually nested.
    """
    indices = nested_fraction_indices(train_df[label_col].to_numpy(), fractions, seed)
    return train_df.iloc[indices[fraction]].reset_index(drop=True)
