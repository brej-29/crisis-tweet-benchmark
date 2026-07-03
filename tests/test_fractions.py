"""Tests for dtc.harness.fractions: nested, seeded, stratified train-fraction
subsampling used by E3 (docs/PLAN.md 1.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtc.harness.fractions import nested_fraction_indices, subsample_train_df


def test_nested_fraction_indices_are_subsets_of_larger_fractions():
    labels = np.array([0] * 500 + [1] * 500)
    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
    result = nested_fraction_indices(labels, fractions, seed=0)
    for smaller, larger in zip(fractions, fractions[1:]):
        assert set(result[smaller].tolist()) <= set(result[larger].tolist())
    assert set(result[1.0].tolist()) == set(range(len(labels)))


def test_nested_fraction_indices_is_deterministic_given_seed():
    labels = np.array([0] * 100 + [1] * 100)
    r1 = nested_fraction_indices(labels, [0.1, 0.5], seed=7)
    r2 = nested_fraction_indices(labels, [0.1, 0.5], seed=7)
    np.testing.assert_array_equal(r1[0.1], r2[0.1])
    np.testing.assert_array_equal(r1[0.5], r2[0.5])


def test_nested_fraction_indices_differ_across_seeds():
    labels = np.array([0] * 100 + [1] * 100)
    r1 = nested_fraction_indices(labels, [0.1], seed=1)
    r2 = nested_fraction_indices(labels, [0.1], seed=2)
    assert not np.array_equal(r1[0.1], r2[0.1])


def test_nested_fraction_indices_are_approximately_stratified():
    labels = np.array([0] * 800 + [1] * 200)
    result = nested_fraction_indices(labels, [0.5], seed=1)
    idx = result[0.5]
    positive_rate = labels[idx].mean()
    assert abs(positive_rate - 0.2) < 0.03


def test_subsample_train_df_returns_expected_row_count_and_nesting():
    df = pd.DataFrame({"text": [f"t{i}" for i in range(100)], "label": [0] * 50 + [1] * 50})
    sub_10 = subsample_train_df(df, "label", fraction=0.1, fractions=[0.1, 1.0], seed=3)
    sub_100 = subsample_train_df(df, "label", fraction=1.0, fractions=[0.1, 1.0], seed=3)
    assert len(sub_10) == 10
    assert len(sub_100) == 100
    assert set(sub_10["text"]) <= set(sub_100["text"])
