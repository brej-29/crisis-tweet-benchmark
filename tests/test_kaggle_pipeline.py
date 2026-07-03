"""Tests for the Kaggle dataset pipeline: dedup policy, stratified split,
and (as an integration test) the full prepare_kaggle script against the
real, locally-provided train.csv.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dtc.data.kaggle import (
    SplitRatios,
    count_exact_duplicate_rows,
    resolve_duplicates,
    stratified_split,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = REPO_ROOT / "data" / "kaggle" / "raw" / "train.csv"


def _load_prepare_kaggle_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_kaggle", REPO_ROOT / "scripts" / "prepare_kaggle.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_kaggle"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Unit tests: dedup policy on a tiny synthetic fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "text": ["a", "a", "b", "b", "b", "c", "d"],
            "target": [1, 1, 0, 1, 0, 1, 0],
            #          ^consistent dup   ^conflicting group (0/1/0)  ^unique  ^unique
        }
    )


def test_resolve_duplicates_keeps_first_of_consistent_duplicate(synthetic_df):
    deduped, dropped = resolve_duplicates(synthetic_df, "text", "target", "id")
    a_rows = deduped[deduped["text"] == "a"]
    assert len(a_rows) == 1
    assert a_rows.iloc[0]["id"] == 1


def test_resolve_duplicates_drops_entire_conflicting_group(synthetic_df):
    deduped, dropped = resolve_duplicates(synthetic_df, "text", "target", "id")
    assert "b" not in set(deduped["text"])
    assert set(dropped["text"]) == {"b"}
    assert len(dropped) == 3
    assert set(dropped["id"]) == {3, 4, 5}


def test_resolve_duplicates_keeps_unique_rows(synthetic_df):
    deduped, _ = resolve_duplicates(synthetic_df, "text", "target", "id")
    assert set(deduped["text"]) == {"a", "c", "d"}
    assert len(deduped) == 3


def test_count_exact_duplicate_rows(synthetic_df):
    # 'a' appears twice (1 duplicate row), 'b' appears three times (2 duplicate rows)
    assert count_exact_duplicate_rows(synthetic_df, "text") == 3


# ---------------------------------------------------------------------------
# Unit tests: stratified split
# ---------------------------------------------------------------------------


def test_stratified_split_no_overlap_and_sizes():
    df = pd.DataFrame(
        {
            "text": [f"t{i}" for i in range(1000)],
            "target": [1 if i % 3 == 0 else 0 for i in range(1000)],
        }
    )
    ratios = SplitRatios(train=0.8, val=0.1, test=0.1)
    train_df, val_df, test_df = stratified_split(df, "target", ratios, seed=123)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    train_texts, val_texts, test_texts = (
        set(train_df["text"]),
        set(val_df["text"]),
        set(test_df["text"]),
    )
    assert train_texts.isdisjoint(val_texts)
    assert train_texts.isdisjoint(test_texts)
    assert val_texts.isdisjoint(test_texts)

    global_rate = df["target"].mean()
    for split_df in (train_df, val_df, test_df):
        assert abs(split_df["target"].mean() - global_rate) < 0.05


def test_stratified_split_is_deterministic_given_seed():
    df = pd.DataFrame(
        {
            "text": [f"t{i}" for i in range(500)],
            "target": [1 if i % 4 == 0 else 0 for i in range(500)],
        }
    )
    ratios = SplitRatios(train=0.8, val=0.1, test=0.1)
    run1 = stratified_split(df, "target", ratios, seed=7)
    run2 = stratified_split(df, "target", ratios, seed=7)
    for a, b in zip(run1, run2):
        pd.testing.assert_frame_equal(a, b)


def test_split_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        SplitRatios(train=0.8, val=0.1, test=0.2)


# ---------------------------------------------------------------------------
# Integration test: full script against the real, locally-provided dataset
# ---------------------------------------------------------------------------

pytestmark_real_data = pytest.mark.skipif(
    not RAW_CSV.exists(), reason="real Kaggle train.csv not present locally"
)


@pytestmark_real_data
def test_prepare_kaggle_end_to_end_no_leakage_and_balance(tmp_path):
    module = _load_prepare_kaggle_module()

    config = {
        "dataset": "kaggle_nlp_getting_started",
        "raw_csv_path": str(RAW_CSV),
        "output_dir": str(tmp_path),
        "seed": 42,
        "text_col": "text",
        "label_col": "target",
        "id_col": "id",
        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
    }
    config_path = tmp_path / "kaggle_test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    manifest = module.main(str(config_path))

    train_df = pd.read_csv(tmp_path / "train.csv")
    val_df = pd.read_csv(tmp_path / "val.csv")
    test_df = pd.read_csv(tmp_path / "test.csv")

    # No exact-text overlap across any pair of splits.
    train_texts, val_texts, test_texts = (
        set(train_df["text"]),
        set(val_df["text"]),
        set(test_df["text"]),
    )
    assert train_texts.isdisjoint(val_texts), "train/val leakage detected"
    assert train_texts.isdisjoint(test_texts), "train/test leakage detected"
    assert val_texts.isdisjoint(test_texts), "val/test leakage detected"

    # Sizes sum to the post-dedup total.
    deduped_total = manifest["deduped_row_count"]
    assert len(train_df) + len(val_df) + len(test_df) == deduped_total

    # Per-split class balance within +/-1% of the post-dedup global rate.
    global_rate = (train_df["target"].sum() + val_df["target"].sum() + test_df["target"].sum()) / deduped_total
    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        rate = split_df["target"].mean()
        assert abs(rate - global_rate) < 0.01, f"{name} class balance drifted: {rate} vs {global_rate}"

    # Measured (not assumed) dedup counts are internally consistent:
    # every row removed between raw and deduped is either the "extra" occurrence
    # of a consistently-labeled duplicate (counted once per group in
    # exact_duplicate_row_count) or belongs to a dropped conflicting group
    # (one group -> its full first-occurrence count is also captured by
    # dropped_conflicting_group_count, since conflicting groups are removed whole).
    assert manifest["raw_row_count"] == len(pd.read_csv(RAW_CSV))
    assert manifest["raw_row_count"] - manifest["deduped_row_count"] == (
        manifest["exact_duplicate_row_count"] + manifest["dropped_conflicting_group_count"]
    )


@pytestmark_real_data
def test_prepare_kaggle_is_reproducible(tmp_path):
    module = _load_prepare_kaggle_module()
    config = {
        "dataset": "kaggle_nlp_getting_started",
        "raw_csv_path": str(RAW_CSV),
        "output_dir": None,
        "seed": 42,
        "text_col": "text",
        "label_col": "target",
        "id_col": "id",
        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
    }

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    for out_dir in (out1, out2):
        config["output_dir"] = str(out_dir)
        config_path = out_dir.parent / f"{out_dir.name}_config.yaml"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)
        module.main(str(config_path))

    manifest1 = (out1 / "manifest.json").read_text(encoding="utf-8")
    manifest2 = (out2 / "manifest.json").read_text(encoding="utf-8")
    assert manifest1 == manifest2, "re-running the pipeline with the same seed must reproduce identical output"