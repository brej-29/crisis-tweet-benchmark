"""Tests for the CrisisLex T6 dataset pipeline: label mapping, event
combination, and (as an integration test) the full prepare_crisislex script
against the real, locally-provided event CSVs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dtc.data.crisislex import LABEL_MAP, load_event_csv, map_labels

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "crisislex" / "raw"
EVENTS = [
    "2012_Sandy_Hurricane",
    "2013_Alberta_Floods",
    "2013_Boston_Bombings",
    "2013_Oklahoma_Tornado",
    "2013_Queensland_Floods",
    "2013_West_Texas_Explosion",
]


def _load_prepare_crisislex_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_crisislex", REPO_ROOT / "scripts" / "prepare_crisislex.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_crisislex"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Unit tests: label mapping
# ---------------------------------------------------------------------------


def test_label_map_values():
    assert LABEL_MAP == {"on-topic": 1, "off-topic": 0}


def test_map_labels_maps_correctly():
    df = pd.DataFrame({"label_raw": ["on-topic", "off-topic", "on-topic"]})
    mapped = map_labels(df)
    assert mapped["label"].tolist() == [1, 0, 1]


def test_map_labels_raises_on_unknown_value():
    df = pd.DataFrame({"label_raw": ["on-topic", "maybe-topic"]})
    with pytest.raises(ValueError):
        map_labels(df)


# ---------------------------------------------------------------------------
# Integration tests: real data
# ---------------------------------------------------------------------------

pytestmark_real_data = pytest.mark.skipif(
    not RAW_DIR.exists(), reason="CrisisLex T6 raw event CSVs not present locally"
)


@pytestmark_real_data
def test_load_event_csv_has_clean_ids_and_text():
    df = load_event_csv(RAW_DIR / f"{EVENTS[0]}-ontopic_offtopic.csv")
    assert list(df.columns) == ["tweet_id", "text", "label_raw"]
    assert not df["tweet_id"].str.startswith("'").any(), "tweet_id should have literal quotes stripped"
    assert set(df["label_raw"].unique()) == {"on-topic", "off-topic"}
    assert df["text"].isna().sum() == 0


@pytestmark_real_data
def test_prepare_crisislex_end_to_end_no_leakage_and_balance(tmp_path):
    module = _load_prepare_crisislex_module()

    config = {
        "dataset": "crisislex_t6",
        "raw_dir": str(RAW_DIR),
        "output_dir": str(tmp_path),
        "seed": 42,
        "text_col": "text",
        "label_col": "label",
        "id_col": "tweet_id",
        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
    }
    config_path = tmp_path / "crisislex_test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    manifest = module.main(str(config_path))

    train_df = pd.read_csv(tmp_path / "train.csv")
    val_df = pd.read_csv(tmp_path / "val.csv")
    test_df = pd.read_csv(tmp_path / "test.csv")

    # event column preserved in every split, values are the 6 known events.
    for split_df in (train_df, val_df, test_df):
        assert "event" in split_df.columns
        assert set(split_df["event"].unique()) <= set(EVENTS)

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
    global_rate = (
        train_df["label"].sum() + val_df["label"].sum() + test_df["label"].sum()
    ) / deduped_total
    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        rate = split_df["label"].mean()
        assert abs(rate - global_rate) < 0.01, f"{name} class balance drifted: {rate} vs {global_rate}"

    # Measured dedup counts are internally consistent (same identity as Kaggle's).
    assert manifest["raw_row_count"] - manifest["deduped_row_count"] == (
        manifest["exact_duplicate_row_count"] + manifest["dropped_conflicting_group_count"]
    )


@pytestmark_real_data
def test_prepare_crisislex_is_reproducible(tmp_path):
    module = _load_prepare_crisislex_module()
    base_config = {
        "dataset": "crisislex_t6",
        "raw_dir": str(RAW_DIR),
        "seed": 42,
        "text_col": "text",
        "label_col": "label",
        "id_col": "tweet_id",
        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
    }

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    for out_dir in (out1, out2):
        config = dict(base_config, output_dir=str(out_dir))
        config_path = out_dir.parent / f"{out_dir.name}_config.yaml"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)
        module.main(str(config_path))

    manifest1 = (out1 / "manifest.json").read_text(encoding="utf-8")
    manifest2 = (out2 / "manifest.json").read_text(encoding="utf-8")
    assert manifest1 == manifest2, "re-running the pipeline with the same seed must reproduce identical output"