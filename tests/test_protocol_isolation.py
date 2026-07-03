"""Protocol A isolation tests (Hard Rule 7).

Two directions, both enforced statically (not just by convention):
  1. dtc.data.protocol_a must never reference the frozen-test guard --
     Protocol A's 90/10 holdout is a different split of a different
     (non-deduped) dataframe, not the Protocol B frozen test split.
  2. No Protocol B code path (dtc.data.kaggle/common, dtc.models.*,
     dtc.train.*) may import dtc.data.protocol_a's split logic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dtc.data import protocol_a

REPO_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_B_SUBTREES = [
    REPO_ROOT / "src" / "dtc" / "data",
    REPO_ROOT / "src" / "dtc" / "models",
    REPO_ROOT / "src" / "dtc" / "train",
]


def test_protocol_a_module_does_not_reference_frozen_test_loader():
    src = (REPO_ROOT / "src" / "dtc" / "data" / "protocol_a.py").read_text(encoding="utf-8")
    assert "frozen_test_loader" not in src


def test_protocol_b_code_does_not_import_protocol_a():
    offenders = []
    for subtree in PROTOCOL_B_SUBTREES:
        if not subtree.exists():
            continue
        for py_file in subtree.rglob("*.py"):
            if py_file.name == "protocol_a.py":
                continue
            text = py_file.read_text(encoding="utf-8")
            if "protocol_a" in text:
                offenders.append(str(py_file.relative_to(REPO_ROOT)))
    assert offenders == [], f"Protocol B code imports/references protocol_a: {offenders}"


def test_protocol_a_split_is_90_10_non_stratified_and_deterministic():
    df = pd.DataFrame({"text": [f"t{i}" for i in range(100)], "target": [1] * 60 + [0] * 40})
    train_df, eval_df = protocol_a.protocol_a_split(df)
    assert len(train_df) == 90
    assert len(eval_df) == 10

    train_df2, eval_df2 = protocol_a.protocol_a_split(df)
    pd.testing.assert_frame_equal(train_df, train_df2)
    pd.testing.assert_frame_equal(eval_df, eval_df2)

    # non-stratified: no guarantee eval_df's class balance matches df's --
    # just assert the split is a real partition (no overlap, full coverage).
    assert set(train_df["text"]) | set(eval_df["text"]) == set(df["text"])
    assert set(train_df["text"]) & set(eval_df["text"]) == set()


def test_mean_token_length_hand_computed():
    # whitespace-token counts: 1, 2, 4 -> mean 2.333..., ceil -> 3
    texts = ["a", "a a", "a a a a"]
    assert protocol_a.mean_token_length(texts) == 3


def test_mean_token_length_supports_custom_token_len_fn():
    texts = ["a", "aa", "aaa"]  # char counts: 1, 2, 3 -> mean 2.0
    assert protocol_a.mean_token_length(texts, token_len_fn=len) == 2
