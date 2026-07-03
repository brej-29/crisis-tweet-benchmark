"""Tests for the frozen-test-set access guard (dtc.eval.frozen_test_loader).

Two layers, matching docs/PLAN.md's standing rule:
  1. Runtime guard: load_frozen_test() raises unless called from an allowed
     evaluation entrypoint (dtc.eval.* or scripts/evaluate_*.py).
  2. Static guard: no file under the training/data/model source tree may
     even import dtc.eval.frozen_test_loader, so a violation is caught by
     CI before the code is ever run.
"""

from pathlib import Path

import pandas as pd
import pytest

from dtc.eval.frozen_test_loader import FrozenTestAccessError, load_frozen_test

REPO_ROOT = Path(__file__).resolve().parents[1]

# Source subtrees that prepare data or select/train models: these must never
# read the frozen test split.
GUARDED_SUBTREES = [
    REPO_ROOT / "src" / "dtc" / "data",
    REPO_ROOT / "src" / "dtc" / "models",
    REPO_ROOT / "src" / "dtc" / "train",
    REPO_ROOT / "src" / "dtc" / "harness",
]


@pytest.fixture
def tiny_csv(tmp_path):
    p = tmp_path / "frozen_test.csv"
    pd.DataFrame({"text": ["a", "b"], "label": [0, 1]}).to_csv(p, index=False)
    return p


def test_disallowed_caller_raises(tiny_csv):
    with pytest.raises(FrozenTestAccessError):
        load_frozen_test(
            tiny_csv,
            _caller_module="dtc.train.train_lstm",
            _caller_file=str(REPO_ROOT / "src" / "dtc" / "train" / "train_lstm.py"),
        )


def test_disallowed_data_prep_caller_raises(tiny_csv):
    with pytest.raises(FrozenTestAccessError):
        load_frozen_test(
            tiny_csv,
            _caller_module="dtc.data.prepare_kaggle",
            _caller_file=str(REPO_ROOT / "src" / "dtc" / "data" / "prepare_kaggle.py"),
        )


def test_allowed_eval_package_caller_succeeds(tiny_csv):
    df = load_frozen_test(
        tiny_csv,
        _caller_module="dtc.eval.run_evaluation",
        _caller_file=str(REPO_ROOT / "src" / "dtc" / "eval" / "run_evaluation.py"),
    )
    assert len(df) == 2


def test_allowed_scripts_evaluate_caller_succeeds(tiny_csv):
    df = load_frozen_test(
        tiny_csv,
        _caller_module="__main__",
        _caller_file=str(REPO_ROOT / "scripts" / "evaluate_lstm.py"),
    )
    assert len(df) == 2


def test_disallowed_scripts_non_evaluate_caller_raises(tiny_csv):
    with pytest.raises(FrozenTestAccessError):
        load_frozen_test(
            tiny_csv,
            _caller_module="__main__",
            _caller_file=str(REPO_ROOT / "scripts" / "train_lstm.py"),
        )


def test_no_training_or_data_module_imports_frozen_test_loader():
    """Static scan: guarded subtrees must not import the frozen test loader."""
    offenders = []
    for subtree in GUARDED_SUBTREES:
        if not subtree.exists():
            continue
        for py_file in subtree.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            if "frozen_test_loader" in text:
                offenders.append(str(py_file.relative_to(REPO_ROOT)))
    assert offenders == [], (
        f"Found forbidden references to frozen_test_loader in training/data/model "
        f"source files: {offenders}"
    )
