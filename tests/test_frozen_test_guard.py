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


@pytest.fixture
def tiny_crisislex_csv(tmp_path):
    """A frozen-test-shaped CSV at a `data/crisislex/test.csv`-like path,
    including the `event` column. The runtime guard is caller-based, not
    path-based, so the CSV's location/shape must not change guard outcomes
    (Task A2 requirement 1) -- these tests exercise that against BOTH
    allowed and disallowed callers, mirroring the kaggle-path tests above.
    """
    p = tmp_path / "data" / "crisislex" / "test.csv"
    p.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "event": ["2013_Oklahoma_Tornado", "2013_Queensland_Floods"],
            "tweet_id": ["1", "2"],
            "text": ["a", "b"],
            "label": [0, 1],
        }
    ).to_csv(p, index=False)
    return p


def test_disallowed_caller_raises_for_crisislex_path(tiny_crisislex_csv):
    with pytest.raises(FrozenTestAccessError):
        load_frozen_test(
            tiny_crisislex_csv,
            _caller_module="dtc.train.train_lstm",
            _caller_file=str(REPO_ROOT / "src" / "dtc" / "train" / "train_lstm.py"),
        )


def test_disallowed_data_prep_caller_raises_for_crisislex_path(tiny_crisislex_csv):
    with pytest.raises(FrozenTestAccessError):
        load_frozen_test(
            tiny_crisislex_csv,
            _caller_module="dtc.data.prepare_crisislex",
            _caller_file=str(REPO_ROOT / "src" / "dtc" / "data" / "prepare_crisislex.py"),
        )


def test_allowed_eval_package_caller_succeeds_for_crisislex_path(tiny_crisislex_csv):
    df = load_frozen_test(
        tiny_crisislex_csv,
        _caller_module="dtc.eval.run_evaluation",
        _caller_file=str(REPO_ROOT / "src" / "dtc" / "eval" / "run_evaluation.py"),
    )
    assert len(df) == 2
    assert "event" in df.columns


def test_allowed_scripts_evaluate_caller_succeeds_for_crisislex_path(tiny_crisislex_csv):
    df = load_frozen_test(
        tiny_crisislex_csv,
        _caller_module="__main__",
        _caller_file=str(REPO_ROOT / "scripts" / "evaluate_lstm.py"),
    )
    assert len(df) == 2


def _scan_for_frozen_test_references(subtrees, base: Path = REPO_ROOT) -> list[str]:
    """Shared by the real static-scan test and the synthetic-fixture test
    below that proves the scan's dataset-genericity, so both exercise
    exactly the same logic (no drift between "real" and "proof" versions).
    """
    offenders = []
    for subtree in subtrees:
        if not subtree.exists():
            continue
        for py_file in subtree.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            if "frozen_test_loader" in text:
                offenders.append(str(py_file.relative_to(base)))
    return offenders


def test_no_training_or_data_module_imports_frozen_test_loader():
    """Static scan: guarded subtrees must not import the frozen test loader."""
    offenders = _scan_for_frozen_test_references(GUARDED_SUBTREES)
    assert offenders == [], (
        f"Found forbidden references to frozen_test_loader in training/data/model "
        f"source files: {offenders}"
    )


def test_static_scan_is_dataset_generic_not_kaggle_only(tmp_path):
    """Proves (rather than assumes) the static scan covers crisislex -- and
    any other dataset -- too: it flags on the mere presence of the string
    'frozen_test_loader' in a guarded-subtree file regardless of which
    dataset's `test.csv` path the offending code reads. A synthetic guarded
    subtree with three offenders (kaggle, crisislex, and an arbitrary third
    dataset -- covering the brief's "and generally <dataset>/test.csv") plus
    one clean file proves both that offenders are caught and clean files
    are left alone.
    """
    guarded = tmp_path / "src" / "dtc" / "train"
    guarded.mkdir(parents=True)

    (guarded / "sneaky_kaggle_reader.py").write_text(
        "from dtc.eval.frozen_test_loader import load_frozen_test\n"
        "load_frozen_test('data/kaggle/test.csv')\n",
        encoding="utf-8",
    )
    (guarded / "sneaky_crisislex_reader.py").write_text(
        "from dtc.eval.frozen_test_loader import load_frozen_test\n"
        "load_frozen_test('data/crisislex/test.csv')\n",
        encoding="utf-8",
    )
    (guarded / "sneaky_other_dataset_reader.py").write_text(
        "from dtc.eval import frozen_test_loader\n"
        "frozen_test_loader.load_frozen_test('data/some_future_dataset/test.csv')\n",
        encoding="utf-8",
    )
    (guarded / "innocent.py").write_text("import pandas as pd\n", encoding="utf-8")

    offenders = _scan_for_frozen_test_references([guarded], base=tmp_path)
    assert set(offenders) == {
        str(Path("src") / "dtc" / "train" / "sneaky_kaggle_reader.py"),
        str(Path("src") / "dtc" / "train" / "sneaky_crisislex_reader.py"),
        str(Path("src") / "dtc" / "train" / "sneaky_other_dataset_reader.py"),
    }
