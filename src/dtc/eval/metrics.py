"""Classification metrics used throughout the project.

`weighted_f1` exists only to reproduce the metric the original
course-derived project reported (`average="weighted"`); it is not used for
any of this project's own headline results, which use macro-F1 and
positive-class F1 instead (see docs/PLAN.md, "Metrics module").
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    confusion_matrix as _sk_confusion_matrix,
)
from sklearn.metrics import (
    precision_recall_fscore_support,
)

ArrayLike = Sequence[int] | np.ndarray


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return float(f1)


def weighted_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Legacy comparison only — matches the original project's reporting convention."""
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return float(f1)


def positive_f1(y_true: ArrayLike, y_pred: ArrayLike, pos_label: int = 1) -> float:
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0
    )
    return float(f1)


def per_class_precision_recall(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, dict[int, float]]:
    labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "precision": dict(zip(labels, (float(p) for p in precision))),
        "recall": dict(zip(labels, (float(r) for r in recall))),
        "f1": dict(zip(labels, (float(x) for x in f1))),
        "support": dict(zip(labels, (int(s) for s in support))),
    }


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, labels: Sequence[int] | None = None) -> np.ndarray:
    if labels is None:
        labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    return _sk_confusion_matrix(y_true, y_pred, labels=labels)


def compute_all_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict:
    """Single entrypoint used by the harness so every ledger entry has the same keys."""
    per_class = per_class_precision_recall(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "positive_f1": positive_f1(y_true, y_pred),
        "weighted_f1_legacy": weighted_f1(y_true, y_pred),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
