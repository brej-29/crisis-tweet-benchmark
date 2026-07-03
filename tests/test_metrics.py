"""Unit tests for dtc.eval.metrics against hand-computed expected values.

Fixture (8 examples, imbalanced 5 positive / 3 negative) is chosen so that
accuracy, macro-F1, and weighted-F1 are all numerically distinct, proving
the three are actually different computations rather than the same number
under three names.

y_true: [1, 1, 1, 1, 1, 0, 0, 0]
y_pred: [1, 1, 1, 0, 0, 0, 0, 1]

Confusion matrix (labels=[0, 1], rows=true, cols=pred):
    true=0 (3 samples: idx 5,6,7): pred = 0,0,1 -> [TN=2, FP=1]
    true=1 (5 samples: idx 0,1,2,3,4): pred = 1,1,1,0,0 -> [FN=2, TP=3]
    cm = [[2, 1],
          [2, 3]]

Class 1: precision=3/4=0.75, recall=3/5=0.6, f1=2/3 (0.666666...)
Class 0: precision=2/4=0.5,  recall=2/3=0.666666..., f1=4/7 (0.571428...)

accuracy    = 5/8            = 0.625
macro_f1    = (2/3 + 4/7)/2  = 13/21  ~= 0.619048
weighted_f1 = (2/3*5+4/7*3)/8 = 53/84 ~= 0.630952
positive_f1 (class 1) = 2/3  ~= 0.666667
"""

import numpy as np
import pytest

from dtc.eval.metrics import (
    accuracy,
    compute_all_metrics,
    confusion_matrix,
    macro_f1,
    per_class_precision_recall,
    positive_f1,
    weighted_f1,
)

Y_TRUE = [1, 1, 1, 1, 1, 0, 0, 0]
Y_PRED = [1, 1, 1, 0, 0, 0, 0, 1]


def test_accuracy():
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(5 / 8)


def test_macro_f1():
    assert macro_f1(Y_TRUE, Y_PRED) == pytest.approx(13 / 21, abs=1e-6)


def test_weighted_f1():
    assert weighted_f1(Y_TRUE, Y_PRED) == pytest.approx(53 / 84, abs=1e-6)


def test_positive_f1():
    assert positive_f1(Y_TRUE, Y_PRED, pos_label=1) == pytest.approx(2 / 3, abs=1e-6)


def test_accuracy_macro_weighted_are_pairwise_distinct():
    a = accuracy(Y_TRUE, Y_PRED)
    m = macro_f1(Y_TRUE, Y_PRED)
    w = weighted_f1(Y_TRUE, Y_PRED)
    assert a != pytest.approx(m, abs=1e-4)
    assert a != pytest.approx(w, abs=1e-4)
    assert m != pytest.approx(w, abs=1e-4)


def test_confusion_matrix():
    cm = confusion_matrix(Y_TRUE, Y_PRED, labels=[0, 1])
    np.testing.assert_array_equal(cm, np.array([[2, 1], [2, 3]]))


def test_per_class_precision_recall():
    result = per_class_precision_recall(Y_TRUE, Y_PRED)
    assert result["precision"][1] == pytest.approx(0.75)
    assert result["recall"][1] == pytest.approx(0.6)
    assert result["precision"][0] == pytest.approx(0.5)
    assert result["recall"][0] == pytest.approx(2 / 3, abs=1e-6)
    assert result["support"][1] == 5
    assert result["support"][0] == 3


def test_compute_all_metrics_shape():
    result = compute_all_metrics(Y_TRUE, Y_PRED)
    assert result["accuracy"] == pytest.approx(5 / 8)
    assert result["macro_f1"] == pytest.approx(13 / 21, abs=1e-6)
    assert result["weighted_f1_legacy"] == pytest.approx(53 / 84, abs=1e-6)
    assert result["positive_f1"] == pytest.approx(2 / 3, abs=1e-6)
    assert result["confusion_matrix"] == [[2, 1], [2, 3]]


def test_perfect_predictions():
    y = [0, 1, 0, 1, 1]
    assert accuracy(y, y) == 1.0
    assert macro_f1(y, y) == 1.0
    assert weighted_f1(y, y) == 1.0
    assert positive_f1(y, y) == 1.0
