"""Floor baselines: majority-class and stratified-random classifiers.

These exist to give every real model a floor to beat. They are
deliberately trivial (sklearn.dummy.DummyClassifier under the hood) --
the point is having them ledgered like any other run, not the modeling.
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier


def fit_majority_class(y_train, seed: int) -> DummyClassifier:
    clf = DummyClassifier(strategy="most_frequent", random_state=seed)
    clf.fit(np.zeros((len(y_train), 1)), y_train)
    return clf


def fit_stratified_random(y_train, seed: int) -> DummyClassifier:
    clf = DummyClassifier(strategy="stratified", random_state=seed)
    clf.fit(np.zeros((len(y_train), 1)), y_train)
    return clf


def predict(clf: DummyClassifier, n: int) -> tuple[np.ndarray, np.ndarray | None]:
    X = np.zeros((n, 1))
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else None
    return y_pred, y_prob