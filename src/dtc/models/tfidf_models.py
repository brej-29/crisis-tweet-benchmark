"""TF-IDF + classical sklearn classifiers: tfidf_mnb, tfidf_logreg.

sklearn models have no early stopping -- not applicable, since these are
not epoch-based iterative-with-a-validation-callback fits the way the
torch models are; documented here rather than silently omitted (Hard Rule
2's per-family early-stopping requirement is scoped to the neural models).
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from dtc.data.text import clean_series
from dtc.models.base import BaseModel, register_model


class _TfidfSklearnModel(BaseModel):
    """Shared fit/predict_proba for TF-IDF + a sklearn classifier.

    The vectorizer is fit on the TRAIN split's cleaned text only (Hard Rule
    3); val/predict text is only ever `.transform()`-ed, never used to fit
    the vocabulary or IDF weights.
    """

    def _build_classifier(self, config: dict, seed: int):
        raise NotImplementedError

    def fit(self, train_df, val_df, config, seed):
        train_texts = clean_series(train_df["text"])
        self.vectorizer = TfidfVectorizer(
            max_features=config.get("max_features", 10000),
            ngram_range=tuple(config.get("ngram_range", (1, 1))),
        )
        X_train = self.vectorizer.fit_transform(train_texts)
        y_train = train_df["label"].to_numpy()
        self.classifier = self._build_classifier(config, seed)
        self.classifier.fit(X_train, y_train)
        return self

    def predict_proba(self, texts):
        cleaned = clean_series(texts)
        X = self.vectorizer.transform(cleaned)
        return self.classifier.predict_proba(X)[:, 1]


@register_model("tfidf_mnb")
class TfidfMultinomialNB(_TfidfSklearnModel):
    def _build_classifier(self, config, seed):
        return MultinomialNB(alpha=config.get("alpha", 1.0))


@register_model("tfidf_logreg")
class TfidfLogisticRegression(_TfidfSklearnModel):
    def _build_classifier(self, config, seed):
        return LogisticRegression(
            C=config.get("C", 1.0),
            max_iter=config.get("max_iter", 1000),
            random_state=seed,
        )
