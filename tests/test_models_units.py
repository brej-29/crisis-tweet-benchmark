"""Unit tests for the nine model implementations: each trains end-to-end on
a tiny synthetic frame and produces well-shaped probabilities. These are
NOT the ledgered smoke runs (Task 8's smoke matrix uses real ~200-example
Kaggle subsets and ledgers each run) -- these are fast, ledger-free
correctness checks of the fit/predict_proba contract itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtc.models.registry import build_model


def _tiny_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    disaster_words = ["fire", "flood", "earthquake", "storm", "explosion"]
    normal_words = ["party", "lunch", "movie", "weekend", "coffee"]
    rows = []
    for i in range(n):
        label = i % 2
        pool = disaster_words if label == 1 else normal_words
        text = " ".join(rng.choice(pool, size=6))
        rows.append({"text": text, "label": label})
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_train_val():
    return _tiny_frame(40, seed=1), _tiny_frame(12, seed=2)


@pytest.mark.parametrize("model_name", ["tfidf_mnb", "tfidf_logreg"])
def test_tfidf_models_fit_and_predict(model_name, tiny_train_val):
    train_df, val_df = tiny_train_val
    model = build_model(model_name)
    model.fit(train_df, val_df, config={}, seed=0)
    probs = model.predict_proba(val_df["text"])
    assert probs.shape == (len(val_df),)
    assert np.all((probs >= 0) & (probs <= 1))


@pytest.mark.parametrize("model_name", ["meanpool_embed", "lstm", "gru", "bilstm", "conv1d"])
def test_vocab_sequence_models_fit_and_predict(model_name, tiny_train_val):
    train_df, val_df = tiny_train_val
    model = build_model(model_name)
    config = {"max_epochs": 3, "patience": 2, "embed_dim": 8, "hidden_size": 8, "num_filters": 4}
    model.fit(train_df, val_df, config=config, seed=0)
    probs = model.predict_proba(val_df["text"])
    assert probs.shape == (len(val_df),)
    assert np.all((probs >= 0) & (probs <= 1))
    assert 0 < len(model.history.train_loss) <= config["max_epochs"]
    assert len(model.history.train_loss) == len(model.history.val_loss)


def test_vocab_sequence_model_vocab_is_train_only(tiny_train_val):
    train_df, val_df = tiny_train_val
    val_df = val_df.copy()
    val_df.loc[0, "text"] = "zzzznevertrain word appears only here"
    model = build_model("meanpool_embed")
    model.fit(train_df, val_df, config={"max_epochs": 1, "patience": 1}, seed=0)
    assert "zzzznevertrain" not in model.vocab
    # OOV val-only words must still predict without crashing (fall back to <unk>)
    probs = model.predict_proba(val_df["text"])
    assert probs.shape == (len(val_df),)


def test_meanpool_embed_respects_no_early_stopping_config(tiny_train_val):
    """Protocol A wires patience=None + restore_best_weights=False through
    model config to reproduce "no early stopping, fixed N epochs"."""
    train_df, val_df = tiny_train_val
    model = build_model("meanpool_embed")
    config = {"max_epochs": 4, "patience": None, "restore_best_weights": False, "embed_dim": 8}
    model.fit(train_df, val_df, config=config, seed=0)
    assert len(model.history.train_loss) == 4
    assert model.history.stopped_epoch == 3


def test_use_frozen_fit_and_predict_with_synthetic_cache(tmp_path, tiny_train_val):
    from dtc.data.use_cache import save_embedding

    train_df, val_df = tiny_train_val
    rng = np.random.RandomState(0)
    for text in pd.concat([train_df["text"], val_df["text"]]).unique():
        save_embedding(tmp_path, text, rng.rand(512).astype(np.float32))

    model = build_model("use_frozen")
    config = {"use_cache_dir": tmp_path, "max_epochs": 3, "patience": 2, "hidden_size": 8}
    model.fit(train_df, val_df, config=config, seed=0)
    probs = model.predict_proba(val_df["text"])
    assert probs.shape == (len(val_df),)
    assert np.all((probs >= 0) & (probs <= 1))


def test_use_frozen_raises_key_error_for_uncached_text(tmp_path, tiny_train_val):
    from dtc.models.use_frozen import UseFrozenModel

    train_df, val_df = tiny_train_val
    model = UseFrozenModel()
    with pytest.raises(KeyError):
        model.fit(train_df, val_df, config={"use_cache_dir": tmp_path, "max_epochs": 1}, seed=0)


@pytest.mark.slow
def test_distilbert_finetune_fit_and_predict(tiny_train_val):
    train_df, val_df = tiny_train_val
    model = build_model("distilbert_finetune")
    config = {"max_epochs": 1, "patience": 1, "batch_size": 8, "max_length": 16}
    model.fit(train_df, val_df, config=config, seed=0)
    probs = model.predict_proba(val_df["text"])
    assert probs.shape == (len(val_df),)
    assert np.all((probs >= 0) & (probs <= 1))
