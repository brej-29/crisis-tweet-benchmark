"""Tests for the model registry (dtc.models.registry / dtc.models.base)."""

from __future__ import annotations

import pytest

from dtc.models.base import MODEL_REGISTRY, BaseModel, build_model, get_model_class, register_model
from dtc.models.registry import ALL_MODEL_NAMES

EXPECTED_MODEL_NAMES = {
    "tfidf_mnb",
    "tfidf_logreg",
    "meanpool_embed",
    "lstm",
    "gru",
    "bilstm",
    "conv1d",
    "use_frozen",
    "distilbert_finetune",
}


def test_all_nine_models_are_registered():
    assert EXPECTED_MODEL_NAMES <= set(ALL_MODEL_NAMES)
    assert len(EXPECTED_MODEL_NAMES) == 9


def test_get_model_class_returns_a_basemodel_subclass():
    for name in EXPECTED_MODEL_NAMES:
        cls = get_model_class(name)
        assert issubclass(cls, BaseModel)
        assert cls.name == name


def test_get_model_class_raises_on_unknown_name():
    with pytest.raises(KeyError):
        get_model_class("not_a_real_model")


def test_build_model_constructs_an_instance():
    instance = build_model("tfidf_mnb")
    assert isinstance(instance, BaseModel)


def test_register_model_rejects_reregistering_a_different_class_under_same_name():
    @register_model("__test_dummy__")
    class _Dummy1(BaseModel):
        def fit(self, train_df, val_df, config, seed):
            return self

        def predict_proba(self, texts):
            return None

    try:
        with pytest.raises(ValueError):

            @register_model("__test_dummy__")
            class _Dummy2(BaseModel):
                def fit(self, train_df, val_df, config, seed):
                    return self

                def predict_proba(self, texts):
                    return None
    finally:
        MODEL_REGISTRY.pop("__test_dummy__", None)
