"""Common model interface + registry (docs/PLAN.md 1.2).

Every model implements:
    fit(train_df, val_df, config, seed) -> self
    predict_proba(texts) -> np.ndarray of P(class=1)

`train_df`/`val_df` are pandas DataFrames with columns "text" (str) and
"label" (int, 0/1) -- callers (scripts, the run driver) are responsible for
renaming dataset-specific columns (e.g. Kaggle's "target") to this shape
before calling fit, so model code stays dataset-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pandas as pd


class BaseModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict, seed: int) -> "BaseModel":
        ...

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def predict(self, texts: Sequence[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(texts) >= threshold).astype(int)


MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(name: str):
    """Class decorator: registers a BaseModel subclass under `name`."""

    def decorator(cls: type[BaseModel]) -> type[BaseModel]:
        if name in MODEL_REGISTRY and MODEL_REGISTRY[name] is not cls:
            raise ValueError(f"model name '{name}' is already registered to {MODEL_REGISTRY[name]}")
        cls.name = name
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> type[BaseModel]:
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(MODEL_REGISTRY)}") from None


def build_model(name: str, **kwargs) -> BaseModel:
    return get_model_class(name)(**kwargs)
