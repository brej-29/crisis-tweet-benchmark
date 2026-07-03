"""Imports every model implementation so its @register_model decorator runs,
then re-exports the populated registry. Scripts/config-driven code should
import from here, not from individual model modules, so all nine models are
always registered together in one place.
"""

from __future__ import annotations

from dtc.models import (  # noqa: F401  (imported for @register_model side effects)
    distilbert_finetune,
    neural_sequence,
    tfidf_models,
    use_frozen,
)
from dtc.models.base import MODEL_REGISTRY, BaseModel, build_model, get_model_class

ALL_MODEL_NAMES = sorted(MODEL_REGISTRY)

__all__ = ["MODEL_REGISTRY", "BaseModel", "build_model", "get_model_class", "ALL_MODEL_NAMES"]
