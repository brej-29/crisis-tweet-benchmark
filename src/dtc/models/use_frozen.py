"""Frozen Universal Sentence Encoder embeddings + a small torch MLP head.

Embeddings are never computed at train/predict time here -- they must
already exist in the cache built by scripts/precompute_use.py (docs/PLAN.md
1.2/Hard Rule 2). Chosen a small torch MLP over sklearn LogisticRegression
(the spec allows either) so this model gets the same early-stopping/
dropout-as-capacity-knob treatment as the other neural models rather than
being a one-off special case -- see docs/DECISIONS.md.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dtc.data.text import clean_series
from dtc.data.use_cache import load_embeddings
from dtc.models.base import BaseModel, register_model
from dtc.models.torch_common import fit_with_early_stopping, get_device, set_seed

USE_EMBED_DIM = 512


class _UseMlp(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(USE_EMBED_DIM, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@register_model("use_frozen")
class UseFrozenModel(BaseModel):
    def fit(self, train_df, val_df, config, seed):
        set_seed(seed)
        cache_dir = config["use_cache_dir"]  # no default: caller must point at a real precomputed cache
        train_texts = clean_series(train_df["text"])
        val_texts = clean_series(val_df["text"])
        X_train = load_embeddings(cache_dir, train_texts).astype(np.float32)
        X_val = load_embeddings(cache_dir, val_texts).astype(np.float32)
        y_train = train_df["label"].to_numpy().astype(np.float32)
        y_val = val_df["label"].to_numpy().astype(np.float32)

        device = get_device()
        batch_size = config.get("batch_size", 32)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=batch_size,
            shuffle=False,
        )

        module = _UseMlp(config.get("hidden_size", 128), config.get("dropout", 0.3))
        optimizer = torch.optim.AdamW(module.parameters(), lr=config.get("lr", 1e-3))
        criterion = nn.BCEWithLogitsLoss()

        self.module, self.history = fit_with_early_stopping(
            module,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            device,
            max_epochs=config.get("max_epochs", 30),
            patience=config.get("patience", 3),
            restore_best_weights=config.get("restore_best_weights", True),
        )
        self.device = device
        self.cache_dir = cache_dir
        return self

    def predict_proba(self, texts):
        cleaned = clean_series(texts)
        X = load_embeddings(self.cache_dir, cleaned).astype(np.float32)
        self.module.eval()
        with torch.no_grad():
            logits = self.module(torch.from_numpy(X).to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs
