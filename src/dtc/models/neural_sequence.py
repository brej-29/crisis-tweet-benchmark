"""From-scratch neural sequence models sharing one vocab/training pipeline:
meanpool_embed, lstm, gru, bilstm, conv1d (docs/PLAN.md 1.2).

All five: trainable embedding table (not pretrained), a project vocab
built TRAIN-only (dtc.data.text.build_vocab, Hard Rule 3), the shared
95th-percentile max_length policy (dtc.data.text.compute_percentile_max_length,
train-only unless pinned via config["max_length"]), AdamW, early stopping
on val loss (patience 3, restore best weights, max_epochs 30), dropout as
the capacity/regularization knob.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dtc.data.text import PAD_ID, build_vocab, clean_series, compute_percentile_max_length, encode_batch
from dtc.models.base import BaseModel, register_model
from dtc.models.torch_common import fit_with_early_stopping, get_device, set_seed


class _VocabSequenceModel(BaseModel):
    """Shared fit/predict_proba: build train-only vocab + max_length, encode,
    train a torch module (defined by `_build_module`) with early stopping.
    """

    def _build_module(self, vocab_size: int, config: dict) -> nn.Module:
        raise NotImplementedError

    def fit(self, train_df, val_df, config, seed):
        set_seed(seed)
        train_texts = clean_series(train_df["text"])
        val_texts = clean_series(val_df["text"])

        self.max_vocab_size = config.get("max_vocab_size", 10000)
        self.vocab = build_vocab(train_texts, max_vocab_size=self.max_vocab_size)
        self.max_length = config.get("max_length") or compute_percentile_max_length(train_texts)

        X_train = encode_batch(train_texts, self.vocab, self.max_length)
        y_train = train_df["label"].to_numpy().astype(np.float32)
        X_val = encode_batch(val_texts, self.vocab, self.max_length)
        y_val = val_df["label"].to_numpy().astype(np.float32)

        device = get_device()
        batch_size = config.get("batch_size", 32)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train).long(), torch.from_numpy(y_train)),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val).long(), torch.from_numpy(y_val)),
            batch_size=batch_size,
            shuffle=False,
        )

        module = self._build_module(len(self.vocab), config)
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
        )
        self.device = device
        return self

    def predict_proba(self, texts):
        cleaned = clean_series(texts)
        X = encode_batch(cleaned, self.vocab, self.max_length)
        self.module.eval()
        with torch.no_grad():
            logits = self.module(torch.from_numpy(X).long().to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs


class _MeanPoolNet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)  # (batch, seq, embed_dim)
        mask = (x != PAD_ID).unsqueeze(-1).float()
        summed = (emb * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        pooled = self.dropout(pooled)
        return self.fc(pooled).squeeze(-1)


@register_model("meanpool_embed")
class MeanPoolEmbedModel(_VocabSequenceModel):
    def _build_module(self, vocab_size, config):
        return _MeanPoolNet(vocab_size, config.get("embed_dim", 100), config.get("dropout", 0.3))


class _RNNNet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, dropout: float, rnn_type: str, bidirectional: bool):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.rnn = rnn_cls(embed_dim, hidden_size, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)
        if self.rnn_type == "lstm":
            _, (h_n, _) = self.rnn(emb)
        else:
            _, h_n = self.rnn(emb)
        # h_n: (num_directions, batch, hidden_size)
        if self.bidirectional:
            pooled = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            pooled = h_n[-1]
        pooled = self.dropout(pooled)
        return self.fc(pooled).squeeze(-1)


@register_model("lstm")
class LSTMModel(_VocabSequenceModel):
    def _build_module(self, vocab_size, config):
        return _RNNNet(
            vocab_size,
            config.get("embed_dim", 100),
            config.get("hidden_size", 64),
            config.get("dropout", 0.3),
            rnn_type="lstm",
            bidirectional=False,
        )


@register_model("gru")
class GRUModel(_VocabSequenceModel):
    def _build_module(self, vocab_size, config):
        return _RNNNet(
            vocab_size,
            config.get("embed_dim", 100),
            config.get("hidden_size", 64),
            config.get("dropout", 0.3),
            rnn_type="gru",
            bidirectional=False,
        )


@register_model("bilstm")
class BiLSTMModel(_VocabSequenceModel):
    def _build_module(self, vocab_size, config):
        return _RNNNet(
            vocab_size,
            config.get("embed_dim", 100),
            config.get("hidden_size", 64),
            config.get("dropout", 0.3),
            rnn_type="lstm",
            bidirectional=True,
        )


class _Conv1DNet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_filters: int, kernel_size: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)  # (batch, embed_dim, seq)
        conv_out = torch.relu(self.conv(emb))  # (batch, num_filters, seq)
        pooled, _ = conv_out.max(dim=2)  # global max-pool over positions
        pooled = self.dropout(pooled)
        return self.fc(pooled).squeeze(-1)


@register_model("conv1d")
class Conv1DModel(_VocabSequenceModel):
    def _build_module(self, vocab_size, config):
        return _Conv1DNet(
            vocab_size,
            config.get("embed_dim", 100),
            config.get("num_filters", 100),
            config.get("kernel_size", 5),
            config.get("dropout", 0.3),
        )
