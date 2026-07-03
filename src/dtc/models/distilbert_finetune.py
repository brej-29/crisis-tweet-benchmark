"""Fine-tuned distilbert-base-uncased, end to end (HF transformers, torch
backend). Truncation length follows the same 95th-percentile-of-train
policy as every other model (dtc.data.text.compute_percentile_max_length),
measured in DistilBERT's own WordPiece token units rather than reusing the
whitespace-token value -- see docs/PROTOCOL.md sec. 2.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from dtc.data.text import clean_series, compute_percentile_max_length
from dtc.models.base import BaseModel, register_model
from dtc.models.torch_common import fit_with_early_stopping_generic, get_device, set_seed

PRETRAINED_NAME = "distilbert-base-uncased"


def _hf_step(model, batch, device):
    input_ids, attention_mask, labels = (t.to(device) for t in batch)
    return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss


@register_model("distilbert_finetune")
class DistilBertFinetuneModel(BaseModel):
    def fit(self, train_df, val_df, config, seed):
        set_seed(seed)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(PRETRAINED_NAME)
        train_texts = clean_series(train_df["text"])
        val_texts = clean_series(val_df["text"])

        self.max_length = config.get("max_length") or compute_percentile_max_length(
            train_texts, token_len_fn=lambda t: len(self.tokenizer.tokenize(t))
        )

        device = get_device()
        self.device = device
        self.model = DistilBertForSequenceClassification.from_pretrained(
            PRETRAINED_NAME, num_labels=2, dropout=config.get("dropout", 0.1)
        ).to(device)

        train_enc = self._encode(train_texts)
        val_enc = self._encode(val_texts)
        y_train = train_df["label"].to_numpy().astype(np.int64)
        y_val = val_df["label"].to_numpy().astype(np.int64)

        batch_size = config.get("batch_size", 16)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            TensorDataset(train_enc["input_ids"], train_enc["attention_mask"], torch.from_numpy(y_train)),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        val_loader = DataLoader(
            TensorDataset(val_enc["input_ids"], val_enc["attention_mask"], torch.from_numpy(y_val)),
            batch_size=batch_size,
            shuffle=False,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.get("lr", 5e-5))
        step = partial(_hf_step, device=device)
        self.model, self.history = fit_with_early_stopping_generic(
            self.model,
            train_loader,
            val_loader,
            optimizer,
            step,
            step,
            max_epochs=config.get("max_epochs", 30),
            patience=config.get("patience", 3),
        )
        return self

    def _encode(self, texts):
        return self.tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

    def predict_proba(self, texts):
        cleaned = clean_series(texts)
        enc = self._encode(cleaned)
        self.model.eval()
        batch_size = 32
        probs = []
        with torch.no_grad():
            for i in range(0, len(cleaned), batch_size):
                input_ids = enc["input_ids"][i : i + batch_size].to(self.device)
                attention_mask = enc["attention_mask"][i : i + batch_size].to(self.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                p = torch.softmax(logits, dim=-1)[:, 1]
                probs.append(p.cpu().numpy())
        return np.concatenate(probs)
