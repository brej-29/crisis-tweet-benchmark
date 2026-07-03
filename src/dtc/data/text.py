"""Uniform text preprocessing protocol, applied identically to every model.

Cleaning is minimal and uniform (docs/PROTOCOL.md): unicode NFC
normalization + whitespace collapse. URLs and @mentions are KEPT as tokens
(a documented choice, not a silent decision -- see docs/PROTOCOL.md and
docs/DECISIONS.md; ablatable later, not decided here).

Sequence-length policy (docs/PLAN.md 1.1): `max_length` for a token-based
model family is the 95th percentile of that family's own token counts on
the KAGGLE TRAIN split only, rounded up -- the original project truncated
at the MEAN, cutting ~half of all tweets; this corrects that confound.
`compute_percentile_max_length` is generic over the token-counting function
so from-scratch models (whitespace tokens) and DistilBERT (WordPiece
tokens) can each compute their own family-specific value from the same
policy, per docs/PROTOCOL.md's "tokenization is inherently model-family
specific, truncation policy is not" distinction.

Everything here (`clean_text`, `build_vocab`, `compute_percentile_max_length`)
must only ever be called with TRAIN-split text (Hard Rule 3) -- callers in
dtc.models.* are responsible for passing the right split.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Callable, Sequence

import numpy as np

_WHITESPACE_RE = re.compile(r"\s+")

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


def clean_text(text: str) -> str:
    """NFC-normalize and collapse whitespace. URLs/@mentions are kept as-is."""
    text = unicodedata.normalize("NFC", str(text))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_series(texts: Sequence[str]) -> list[str]:
    return [clean_text(t) for t in texts]


def whitespace_token_count(text: str) -> int:
    return len(text.split())


def whitespace_tokenize(text: str) -> list[str]:
    return text.split()


def compute_percentile_max_length(
    texts: Sequence[str],
    percentile: float = 95.0,
    token_len_fn: Callable[[str], int] | None = None,
) -> int:
    """95th-percentile token count, rounded up. Call on TRAIN-split text only.

    `token_len_fn` defaults to whitespace-token counting; pass a
    tokenizer-specific counter (e.g. `lambda t: len(hf_tokenizer.tokenize(t))`)
    to compute the same policy in a different family's token units.
    """
    if token_len_fn is None:
        token_len_fn = whitespace_token_count
    counts = [token_len_fn(t) for t in texts]
    value = np.percentile(counts, percentile)
    return int(math.ceil(value))


def build_vocab(texts: Sequence[str], max_vocab_size: int = 10000) -> dict[str, int]:
    """Top-N whitespace tokens by frequency on the given (TRAIN-only) texts.

    Index 0 = PAD, 1 = UNK, then tokens in descending-frequency order.
    """
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(whitespace_tokenize(text))
    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, _ in counter.most_common(max_vocab_size):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    """Tokenize on whitespace, map to vocab ids (UNK for OOV), pad/truncate."""
    unk_id = vocab.get(UNK_TOKEN, UNK_ID)
    pad_id = vocab.get(PAD_TOKEN, PAD_ID)
    ids = [vocab.get(tok, unk_id) for tok in whitespace_tokenize(text)]
    ids = ids[:max_length]
    ids = ids + [pad_id] * (max_length - len(ids))
    return ids


def encode_batch(texts: Sequence[str], vocab: dict[str, int], max_length: int) -> np.ndarray:
    return np.array([encode(t, vocab, max_length) for t in texts], dtype=np.int64)
