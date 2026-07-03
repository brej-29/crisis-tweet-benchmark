"""Tests for the uniform preprocessing protocol (dtc.data.text).

Covers: cleaning (NFC + whitespace collapse, URL/@mention preservation),
percentile-based max_length computation (generic over token-counting
function, so it can be reused for WordPiece as well as whitespace tokens),
and train-only vocab construction (Hard Rule 3: a vocab test that would
catch a vocab "fit" on val text).
"""

from __future__ import annotations

import numpy as np

from dtc.data.text import (
    PAD_ID,
    UNK_ID,
    build_vocab,
    clean_series,
    clean_text,
    compute_percentile_max_length,
    encode,
    encode_batch,
    whitespace_token_count,
)


def test_clean_text_collapses_whitespace_and_normalizes_unicode():
    assert clean_text("hello    world\t\n  again") == "hello world again"
    # NFC normalization: combining-accent form -> composed form, same rendered text
    decomposed = "café"  # 'e' + combining acute accent
    composed = "café"
    assert clean_text(decomposed) == composed


def test_clean_text_keeps_urls_and_mentions_as_tokens():
    text = "Fire near @NWS see http://example.com/alert now"
    cleaned = clean_text(text)
    assert "@NWS" in cleaned
    assert "http://example.com/alert" in cleaned


def test_clean_series_applies_to_each_element():
    out = clean_series(["a   b", "c\n\nd"])
    assert out == ["a b", "c d"]


def test_compute_percentile_max_length_hand_computed():
    # whitespace-token counts: 1, 2, 3, ..., 20 -> 95th percentile (linear
    # interpolation, numpy default) is 19.05, rounded up to 20.
    texts = [" ".join(["tok"] * n) for n in range(1, 21)]
    assert compute_percentile_max_length(texts) == 20


def test_compute_percentile_max_length_supports_custom_token_len_fn():
    texts = ["a", "aa", "aaa"]
    # custom counter: character count instead of whitespace tokens
    value = compute_percentile_max_length(texts, percentile=100.0, token_len_fn=len)
    assert value == 3


def test_whitespace_token_count():
    assert whitespace_token_count("a b  c") == 3
    assert whitespace_token_count("") == 0


def test_build_vocab_is_train_only_and_does_not_leak_val_vocabulary():
    train_texts = ["fire alarm", "flood warning", "fire truck"]
    val_only_word = "zzzznevertrain"
    vocab = build_vocab(train_texts, max_vocab_size=100)
    assert val_only_word not in vocab
    assert "fire" in vocab  # appears twice in train, should be included


def test_build_vocab_reserves_pad_and_unk_ids():
    vocab = build_vocab(["fire alarm flood"], max_vocab_size=100)
    assert vocab["<pad>"] == PAD_ID
    assert vocab["<unk>"] == UNK_ID


def test_build_vocab_respects_max_vocab_size():
    texts = ["a b c d e f g h i j"]
    vocab = build_vocab(texts, max_vocab_size=3)
    # 2 reserved (pad, unk) + 3 top tokens = 5
    assert len(vocab) == 5


def test_encode_pads_short_sequences_and_truncates_long_ones():
    vocab = build_vocab(["fire alarm flood warning"], max_vocab_size=100)
    short = encode("fire alarm", vocab, max_length=5)
    assert len(short) == 5
    assert short[-1] == PAD_ID  # padded at the end

    long_ids = encode("fire alarm flood warning extra words here", vocab, max_length=3)
    assert len(long_ids) == 3


def test_encode_maps_out_of_vocab_tokens_to_unk():
    vocab = build_vocab(["fire alarm"], max_vocab_size=100)
    ids = encode("totally unseen words here", vocab, max_length=4)
    assert all(i == UNK_ID for i in ids)


def test_encode_batch_returns_expected_shape():
    vocab = build_vocab(["fire alarm flood"], max_vocab_size=100)
    arr = encode_batch(["fire alarm", "flood"], vocab, max_length=4)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 4)
