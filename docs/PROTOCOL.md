# PROTOCOL.md — Phase 1 preprocessing protocol

Implemented once in `src/dtc/data/text.py` (whitespace-token utilities used
by every from-scratch model) plus per-family additions noted below. Applies
to the KAGGLE dataset (Phase 1's controlled core benchmark); CrisisLex
training happens in Phase 2 and will reuse this same protocol.

## 1. Uniform minimal cleaning (identical for every model)

`dtc.data.text.clean_text`: unicode NFC normalization, then whitespace
collapse (`\s+` -> single space, stripped). Nothing else.

**URLs and @mentions are KEPT as tokens** — not stripped, not replaced with
placeholders. This is a documented choice, not a silent decision:
- Disaster tweets often carry information in URLs (news links) and
  @mentions (official accounts, e.g. @NWS, @FEMA), so stripping them
  discards signal a real classifier could use.
- It is ablatable later (a URL/mention-stripping variant is a cheap,
  well-defined follow-up experiment) but is not run in Phase 1.

## 2. Sequence length: the 95th-percentile policy

Per `docs/PLAN.md` 1.1: the original (audited) project truncated at the
**mean** token length, which cut roughly half of all tweets. This project
instead uses the **95th percentile** of token counts, computed on the
**KAGGLE TRAIN split only** (Hard Rule 3), rounded up.

- **Measured value (whitespace tokens, Kaggle train split, n=5,988 rows,
  post-cleaning)**: 95th percentile = 24 (mean 14.93, median 15, max 31).
  This is `max_length` for every from-scratch token-based model
  (`meanpool_embed`, `lstm`, `gru`, `bilstm`, `conv1d`) and is recorded in
  each of those models' configs as `max_length: 24`.
- **DistilBERT** measures the *same 95th-percentile policy* but in its own
  WordPiece tokenizer's units, computed train-only via
  `dtc.data.text.compute_percentile_max_length(train_texts, token_len_fn=lambda
  t: len(tokenizer.tokenize(t)))`. WordPiece subword tokenization produces a
  different (typically higher) token count than whitespace tokenization for
  the same text, so this value is computed and recorded separately rather
  than reusing 24. See `src/dtc/models/distilbert_finetune.py` for the
  computed value once tuning/final runs record it in the ledger's config
  snapshot (every run's `config` field includes the `max_length` actually
  used, so no number here can silently drift from what ran).
- **USE (`use_frozen`)** consumes cleaned raw strings directly — the USE
  module handles its own internal tokenization and has no user-facing
  `max_length` knob, so no truncation length applies to it.

## 3. Vocab (from-scratch token-based models)

`dtc.data.text.build_vocab`: top-N whitespace tokens by frequency, built
from the **TRAIN split only** (Hard Rule 3). Default `max_vocab_size =
10000` (a tunable knob — see `configs/tuning/*.yaml` for which models
sweep it). Index 0 = `<pad>`, index 1 = `<unk>`; out-of-vocabulary tokens
at encode time map to `<unk>`.

Shared by: `meanpool_embed`, `lstm`, `gru`, `bilstm`, `conv1d`.

## 4. Per-model-family tokenization differences (inherent, not a confound)

Per `docs/PLAN.md` 1.1, tokenization is necessarily family-specific; only
the truncation *policy* (95th percentile, train-only) is required to
match, not the literal tokenizer:

| Family | Tokenization | Fit on |
|---|---|---|
| `tfidf_mnb`, `tfidf_logreg` | scikit-learn `TfidfVectorizer` (its own internal tokenizer/n-gram vocab) | TRAIN only |
| `meanpool_embed`, `lstm`, `gru`, `bilstm`, `conv1d` | whitespace tokens, project vocab (`dtc.data.text.build_vocab`) | TRAIN only |
| `use_frozen` | Universal Sentence Encoder's internal tokenization (opaque, frozen weights) | not fit — pretrained, frozen |
| `distilbert_finetune` | `distilbert-base-uncased` WordPiece tokenizer (pretrained vocab, not fit on this data) | tokenizer vocab is pretrained; only the 95th-pct max_length is measured train-only |

## 5. Frozen-test discipline

None of the fitting described above (vocab, TF-IDF, percentile
max_length) may ever see val or test text — every function in
`dtc.data.text` takes an explicit `texts` argument, and every model's
`fit()` is responsible for passing only the train split. `docs/DECISIONS.md`
records the specific test that guards this
(`tests/test_text_preprocessing.py::test_build_vocab_is_train_only_and_does_not_leak_val_vocabulary`)
and the broader frozen-test-set guard is `dtc.eval.frozen_test_loader`
(unrelated mechanism, same discipline).
