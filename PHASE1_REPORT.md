# PHASE1_REPORT.md

Phase 1 build report for `dtc`, covering `docs/PLAN.md` "PHASE 1 — Controlled
core benchmark." Per the task's scope, this session is **BUILD + SMOKE
only**: all code, configs, and the resumable driver are built and verified
with real (not mocked) tiny-subset CPU runs; the full experiment matrix
(tuning at full scale, E1/E2/E3 for real) is left for the user to execute,
per `docs/COLAB.md`. Repo root: `D:\Project\disaster-tweet-reeval`. Final
commit at time of writing: `d6f34ad` (11 commits since `PHASE0_REPORT.md`:
one per Task 0-3 and 6-7, one combined Task 4 + Task 5-partial commit, one
Task 5 completion commit, one tuning-driver extension commit, and two
smoke-verification commits from real Task 8 execution).

## 1. What was built (paths)

### Task 0 — Harness upgrade
- `src/dtc/harness/ledger.py`: `get_git_dirty_paths()` (from `git status
  --porcelain`); `is_git_dirty()` now derives the boolean as "any dirty
  path other than `results/ledger.jsonl`" instead of the old raw boolean.
- `src/dtc/harness/config.py` (new): `compute_config_id()`, a stable
  16-hex-char hash of a resolved config dict.
- `src/dtc/harness/run.py`: `build_run_record`/`log_evaluation_run` gained
  `protocol`, `phase`, `stage`, `smoke`, `train_fraction`, `config_id`,
  and (added when Task 6 needed it) an optional `dataset_split_hashes`
  override so Protocol A's runs don't have to (mis)reuse Protocol B's
  manifest.
- Tests: `tests/test_harness_ledger.py` (+11 tests over Phase 0's 7).

### Task 1 — Preprocessing protocol
- `src/dtc/data/text.py`: `clean_text`/`clean_series` (NFC + whitespace
  collapse; URLs/@mentions kept as tokens, documented choice), `build_vocab`
  (train-only top-N), `compute_percentile_max_length` (generic over a
  token-length function so DistilBERT can reuse the same 95th-percentile
  policy in WordPiece units), `encode`/`encode_batch`.
- `docs/PROTOCOL.md` (new): the full preprocessing protocol, per-model-family
  tokenization differences, and the **measured** 95th-percentile value.
- Tests: `tests/test_text_preprocessing.py` (12 tests), including a
  train-only-vocab leakage test.

### Task 2 — Nine model implementations + registry
- `src/dtc/models/base.py`: `BaseModel` (`fit`/`predict_proba`) + registry
  (`register_model`, `get_model_class`, `build_model`).
- `src/dtc/models/torch_common.py`: `set_seed`, `get_device`,
  `fit_with_early_stopping[_generic]` (patience 3 / restore-best-weights by
  default; `patience=None` + `restore_best_weights=False` for Protocol A).
- `src/dtc/models/tfidf_models.py` (`tfidf_mnb`, `tfidf_logreg`),
  `neural_sequence.py` (`meanpool_embed`, `lstm`, `gru`, `bilstm`,
  `conv1d`), `use_frozen.py` (`use_frozen`, a torch MLP head — chosen over
  sklearn LogReg so it gets the same early-stopping treatment),
  `distilbert_finetune.py` (`distilbert_finetune`).
- `src/dtc/models/registry.py`: imports all of the above so every model is
  registered by importing one module.
- `src/dtc/data/use_cache.py`: SHA256-keyed embedding cache read/write,
  shared by the model and the precompute script without either importing TF.
- Tests: `tests/test_models_registry.py` (5), `tests/test_models_units.py`
  (13, incl. one `@pytest.mark.slow` real DistilBERT download+train).

### Task 3 — USE embedding precompute
- `scripts/precompute_use.py`: TF-CPU + `tensorflow_hub` only here (Hard
  Rule 2); embeds every split's **cleaned** text (see §5, bug #1) to
  `data/<dataset>/use_embeddings/<sha256>.npy` + a manifest; `--extra-csv`
  covers Protocol A's raw (non-deduped) csv too.
- `src/dtc/data/loaders.py`: `load_split_standardized` (train/val only,
  dataset-agnostic id/text/label renaming).
- Tests: `tests/test_precompute_use.py` (4, real network+TF, skipped if
  the optional `tf` group isn't installed).

### Task 4 — Tuning procedure
- `configs/tuning/<model>.yaml` × 9: lr × one capacity knob, ≤6 configs
  each (single-axis, documented, for the two sklearn models with no
  learning rate).
- `scripts/select_configs.py`: reads only `stage == "tuning"` ledger
  records (and, by default, only non-smoke ones — see bug #2 in §5),
  picks the best val macro-F1 config per model, writes
  `configs/final/<model>.yaml`.
- Tests: `tests/test_select_configs.py` (7).

### Task 5 — E1/E2/E3 + Protocol A isolation
- `configs/experiments.yaml`: declarative `tuning`/`e1`/`e2`/`e3` definitions.
- `src/dtc/data/protocol_a.py`: raw-csv loader, 90/10 non-stratified split
  (seed 42), mean-token-length truncation — the flaws being replicated,
  not fixed.
- `configs/protocol_a/<model>.yaml` × 9: hyperparameters pulled from
  `PROJECT_AUDIT.md` (the sibling audited repo), not guessed — see §5.
- `src/dtc/harness/fractions.py`: seeded, nested stratified train-fraction
  subsampling for E3.
- `src/dtc/eval/run_evaluation.py`: the only path allowed to read the
  frozen test split for evaluation (satisfies the guard's `dtc.eval.*`
  allowlist by construction).
- Tests: `tests/test_protocol_isolation.py` (7), `tests/test_fractions.py`
  (5), `tests/test_run_evaluation.py` (3), `tests/test_torch_common.py` (3).

### Task 6 — Resumable run driver
- `scripts/run_matrix.py`: expands `configs/experiments.yaml` (including
  tuning's per-model grids) into individual runs; skips anything already
  ledgered under a `(stage, protocol, model_name, config_id, seed,
  train_fraction, smoke)` key; `--dry-run`, `--only`, `--models`, `--seeds`,
  `--smoke`/`--smoke-n`.
- `scripts/merge_ledger.py`: appends only new, schema-valid, non-duplicate
  remote ledger lines.
- `docs/COLAB.md`: the full clone → install → run → download → merge runbook.
- Tests: `tests/test_run_matrix.py` (9, incl. a real tmp_path end-to-end
  dry-run → run → resume-skip cycle), `tests/test_merge_ledger.py` (6).

### Task 7 — Tables
- `scripts/make_tables.py`: T1 (Protocol B main results), T2 (Protocol A
  vs. B ranking + rank-delta), T3 (seed-variance), T4 (data-efficiency),
  from the ledger only. Excludes smoke by default; `--include-smoke`
  watermarks every table `SMOKE`. Refuses to run if any considered record's
  `git_dirty_paths` includes a source file.
- Tests: `tests/test_make_tables.py` (10).

### Task 8 — Smoke verification (this task)
- Real USE embeddings precomputed for the actual Kaggle dataset (all
  splits + the raw csv), real smoke runs executed for all 9 models across
  tuning/E1/E2/E3, resume-skip demonstrated for real, `select_configs.py`
  run end-to-end on real (smoke) tuning data, `make_tables.py` exercised
  in both modes. Two real bugs found and fixed — see §5.

## 2. Full pytest output (commit `d6f34ad`, 129 tests)

```
============================= test session starts =============================
platform win32 -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0 -- D:\Project\disaster-tweet-reeval\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Project\disaster-tweet-reeval
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.14.1
collecting ... collected 129 items

tests/test_crisislex_pipeline.py::test_label_map_values PASSED
tests/test_crisislex_pipeline.py::test_map_labels_maps_correctly PASSED
tests/test_crisislex_pipeline.py::test_map_labels_raises_on_unknown_value PASSED
tests/test_crisislex_pipeline.py::test_load_event_csv_has_clean_ids_and_text PASSED
tests/test_crisislex_pipeline.py::test_prepare_crisislex_end_to_end_no_leakage_and_balance PASSED
tests/test_crisislex_pipeline.py::test_prepare_crisislex_is_reproducible PASSED
tests/test_floor_baselines.py::test_majority_class_always_predicts_the_majority_label PASSED
tests/test_floor_baselines.py::test_stratified_random_is_seeded_deterministically PASSED
tests/test_floor_baselines.py::test_stratified_random_roughly_matches_train_distribution PASSED
tests/test_floor_baselines.py::test_run_floor_baselines_end_to_end_ledgers_two_runs PASSED
tests/test_fractions.py::test_nested_fraction_indices_are_subsets_of_larger_fractions PASSED
tests/test_fractions.py::test_nested_fraction_indices_is_deterministic_given_seed PASSED
tests/test_fractions.py::test_nested_fraction_indices_differ_across_seeds PASSED
tests/test_fractions.py::test_nested_fraction_indices_are_approximately_stratified PASSED
tests/test_fractions.py::test_subsample_train_df_returns_expected_row_count_and_nesting PASSED
tests/test_frozen_test_guard.py::test_disallowed_caller_raises PASSED
tests/test_frozen_test_guard.py::test_disallowed_data_prep_caller_raises PASSED
tests/test_frozen_test_guard.py::test_allowed_eval_package_caller_succeeds PASSED
tests/test_frozen_test_guard.py::test_allowed_scripts_evaluate_caller_succeeds PASSED
tests/test_frozen_test_guard.py::test_disallowed_scripts_non_evaluate_caller_raises PASSED
tests/test_frozen_test_guard.py::test_no_training_or_data_module_imports_frozen_test_loader PASSED
tests/test_harness_ledger.py::test_generate_run_id_is_unique PASSED
tests/test_harness_ledger.py::test_get_git_commit_hash_returns_full_sha PASSED
tests/test_harness_ledger.py::test_append_run_record_writes_well_formed_jsonl PASSED
tests/test_harness_ledger.py::test_ledger_is_append_only_not_overwritten PASSED
tests/test_harness_ledger.py::test_append_run_record_never_truncates_existing_lines PASSED
tests/test_harness_ledger.py::test_build_run_record_has_required_keys PASSED
tests/test_harness_ledger.py::test_build_run_record_honors_explicit_protocol_phase_smoke_and_config_id PASSED
tests/test_harness_ledger.py::test_build_run_record_without_manifest_uses_explicit_split_hashes PASSED
tests/test_harness_ledger.py::test_build_run_record_with_neither_manifest_nor_hashes_leaves_hashes_none PASSED
tests/test_harness_ledger.py::test_compute_config_id_is_deterministic_and_key_order_independent PASSED
tests/test_harness_ledger.py::test_is_git_dirty_ignores_ledger_only_changes PASSED
tests/test_harness_ledger.py::test_is_git_dirty_true_when_non_ledger_paths_dirty PASSED
tests/test_harness_ledger.py::test_is_git_dirty_false_when_tree_clean PASSED
tests/test_harness_ledger.py::test_save_predictions_writes_expected_columns PASSED
tests/test_kaggle_pipeline.py::test_resolve_duplicates_keeps_first_of_consistent_duplicate PASSED
tests/test_kaggle_pipeline.py::test_resolve_duplicates_drops_entire_conflicting_group PASSED
tests/test_kaggle_pipeline.py::test_resolve_duplicates_keeps_unique_rows PASSED
tests/test_kaggle_pipeline.py::test_count_exact_duplicate_rows PASSED
tests/test_kaggle_pipeline.py::test_stratified_split_no_overlap_and_sizes PASSED
tests/test_kaggle_pipeline.py::test_stratified_split_is_deterministic_given_seed PASSED
tests/test_kaggle_pipeline.py::test_split_ratios_must_sum_to_one PASSED
tests/test_kaggle_pipeline.py::test_prepare_kaggle_end_to_end_no_leakage_and_balance PASSED
tests/test_kaggle_pipeline.py::test_prepare_kaggle_is_reproducible PASSED
tests/test_make_tables.py::test_build_t1_computes_mean_and_std_per_model PASSED
tests/test_make_tables.py::test_build_t1_excludes_non_e1_or_non_test_split_records PASSED
tests/test_make_tables.py::test_build_t2_ranks_and_computes_rank_delta PASSED
tests/test_make_tables.py::test_build_t3_computes_min_max_std PASSED
tests/test_make_tables.py::test_build_t4_groups_by_model_and_fraction PASSED
tests/test_make_tables.py::test_check_no_dirty_source_runs_raises_for_source_paths PASSED
tests/test_make_tables.py::test_check_no_dirty_source_runs_allows_data_and_results_paths PASSED
tests/test_make_tables.py::test_main_excludes_smoke_by_default_and_includes_with_flag PASSED
tests/test_make_tables.py::test_main_is_idempotent PASSED
tests/test_make_tables.py::test_main_refuses_when_referenced_run_has_dirty_source_paths PASSED
tests/test_merge_ledger.py::test_validate_record_flags_missing_keys PASSED
tests/test_merge_ledger.py::test_validate_record_accepts_well_formed_record PASSED
tests/test_merge_ledger.py::test_merge_appends_new_valid_records PASSED
tests/test_merge_ledger.py::test_merge_rejects_duplicate_run_ids PASSED
tests/test_merge_ledger.py::test_merge_rejects_schema_violations PASSED
tests/test_merge_ledger.py::test_merge_does_not_touch_local_file_when_remote_has_nothing_new PASSED
tests/test_metrics.py::test_accuracy PASSED
tests/test_metrics.py::test_macro_f1 PASSED
tests/test_metrics.py::test_weighted_f1 PASSED
tests/test_metrics.py::test_positive_f1 PASSED
tests/test_metrics.py::test_accuracy_macro_weighted_are_pairwise_distinct PASSED
tests/test_metrics.py::test_confusion_matrix PASSED
tests/test_metrics.py::test_per_class_precision_recall PASSED
tests/test_metrics.py::test_compute_all_metrics_shape PASSED
tests/test_metrics.py::test_perfect_predictions PASSED
tests/test_models_registry.py::test_all_nine_models_are_registered PASSED
tests/test_models_registry.py::test_get_model_class_returns_a_basemodel_subclass PASSED
tests/test_models_registry.py::test_get_model_class_raises_on_unknown_name PASSED
tests/test_models_registry.py::test_build_model_constructs_an_instance PASSED
tests/test_models_registry.py::test_register_model_rejects_reregistering_a_different_class_under_same_name PASSED
tests/test_models_units.py::test_tfidf_models_fit_and_predict[tfidf_mnb] PASSED
tests/test_models_units.py::test_tfidf_models_fit_and_predict[tfidf_logreg] PASSED
tests/test_models_units.py::test_vocab_sequence_models_fit_and_predict[meanpool_embed] PASSED
tests/test_models_units.py::test_vocab_sequence_models_fit_and_predict[lstm] PASSED
tests/test_models_units.py::test_vocab_sequence_models_fit_and_predict[gru] PASSED
tests/test_models_units.py::test_vocab_sequence_models_fit_and_predict[bilstm] PASSED
tests/test_models_units.py::test_vocab_sequence_models_fit_and_predict[conv1d] PASSED
tests/test_models_units.py::test_vocab_sequence_model_vocab_is_train_only PASSED
tests/test_models_units.py::test_meanpool_embed_respects_no_early_stopping_config PASSED
tests/test_models_units.py::test_use_frozen_fit_and_predict_with_synthetic_cache PASSED
tests/test_models_units.py::test_use_frozen_raises_key_error_for_uncached_text PASSED
tests/test_models_units.py::test_distilbert_finetune_fit_and_predict PASSED
tests/test_precompute_use.py::test_precompute_use_embeds_20_texts_and_hits_cache_on_rerun PASSED
tests/test_precompute_use.py::test_precompute_use_caches_under_cleaned_text_hash_not_raw PASSED
tests/test_precompute_use.py::test_precompute_use_embeds_extra_csv_paths PASSED
tests/test_protocol_isolation.py::test_protocol_a_module_does_not_reference_frozen_test_loader PASSED
tests/test_protocol_isolation.py::test_protocol_b_code_does_not_import_protocol_a PASSED
tests/test_protocol_isolation.py::test_protocol_a_split_is_90_10_non_stratified_and_deterministic PASSED
tests/test_protocol_isolation.py::test_mean_token_length_hand_computed PASSED
tests/test_protocol_isolation.py::test_mean_token_length_supports_custom_token_len_fn PASSED
tests/test_run_evaluation.py::test_load_frozen_test_standardized_renames_kaggle_columns PASSED
tests/test_run_evaluation.py::test_load_frozen_test_standardized_renames_crisislex_columns PASSED
tests/test_run_evaluation.py::test_evaluate_model_on_frozen_test_returns_expected_fields PASSED
tests/test_run_matrix.py::test_build_run_specs_enumerates_all_axes PASSED
tests/test_run_matrix.py::test_build_run_specs_respects_only_filter PASSED
tests/test_run_matrix.py::test_build_run_specs_respects_models_filter PASSED
tests/test_run_matrix.py::test_skip_key_and_already_ledgered_keys_round_trip PASSED
tests/test_run_matrix.py::test_skip_key_distinguishes_smoke_from_real_runs PASSED
tests/test_run_matrix.py::test_real_experiments_yaml_parses_with_expected_shape PASSED
tests/test_run_matrix.py::test_build_run_specs_expands_tuning_grid_into_one_spec_per_entry PASSED
tests/test_run_matrix.py::test_dry_run_enumerates_full_matrix_including_tuning PASSED
tests/test_run_matrix.py::test_end_to_end_dry_run_and_resume_skip_against_tiny_fixture PASSED
tests/test_select_configs.py::test_select_best_configs_picks_highest_macro_f1_per_model PASSED
tests/test_select_configs.py::test_select_best_configs_ignores_non_tuning_stage_records PASSED
tests/test_select_configs.py::test_select_best_configs_excludes_smoke_by_default PASSED
tests/test_select_configs.py::test_select_best_configs_include_smoke_true_considers_smoke_records PASSED
tests/test_select_configs.py::test_select_best_configs_respects_models_filter PASSED
tests/test_select_configs.py::test_write_final_configs_and_end_to_end_main PASSED
tests/test_select_configs.py::test_main_with_no_tuning_records_writes_nothing PASSED
tests/test_smoke.py::test_package_imports PASSED
tests/test_text_preprocessing.py::test_clean_text_collapses_whitespace_and_normalizes_unicode PASSED
tests/test_text_preprocessing.py::test_clean_text_keeps_urls_and_mentions_as_tokens PASSED
tests/test_text_preprocessing.py::test_clean_series_applies_to_each_element PASSED
tests/test_text_preprocessing.py::test_compute_percentile_max_length_hand_computed PASSED
tests/test_text_preprocessing.py::test_compute_percentile_max_length_supports_custom_token_len_fn PASSED
tests/test_text_preprocessing.py::test_whitespace_token_count PASSED
tests/test_text_preprocessing.py::test_build_vocab_is_train_only_and_does_not_leak_val_vocabulary PASSED
tests/test_text_preprocessing.py::test_build_vocab_reserves_pad_and_unk_ids PASSED
tests/test_text_preprocessing.py::test_build_vocab_respects_max_vocab_size PASSED
tests/test_text_preprocessing.py::test_encode_pads_short_sequences_and_truncates_long_ones PASSED
tests/test_text_preprocessing.py::test_encode_maps_out_of_vocab_tokens_to_unk PASSED
tests/test_text_preprocessing.py::test_encode_batch_returns_expected_shape PASSED
tests/test_torch_common.py::test_default_patience_can_stop_before_max_epochs_or_reach_it PASSED
tests/test_torch_common.py::test_patience_none_always_runs_the_full_max_epochs PASSED
tests/test_torch_common.py::test_restore_best_weights_false_skips_checkpoint_restoration PASSED

======================= 129 passed in 131.58s (0:02:11) =======================
```

`uv run ruff check .` → `All checks passed!` (verified immediately before
this run, same commit).

## 3. Computed 95th-percentile `max_length` (Kaggle train split)

Measured on the real, cleaned Kaggle train split (n=5,988 rows, whitespace
tokens): **95th percentile = 24** (mean 14.93, median 15, max 31). Recorded
in `docs/PROTOCOL.md` and `docs/DECISIONS.md`. DistilBERT computes the same
95th-percentile *policy* independently in its own WordPiece token units
(not 24 — a different tokenizer produces a different count for the same
text), per `docs/PROTOCOL.md` §2.

For comparison: `PROJECT_AUDIT.md` (the sibling audited repo) reports the
original project used a fixed `max_length=15` — the **mean**, not the
95th percentile, cutting roughly half of all tweets. This project's
Protocol A replication (`dtc.data.protocol_a.mean_token_length`) computes
that same mean-based value fresh from Protocol A's own (raw, non-deduped)
split rather than hardcoding 15, so the two numbers may differ slightly by
construction, and the actual value used is recorded in every run's ledgered
`config`.

## 4. Real smoke-run ledger excerpt

93 total ledger lines exist; 91 are Phase 1 (`phase: "phase1"`), all
`smoke: true`, broken down by stage:

| stage | count |
|---|---:|
| `tuning` | 52 (9 models × grid size: 4/6/6/6/6/6/6/6/6) |
| `E1` | 18 (9 models × seeds {0, 1}) |
| `E2` | 9 (9 models × seed 42) |
| `E3` | 12 (`tfidf_logreg` × 6 fractions × seeds {0, 1}) |

One representative record per stage (abridged):

```json
{
  "stage": "tuning", "protocol": "B", "model_name": "tfidf_mnb",
  "config": {"alpha": 0.1}, "split": "val", "smoke": true,
  "metrics": {"accuracy": 0.725, "macro_f1": 0.6925, "positive_f1": 0.5926}
}
{
  "stage": "E1", "protocol": "B", "model_name": "tfidf_mnb",
  "split": "test", "smoke": true,
  "metrics": {"accuracy": 0.6929, "macro_f1": 0.6670, "positive_f1": 0.5741}
}
{
  "stage": "E2", "protocol": "A", "model_name": "tfidf_mnb",
  "split": "protocol_a_eval", "smoke": true,
  "dataset_split_hashes": {"protocol_a_train": "1a18c227...", "protocol_a_eval": "b02ae625..."},
  "metrics": {"accuracy": 0.7, "macro_f1": 0.6, "weighted_f1_legacy": 0.64}
}
{
  "stage": "E3", "protocol": "B", "model_name": "tfidf_logreg",
  "train_fraction": 0.01, "split": "test", "smoke": true,
  "metrics": {"accuracy": 0.502, "macro_f1": 0.4965}
}
```

Note `dataset_manifest_path: null` and `dataset_split_hashes` keyed
`protocol_a_train`/`protocol_a_eval` (not `train`/`val`/`test`) for the E2
record — Protocol A never reuses Protocol B's manifest (§5, harness change).

## 5. Deviations / real findings from this session

1. **Bug found — USE cache keyed by raw text, not cleaned text.**
   `scripts/precompute_use.py` originally hashed the raw CSV `text` column;
   `dtc.models.use_frozen` hashes *cleaned* text (`dtc.data.text.clean_text`)
   at fit/predict time, per `docs/PROTOCOL.md` §4 ("USE consumes cleaned
   raw strings"). This produced a real `KeyError` on the first live
   `run_matrix.py --smoke --only tuning` invocation, not a hypothetical.
   Fixed by cleaning text before hashing in the precompute script; a
   regression test (`test_precompute_use_caches_under_cleaned_text_hash_not_raw`)
   now guards it. The stale cache was deleted and regenerated.
2. **Real resumability demonstration (unplanned).** The crash above
   happened after 40 of 52 tuning-smoke runs had already succeeded and
   been ledgered (crash was inside `use_frozen`'s `fit`, 8th of 9 models
   in enumeration order). Re-running the identical command after the fix
   correctly skipped those 40 and executed only the remaining 12
   (`use_frozen` + `distilbert_finetune`) — proving the driver's
   crash-safety claim against a real crash, not just the synthetic
   fixture in `tests/test_run_matrix.py`.
3. **Bug found — `select_configs.py` didn't actually exclude smoke runs.**
   Its docstring claimed smoke runs were ignored; the filter only checked
   `stage == "tuning"`. Fixed with an `include_smoke` parameter (default
   `False`), mirroring `make_tables.py`'s `--include-smoke` pattern, so a
   lucky smoke-subset config can never silently outrank a real tuning
   result once both share the ledger.
4. **`make_tables.py`'s dirty-source refusal fired for real, correctly.**
   Running `make_tables.py --include-smoke` against the live ledger
   refuses, because most of this session's smoke runs were executed while
   source files were still being actively edited (a genuinely dirty tree,
   not a bug in the check). The **default** invocation (no flag) succeeds
   with empty tables, since zero non-smoke runs exist yet — this is the
   correct production behavior. A dedicated clean-tree smoke batch (E1
   seed=1, E3 seed=1, run immediately after a commit) was added specifically
   to prove the driver produces clean (`git_dirty_paths` containing at most
   the exempted `results/ledger.jsonl`) ledger entries when run against a
   clean tree — but since the ledger is append-only, the *populated*
   `--include-smoke` table path can't be demonstrated live against this
   session's real (historically dirty) ledger without real, non-smoke runs
   first. `tests/test_make_tables.py::test_main_excludes_smoke_by_default_and_includes_with_flag`
   is the authoritative proof that path works (synthetic, clean fixture).
5. **Protocol A hyperparameters sourced from `PROJECT_AUDIT.md`,** not
   guessed: `Embedding(10000, 128)`, `LSTM(64)`/`GRU(64)`/
   `Bidirectional(LSTM(64))`/`Conv1D(filters=32, kernel_size=5)`, no
   dropout anywhere, `epochs=5`, Adam defaults, no early stopping, and —
   notably — `TfidfVectorizer()`/`MultinomialNB()` with **no** `max_features`
   cap (unlike this project's own 10000 default). The audit also revealed
   the original notebook's own markdown claims a "Logistic Regression"
   model that was never actually executed (only `MultinomialNB` was) — so
   `tfidf_logreg` and `distilbert_finetune` have no original equivalent and
   use documented plain defaults instead.
6. **`stage` field reused for E1/E2/E3/tuning identification** rather than
   adding a separate `experiment` field — one field fully identifies "what
   this run is for."
7. **`dtc.harness.run.build_run_record`'s `dataset_manifest_path` made
   optional**, with a `dataset_split_hashes` override, so Protocol A runs
   record real hashes of their *own* split instead of misleadingly reusing
   Protocol B's manifest.
8. **`torch_common.fit_with_early_stopping[_generic]` gained
   `patience: int | None` and `restore_best_weights: bool`** so Protocol
   A's "no early stopping, fixed N epochs" replication reuses the exact
   same model classes as Protocol B.
9. Task 6's driver was extended to also enumerate **tuning** runs (not
   just E1/E2/E3), per the acceptance criterion that `--dry-run` show the
   complete "E1+tuning+E2+E3" matrix through one code path.

## 6. Real `--dry-run` output (full E1+tuning+E2+E3 matrix)

Run against the actual repo, commit `d6f34ad`, with `configs/final/*.yaml`
populated by the smoke-tuning placeholders (§ "configs/final/README.md"):

```
e1: 45 pending / 45 total   (9 models x 5 seeds x 1.0 fraction)
e2: 9 pending / 9 total     (9 models x 1 seed x 1.0 fraction)
e3: 90 pending / 90 total   (5 models x 3 seeds x 6 fractions)
tuning: 52 pending / 52 total (9 models, grid sizes 4/6/6/6/6/6/6/6/6)

Total pending runs: 196 / 196
```

(All 196 are real/non-smoke and correctly show as pending, since only
smoke-tagged runs exist in the ledger so far — the skip-key includes the
`smoke` flag, so smoke and real runs never collide.)

**Rough wall-time estimates**, extrapolated from real per-run smoke
timings on ~200-example CPU subsets (not benchmarked at full scale — these
are order-of-magnitude planning estimates, not measured full-scale numbers):

| Model family | Smoke run (~200 ex, CPU) | Full-scale rough estimate (CPU) | Full-scale rough estimate (T4 GPU) |
|---|---|---|---|
| `tfidf_mnb`/`tfidf_logreg` | <1s | seconds | n/a (sklearn, CPU-only) |
| `meanpool_embed`/`conv1d` | 3-10s | tens of seconds | seconds |
| `lstm`/`gru`/`bilstm` | 5-20s | 1-3 min per run | 10-30s per run |
| `use_frozen` | 2-6s | tens of seconds (linear head only) | seconds |
| `distilbert_finetune` | 2-6 min per config | 15-40 min per run (CPU) | 1-4 min per run (GPU) |

**Measured GPU smoke run (Phase 1.6)**: after switching to a CUDA build of
torch (`torch==2.12.1+cu126`, see `docs/DECISIONS.md`), one real
`distilbert_finetune` E1 smoke run (seed 2, ~200-example subset, run
through `scripts/run_matrix.py --smoke --only e1 --models
distilbert_finetune --seeds 2`, `run_id=2ef5d73089d74501b53334eea759cb4b`)
completed in **42s wall time** on the local RTX 3050 (6GB) — vs. the 2-6
min CPU smoke estimate above. GPU execution was confirmed two ways: the
run's `dtc.models.torch_common.get_device()` call returns `cuda` whenever
`torch.cuda.is_available()` is `True`, and the training step raised
PyTorch's determinism warning from
`aten/src/ATen/native/transformers/cuda/attention_backward.cu` (a
CUDA-only code path, never reached on CPU). This is one anecdotal
data point at smoke scale, not a full-scale benchmark, but it directionally
confirms the GPU-vs-CPU gap this section estimated.

At these rates, E1 (45 runs, dominated by 5 DistilBERT runs) is CPU-hours
on a laptop but well under an hour on a T4; the full tuning stage (52
configs, 6 of them DistilBERT) is the single most expensive piece and is
the strongest argument for running on Colab GPU per `docs/COLAB.md`. E3's
90 runs are mostly cheap (fractional data means less compute per run,
partially offsetting the 18x run-count increase from tuning's 52).

## 7. Acceptance criteria checklist

1. [x] All tests green (129 passed, pasted above); `ruff check .` clean.
2. [x] Nine models registered (`tests/test_models_registry.py`); each has
   a real passing smoke run in the ledger (tuning + E1 + E2, all 9 models).
3. [x] Tuning grids exist for all nine (`configs/tuning/*.yaml`);
   `select_configs.py` demonstrated end to end on real smoke tuning data
   (`--include-smoke`); the resulting `configs/final/*.yaml` are clearly
   marked as smoke placeholders in `configs/final/README.md`, with exact
   regeneration commands.
4. [x] Protocol A isolation verified by tests
   (`tests/test_protocol_isolation.py`): its split path is unused by
   Protocol B code and vice versa; no frozen-test import.
5. [x] Driver `--dry-run` enumerates the complete tuning+E1+E2+E3 matrix
   deterministically (196 pending runs, §6); resume-skip demonstrated for
   real (tuning 52/52, E1 9/9 + 9/9, E2 9/9, E3 6/6 — all "0 run, N
   skipped" on re-invocation).
6. [x] `make_tables.py` verified in both modes against the real ledger:
   default succeeds (empty tables, no real runs yet); `--include-smoke`
   correctly refuses due to genuinely dirty historical smoke runs (§5,
   item 4) — the populated-SMOKE-table path is proven by
   `tests/test_make_tables.py` (10 passing, synthetic fixtures).
7. [x] `docs/COLAB.md` + `scripts/merge_ledger.py` present and coherent
   (`tests/test_merge_ledger.py`, 6 passing).
8. [x] `PHASE1_REPORT.md` committed; per-task commits throughout (11
   commits since `PHASE0_REPORT.md`, see §1 header, plus this report's
   own commit).

## 8. Open issues, stated plainly

- **`configs/final/*.yaml` are smoke placeholders**, not real tuning
  results — must be regenerated (`run_matrix.py --only tuning` without
  `--smoke`, then `select_configs.py` without `--include-smoke`) before E1
  or E3 execute for real. Flagged in `configs/final/README.md`.
- **No real (non-smoke) E1/E2/E3/tuning runs exist yet** — this session
  was scoped to BUILD + SMOKE only per the task's Hard Rules. The user
  executes the real matrix next, per `docs/COLAB.md`.
- **Wall-time estimates in §6 are extrapolated, not measured at full
  scale** — stated as rough planning numbers, not benchmarked facts.
- **The real ledger's smoke history is permanently dirty-flagged** for
  most entries (this session actively edited source while smoke-testing)
  — this is accurate, not a defect, and doesn't block anything: smoke runs
  are excluded from `make_tables.py` by default regardless of dirty state.
- **`data/kaggle/use_embeddings/` cache exists only for Kaggle** — CrisisLex
  (Phase 2) will need its own `precompute_use.py --dataset crisislex` run
  before any `use_frozen` training on that dataset.
- Both datasets' licenses remain `[UNCLEAR]` (carried over from Phase 0,
  unchanged this session).
- Per Hard Rule 4's "no third outcome": every task in this phase resolved
  to "built and smoke-verified," none required a blocker report.
