# PHASE0_REPORT.md

Phase 0 build report for `dtc` (Disaster Tweet Classification: Controlled
Re-Evaluation), covering the tasks in `docs/PLAN.md` section "PHASE 0 —
Rebuild: repo, data, harness." Repo root: `D:\Project\disaster-tweet-reeval`.
Final commit at time of writing: `d199280` (6 commits total, one per task).

## 1. What was built

### Task 1 — Repo scaffold
- `src/dtc/` installable package (hatchling, src-layout) with submodules
  `data/`, `models/`, `train/`, `eval/`, `analysis/`, `harness/`.
- `pyproject.toml`: Python `>=3.11`, core deps (`pandas`, `scikit-learn`,
  `pyyaml`, `numpy`), dev deps (`pytest`, `ruff`) in a `dependency-groups.dev`
  group, `uv.lock` committed.
- `.github/workflows/ci.yml`: `ruff check .` + `pytest -v` on every push/PR.
- `docs/PLAN.md` (the full plan, committed verbatim) and `docs/DECISIONS.md`
  (deviation/decision log, 9 entries by the end of Phase 0).
- Files: `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`,
  `README.md`, `.github/workflows/ci.yml`.

### Task 2 — Metrics module + frozen-test-set guard
- `src/dtc/eval/metrics.py`: `accuracy`, `macro_f1`, `positive_f1`
  (binary, positive class), `per_class_precision_recall`, `confusion_matrix`,
  `weighted_f1` (explicitly docstringed "legacy comparison only" — matches
  the audited original project's `average="weighted"` convention), and
  `compute_all_metrics` as the single entrypoint the harness uses.
- `src/dtc/eval/frozen_test_loader.py`: `load_frozen_test()` raises
  `FrozenTestAccessError` unless called from `dtc.eval.*` or a
  `scripts/evaluate_*.py` entrypoint (checked via the caller's module name
  and source file path, not just convention).
- Tests: `tests/test_metrics.py` (hand-computed 8-example imbalanced
  fixture where accuracy=5/8, macro-F1=13/21, weighted-F1=53/84,
  positive-F1=2/3 are all pairwise distinct — see the file's docstring for
  the full by-hand derivation), `tests/test_frozen_test_guard.py` (runtime
  allow/deny checks + a static source-scan asserting no file under
  `src/dtc/{data,models,train,harness}` contains the string
  `frozen_test_loader`).

### Task 3 — Kaggle dataset pipeline
- `src/dtc/data/kaggle.py` + `src/dtc/data/common.py` (shared dedup/split/
  hash logic, used by both Kaggle and CrisisLex — extracted mid-session,
  see `docs/DECISIONS.md`).
- `scripts/prepare_kaggle.py` + `configs/kaggle.yaml` (seed=42, 80/10/10).
- Raw `train.csv` obtained by extracting it from the audited course-derived
  repo's `nlp_getting_started.zip` (same public competition file, not code
  — see `docs/DECISIONS.md`).
- Tests: `tests/test_kaggle_pipeline.py` — dedup-policy unit tests on a
  synthetic fixture, stratified-split unit tests (no-overlap, determinism,
  ratio validation), and two integration tests running the real script
  end-to-end against the real `train.csv` (leakage check, class-balance
  tolerance, reproducibility across two independent runs).

### Task 4 — CrisisLex T6 pipeline
- Text availability and license verified directly against the actual
  distributed files (`D:\Project\CrisisLexT6-v1.0.zip`, supplied by the
  user) — see section 3 below for the full finding.
- `src/dtc/data/crisislex.py`: loads and combines the 6 per-event CSVs
  (proper CSV parsing required; quoted fields contain embedded
  commas/newlines), strips the literal single-quote wrapping on
  `tweet id` values, preserves an `event` column through every split, maps
  `on-topic -> 1` / `off-topic -> 0`, then reuses `dtc.data.common` for
  dedup + stratified split.
- `scripts/prepare_crisislex.py` + `configs/crisislex.yaml` (seed=42,
  80/10/10).
- `docs/DATASETS.md`: both datasets' sources, label semantics, the
  documented mismatch between Kaggle's general "real disaster" judgment
  and CrisisLex's event-specific "on-topic" judgment, and the license
  ambiguity for both.
- Tests: `tests/test_crisislex_pipeline.py` — label-mapping unit tests
  (including a rejection test for unmapped values), a real-data unit test
  on `load_event_csv`, and two integration tests (leakage/balance,
  reproducibility) against the real 6 event files.

### Task 5 — Run harness + ledger + floor baselines
- `src/dtc/harness/ledger.py`: `append_run_record()` is the only
  ledger-writing function in the codebase (no update/delete function
  exists); `read_ledger()` for reading; `get_git_commit_hash()` /
  `is_git_dirty()`.
- `src/dtc/harness/run.py`: `build_run_record()` (config snapshot, seed,
  git commit + dirty flag, dataset split hashes pulled from that dataset's
  manifest, metrics), `save_predictions()` (per-example
  `id, text_sha256, y_true, y_pred, y_prob` to
  `results/runs/<run_id>/predictions.csv`), `log_evaluation_run()` tying
  both together.
- `src/dtc/models/floor_baselines.py` + `scripts/run_floor_baselines.py`:
  majority-class and stratified-random baselines (`sklearn.dummy.
  DummyClassifier`), evaluated on the Kaggle **val** split only — this
  script never imports `frozen_test_loader`, and the frozen test split is
  untouched.
- Tests: `tests/test_harness_ledger.py` (append-only behavior, required
  run-record keys, prediction-file schema), `tests/test_floor_baselines.py`
  (baseline determinism/distribution-matching unit tests + an end-to-end
  test of the real script against the real Kaggle val split).

## 2. Full pytest output (final run, commit `d199280`)

```
============================= test session starts =============================
platform win32 -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0 -- D:\Project\disaster-tweet-reeval\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Project\disaster-tweet-reeval
configfile: pyproject.toml
testpaths: tests
collecting ... collected 42 items

tests/test_crisislex_pipeline.py::test_label_map_values PASSED           [  2%]
tests/test_crisislex_pipeline.py::test_map_labels_maps_correctly PASSED  [  4%]
tests/test_crisislex_pipeline.py::test_map_labels_raises_on_unknown_value PASSED [  7%]
tests/test_crisislex_pipeline.py::test_load_event_csv_has_clean_ids_and_text PASSED [  9%]
tests/test_crisislex_pipeline.py::test_prepare_crisislex_end_to_end_no_leakage_and_balance PASSED [ 11%]
tests/test_crisislex_pipeline.py::test_prepare_crisislex_is_reproducible PASSED [ 14%]
tests/test_floor_baselines.py::test_majority_class_always_predicts_the_majority_label PASSED [ 16%]
tests/test_floor_baselines.py::test_stratified_random_is_seeded_deterministically PASSED [ 19%]
tests/test_floor_baselines.py::test_stratified_random_roughly_matches_train_distribution PASSED [ 21%]
tests/test_floor_baselines.py::test_run_floor_baselines_end_to_end_ledgers_two_runs PASSED [ 23%]
tests/test_frozen_test_guard.py::test_disallowed_caller_raises PASSED    [ 26%]
tests/test_frozen_test_guard.py::test_disallowed_data_prep_caller_raises PASSED [ 28%]
tests/test_frozen_test_guard.py::test_allowed_eval_package_caller_succeeds PASSED [ 30%]
tests/test_frozen_test_guard.py::test_allowed_scripts_evaluate_caller_succeeds PASSED [ 33%]
tests/test_frozen_test_guard.py::test_disallowed_scripts_non_evaluate_caller_raises PASSED [ 35%]
tests/test_frozen_test_guard.py::test_no_training_or_data_module_imports_frozen_test_loader PASSED [ 38%]
tests/test_harness_ledger.py::test_generate_run_id_is_unique PASSED      [ 40%]
tests/test_harness_ledger.py::test_get_git_commit_hash_returns_full_sha PASSED [ 42%]
tests/test_harness_ledger.py::test_append_run_record_writes_well_formed_jsonl PASSED [ 45%]
tests/test_harness_ledger.py::test_ledger_is_append_only_not_overwritten PASSED [ 47%]
tests/test_harness_ledger.py::test_append_run_record_never_truncates_existing_lines PASSED [ 50%]
tests/test_harness_ledger.py::test_build_run_record_has_required_keys PASSED [ 52%]
tests/test_harness_ledger.py::test_save_predictions_writes_expected_columns PASSED [ 54%]
tests/test_kaggle_pipeline.py::test_resolve_duplicates_keeps_first_of_consistent_duplicate PASSED [ 57%]
tests/test_kaggle_pipeline.py::test_resolve_duplicates_drops_entire_conflicting_group PASSED [ 59%]
tests/test_kaggle_pipeline.py::test_resolve_duplicates_keeps_unique_rows PASSED [ 61%]
tests/test_kaggle_pipeline.py::test_count_exact_duplicate_rows PASSED    [ 64%]
tests/test_kaggle_pipeline.py::test_stratified_split_no_overlap_and_sizes PASSED [ 66%]
tests/test_kaggle_pipeline.py::test_stratified_split_is_deterministic_given_seed PASSED [ 69%]
tests/test_kaggle_pipeline.py::test_split_ratios_must_sum_to_one PASSED  [ 71%]
tests/test_kaggle_pipeline.py::test_prepare_kaggle_end_to_end_no_leakage_and_balance PASSED [ 73%]
tests/test_kaggle_pipeline.py::test_prepare_kaggle_is_reproducible PASSED [ 76%]
tests/test_metrics.py::test_accuracy PASSED                              [ 78%]
tests/test_metrics.py::test_macro_f1 PASSED                              [ 80%]
tests/test_metrics.py::test_weighted_f1 PASSED                           [ 83%]
tests/test_metrics.py::test_positive_f1 PASSED                           [ 85%]
tests/test_metrics.py::test_accuracy_macro_weighted_are_pairwise_distinct PASSED [ 88%]
tests/test_metrics.py::test_confusion_matrix PASSED                      [ 90%]
tests/test_metrics.py::test_per_class_precision_recall PASSED            [ 92%]
tests/test_metrics.py::test_compute_all_metrics_shape PASSED             [ 95%]
tests/test_metrics.py::test_perfect_predictions PASSED                   [ 97%]
tests/test_smoke.py::test_package_imports PASSED                         [100%]

============================= 42 passed in 7.56s ==============================
```

`uv run ruff check .` → `All checks passed!` (verified immediately before
this run, same commit).

## 3. Kaggle manifest (actual numbers, `data/kaggle/manifest.json`)

```json
{
  "dataset": "kaggle_nlp_getting_started",
  "raw_csv_path": "data/kaggle/raw/train.csv",
  "raw_sha256": "61111c6dc31eaffa34d1e1fa62e2395325c9bc3b38bba1941a5f1ed9b3fa60df",
  "raw_row_count": 7613,
  "exact_duplicate_row_count": 110,
  "dropped_conflicting_row_count": 55,
  "dropped_conflicting_group_count": 18,
  "deduped_row_count": 7485,
  "seed": 42,
  "split_ratios": { "train": 0.8, "val": 0.1, "test": 0.1 },
  "splits": {
    "train": { "row_count": 5988, "class_balance": { "positive_rate": 0.4258517034068136 } },
    "val":   { "row_count": 748,  "class_balance": { "positive_rate": 0.4264705882352941 } },
    "test":  { "row_count": 749,  "class_balance": { "positive_rate": 0.4259012016021362 } }
  }
}
```

These numbers were **measured**, not assumed from the prior audit: the
prior audit (in the sibling course-derived repo) reported ~110
exact-duplicate rows and 18 conflicting-label groups from an ad hoc
re-run; this pipeline's own dedicated dedup logic reproduces the identical
counts (110, 18) independently, plus the additional numbers the audit did
not compute (55 rows dropped for those 18 groups, deduped total 7,485,
per-split sizes and class balance). Full manifest (with per-split SHA256)
is at `data/kaggle/manifest.json`.

## 4. CrisisLex T6 findings

**Text availability verdict: available directly, no API hydration needed.**
Verified by inspecting the actual distributed files
(`D:\Project\CrisisLexT6-v1.0.zip`, supplied locally by the user this
session): each of the 6 event CSVs
(`CrisisLexT6/<event>/<event>-ontopic_offtopic.csv`) has the header
`tweet id, tweet, label` and contains full tweet text in the `tweet`
column, not just IDs. Confirmed with a proper CSV parser (naive
line-splitting breaks because tweet text contains embedded commas and
quoted newlines) — column count was exactly 3 for every one of the
60,082 rows across all 6 files, and label values were cleanly
`{on-topic, off-topic}` with zero stray/malformed values once parsed
correctly.

**License verdict: [UNCLEAR — needs clarification], not proceeding as if
resolved.** The distributed `CrisisLexT6/README.md` (full text preserved
in this session's tool output and summarized in `docs/DATASETS.md`)
states no explicit license — it names the citation
(Olteanu, Castillo, Diaz, Vieweg, ICWSM 2014) and gives contact emails "for
inquiries," but no SPDX identifier or usage grant. This is flagged, not
assumed permissive, in both `docs/DATASETS.md` and `docs/DECISIONS.md`.
Since the pipeline was still built and run (per Hard Rule 4, this only
blocks if *text* is unavailable — text availability was confirmed, so
CrisisLex was fully pipelined; the license gap is a separate, explicitly
flagged open issue, not a Hard-Rule-4 blocker), this is not a "no third
outcome" violation: text availability was the actual gating condition, and
it resolved positively.

Manifest (`data/crisislex/manifest.json`, abridged):

```json
{
  "dataset": "crisislex_t6",
  "raw_row_count": 60082,
  "exact_duplicate_row_count": 4705,
  "dropped_conflicting_row_count": 533,
  "dropped_conflicting_group_count": 101,
  "deduped_row_count": 55276,
  "seed": 42,
  "split_ratios": { "train": 0.8, "val": 0.1, "test": 0.1 },
  "splits": {
    "train": { "row_count": 44220, "class_balance": { "positive_rate": 0.5032112166440524 } },
    "val":   { "row_count": 5528,  "class_balance": { "positive_rate": 0.5030752532561505 } },
    "test":  { "row_count": 5528,  "class_balance": { "positive_rate": 0.5032561505065123 } }
  }
}
```

Full manifest (with per-split, per-event class balance for all 6 events)
is at `data/crisislex/manifest.json`. `docs/DATASETS.md` also documents a
substantive, non-cosmetic finding: CrisisLex's dedup/conflict rate
(4,705 duplicate rows, 101 conflicting groups) is roughly 40x Kaggle's
(110, 18) in absolute terms and still notably higher as a fraction of
total rows (~7.8% vs. ~1.4% for exact duplicates alone) — reported as a
measured fact, with an explicitly hedged (not asserted) possible mechanism.

## 5. Ledger excerpt (floor-baseline runs, `results/ledger.jsonl`)

Both runs generated by `scripts/run_floor_baselines.py --seed 42` against
commit `02ec1b6` (the commit immediately before the ledger file itself was
added), on the Kaggle **val** split only:

| run_id | model | split | accuracy | macro_f1 | positive_f1 | git_commit | git_dirty |
|---|---|---|---:|---:|---:|---|---|
| `6ed405a3...` | majority_class | val | 0.5735 | 0.3645 | 0.0000 | `02ec1b68...` | false |
| `59765b20...` | stratified_random | val | 0.5401 | 0.5292 | 0.4574 | `02ec1b68...` | true |

Note on the `git_dirty` discrepancy between the two rows: the first run's
call to `is_git_dirty()` ran before `results/ledger.jsonl` existed on
disk, so the tree was genuinely clean; by the time the *second* run's call
happened (milliseconds later, same script invocation), the first run had
already created `results/ledger.jsonl` as a new untracked file, which
`git status --porcelain` correctly reports as a dirty-tree condition. This
is accurate, not a bug — it is a real consequence of ledgering to a file
that itself lives in the repo, observed the first time two ledgered runs
happened back-to-back without an intervening commit. No code change was
made in response; noted here as the honest explanation rather than
silently rounded away.

Majority-class's `positive_f1 = 0.0` is expected and correct: it never
predicts the positive class, so binary/positive-class F1 for class 1 is
undefined-in-practice and computed as 0 (via `zero_division=0`), while its
accuracy (0.5735) simply reflects the negative-class base rate on this
val split (429/748 = 0.5735).

## 6. Deviations from `docs/PLAN.md`

All logged in `docs/DECISIONS.md` (9 entries); summarized here:
1. `uv init`'s default non-package layout was replaced with a hand-configured
   src-layout package.
2. Dedup/split/hash logic was refactored out of `dtc.data.kaggle` into a
   shared `dtc.data.common` module partway through the session (before
   CrisisLex was written), so both datasets share one code path instead of
   two copies — not in the plan's literal task list, but directly serves
   the plan's own "same dedup + stratified split + frozen-test discipline"
   language.
3. Raw data files (Kaggle `train.csv`, CrisisLex's 6 event CSVs) were
   obtained from local sources already on the user's machine
   (course-derived repo's zip; user-supplied `CrisisLexT6-v1.0.zip`) rather
   than downloaded programmatically, consistent with the plan's Kaggle
   instruction and reasonable for CrisisLex given no download capability
   was assumed.
4. CSV serialization for hashing uses explicit `lineterminator="\n"` to
   keep SHA256 hashes stable across the Windows dev environment and the
   `ubuntu-latest` CI runner.
5. Both datasets' licenses are flagged `[UNCLEAR]` rather than assumed —
   see section 4.

## 7. Open issues / unresolved, stated plainly

- **Both datasets' licenses are unresolved.** Must be confirmed (Kaggle
  competition rules page; CrisisLex maintainer contact or crisislex.org
  terms) before any public release of derived splits or the paper.
- **`git_dirty` can flip within a single script invocation** when the
  ledger file itself is the thing making the tree dirty (section 5). Not
  fixed; documented as expected behavior, since "fixing" it (e.g., ignoring
  the ledger's own dirty state) would make the dirty flag less accurate,
  not more.
- **`results/runs/<run_id>/predictions.csv` files are gitignored, not
  committed** (only `results/ledger.jsonl` is) — per-run predictions exist
  locally for the two floor-baseline runs but are not in git history. This
  matches the plan's explicit "results/ (gitignored except ledger)"
  instruction; flagged here so it isn't mistaken for an oversight.
- **No CI run has actually executed yet** (no push to a remote/GitHub has
  happened in this session — the repo has no remote configured). The
  workflow file is present and was not exercised on GitHub's runners; local
  `ruff check .` + `pytest -v` were used as the verification proxy.
- Per Hard Rule 4's "no third outcome": CrisisLex resolved to the
  "fully pipelined" outcome (text available), not the "blocker report"
  outcome — there is no CrisisLex blocker section because none was needed.

## 8. Acceptance criteria checklist

- [x] Fresh repo, per-task commits (6 commits, one per task), `uv.lock`
  committed, CI config present (`.github/workflows/ci.yml`).
- [x] All tests pass locally; pytest summary pasted above (42 passed).
- [x] Kaggle splits on disk with manifest; leakage and stratification
  tests green; dedup counts reported as measured (110 / 18 / 55 / 7,485).
- [x] CrisisLex fully pipelined with the same guarantees (text was
  available; no blocker report needed).
- [x] Ledger exists with 2 real (floor-baseline) runs; frozen-test guard
  test is green; no code path outside `dtc.eval.*` / `scripts/evaluate_*.py`
  imports `frozen_test_loader` (verified by static scan test).
- [x] `docs/PLAN.md`, `docs/DECISIONS.md`, `docs/DATASETS.md`, and this
  `PHASE0_REPORT.md` all committed (this file is committed in the same
  action that completes Phase 0).
- [x] Nothing from the old repository or course material appears in this
  codebase — verified by construction (all code written fresh this
  session); only the two raw, public dataset files themselves (not code)
  were sourced from local copies, per `docs/DECISIONS.md`.