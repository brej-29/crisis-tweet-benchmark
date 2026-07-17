# PHASE2_REPORT.md — Stage A (build + smoke)

Phase 2 build for the CrisisLex in-domain + cross-dataset experiments (E4/E5), per the
staged Phase 2+3 plan. Real E4/E5 execution is Stage B and is appended as an execution
section below once run.

## 1. Built artifacts (commits d7007ed..1a1c93b)

| Task | Commits | Summary |
|---|---|---|
| (pre) | `d7007ed` | Phase 1 run ledger committed (196 real runs: tuning 52 / E1 45 / E2 9 / E3 90) |
| A1 | `dfc860c`, `e5899b6`, `4d39bf1`, `10e7ea5` | Run records gain `train_dataset`/`eval_dataset`/`training_id`; read-time-only backfill in `read_ledger` (old records → kaggle/kaggle; file bytes never touched); `merge_ledger` validation extended; skip-key now (stage, protocol, model, config_id, seed, fraction, smoke, train_dataset, eval_dataset); driver `eval_datasets` support — one training emits one ledger record per eval dataset sharing a `training_id`, resuming fills only missing eval records; `use_frozen` multi-cache lookup via `extra_cache_dirs` attribute (outside config, so config_ids stay comparable to E1); guard refusing `eval_datasets` on protocol-A/tuning experiments |
| A2 | `59e838b` | Frozen-test guard coverage proven dataset-generic (crisislex + synthetic third dataset in static scan); `event` column carried end-to-end into crisislex-eval `predictions.csv` via `extra_columns` (ledger schema unchanged; kaggle predictions unchanged) |
| A3 | (no code) | CrisisLex USE cache precomputed: 55,276 embeddings (train 44,220 / val 5,528 / test 5,528), cleaned-text keyed; cache-hit re-run = 0 new; `torch.cuda.is_available()` True re-verified after sync. First attempt failed on a corrupted tfhub temp cache (partial download); cleared and re-downloaded |
| A4 | `1d3ff8e`, `e739aa2` | `e4`/`e5` in configs/experiments.yaml (9 models × seeds 0–4, config_source final, dual eval; `phase: phase2` via new optional per-experiment `phase` key, default `phase1`); CrisisLex measured max_lengths in PROTOCOL.md §2; smoke ledger records committed |
| A5 | `1a1c93b` | make_tables T5 (cross-dataset matrix + OOD deltas), T6 (per-event from prediction files), T7 (E5-vs-E1 reproducibility, report-only); config-uniqueness guards per (model, train_dataset) for T5/T6 and per (model, stage) for T7; `--only-config-ids-from` now resolves use_frozen's config_id per train_dataset (E4 use_frozen records would otherwise be silently dropped) |

Each task was implemented and reviewed by separate fresh-context agents; findings fixed
and re-reviewed (A1: 1 Important fix — refuse multi-eval on protocol-A/tuning; A2/A4/A5: no findings).

## 2. Test / lint status

- `uv run python -m pytest -q` → **190 passed** (112.8 s; the former `tensorflow_hub` skip now runs since the tf extra is installed locally).
- `uv run ruff check .` → clean (verified at every task commit).
- CI status at push: see §6.

## 3. CrisisLex measured values (train-only, recorded in docs/PROTOCOL.md §2)

- Whitespace tokens (n=44,220 train rows, post-cleaning): **95th pct = 24** (mean 14.79, median 15, max 43).
- DistilBERT WordPiece: **95th pct = 45** (mean 27.34, median 28, max 96).
- Kaggle WordPiece, measured-now for completeness (n=5,988): **95th pct = 51** (mean 31.13, median 31, max 80).
- These are computed at fit time by the models (train-only policy); they are NOT pinned in
  `configs/final/*.yaml`, which were verified to contain no dataset-derived values
  (the `use_cache_dir` in `use_frozen.yaml` is inert — the driver injects the per-dataset
  path before hashing).

## 4. Dry-run counts (real ledger, non-smoke)

```
tuning: 0 pending / 52   e1: 0 / 45   e2: 0 / 9   e3: 0 / 90
e4: 45 pending / 45      e5: 45 pending / 45
Total pending runs: 90 / 286        (= 90 trainings → 180 eval records)
```

## 5. Smoke verification (real ledger, smoke=true)

18 smoke trainings (9 models × e4/e5, seed 0) → 36 records: every training has exactly 2
records sharing `training_id` with distinct `run_id`s and correct train/eval dataset pairs;
`predictions.csv` present for all 36; `event` column present iff eval_dataset=crisislex;
`use_frozen` succeeded in both directions against the real caches (cross-dataset cache
fallback). A killed mid-run distilbert smoke left no partial ledger line and the relaunch
skipped the 16 completed records — live confirmation of driver resumability.

## 6. Wall-time estimates for real E4/E5 (extrapolated from Phase 1 ledger timestamps)

E1 per-run wall times (gaps between consecutive E1 ledger records, same driver invocation;
Kaggle scale, ~6.0k train rows, RTX 3050 6GB):

| Family | E1 median/run | E5 est. (Kaggle-scale ×5 seeds) | E4 est. (CrisisLex ≈ 7.4× train, ×5 seeds) |
|---|---|---|---|
| tfidf_mnb / tfidf_logreg | 0.5–1.1 s | < 1 min | ~1–2 min |
| meanpool/lstm/gru/bilstm/conv1d | 4.4–7.4 s | ~3 min total | ~15–35 min total (5 models) |
| use_frozen | 8.3 s | < 1 min | ~5 min |
| distilbert_finetune | **152 s** | **~13 min** | **~75–110 min** (≈ 15–22 min/run) |

Totals: **E5 ≈ 20 min**, **E4 ≈ 1.5–2.5 h** (distilbert is the tail; early stopping on the
larger CrisisLex val split adds uncertainty). Both fit a single local session; the driver
resumes from the ledger if interrupted.

## 7. Deviations / decisions this stage

All logged in docs/DECISIONS.md (2026-07-16 entries): read-time backfill; partial-crash
retrain-fills-gap semantics; `extra_cache_dirs` outside config; multi-eval refusal on
protocol-A/tuning; `phase` key; PROTOCOL.md numbers not pinned in configs; kaggle WordPiece
measured-now provenance; T5/T6/T7 guard groupings; per-train_dataset use_frozen config
resolution.

## 8. Stage B — real execution (E4/E5, commit `856aace`)

Preflight: clean tree at every launch boundary, `torch.cuda.is_available()` True (RTX 3050
6GB Laptop GPU), confirmed on the commit that ran.

- **E5 first** (`--only e5`, Kaggle-scale, 45 trainings / 90 eval records): completed in
  one invocation, 45 run / 0 skipped.
- **T7 inspected before E4** (per the gate): every model/seed's E5-kaggle-eval macro-F1
  equals its E1 macro-F1 exactly (delta = 0.0000 across all 45 rows, all 9 models
  including the CUDA-nondeterministic neural families). No anomaly — proceeded to E4.
- **E4** (`--only e4`, CrisisLex-scale, ~7.4x the Kaggle train size, 45 trainings / 90
  eval records): completed in one background invocation, 45 run / 0 skipped. The
  distilbert_finetune tail dominated wall time as estimated in §6.
- `make_tables.py` regenerated all seven tables from the now-complete ledger (506 lines).
  T1–T4 are byte-identical to their pre-Stage-B content (verified via `git diff
  results/tables/` before committing — no diff, confirming E4/E5 additions did not
  perturb Phase-1-scoped tables). `results/tables/` is gitignored by design (`results/*`
  with an explicit `!results/ledger.jsonl` exception) — only the ledger is committed;
  tables regenerate deterministically from it.
- Final dry-run against the completed real ledger: `tuning 0/52, e1 0/45, e2 0/9, e3
  0/90, e4 0/45, e5 0/45` — **Total pending: 0/286**.

### Run counts

90 real trainings (45 E4 + 45 E5) x dual eval = 180 real eval records, plus the 36 smoke
records from Stage A (unchanged). Ledger: 196 (Phase 1) + 36 (Stage A smoke) + 180
(Stage B real) = 412 non-legacy new lines since Phase 1's close, 506 total lines.

### Anomalies

None. T7 deltas are exactly 0.0000 for all 45 (model, seed) pairs — stronger
reproducibility than the "expect ~0, small nonzero for CUDA-nondeterministic models"
default expectation, plausibly because this GPU/driver/torch combination's kernels for
these specific ops and batch sizes happened to be deterministic in practice, or the
nondeterministic kernels aren't on the hot path these models exercise; stated as an
observation, not asserted as guaranteed for future re-runs.

### T5/T6/T7 highlights (stated neutrally, no narrative spin)

- **T5** (cross-dataset matrix): every model's `Train=X, Eval=X` (in-domain) score
  exceeds its `Train=Y, Eval=X` (out-of-domain) score; OOD deltas are negative for every
  model in both directions (range: eval=kaggle deltas −0.046 to −0.217; eval=crisislex
  deltas −0.115 to −0.311). CrisisLex-trained/CrisisLex-evaluated in-domain macro-F1
  (0.88–0.95) is uniformly higher than Kaggle-trained/Kaggle-evaluated in-domain
  macro-F1 (0.69–0.81) across every model.
- **T6** (per-event breakdown): per-event macro-F1 varies by CrisisLex event and by
  which dataset trained the model; for kaggle-trained models the lowest per-event score
  is consistently `2012_Sandy_Hurricane` (range 0.46–0.77 across models) and the highest
  is consistently `2013_Queensland_Floods` or `2013_West_Texas_Explosion` (range
  0.73–0.96); crisislex-trained models show a narrower per-event range (0.74–0.97).
- **T7** (reproducibility): 45/45 (model, seed) deltas exactly 0.0000; see anomalies
  note above.

## 9. CHECKPOINT 1

Stage A + Stage B complete, committed (`856aace`), pushed. Per the plan: **stop here**.
Paste this report (§1–9) plus `results/tables/T5_cross_dataset.md`,
`T6_per_event.md`, and `T7_reproducibility.md` to the architect chat for the Phase 2
gate before Stage C proceeds.
