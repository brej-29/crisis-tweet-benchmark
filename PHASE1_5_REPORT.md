# PHASE1_5_REPORT.md

Brief report for Phase 1.5 (small maintenance, pre-execution), covering
the two hazards identified at the Phase 1 gate. Scope: additive/guard
changes only, no model/training/driver-enumeration/ledger-schema changes,
per Hard Rule 1.

## What changed

### Preamble (before the two tasks)
- Pushed the repo to `https://github.com/brej-29/crisis-tweet-benchmark`
  (branch `master`, tracking set up).
- **Local folder rename deferred to the user.** Confirmed empirically
  that the harness pins this session's Bash working directory to the
  original path and resets it before every command (a `cd` in one command
  did not persist to the next), so renaming the active project folder
  mid-session would break Bash/PowerShell for the rest of the session.
  Asked and the user chose to skip it this session; they'll rename
  `disaster-tweet-reeval` -> `crisis-tweet-benchmark` between sessions.
- **CUDA check**: an NVIDIA RTX 3050 (6GB VRAM) is present locally
  (`nvidia-smi` confirms it, driver supports CUDA 13.0), but the installed
  `torch` is a CPU-only build (`torch.cuda.is_available()` is `False`).
  Real training will need either a CUDA-enabled torch reinstall locally
  or Colab, per `docs/COLAB.md`.

### Task 1 — Config-uniqueness guard in `scripts/make_tables.py`
- **T1** (`build_t1_protocol_b_main`): now calls
  `check_single_config_per_model`, which raises `MixedConfigError` naming
  the model(s) and distinct `config_id`s if any model has more than one
  among its non-smoke E1 records. (This guard existed since the
  mid-Phase-1 follow-up request; Task 1 extended it.)
- **T4** (`build_t4_data_efficiency`): new
  `check_single_config_per_model_fraction`, the same guard grouped by
  `(model_name, train_fraction)` instead of `model_name` alone (E3
  legitimately trains the same model at different fractions; the signal
  to catch is two different configs at the *same* fraction).
- **T2/T3**: no separate guard call added — they inherit protection
  because `main()`'s table dict evaluates T1 first (Python evaluates
  dict-literal values left to right), so a `MixedConfigError` in T1 aborts
  the whole `main()` call before T2/T3/T4 run.
- **Recovery mechanism**: `--only-config-ids-from <dir>` (new
  `resolve_final_config_ids()` + `filter_to_final_config_ids()`). Resolves
  each model's CURRENT config_id from `<dir>/<model>.yaml` (typically
  `configs/final/`) and filters every table's input records down to that
  config_id, dropping superseded ones — without editing or deleting any
  ledger line. Chose this over a manual `--config-id-allowlist
  model=config_id` flag since `configs/final/` is already the project's
  source of truth for "what config should E1/E3 use." One wrinkle:
  `use_frozen`'s real config_id includes an injected `use_cache_dir` that
  isn't in the bare YAML file; `resolve_final_config_ids()` replicates
  just that one injection locally (not by importing `scripts/run_matrix.py`,
  per Hard Rule 1).
- 6 new tests (`tests/test_make_tables.py`): T4 mixed-config refusal,
  T4 same-model-different-fraction is fine, T4 smoke exemption,
  `resolve_final_config_ids` (incl. the use_frozen cache-dir injection),
  `filter_to_final_config_ids` (drops superseded / keeps current /
  passes through unknown models), and an end-to-end
  `--only-config-ids-from` recovery from a genuinely mixed synthetic
  ledger. All 19 tests in that file pass; 138 total in the suite.

### Task 2 — `docs/COLAB.md` audit and patch
Verified against the fact that `data/` is gitignored (a fresh Colab clone
has no raw CSVs, no split CSVs, no USE cache) and found it jumped straight
from "install deps" to "run the driver." Patched to add, in order:
1. Uploading the two raw files to exact expected paths — and discovered
   `dtc.data.crisislex.load_and_combine_events` expects the 6 CrisisLex
   event CSVs FLAT in `data/crisislex/raw/`, not nested one level deeper
   the way the distributed zip has them; the runbook now flattens on
   extraction.
2. Regenerating splits, then a MANIFEST IDENTITY CHECK
   (`git status --porcelain data/` must be empty) with an explicit STOP
   instruction if it isn't.
3. Running `scripts/precompute_use.py` (with `--extra-csv` for Protocol
   A's raw split) before any `use_frozen` run.
4. The clean-tree requirement, tied explicitly to `make_tables.py`'s
   dirty-source-path refusal.
5. Two-round sequencing (tuning -> local merge/select/commit/push ->
   E1/E2/E3), stated inline since `EXECUTION_RUNBOOK.md` doesn't exist in
   this repo — ties directly to Task 1's `MixedConfigError` guard.
6. Bringing back both `results/ledger.jsonl` and `results/runs/<run_id>/`
   after each round (was already present, kept and clarified).
- Updated the clone URL from a placeholder to the real remote.

## pytest summary

```
138 passed in 59.85s
```

`uv run ruff check .` → `All checks passed!`

## Deviations / open issues

- Local folder is still named `disaster-tweet-reeval` (user deferred the
  rename; see preamble above). `docs/COLAB.md`'s clone URL and directory
  name already reference `crisis-tweet-benchmark` (the remote's actual
  name), so this is only a local-filesystem cosmetic mismatch, not a
  functional one.
- No CUDA-enabled torch locally; real training runs need either a torch
  reinstall with CUDA wheels or Colab.
- Both datasets' licenses remain `[UNCLEAR]` (unchanged, carried over from
  Phase 0).
