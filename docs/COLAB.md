# COLAB.md — running the full experiment matrix on Colab GPU

This is the runbook for executing tuning/E1/E2/E3 (`configs/experiments.yaml`)
on a Colab GPU runtime, then bringing the results back into this repo.
Build + smoke happened locally on CPU (see `PHASE1_REPORT.md`); this
document is for the full-scale runs, which the build session deliberately
did not execute (Hard Rule: "This phase is BUILD + SMOKE only").

`data/` is gitignored (Phase 0 policy: raw/split CSVs and the USE
embedding cache are never committed, only small manifests/hashes are) --
**a fresh Colab clone has NO raw CSVs, NO split CSVs, and NO USE embedding
cache.** Steps 3-5 below rebuild all of that before any training runs.

## 1. Clone at a specific commit

Pin the exact commit you smoke-tested locally so the Colab run isn't
silently built on different code:

```python
!git clone https://github.com/brej-29/crisis-tweet-benchmark.git
%cd crisis-tweet-benchmark
!git checkout <commit-sha>   # the commit PHASE1_REPORT.md (or your latest local commit) was written against
```

Confirm the commit matches your local `git rev-parse HEAD` before
trusting any resulting numbers.

## 2. Install dependencies

Colab's environment isn't uv-managed, so use a pip install pinned from the
committed lockfile rather than trusting Colab's preinstalled versions:

```bash
!pip install uv
!uv export --frozen --no-dev -o requirements.txt   # from the local repo, commit this file if you want it versioned
!pip install -r requirements.txt
!pip install -e .
```

You will need the `tf` extra too — Colab needs `tensorflow`/
`tensorflow-hub` for step 5's USE embedding precompute, even though no
model *trains* in TF (Hard Rule 2):

```bash
!pip install tensorflow tensorflow-hub
```

Verify GPU visibility before running anything:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

If this prints `False`, stop here — you're not actually using the GPU
runtime (Runtime -> Change runtime type -> GPU in the Colab menu), and
training will silently fall back to Colab's CPU.

## 3. Upload the raw data files

Neither raw dataset is committed to git. Upload both, to these **exact**
paths (create the directories if they don't exist):

| File | Expected path |
|---|---|
| Kaggle `train.csv` | `data/kaggle/raw/train.csv` |
| CrisisLexT6 zip | extracted so the 6 event CSVs live FLAT at `data/crisislex/raw/<event>-ontopic_offtopic.csv` (no nested `CrisisLexT6/<event>/` subfolder — `dtc.data.crisislex.load_and_combine_events` reads `raw_dir / f"{event}-ontopic_offtopic.csv"` directly; the distributed zip nests them one level deeper, so flatten on extraction) |

```python
from google.colab import files
uploaded = files.upload()  # pick train.csv, then move/rename it into place
!mkdir -p data/kaggle/raw data/crisislex/raw
!mv train.csv data/kaggle/raw/train.csv
!mv CrisisLexT6-v1.0.zip /tmp/CrisisLexT6-v1.0.zip
!unzip -o /tmp/CrisisLexT6-v1.0.zip -d /tmp/crisislex_extract
!find /tmp/crisislex_extract -name "*-ontopic_offtopic.csv" -exec cp {} data/crisislex/raw/ \;
!ls data/crisislex/raw/  # sanity check: 6 files, no subdirectories
```

(Phase 1's E1/E2/E3 only need the Kaggle raw file. Skip the CrisisLex
upload if you are not also doing Phase 2 work on this Colab runtime.)

## 4. Regenerate the splits, then the MANIFEST IDENTITY CHECK

```bash
!uv run python scripts/prepare_kaggle.py
!uv run python scripts/prepare_crisislex.py   # only if you uploaded CrisisLex in step 3
```

**Then, before doing anything else:**

```bash
!git status --porcelain data/
```

This must print **nothing** for the manifest/dropped-conflicting files
(`data/kaggle/manifest.json`, `data/kaggle/dropped_conflicting.csv`, and
the CrisisLex equivalents if regenerated) — those are the only `data/`
files tracked by git, and regenerating the splits on Colab must reproduce
them byte-for-byte (same seed, same dedup logic, same raw file).

**If `git status --porcelain data/` shows ANY of those files as modified:
STOP and report it. Do not proceed to training.** A manifest diff means
the splits differ between your local machine and Colab (e.g. a different
raw `train.csv`, a pandas/sklearn version skew affecting the stratified
split, or line-ending differences) — every downstream run would be
trained/evaluated on a different dataset than your local smoke runs, and
the ledger's `dataset_split_hashes` would silently misrepresent the truth.

## 5. Precompute USE embeddings

Required before any `use_frozen` run (tuning, E1, or E3) — `use_frozen`
never computes embeddings itself, only reads this cache (Hard Rule 2):

```bash
!uv run python scripts/precompute_use.py --dataset kaggle --extra-csv data/kaggle/raw/train.csv
```

The `--extra-csv` covers Protocol A's raw (non-deduped) split too, so E2's
`use_frozen` run doesn't hit a cache-miss `KeyError` for rows Protocol B's
dedup dropped (see `docs/DECISIONS.md`, 2026-07-04). This step needs
network access to `tfhub.dev` on first run (downloads the ~1GB USE model);
re-runs are fast (cache hits).

## 6. The clean-tree requirement

**Before running anything in step 7, commit or discard any local edits:**

```bash
!git status --porcelain
```

Must print nothing (aside from `data/` and `results/` paths, which are
gitignored anyway and don't show up here). Every real run's ledger record
captures `git_dirty_paths` at the moment it ran; `scripts/make_tables.py`
refuses to build any table referencing a run whose `git_dirty_paths`
includes a source file. Training against an uncommitted source edit
doesn't just risk a `make_tables.py` refusal later — it means the run
isn't reproducible from a commit at all. If you need to change something,
commit it (or push it) first, then re-clone/pull on Colab before training.

## 7. Two-round sequencing: tuning first, then E1/E2/E3

`configs/final/<model>.yaml` (used by E1 and E3) must hold REAL
(non-smoke) tuning results before E1/E3 run for real -- otherwise E1/E3
ledger entries would use `configs/final/`'s *current* smoke-placeholder
config, and a later re-run of `select_configs.py` on real tuning data
would silently invalidate them (`scripts/make_tables.py`'s
`MixedConfigError` guard exists precisely to catch this — see
`docs/DECISIONS.md`, Phase 1.5). Sequence Colab work in two rounds, not
one, with a local round-trip in between:

**Round 1 — tuning (Colab):**
```bash
!uv run python scripts/run_matrix.py --only tuning
```
Download `results/ledger.jsonl` and `results/runs/` (step 8). Locally:
```bash
uv run python scripts/merge_ledger.py --remote /path/to/downloaded/ledger.jsonl
uv run python scripts/select_configs.py          # NOT --include-smoke -- real tuning data now exists
git add results/ledger.jsonl configs/final/
git commit -m "Real tuning results; regenerate configs/final/ from real data"
git push
```

**Round 2 — E1/E2/E3 (Colab):**
```bash
!git pull   # picks up the real configs/final/*.yaml and merged ledger from round 1
!uv run python scripts/run_matrix.py --only e1 e2 e3
```

`--only e1` / `--only e3` / `--models lstm gru` further splits either
round across multiple Colab sessions if you hit usage limits — the
driver's skip logic (keyed on stage/protocol/model_name/config_id/seed/
train_fraction) makes this safe to interleave in any order **within** a
round. Do not skip straight to round 2 without completing the local
round-trip: E1/E3 run against whatever `configs/final/` currently
contains on that Colab checkout, smoke placeholders included.

If Colab disconnects mid-run (either round): just re-run the same
command. The run that was killed left no partial ledger line (a ledger
line is only appended after a run's metrics are fully computed), so it
will be re-attempted, not silently skipped as "done."

## 8. Bring results back to the local repo

After **each** round, download both:
- `results/ledger.jsonl` (the new lines only, or the whole file — merge
  handles both)
- every new `results/runs/<run_id>/` directory (needed for McNemar's/error
  analysis in Phase 3 — the ledger alone is not enough, it doesn't carry
  per-example predictions)

```python
from google.colab import files
files.download('results/ledger.jsonl')
!zip -r runs.zip results/runs/
files.download('runs.zip')
```

Locally, merge the ledger (never just overwrite `results/ledger.jsonl` —
that would violate the append-only discipline and could silently drop
runs that happened locally in the meantime):

```bash
uv run python scripts/merge_ledger.py --remote /path/to/downloaded/ledger.jsonl
```

`merge_ledger.py` rejects duplicate `run_id`s (already-present lines are
left untouched) and any record missing a required ledger key, and reports
counts for accepted/duplicate/invalid so you can see exactly what
happened. Then extract `runs.zip` into `results/runs/` locally (predictions
are gitignored, so just unzip in place — no merge logic needed there).

## 9. Regenerate tables

Once the ledger is merged:

```bash
uv run python scripts/make_tables.py
```

This reads only from `results/ledger.jsonl` and refuses to emit tables if
any referenced run's `git_dirty_paths` includes source files, so a report
generated from a dirty-tree run will visibly fail rather than silently
produce numbers that don't trace to a clean commit. It also refuses (per
model, for T1; per model+fraction, for T4) if non-smoke E1/E3 records
disagree on `config_id` for the same model — if that fires because an
earlier smoke placeholder run is still sitting in the ledger alongside a
real one, use `--only-config-ids-from configs/final` to filter to each
model's current config instead of editing or deleting any ledger line.
