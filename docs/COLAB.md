# COLAB.md — running the full experiment matrix on Colab GPU

This is the runbook for executing E1/E2/E3 (`configs/experiments.yaml`) on
a Colab GPU runtime, then bringing the results back into this repo. Build +
smoke happened locally on CPU (see `PHASE1_REPORT.md`); this document is
for the full-scale runs, which this session deliberately did not execute
(Hard Rule: "This phase is BUILD + SMOKE only").

## 1. Clone at a specific commit

Pin the exact commit you smoke-tested locally so the Colab run isn't
silently built on different code:

```python
!git clone https://github.com/<you>/disaster-tweet-reeval.git
%cd disaster-tweet-reeval
!git checkout <commit-sha>   # the commit PHASE1_REPORT.md was written against
```

If the repo isn't pushed to a remote yet, upload it as a zip and `!unzip`
instead — the important thing is that the commit hash in the Colab copy
matches a real commit in your local history (check with `git rev-parse
HEAD` on both sides before trusting any resulting numbers).

## 2. Install dependencies

Colab's environment isn't uv-managed, so use a pip install pinned from the
committed lockfile rather than trusting Colab's preinstalled versions:

```bash
!pip install uv
!uv export --frozen --no-dev -o requirements.txt   # from the local repo, commit this file if you want it versioned
!pip install -r requirements.txt
!pip install -e .
```

If you also need the USE embedding cache and it wasn't computed locally
(or needs re-running for CrisisLex/Protocol A raw text), install the `tf`
extra too:

```bash
!pip install tensorflow tensorflow-hub
```

Verify GPU visibility before running anything:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

## 3. Run the driver

```bash
# Full dry run first -- sanity-check the pending count before spending GPU time
!uv run python scripts/run_matrix.py --dry-run

# Then the real matrix (resumable -- safe to re-run after a Colab
# disconnect; already-ledgered runs are skipped automatically)
!uv run python scripts/run_matrix.py --only e1 e2 e3
```

Use `--only e1` / `--only e3` / `--models lstm gru` to split the matrix
across multiple Colab sessions if you hit usage limits — the driver's skip
logic (keyed on stage/protocol/model_name/config_id/seed/train_fraction)
makes this safe to interleave in any order.

If Colab disconnects mid-run: just re-run the same command. The run that
was killed left no partial ledger line (a ledger line is only appended
after a run's metrics are fully computed), so it will be re-attempted, not
silently skipped as "done."

## 4. Bring results back to the local repo

Download two things from the Colab runtime:
- `results/ledger.jsonl` (the new lines only, or the whole file — merge
  handles both)
- `results/runs/<run_id>/predictions.csv` for each new run (needed for
  McNemar's/error analysis in Phase 3)

```python
from google.colab import files
files.download('results/ledger.jsonl')
# zip results/runs/ if there are many run directories
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

## 5. Regenerate tables

Once the ledger is merged:

```bash
uv run python scripts/make_tables.py
```

This reads only from `results/ledger.jsonl` and refuses to emit tables if
any referenced run's `git_dirty_paths` includes source files, so a report
generated from a dirty-tree run will visibly fail rather than silently
produce numbers that don't trace to a clean commit.
