# configs/final/ — SMOKE PLACEHOLDERS, not real tuning results

The nine YAML files in this directory were produced by:

```
uv run python scripts/run_matrix.py --smoke --only tuning
uv run python scripts/select_configs.py --include-smoke
```

`--smoke` trains each of the 52 tuning-grid configs on a ~200-example
random subset of the Kaggle train split for 1-2 effective epochs (before
early stopping typically kicks in). This demonstrates the tuning ->
selection pipeline works end to end (Phase 1 Task 4/8 acceptance
criteria), but the winning configs here reflect noise from a tiny
data subset, not real hyperparameter tuning on the full ~5,988-row train
split.

`select_configs.py` excludes smoke runs by default -- `--include-smoke`
had to be passed explicitly to select from smoke-only tuning data, which
is exactly the signal that these are placeholders, not real results.

**Before running E1 or E3 for real**, regenerate these files from real
tuning runs:

```
uv run python scripts/run_matrix.py --only tuning
uv run python scripts/select_configs.py
```

(no `--smoke`, no `--include-smoke`). This overwrites every file in this
directory with configs selected from real, non-smoke tuning-stage ledger
entries only.
