# dtc — Disaster Tweet Classification: Controlled Re-Evaluation

A from-scratch, controlled re-evaluation of text-classification architectures for
disaster tweet detection. See [`docs/PLAN.md`](docs/PLAN.md) for the full
multi-phase research plan, [`docs/DECISIONS.md`](docs/DECISIONS.md) for the
running log of deviations/decisions, and `PHASE0_REPORT.md` (once written) for
the Phase 0 build report.

This repository contains no code carried over from any prior/course-derived
project. All pipeline logic lives in `src/dtc/`; `notebooks/` is for
figure/analysis generation only, reading from saved results.

## Setup

```
uv sync
uv run pytest
```
