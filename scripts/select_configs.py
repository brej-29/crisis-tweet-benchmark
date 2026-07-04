"""Freezes each model's winning tuning config into configs/final/<model>.yaml.

Reads ONLY ledger entries with stage == "tuning" (docs/PLAN.md 1.4/Task 4)
-- no human-in-the-loop number editing. The winner per model is whichever
tuning-stage ledger entry has the best val macro-F1; ties broken by
earliest run (stable sort + first `max` match).

Usage:
    uv run python scripts/select_configs.py
    uv run python scripts/select_configs.py --models lstm gru
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from dtc.harness.ledger import read_ledger

REPO_ROOT = Path(__file__).resolve().parents[1]


def select_best_configs(
    ledger_records: list[dict], models: list[str] | None = None, include_smoke: bool = False
) -> dict[str, dict]:
    """Per model_name, the config of the tuning-stage record with the
    highest val macro-F1. Only records with stage == "tuning" are
    considered; anything else (final runs, floor baselines) is ignored
    regardless of model_name. Smoke runs are excluded by default -- once
    real tuning runs exist in the same ledger, a lucky smoke-subset result
    must not be able to outrank/contaminate a real selection; pass
    `include_smoke=True` only to demonstrate the pipeline before real
    tuning runs exist (see configs/final/README.md).
    """
    tuning_records = [
        r for r in ledger_records if r.get("stage") == "tuning" and (include_smoke or not r.get("smoke", False))
    ]
    by_model: dict[str, list[dict]] = {}
    for record in tuning_records:
        by_model.setdefault(record["model_name"], []).append(record)

    winners = {}
    for model_name, records in by_model.items():
        if models is not None and model_name not in models:
            continue
        best = max(records, key=lambda r: r["metrics"]["macro_f1"])
        winners[model_name] = best["config"]
    return winners


def write_final_configs(winners: dict[str, dict], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for model_name, config in winners.items():
        out_path = output_dir / f"{model_name}.yaml"
        out_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        written[model_name] = out_path
    return written


def main(
    ledger_path: str | Path,
    output_dir: str | Path,
    models: list[str] | None = None,
    include_smoke: bool = False,
) -> dict[str, Path]:
    records = read_ledger(ledger_path)
    winners = select_best_configs(records, models, include_smoke=include_smoke)
    return write_final_configs(winners, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=REPO_ROOT / "results" / "ledger.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "configs" / "final")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--include-smoke", action="store_true", help="Include smoke tuning runs (demo only)")
    args = parser.parse_args()
    result = main(args.ledger, args.output_dir, args.models, include_smoke=args.include_smoke)
    if not result:
        print("No tuning-stage ledger entries found -- nothing selected.")
    for model_name, path in sorted(result.items()):
        print(f"{model_name}: {path}")
