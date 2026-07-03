"""Run majority-class and stratified-random floor baselines on the Kaggle
VALIDATION split (never the frozen test split) and append both to the
results ledger.

Usage:
    uv run python scripts/run_floor_baselines.py [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dtc.harness.run import log_evaluation_run
from dtc.models.floor_baselines import fit_majority_class, fit_stratified_random, predict

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(seed: int = 42) -> list[dict]:
    train_df = pd.read_csv(REPO_ROOT / "data" / "kaggle" / "train.csv")
    val_df = pd.read_csv(REPO_ROOT / "data" / "kaggle" / "val.csv")

    ledger_path = REPO_ROOT / "results" / "ledger.jsonl"
    results_dir = REPO_ROOT / "results"
    manifest_path = REPO_ROOT / "data" / "kaggle" / "manifest.json"

    y_train = train_df["target"].to_numpy()
    y_val = val_df["target"].to_numpy()

    records = []

    majority_clf = fit_majority_class(y_train, seed)
    y_pred, y_prob = predict(majority_clf, len(val_df))
    record = log_evaluation_run(
        repo_root=REPO_ROOT,
        ledger_path=ledger_path,
        results_dir=results_dir,
        model_name="majority_class",
        dataset="kaggle_nlp_getting_started",
        split="val",
        seed=seed,
        config={"strategy": "most_frequent"},
        dataset_manifest_path=manifest_path,
        ids=val_df["id"],
        texts=val_df["text"],
        y_true=y_val,
        y_pred=y_pred,
        y_prob=y_prob,
    )
    records.append(record)

    stratified_clf = fit_stratified_random(y_train, seed)
    y_pred, y_prob = predict(stratified_clf, len(val_df))
    record = log_evaluation_run(
        repo_root=REPO_ROOT,
        ledger_path=ledger_path,
        results_dir=results_dir,
        model_name="stratified_random",
        dataset="kaggle_nlp_getting_started",
        split="val",
        seed=seed,
        config={"strategy": "stratified"},
        dataset_manifest_path=manifest_path,
        ids=val_df["id"],
        texts=val_df["text"],
        y_true=y_val,
        y_pred=y_pred,
        y_prob=y_prob,
    )
    records.append(record)

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = main(args.seed)
    for r in results:
        print(f"{r['model_name']}: run_id={r['run_id']} metrics={r['metrics']}")