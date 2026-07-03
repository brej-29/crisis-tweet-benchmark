"""Run-record construction and per-example prediction storage.

Every evaluated run gets: a unique run_id, a config snapshot, the current
git commit hash, the seed used, the dataset manifest's split hashes, a
timestamp, and its computed metrics -- appended to results/ledger.jsonl via
dtc.harness.ledger.append_run_record. Per-example predictions are written
separately to results/runs/<run_id>/predictions.csv, since the ledger is
meant to stay small and diffable.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from dtc.eval.metrics import compute_all_metrics
from dtc.harness.config import compute_config_id
from dtc.harness.ledger import (
    append_run_record,
    generate_run_id,
    get_git_commit_hash,
    get_git_dirty_paths,
    is_git_dirty,
)


def _load_dataset_manifest(manifest_path: str | Path) -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def _split_hashes(manifest: dict) -> dict:
    return {name: info["sha256"] for name, info in manifest.get("splits", {}).items()}


def build_run_record(
    *,
    run_id: str,
    repo_root: str | Path,
    model_name: str,
    dataset: str,
    split: str,
    seed: int,
    config: dict,
    metrics: dict,
    dataset_manifest_path: str | Path,
    protocol: str | None = None,
    phase: str = "phase0",
    smoke: bool = False,
    train_fraction: float = 1.0,
    config_id: str | None = None,
) -> dict:
    manifest = _load_dataset_manifest(dataset_manifest_path)
    try:
        manifest_path_str = Path(dataset_manifest_path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        manifest_path_str = str(dataset_manifest_path)
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(repo_root),
        "git_dirty": is_git_dirty(repo_root),
        "git_dirty_paths": get_git_dirty_paths(repo_root),
        "model_name": model_name,
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "config": config,
        "config_id": config_id or compute_config_id(config),
        "protocol": protocol,
        "phase": phase,
        "smoke": smoke,
        "train_fraction": train_fraction,
        "dataset_manifest_path": manifest_path_str,
        "dataset_split_hashes": _split_hashes(manifest),
        "metrics": metrics,
    }


def save_predictions(
    *,
    run_id: str,
    results_dir: str | Path,
    ids,
    texts,
    y_true,
    y_pred,
    y_prob=None,
) -> Path:
    run_dir = Path(results_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    text_hashes = [hashlib.sha256(str(t).encode("utf-8")).hexdigest() for t in texts]
    df = pd.DataFrame(
        {
            "id": list(ids),
            "text_sha256": text_hashes,
            "y_true": list(y_true),
            "y_pred": list(y_pred),
            "y_prob": list(y_prob) if y_prob is not None else [None] * len(list(ids)),
        }
    )
    out_path = run_dir / "predictions.csv"
    df.to_csv(out_path, index=False, lineterminator="\n")
    return out_path


def log_evaluation_run(
    *,
    repo_root: str | Path,
    ledger_path: str | Path,
    results_dir: str | Path,
    model_name: str,
    dataset: str,
    split: str,
    seed: int,
    config: dict,
    dataset_manifest_path: str | Path,
    ids,
    texts,
    y_true,
    y_pred,
    y_prob=None,
    protocol: str | None = None,
    phase: str = "phase0",
    smoke: bool = False,
    train_fraction: float = 1.0,
    config_id: str | None = None,
) -> dict:
    """Compute metrics, save per-example predictions, and append one ledger line.

    Returns the run record that was appended.
    """
    run_id = generate_run_id()
    metrics = compute_all_metrics(y_true, y_pred)
    save_predictions(
        run_id=run_id,
        results_dir=results_dir,
        ids=ids,
        texts=texts,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )
    record = build_run_record(
        run_id=run_id,
        repo_root=repo_root,
        model_name=model_name,
        dataset=dataset,
        split=split,
        seed=seed,
        config=config,
        metrics=metrics,
        dataset_manifest_path=dataset_manifest_path,
        protocol=protocol,
        phase=phase,
        smoke=smoke,
        train_fraction=train_fraction,
        config_id=config_id,
    )
    append_run_record(ledger_path, record)
    return record