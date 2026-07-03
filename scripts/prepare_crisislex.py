"""Thin CLI entrypoint: prepare the CrisisLex T6 dataset (label map + dedup +
stratified split).

Usage:
    uv run python scripts/prepare_crisislex.py [--config configs/crisislex.yaml]

All actual pipeline logic lives in dtc.data.crisislex; this script only
reads config, calls that logic, and writes files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from dtc.data.crisislex import (
    EVENTS,
    SplitRatios,
    build_manifest,
    count_exact_duplicate_rows,
    dataframe_csv_bytes,
    load_and_combine_events,
    map_labels,
    resolve_duplicates,
    sha256_bytes,
    sha256_file,
    stratified_split,
)


def main(config_path: str) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    with open(repo_root / config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = repo_root / cfg["raw_dir"] if not Path(cfg["raw_dir"]).is_absolute() else Path(cfg["raw_dir"])
    output_dir = repo_root / cfg["output_dir"] if not Path(cfg["output_dir"]).is_absolute() else Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_file_hashes = {
        event: sha256_file(raw_dir / f"{event}-ontopic_offtopic.csv") for event in EVENTS
    }

    combined = load_and_combine_events(raw_dir, EVENTS)
    raw_row_count = len(combined)

    combined = map_labels(combined, label_raw_col="label_raw")

    text_col, label_col, id_col = cfg["text_col"], cfg["label_col"], cfg["id_col"]

    exact_dup_count = count_exact_duplicate_rows(combined, text_col)
    deduped, dropped_conflicting = resolve_duplicates(combined, text_col, label_col, id_col)
    dropped_conflicting_group_count = (
        dropped_conflicting[text_col].nunique() if len(dropped_conflicting) else 0
    )

    dropped_path = output_dir / "dropped_conflicting.csv"
    dropped_conflicting.to_csv(dropped_path, index=False, lineterminator="\n")

    ratios = SplitRatios(**cfg["split"])
    seed = cfg["seed"]
    train_df, val_df, test_df = stratified_split(deduped, label_col, ratios, seed)
    splits = {"train": train_df, "val": val_df, "test": test_df}

    split_hashes = {}
    for name, split_df in splits.items():
        split_path = output_dir / f"{name}.csv"
        split_df.to_csv(split_path, index=False, lineterminator="\n")
        split_hashes[name] = sha256_bytes(dataframe_csv_bytes(split_df))

    manifest = build_manifest(
        raw_dir=cfg["raw_dir"],
        raw_file_hashes=raw_file_hashes,
        raw_row_count=raw_row_count,
        dropped_conflicting_count=len(dropped_conflicting),
        dropped_conflicting_group_count=int(dropped_conflicting_group_count),
        exact_duplicate_row_count=exact_dup_count,
        deduped_row_count=len(deduped),
        seed=seed,
        ratios=ratios,
        splits=splits,
        split_hashes=split_hashes,
        label_col=label_col,
    )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crisislex.yaml")
    args = parser.parse_args()
    result = main(args.config)
    print(json.dumps(result, indent=2))