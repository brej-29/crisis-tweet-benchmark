"""Precompute Universal Sentence Encoder (USE) embeddings for every split of
a dataset, cached to data/<dataset>/use_embeddings/<sha256>.npy.

TF-CPU + tensorflow_hub ONLY here (Hard Rule 2 -- no model trains in TF);
this script's only job is to build a reusable feature cache that
dtc.models.use_frozen (torch) reads via dtc.data.use_cache, so its own
import never touches TF.

This script reads split CSVs directly (its own path), NOT via
dtc.eval.frozen_test_loader. The frozen-test GUARD governs who may read
test *labels* for evaluation/model-selection purposes; this script reads
only the `text` column of every split (train/val/test, for whichever
dataset is given) to build a feature cache, and never reads or returns the
label column -- see docs/DECISIONS.md for this documented nuance.

Usage:
    uv run --extra tf python scripts/precompute_use.py --dataset kaggle
    uv run --extra tf python scripts/precompute_use.py --dataset crisislex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dtc.data.use_cache import embedding_path, hash_of_hashes, save_embedding, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
USE_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
SPLITS = ("train", "val", "test")


def load_split_texts(dataset: str, split: str) -> pd.Series:
    path = REPO_ROOT / "data" / dataset / f"{split}.csv"
    return pd.read_csv(path)["text"]


def embed_and_cache(texts, cache_dir: Path, encoder) -> int:
    """Embeds only texts not already cached (cache hit on re-run). Returns
    the number of NEW embeddings computed this call."""
    to_compute = [t for t in texts if not embedding_path(cache_dir, t).exists()]
    if not to_compute:
        return 0
    vectors = encoder(to_compute).numpy()
    for text, vector in zip(to_compute, vectors):
        save_embedding(cache_dir, text, vector)
    return len(to_compute)


def main(dataset: str, splits: tuple[str, ...] = SPLITS) -> dict:
    import tensorflow_hub as hub  # deferred: keeps TF out of any non-TF import path

    cache_dir = REPO_ROOT / "data" / dataset / "use_embeddings"
    encoder = hub.load(USE_MODEL_URL)

    all_texts: list[str] = []
    new_counts = {}
    for split in splits:
        texts = load_split_texts(dataset, split)
        all_texts.extend(texts.tolist())
        new_counts[split] = embed_and_cache(texts, cache_dir, encoder)

    manifest_path = write_manifest(
        cache_dir,
        use_model_url=USE_MODEL_URL,
        count=len(set(all_texts)),
        hash_of_hashes=hash_of_hashes(all_texts),
    )
    return {
        "cache_dir": str(cache_dir),
        "manifest_path": str(manifest_path),
        "new_embeddings_by_split": new_counts,
        "total_unique_texts": len(set(all_texts)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["kaggle", "crisislex"], required=True)
    args = parser.parse_args()
    result = main(args.dataset)
    print(result)
