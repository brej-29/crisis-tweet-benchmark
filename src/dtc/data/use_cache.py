"""Shared cache read/write for precomputed Universal Sentence Encoder
embeddings. `scripts/precompute_use.py` (TF, writes) and
`dtc.models.use_frozen` (torch, reads) both import this module; neither the
model code nor this module imports TensorFlow, keeping TF confined to the
one precompute script per Hard Rule 2.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def text_sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def embedding_path(cache_dir: str | Path, text: str) -> Path:
    return Path(cache_dir) / f"{text_sha256(text)}.npy"


def load_embedding(cache_dir: str | Path, text: str) -> np.ndarray:
    path = embedding_path(cache_dir, text)
    if not path.exists():
        raise KeyError(
            f"No cached USE embedding for text hash {path.stem} in {cache_dir}; "
            "run scripts/precompute_use.py for this dataset/split first."
        )
    return np.load(path)


def load_embeddings(cache_dir: str | Path, texts: Sequence[str]) -> np.ndarray:
    return np.stack([load_embedding(cache_dir, t) for t in texts])


def load_embedding_multi(cache_dirs: Sequence[str | Path], text: str) -> np.ndarray:
    """Looks the text up in each cache dir in order (first hit wins).
    Cross-dataset evaluation (Phase 2 E4/E5) predicts on texts cached under
    the EVAL dataset's dir, not the train dataset's -- callers pass the
    train cache first, then the extra eval caches.
    """
    for cache_dir in cache_dirs:
        path = embedding_path(cache_dir, text)
        if path.exists():
            return np.load(path)
    dirs = ", ".join(str(d) for d in cache_dirs)
    raise KeyError(
        f"No cached USE embedding for text hash {text_sha256(text)} in any of [{dirs}]; "
        "run scripts/precompute_use.py for the missing dataset/split first."
    )


def load_embeddings_multi(cache_dirs: Sequence[str | Path], texts: Sequence[str]) -> np.ndarray:
    return np.stack([load_embedding_multi(cache_dirs, t) for t in texts])


def save_embedding(cache_dir: str | Path, text: str, vector: np.ndarray) -> Path:
    path = embedding_path(cache_dir, text)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(vector, dtype=np.float32))
    return path


def write_manifest(cache_dir: str | Path, *, use_model_url: str, count: int, hash_of_hashes: str) -> Path:
    manifest = {"use_model_url": use_model_url, "count": count, "hash_of_hashes": hash_of_hashes}
    path = Path(cache_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_manifest(cache_dir: str | Path) -> dict:
    path = Path(cache_dir) / "manifest.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def hash_of_hashes(texts: Sequence[str]) -> str:
    """A single hash summarizing the exact set of texts embedded, so the cache
    manifest can be checked for staleness against a given split's contents."""
    hashes = sorted(text_sha256(t) for t in texts)
    return hashlib.sha256("".join(hashes).encode("utf-8")).hexdigest()
