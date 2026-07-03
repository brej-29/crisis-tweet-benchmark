"""Tests for scripts/precompute_use.py: a real (not mocked) 20-text smoke
embed against the actual USE model, verifying 512-dim shape and a cache
hit (no recompute) on re-run. Requires the optional `tf` dependency group
(tensorflow + tensorflow-hub) and network access to tfhub.dev on first
run; skipped otherwise so the TF-free core test suite still passes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("tensorflow_hub")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_precompute_use_module():
    spec = importlib.util.spec_from_file_location("precompute_use", REPO_ROOT / "scripts" / "precompute_use.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["precompute_use"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_precompute_use_embeds_20_texts_and_hits_cache_on_rerun(tmp_path, monkeypatch):
    module = _load_precompute_use_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    dataset_dir = tmp_path / "data" / "smoke_dataset"
    dataset_dir.mkdir(parents=True)
    texts = [f"disaster tweet number {i} about a flood" for i in range(20)]
    for split, n in (("train", 20), ("val", 0), ("test", 0)):
        pd.DataFrame({"text": texts if split == "train" else []}).to_csv(dataset_dir / f"{split}.csv", index=False)

    result_first = module.main("smoke_dataset", splits=("train", "val", "test"))
    assert result_first["new_embeddings_by_split"]["train"] == 20
    assert result_first["total_unique_texts"] == 20

    from dtc.data.use_cache import load_embeddings

    cache_dir = dataset_dir / "use_embeddings"
    embeddings = load_embeddings(cache_dir, texts)
    assert embeddings.shape == (20, 512)

    result_second = module.main("smoke_dataset", splits=("train", "val", "test"))
    assert result_second["new_embeddings_by_split"]["train"] == 0  # cache hit, nothing recomputed


@pytest.mark.slow
def test_precompute_use_embeds_extra_csv_paths(tmp_path, monkeypatch):
    module = _load_precompute_use_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    dataset_dir = tmp_path / "data" / "smoke_dataset2"
    dataset_dir.mkdir(parents=True)
    for split in ("train", "val", "test"):
        pd.DataFrame({"text": []}).to_csv(dataset_dir / f"{split}.csv", index=False)

    extra_csv = tmp_path / "extra_raw.csv"
    pd.DataFrame({"text": ["extra text one", "extra text two"]}).to_csv(extra_csv, index=False)

    result = module.main("smoke_dataset2", splits=("train", "val", "test"), extra_csv_paths=(extra_csv,))
    assert result["new_embeddings_by_split"][str(extra_csv)] == 2
    assert result["total_unique_texts"] == 2
