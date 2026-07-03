"""Resumable, config-driven experiment matrix runner (docs/PLAN.md Task 6).

Reads configs/experiments.yaml (E1/E2/E3, declaratively), expands it into
individual runs (stage, protocol, model, seed, train_fraction), skips
anything already present in the ledger as a run with a matching (stage,
protocol, model_name, config_id, seed, train_fraction, smoke) key, and
executes the rest sequentially -- ledgering each immediately on
completion. A killed run leaves no partial ledger line, since
dtc.harness.run.log_evaluation_run only appends after metrics are
computed (Hard Rule: crash-safe by construction, not by a lock file).

Usage:
    uv run python scripts/run_matrix.py --dry-run
    uv run python scripts/run_matrix.py --only e1 --models lstm gru
    uv run python scripts/run_matrix.py --smoke --smoke-n 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from dtc.data.common import dataframe_csv_bytes, sha256_bytes
from dtc.data.loaders import load_split_standardized
from dtc.data.protocol_a import load_raw_kaggle, mean_token_length, protocol_a_split
from dtc.eval.run_evaluation import evaluate_model_on_frozen_test
from dtc.harness.config import compute_config_id
from dtc.harness.fractions import subsample_train_df
from dtc.harness.ledger import read_ledger
from dtc.harness.run import log_evaluation_run
from dtc.models.registry import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_CONFIG_PATH = REPO_ROOT / "configs" / "experiments.yaml"

VOCAB_MODELS = {"meanpool_embed", "lstm", "gru", "bilstm", "conv1d"}
MAX_LENGTH_MODELS = VOCAB_MODELS | {"distilbert_finetune"}


def load_experiments_config(path: str | Path = EXPERIMENTS_CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_yaml_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No config at {path}. For E1/E3, run scripts/select_configs.py first; "
            "for E2, configs/protocol_a/<model>.yaml should already exist."
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_run_specs(
    experiments: dict, only: list[str] | None = None, models_filter: list[str] | None = None
) -> list[dict]:
    """Flattens configs/experiments.yaml into one dict per individual run
    (enumeration axes only -- config resolution happens later, per-run,
    since it can depend on data (Protocol A's mean_token_length))."""
    specs = []
    for exp_key in sorted(experiments):
        if only is not None and exp_key not in only:
            continue
        exp = experiments[exp_key]
        models = [m for m in exp["models"] if models_filter is None or m in models_filter]
        seeds = exp["seeds"]
        fractions = exp.get("train_fractions", [exp.get("train_fraction", 1.0)])
        for model_name in models:
            for seed in seeds:
                for fraction in fractions:
                    specs.append(
                        {
                            "experiment_key": exp_key,
                            "stage": exp["stage"],
                            "protocol": exp["protocol"],
                            "dataset": exp["dataset"],
                            "model_name": model_name,
                            "config_source": exp["config_source"],
                            "seed": seed,
                            "train_fraction": fraction,
                            "all_fractions": fractions,
                        }
                    )
    return specs


def _skip_key(spec: dict, config_id: str, smoke: bool) -> tuple:
    return (spec["stage"], spec["protocol"], spec["model_name"], config_id, spec["seed"], spec["train_fraction"], smoke)


def _already_ledgered_keys(ledger_records: list[dict]) -> set[tuple]:
    keys = set()
    for r in ledger_records:
        keys.add(
            (
                r.get("stage"),
                r.get("protocol"),
                r.get("model_name"),
                r.get("config_id"),
                r.get("seed"),
                r.get("train_fraction"),
                r.get("smoke"),
            )
        )
    return keys


def _prepare_protocol_b_data(repo_root: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = load_split_standardized(repo_root, dataset, "train")
    val_df = load_split_standardized(repo_root, dataset, "val")
    return train_df, val_df


def _prepare_protocol_a_data(repo_root: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = repo_root / "data" / dataset / "raw" / "train.csv"
    raw_df = load_raw_kaggle(raw_path)
    train_df, eval_df = protocol_a_split(raw_df)
    train_df = train_df.rename(columns={"target": "label"})[["id", "text", "label"]]
    eval_df = eval_df.rename(columns={"target": "label"})[["id", "text", "label"]]
    return train_df, eval_df


def _protocol_a_max_length_override(model_name: str, train_texts) -> dict:
    if model_name not in MAX_LENGTH_MODELS:
        return {}
    if model_name == "distilbert_finetune":
        from transformers import DistilBertTokenizerFast

        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        value = mean_token_length(train_texts, token_len_fn=lambda t: len(tokenizer.tokenize(t)))
    else:
        value = mean_token_length(train_texts)
    return {"max_length": value}


def resolve_config(spec: dict, repo_root: Path, train_texts) -> dict:
    configs_dir = repo_root / "configs" / spec["config_source"]
    config = dict(_load_yaml_config(configs_dir / f"{spec['model_name']}.yaml"))

    if spec["model_name"] == "use_frozen":
        config["use_cache_dir"] = str(repo_root / "data" / spec["dataset"] / "use_embeddings")

    if spec["protocol"] == "A":
        config.update(_protocol_a_max_length_override(spec["model_name"], train_texts))

    return config


def execute_run(spec: dict, repo_root: Path, *, smoke: bool, smoke_n: int) -> dict:
    dataset = spec["dataset"]
    protocol = spec["protocol"]
    seed = spec["seed"]

    if protocol == "A":
        train_df, eval_df = _prepare_protocol_a_data(repo_root, dataset)
    else:
        train_df, eval_df = _prepare_protocol_b_data(repo_root, dataset)

    if smoke:
        train_df = train_df.sample(n=min(len(train_df), smoke_n), random_state=seed).reset_index(drop=True)
        eval_df = eval_df.sample(n=min(len(eval_df), max(20, smoke_n // 5)), random_state=seed).reset_index(drop=True)

    if spec["train_fraction"] < 1.0:
        train_df = subsample_train_df(train_df, "label", spec["train_fraction"], spec["all_fractions"], seed)

    config = resolve_config(spec, repo_root, train_df["text"])
    config_id = compute_config_id(config)

    model = build_model(spec["model_name"])
    model.fit(train_df, eval_df, config=config, seed=seed)

    common_kwargs = dict(
        repo_root=repo_root,
        ledger_path=repo_root / "results" / "ledger.jsonl",
        results_dir=repo_root / "results",
        model_name=spec["model_name"],
        dataset=dataset,
        seed=seed,
        config=config,
        protocol=protocol,
        phase="phase1",
        stage=spec["stage"],
        smoke=smoke,
        train_fraction=spec["train_fraction"],
        config_id=config_id,
    )

    if protocol == "A":
        y_prob = model.predict_proba(eval_df["text"])
        y_pred = (y_prob >= 0.5).astype(int)
        split_hashes = {
            "protocol_a_train": sha256_bytes(dataframe_csv_bytes(train_df)),
            "protocol_a_eval": sha256_bytes(dataframe_csv_bytes(eval_df)),
        }
        record = log_evaluation_run(
            **common_kwargs,
            split="protocol_a_eval",
            ids=eval_df["id"],
            texts=eval_df["text"],
            y_true=eval_df["label"].to_numpy(),
            y_pred=y_pred,
            y_prob=y_prob,
            dataset_split_hashes=split_hashes,
        )
    else:
        eval_fields = evaluate_model_on_frozen_test(model, repo_root, dataset)
        manifest_path = repo_root / "data" / dataset / "manifest.json"
        record = log_evaluation_run(
            **common_kwargs,
            split="test",
            dataset_manifest_path=manifest_path,
            **eval_fields,
        )

    return record


def main(
    *,
    experiments_config_path: str | Path = EXPERIMENTS_CONFIG_PATH,
    repo_root: Path = REPO_ROOT,
    only: list[str] | None = None,
    models_filter: list[str] | None = None,
    dry_run: bool = False,
    smoke: bool = False,
    smoke_n: int = 200,
) -> list[dict]:
    experiments = load_experiments_config(experiments_config_path)
    specs = build_run_specs(experiments, only=only, models_filter=models_filter)

    ledger_path = repo_root / "results" / "ledger.jsonl"
    ledgered_keys = _already_ledgered_keys(read_ledger(ledger_path))

    results = []
    for spec in specs:
        # Protocol B's config never depends on data (only Protocol A's
        # models need train texts, to compute the mean-token max_length
        # override), so we only pay for a CSV load here when it's actually
        # needed -- keeps --dry-run fast across the full E1/E2/E3 matrix.
        train_texts = None
        if spec["protocol"] == "A":
            train_df, _ = _prepare_protocol_a_data(repo_root, spec["dataset"])
            if spec["train_fraction"] < 1.0:
                train_df = subsample_train_df(
                    train_df, "label", spec["train_fraction"], spec["all_fractions"], spec["seed"]
                )
            train_texts = train_df["text"]

        config = resolve_config(spec, repo_root, train_texts)
        config_id = compute_config_id(config)
        key = _skip_key(spec, config_id, smoke)

        pending = key not in ledgered_keys
        entry = {**spec, "config_id": config_id, "would_run": pending}

        if dry_run:
            results.append(entry)
            continue

        if not pending:
            results.append({**entry, "skipped": True})
            continue

        record = execute_run(spec, repo_root, smoke=smoke, smoke_n=smoke_n)
        ledgered_keys.add(key)
        results.append({**entry, "skipped": False, "run_id": record["run_id"], "metrics": record["metrics"]})

    return results


def _print_dry_run_summary(results: list[dict]) -> None:
    by_exp: dict[str, list[dict]] = {}
    for r in results:
        by_exp.setdefault(r["experiment_key"], []).append(r)

    total_pending = 0
    for exp_key, rows in sorted(by_exp.items()):
        pending = [r for r in rows if r["would_run"]]
        total_pending += len(pending)
        print(f"{exp_key}: {len(pending)} pending / {len(rows)} total")
        for r in pending:
            print(
                f"  [{r['stage']}] {r['model_name']} seed={r['seed']} "
                f"fraction={r['train_fraction']} config_id={r['config_id']}"
            )
    print(f"\nTotal pending runs: {total_pending} / {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Experiment keys to include, e.g. e1 e2")
    parser.add_argument("--models", nargs="*", default=None, help="Model names to include")
    parser.add_argument("--smoke", action="store_true", help="Tiny-subset smoke mode, tagged smoke=true")
    parser.add_argument("--smoke-n", type=int, default=200)
    args = parser.parse_args()

    outcome = main(
        only=args.only,
        models_filter=args.models,
        dry_run=args.dry_run,
        smoke=args.smoke,
        smoke_n=args.smoke_n,
    )

    if args.dry_run:
        _print_dry_run_summary(outcome)
    else:
        n_run = sum(1 for r in outcome if "run_id" in r)
        n_skipped = sum(1 for r in outcome if r.get("skipped") is True)
        for r in outcome:
            if "run_id" in r:
                print(f"[{r['stage']}] {r['model_name']} seed={r['seed']} -> run_id={r['run_id']}")
        print(f"\n{n_run} run, {n_skipped} skipped, {len(outcome)} total.")
