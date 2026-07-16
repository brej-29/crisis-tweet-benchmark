"""Resumable, config-driven experiment matrix runner (docs/PLAN.md Task 6).

Reads configs/experiments.yaml (E1-E5, declaratively), expands it into
individual TRAINING runs (stage, protocol, model, seed, train_fraction),
skips any eval already present in the ledger as a run with a matching
(stage, protocol, model_name, config_id, seed, train_fraction, smoke,
train_dataset, eval_dataset) key, and executes the rest sequentially --
ledgering each immediately on completion. A killed run leaves no partial
ledger line, since dtc.harness.run.log_evaluation_run only appends after
metrics are computed (Hard Rule: crash-safe by construction, not by a
lock file).

Phase 2 cross-dataset experiments (E4/E5) declare `eval_datasets: [a, b]`:
ONE training per spec, evaluated on BOTH frozen tests with the in-memory
model (no persistence/reload), emitting one ledger record per eval_dataset
sharing a training_id. A training is pending if ANY of its per-eval keys
is missing; execution fills only the missing eval records.

An experiment may also declare `phase: phase2` (Task A4), tagging every
ledger record it produces via `dtc.harness.run.log_evaluation_run`'s
`phase` field; absent, it defaults to "phase1" (Phase 1's tuning/e1/e2/e3
keep their existing tag, unchanged).

Usage:
    uv run python scripts/run_matrix.py --dry-run
    uv run python scripts/run_matrix.py --only e1 --models lstm gru
    uv run python scripts/run_matrix.py --smoke --smoke-n 200
"""

from __future__ import annotations

import argparse
import uuid
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


def _load_tuning_grid(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"No tuning grid at {path} (docs/PLAN.md Task 4).")
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    return doc["grid"]


def build_run_specs(
    experiments: dict,
    only: list[str] | None = None,
    models_filter: list[str] | None = None,
    seeds_filter: list[int] | None = None,
    repo_root: Path = REPO_ROOT,
) -> list[dict]:
    """Flattens configs/experiments.yaml into one dict per individual run
    (enumeration axes only -- most config resolution happens later, per-run,
    since it can depend on data (Protocol A's mean_token_length)). The one
    exception is `config_source: tuning`: each model's grid is read here
    (cheap, static YAML) so one run spec is produced per grid entry, not
    per model.
    """
    specs = []
    for exp_key in sorted(experiments):
        if only is not None and exp_key not in only:
            continue
        exp = experiments[exp_key]
        eval_datasets = list(exp.get("eval_datasets") or [exp["dataset"]])
        if len(eval_datasets) > 1 and (exp["protocol"] == "A" or exp["stage"] == "tuning"):
            # Only Protocol B non-tuning stages loop over eval datasets in
            # execute_run; the Protocol A / tuning branches always emit
            # exactly one record, so extra eval datasets would be silently
            # marked ledgered without ever being evaluated.
            raise ValueError(
                f"experiment '{exp_key}': eval_datasets={eval_datasets} is only supported "
                "for protocol-B non-tuning stages (protocol-A and tuning runs evaluate on "
                "a single dataset and emit exactly one ledger record per training)."
            )
        models = [m for m in exp["models"] if models_filter is None or m in models_filter]
        seeds = [s for s in exp["seeds"] if seeds_filter is None or s in seeds_filter]
        fractions = exp.get("train_fractions", [exp.get("train_fraction", 1.0)])
        config_source = exp["config_source"]
        for model_name in models:
            grid_configs: list[dict | None]
            if config_source == "tuning":
                grid_configs = _load_tuning_grid(repo_root / "configs" / "tuning" / f"{model_name}.yaml")
            else:
                grid_configs = [None]  # resolved later from a single config file
            for grid_index, grid_config in enumerate(grid_configs):
                for seed in seeds:
                    for fraction in fractions:
                        specs.append(
                            {
                                "experiment_key": exp_key,
                                "stage": exp["stage"],
                                "protocol": exp["protocol"],
                                "dataset": exp["dataset"],
                                "phase": exp.get("phase", "phase1"),
                                # one spec per TRAINING run: E4/E5 evaluate the
                                # same in-memory model on several frozen tests
                                "train_dataset": exp["dataset"],
                                "eval_datasets": list(eval_datasets),
                                "model_name": model_name,
                                "config_source": config_source,
                                "grid_index": grid_index if config_source == "tuning" else None,
                                "grid_config": grid_config,
                                "seed": seed,
                                "train_fraction": fraction,
                                "all_fractions": fractions,
                            }
                        )
    return specs


def _skip_key(spec: dict, config_id: str, smoke: bool, eval_dataset: str) -> tuple:
    # One key per EVAL record (not per training): E4/E5 emit several records
    # per training, and each is skipped/pending independently. Old (Phase 1)
    # ledger records match via read_ledger's read-time backfill of
    # train_dataset/eval_dataset -> "kaggle".
    return (
        spec["stage"],
        spec["protocol"],
        spec["model_name"],
        config_id,
        spec["seed"],
        spec["train_fraction"],
        smoke,
        spec["train_dataset"],
        eval_dataset,
    )


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
                r.get("train_dataset"),
                r.get("eval_dataset"),
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
    if spec["config_source"] == "tuning":
        config = dict(spec["grid_config"])
    else:
        configs_dir = repo_root / "configs" / spec["config_source"]
        config = dict(_load_yaml_config(configs_dir / f"{spec['model_name']}.yaml"))

    if spec["model_name"] == "use_frozen":
        config["use_cache_dir"] = str(repo_root / "data" / spec["dataset"] / "use_embeddings")

    if spec["protocol"] == "A":
        config.update(_protocol_a_max_length_override(spec["model_name"], train_texts))

    return config


def execute_run(
    spec: dict, repo_root: Path, *, smoke: bool, smoke_n: int, pending_eval_datasets: list[str] | None = None
) -> list[dict]:
    """Trains once and returns the list of ledger records emitted (one per
    eval dataset; single-element for tuning/Protocol A/single-eval specs).

    `pending_eval_datasets`: which of spec["eval_datasets"] still need a
    record (default: all of them). On resume after a crash between two
    evals, the training is redone and ONLY the gap is filled -- the two
    records then carry different training_ids (provenance only, see
    docs/DECISIONS.md).
    """
    dataset = spec["dataset"]
    train_dataset = spec["train_dataset"]
    if pending_eval_datasets is None:
        pending_eval_datasets = list(spec["eval_datasets"])
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

    # Cross-dataset eval (E4/E5): use_frozen's predict-time embedding lookups
    # must also see the OTHER dataset's cache. Set post-fit as a plain
    # attribute, never via config -- config_id must stay identical to the
    # single-dataset case (docs/DECISIONS.md).
    if spec["model_name"] == "use_frozen" and len(spec["eval_datasets"]) > 1:
        model.extra_cache_dirs = [
            repo_root / "data" / ds / "use_embeddings" for ds in spec["eval_datasets"] if ds != train_dataset
        ]

    common_kwargs = dict(
        repo_root=repo_root,
        ledger_path=repo_root / "results" / "ledger.jsonl",
        results_dir=repo_root / "results",
        model_name=spec["model_name"],
        dataset=dataset,
        seed=seed,
        config=config,
        protocol=protocol,
        phase=spec["phase"],
        stage=spec["stage"],
        smoke=smoke,
        train_fraction=spec["train_fraction"],
        config_id=config_id,
        train_dataset=train_dataset,
        training_id=uuid.uuid4().hex,  # one per TRAINING; shared by all its eval records
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
            eval_dataset=dataset,
            ids=eval_df["id"],
            texts=eval_df["text"],
            y_true=eval_df["label"].to_numpy(),
            y_pred=y_pred,
            y_prob=y_prob,
            dataset_split_hashes=split_hashes,
        )
    elif spec["stage"] == "tuning":
        # Task 4: selection metric is val macro-F1 -- evaluate on val
        # directly (never the frozen test split; tuning is a
        # model-selection step, not a final report).
        y_prob = model.predict_proba(eval_df["text"])
        y_pred = (y_prob >= 0.5).astype(int)
        manifest_path = repo_root / "data" / dataset / "manifest.json"
        record = log_evaluation_run(
            **common_kwargs,
            split="val",
            eval_dataset=dataset,
            ids=eval_df["id"],
            texts=eval_df["text"],
            y_true=eval_df["label"].to_numpy(),
            y_pred=y_pred,
            y_prob=y_prob,
            dataset_manifest_path=manifest_path,
        )
    else:
        records = []
        for eval_ds in pending_eval_datasets:
            # eval_fields carries ids/texts/y_true/y_pred/y_prob plus, when the
            # eval dataset has passthrough columns (CrisisLex's `event`), an
            # optional "extra_columns" key -- **eval_fields routes that key
            # straight into log_evaluation_run's extra_columns param, so no
            # per-dataset special-casing is needed here (docs/DECISIONS.md).
            eval_fields = evaluate_model_on_frozen_test(model, repo_root, eval_ds)
            manifest_path = repo_root / "data" / eval_ds / "manifest.json"
            records.append(
                log_evaluation_run(
                    **common_kwargs,
                    split="test",
                    eval_dataset=eval_ds,
                    dataset_manifest_path=manifest_path,
                    **eval_fields,
                )
            )
        return records

    return [record]


def main(
    *,
    experiments_config_path: str | Path = EXPERIMENTS_CONFIG_PATH,
    repo_root: Path = REPO_ROOT,
    only: list[str] | None = None,
    models_filter: list[str] | None = None,
    seeds_filter: list[int] | None = None,
    dry_run: bool = False,
    smoke: bool = False,
    smoke_n: int = 200,
) -> list[dict]:
    experiments = load_experiments_config(experiments_config_path)
    specs = build_run_specs(
        experiments, only=only, models_filter=models_filter, seeds_filter=seeds_filter, repo_root=repo_root
    )

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
        per_eval_keys = {ds: _skip_key(spec, config_id, smoke, ds) for ds in spec["eval_datasets"]}

        # A training is pending if ANY of its per-eval records is missing;
        # execution then trains once and emits ONLY the missing records.
        pending_evals = [ds for ds in spec["eval_datasets"] if per_eval_keys[ds] not in ledgered_keys]
        pending = bool(pending_evals)
        entry = {**spec, "config_id": config_id, "would_run": pending, "pending_eval_datasets": pending_evals}

        if dry_run:
            results.append(entry)
            continue

        if not pending:
            results.append({**entry, "skipped": True})
            continue

        records = execute_run(spec, repo_root, smoke=smoke, smoke_n=smoke_n, pending_eval_datasets=pending_evals)
        for ds in pending_evals:
            ledgered_keys.add(per_eval_keys[ds])
        results.append(
            {
                **entry,
                "skipped": False,
                "run_id": records[-1]["run_id"],
                "run_ids": [r["run_id"] for r in records],
                "metrics": records[-1]["metrics"],
            }
        )

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
            line = (
                f"  [{r['stage']}] {r['model_name']} seed={r['seed']} "
                f"fraction={r['train_fraction']} config_id={r['config_id']}"
            )
            if len(r["eval_datasets"]) > 1:
                # dual-eval spec: show which of its eval records are missing
                line += f" pending_evals={','.join(r['pending_eval_datasets'])}"
            print(line)
    print(f"\nTotal pending runs: {total_pending} / {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Experiment keys to include, e.g. e1 e2")
    parser.add_argument("--models", nargs="*", default=None, help="Model names to include")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Seeds to include")
    parser.add_argument("--smoke", action="store_true", help="Tiny-subset smoke mode, tagged smoke=true")
    parser.add_argument("--smoke-n", type=int, default=200)
    args = parser.parse_args()

    outcome = main(
        only=args.only,
        models_filter=args.models,
        seeds_filter=args.seeds,
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
