"""Generates Phase 1 results tables T1-T4 from results/ledger.jsonl only --
never hand-typed (docs/PLAN.md Task 7). Idempotent: re-running with the
same ledger state overwrites the same files with identical content.
Excludes smoke runs by default. Refuses to emit any table referencing a
run whose git_dirty_paths includes a source file (ledger hygiene: every
number must trace to a run against a clean commit). T1 additionally
refuses if any model has more than one distinct config_id among its
non-smoke E1 records, and T4 the same per (model, train_fraction) --
mixing configs would silently average runs trained under different final
configs (the ledger is append-only, so a config regenerated after some
runs already happened doesn't erase the old runs' entries).

Use `--only-config-ids-from configs/final` to resolve, per model, the
config_id that model's CURRENT configs/final/<model>.yaml would produce,
and filter every table's input records down to just that config_id --
this is how you recover from the mixed-config refusal once a real config
has superseded an earlier (e.g. smoke-placeholder) one, without editing
or deleting any ledger line (docs/DECISIONS.md; Phase 1.5).

Usage:
    uv run python scripts/make_tables.py
    uv run python scripts/make_tables.py --include-smoke   # demo only, watermarked SMOKE
    uv run python scripts/make_tables.py --only-config-ids-from configs/final
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import yaml

from dtc.harness.config import compute_config_id
from dtc.harness.ledger import read_ledger

REPO_ROOT = Path(__file__).resolve().parents[1]
_NON_SOURCE_PREFIXES = ("results/", "data/")


class DirtyLedgerError(RuntimeError):
    """Raised when a referenced run's git_dirty_paths includes source files."""


class MixedConfigError(RuntimeError):
    """Raised when a model (T1) or a (model, train_fraction) pair (T4) has
    more than one distinct config_id among its non-smoke records --
    pooling those into one mean/std would silently average runs trained
    under different final configs."""


def _is_source_path(path: str) -> bool:
    return not path.startswith(_NON_SOURCE_PREFIXES)


def check_single_config_per_model(e1_records: list[dict]) -> None:
    """Smoke E1 records are exempt: they may legitimately use different
    (placeholder) configs across a session, so only non-smoke records are
    checked for config consistency."""
    non_smoke = [r for r in e1_records if not r.get("smoke", False)]
    by_model: dict[str, set[str]] = {}
    for r in non_smoke:
        by_model.setdefault(r["model_name"], set()).add(r.get("config_id"))
    offenders = {model: sorted(ids) for model, ids in by_model.items() if len(ids) > 1}
    if offenders:
        details = "; ".join(f"{model}: {ids}" for model, ids in sorted(offenders.items()))
        raise MixedConfigError(
            f"Refusing to build T1: model(s) with multiple distinct config_ids among non-smoke E1 records: {details}. "
            "Use --only-config-ids-from configs/final to filter to each model's current config."
        )


def check_single_config_per_model_fraction(e3_records: list[dict]) -> None:
    """T4's analogue of check_single_config_per_model, grouped by
    (model_name, train_fraction) instead of model_name alone. Smoke E3
    records are exempt, same rationale."""
    non_smoke = [r for r in e3_records if not r.get("smoke", False)]
    by_key: dict[tuple[str, float], set[str]] = {}
    for r in non_smoke:
        key = (r["model_name"], r["train_fraction"])
        by_key.setdefault(key, set()).add(r.get("config_id"))
    offenders = {key: sorted(ids) for key, ids in by_key.items() if len(ids) > 1}
    if offenders:
        details = "; ".join(f"{model}@{fraction}: {ids}" for (model, fraction), ids in sorted(offenders.items()))
        raise MixedConfigError(
            f"Refusing to build T4: model/fraction combo(s) with multiple distinct config_ids among non-smoke "
            f"E3 records: {details}. Use --only-config-ids-from configs/final to filter to each model's current config."
        )


def resolve_final_config_ids(
    final_configs_dir: str | Path, *, dataset: str = "kaggle", repo_root: str | Path = REPO_ROOT
) -> dict[str, str]:
    """For each `<final_configs_dir>/<model>.yaml`, computes the config_id
    a real (non-smoke) run using that exact config would produce.

    Replicates the one piece of scripts/run_matrix.py's `resolve_config()`
    injection that affects `use_frozen`'s config_id (its `use_cache_dir`)
    without importing run_matrix.py itself -- Phase 1.5 Hard Rule 1 is
    additive/guard changes only, no driver changes. Models with no
    `<model>.yaml` file in `final_configs_dir` are simply absent from the
    returned dict.
    """
    final_configs_dir = Path(final_configs_dir)
    result = {}
    for path in sorted(final_configs_dir.glob("*.yaml")):
        model_name = path.stem
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if model_name == "use_frozen":
            config = {**config, "use_cache_dir": str(Path(repo_root) / "data" / dataset / "use_embeddings")}
        result[model_name] = compute_config_id(config)
    return result


def filter_to_final_config_ids(records: list[dict], final_config_ids: dict[str, str]) -> list[dict]:
    """Keeps a record if its model has no resolvable "current" config (no
    file in the given final_configs_dir -- passed through unfiltered, so
    the mixed-config check still catches genuine ambiguity for it) or if
    its config_id matches that model's current final config; drops
    records whose config_id has been superseded by a later regeneration
    of configs/final/<model>.yaml.
    """
    kept = []
    for r in records:
        expected = final_config_ids.get(r.get("model_name"))
        if expected is None or r.get("config_id") == expected:
            kept.append(r)
    return kept


def check_no_dirty_source_runs(records: list[dict]) -> None:
    offenders = []
    for r in records:
        dirty_source_paths = [p for p in r.get("git_dirty_paths") or [] if _is_source_path(p)]
        if dirty_source_paths:
            offenders.append((r.get("run_id"), dirty_source_paths))
    if offenders:
        details = "; ".join(f"{run_id}: {paths}" for run_id, paths in offenders)
        raise DirtyLedgerError(f"Refusing to generate tables: dirty source paths found in runs: {details}")


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def build_t1_protocol_b_main(records: list[dict]) -> list[dict]:
    """Per model: test accuracy/macro-F1/positive-F1 mean +- std over seeds."""
    e1 = [r for r in records if r.get("stage") == "E1" and r.get("protocol") == "B" and r.get("split") == "test"]
    check_single_config_per_model(e1)
    by_model: dict[str, list[dict]] = {}
    for r in e1:
        by_model.setdefault(r["model_name"], []).append(r)

    rows = []
    for model_name, runs in sorted(by_model.items()):
        acc_mean, acc_std = _mean_std([r["metrics"]["accuracy"] for r in runs])
        macro_mean, macro_std = _mean_std([r["metrics"]["macro_f1"] for r in runs])
        pos_mean, pos_std = _mean_std([r["metrics"]["positive_f1"] for r in runs])
        rows.append(
            {
                "model": model_name,
                "n_seeds": len(runs),
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "macro_f1_mean": macro_mean,
                "macro_f1_std": macro_std,
                "positive_f1_mean": pos_mean,
                "positive_f1_std": pos_std,
            }
        )
    return rows


def build_t2_protocol_comparison(records: list[dict]) -> list[dict]:
    """Side-by-side model rankings under each protocol's own headline metric
    (Protocol B: macro-F1; Protocol A: legacy weighted-F1), with rank delta."""
    e1 = [r for r in records if r.get("stage") == "E1" and r.get("protocol") == "B" and r.get("split") == "test"]
    e2 = [r for r in records if r.get("stage") == "E2" and r.get("protocol") == "A"]

    b_scores: dict[str, list[float]] = {}
    for r in e1:
        b_scores.setdefault(r["model_name"], []).append(r["metrics"]["macro_f1"])
    a_scores: dict[str, float] = {r["model_name"]: r["metrics"]["weighted_f1_legacy"] for r in e2}

    b_mean = {m: statistics.mean(v) for m, v in b_scores.items()}
    b_rank = {m: i + 1 for i, m in enumerate(sorted(b_mean, key=lambda m: -b_mean[m]))}
    a_rank = {m: i + 1 for i, m in enumerate(sorted(a_scores, key=lambda m: -a_scores[m]))}

    rows = []
    for model_name in sorted(set(b_mean) | set(a_scores)):
        rank_b = b_rank.get(model_name)
        rank_a = a_rank.get(model_name)
        delta = (rank_a - rank_b) if (rank_a is not None and rank_b is not None) else None
        rows.append(
            {
                "model": model_name,
                "protocol_b_macro_f1": b_mean.get(model_name),
                "protocol_b_rank": rank_b,
                "protocol_a_weighted_f1": a_scores.get(model_name),
                "protocol_a_rank": rank_a,
                "rank_delta": delta,
            }
        )
    return rows


def build_t3_seed_variance(records: list[dict]) -> list[dict]:
    e1 = [r for r in records if r.get("stage") == "E1" and r.get("protocol") == "B" and r.get("split") == "test"]
    by_model: dict[str, list[float]] = {}
    for r in e1:
        by_model.setdefault(r["model_name"], []).append(r["metrics"]["macro_f1"])

    rows = []
    for model_name, values in sorted(by_model.items()):
        rows.append(
            {
                "model": model_name,
                "n_seeds": len(values),
                "min_macro_f1": min(values),
                "max_macro_f1": max(values),
                "std_macro_f1": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        )
    return rows


def build_t4_data_efficiency(records: list[dict]) -> list[dict]:
    e3 = [r for r in records if r.get("stage") == "E3"]
    check_single_config_per_model_fraction(e3)
    by_key: dict[tuple[str, float], list[float]] = {}
    for r in e3:
        key = (r["model_name"], r["train_fraction"])
        by_key.setdefault(key, []).append(r["metrics"]["macro_f1"])

    rows = []
    for (model_name, fraction), values in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        mean, std = _mean_std(values)
        rows.append(
            {"model": model_name, "train_fraction": fraction, "n_seeds": len(values), "macro_f1_mean": mean, "macro_f1_std": std}
        )
    return rows


def _fmt(x, digits: int = 4) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _fmt_mean_std(mean, std, digits: int = 4) -> str:
    return f"{_fmt(mean, digits)} ± {_fmt(std, digits)}"


def _render_markdown_table(title: str, headers: list[str], rows: list[list], banner: str | None) -> str:
    lines = [f"# {title}", ""]
    if banner:
        lines += [f"**{banner}**", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines.append("")
    return "\n".join(lines)


def render_t1(rows: list[dict], banner: str | None) -> str:
    headers = ["Model", "n_seeds", "Accuracy", "Macro-F1", "Positive-F1"]
    table_rows = [
        [r["model"], r["n_seeds"], _fmt_mean_std(r["accuracy_mean"], r["accuracy_std"]),
         _fmt_mean_std(r["macro_f1_mean"], r["macro_f1_std"]), _fmt_mean_std(r["positive_f1_mean"], r["positive_f1_std"])]
        for r in rows
    ]
    return _render_markdown_table("T1 — Protocol B main results (frozen Kaggle test)", headers, table_rows, banner)


def render_t2(rows: list[dict], banner: str | None) -> str:
    headers = ["Model", "Protocol B macro-F1", "Protocol B rank", "Protocol A weighted-F1", "Protocol A rank", "Rank delta (A - B)"]
    table_rows = [
        [
            r["model"],
            _fmt(r["protocol_b_macro_f1"]),
            r["protocol_b_rank"] if r["protocol_b_rank"] is not None else "—",
            _fmt(r["protocol_a_weighted_f1"]),
            r["protocol_a_rank"] if r["protocol_a_rank"] is not None else "—",
            r["rank_delta"] if r["rank_delta"] is not None else "—",
        ]
        for r in rows
    ]
    return _render_markdown_table("T2 — Protocol A vs. Protocol B ranking comparison", headers, table_rows, banner)


def render_t3(rows: list[dict], banner: str | None) -> str:
    headers = ["Model", "n_seeds", "Min macro-F1", "Max macro-F1", "Std macro-F1"]
    table_rows = [
        [r["model"], r["n_seeds"], _fmt(r["min_macro_f1"]), _fmt(r["max_macro_f1"]), _fmt(r["std_macro_f1"])] for r in rows
    ]
    return _render_markdown_table("T3 — Seed-variance (Protocol B, frozen test)", headers, table_rows, banner)


def render_t4(rows: list[dict], banner: str | None) -> str:
    headers = ["Model", "Train fraction", "n_seeds", "Macro-F1"]
    table_rows = [
        [r["model"], r["train_fraction"], r["n_seeds"], _fmt_mean_std(r["macro_f1_mean"], r["macro_f1_std"])] for r in rows
    ]
    return _render_markdown_table("T4 — Data efficiency (Protocol B)", headers, table_rows, banner)


def main(
    ledger_path: str | Path = REPO_ROOT / "results" / "ledger.jsonl",
    output_dir: str | Path = REPO_ROOT / "results" / "tables",
    include_smoke: bool = False,
    only_config_ids_from: str | Path | None = None,
) -> dict[str, Path]:
    records = read_ledger(ledger_path)
    if not include_smoke:
        records = [r for r in records if not r.get("smoke", False)]
    if only_config_ids_from is not None:
        final_config_ids = resolve_final_config_ids(only_config_ids_from)
        records = filter_to_final_config_ids(records, final_config_ids)
    check_no_dirty_source_runs(records)

    banner = "SMOKE DATA -- placeholder, not real results" if include_smoke else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "T1": (render_t1(build_t1_protocol_b_main(records), banner), "T1_protocol_b_main.md"),
        "T2": (render_t2(build_t2_protocol_comparison(records), banner), "T2_protocol_comparison.md"),
        "T3": (render_t3(build_t3_seed_variance(records), banner), "T3_seed_variance.md"),
        "T4": (render_t4(build_t4_data_efficiency(records), banner), "T4_data_efficiency.md"),
    }

    written = {}
    for name, (content, filename) in tables.items():
        path = output_dir / filename
        path.write_text(content, encoding="utf-8")
        written[name] = path
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=REPO_ROOT / "results" / "ledger.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    parser.add_argument("--include-smoke", action="store_true", help="Include smoke runs, watermarked SMOKE (demo only)")
    parser.add_argument(
        "--only-config-ids-from",
        type=Path,
        default=None,
        help="Directory of <model>.yaml configs (e.g. configs/final) to resolve each model's CURRENT "
        "config_id from; records using a different (superseded) config_id are filtered out before "
        "aggregation, instead of refusing on mixed config_ids.",
    )
    args = parser.parse_args()
    result = main(
        args.ledger, args.output_dir, include_smoke=args.include_smoke, only_config_ids_from=args.only_config_ids_from
    )
    for name, path in result.items():
        print(f"{name}: {path}")
