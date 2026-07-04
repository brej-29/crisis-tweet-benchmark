"""Generates Phase 1 results tables T1-T4 from results/ledger.jsonl only --
never hand-typed (docs/PLAN.md Task 7). Idempotent: re-running with the
same ledger state overwrites the same files with identical content.
Excludes smoke runs by default. Refuses to emit any table referencing a
run whose git_dirty_paths includes a source file (ledger hygiene: every
number must trace to a run against a clean commit).

Usage:
    uv run python scripts/make_tables.py
    uv run python scripts/make_tables.py --include-smoke   # demo only, watermarked SMOKE
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from dtc.harness.ledger import read_ledger

REPO_ROOT = Path(__file__).resolve().parents[1]
_NON_SOURCE_PREFIXES = ("results/", "data/")


class DirtyLedgerError(RuntimeError):
    """Raised when a referenced run's git_dirty_paths includes source files."""


def _is_source_path(path: str) -> bool:
    return not path.startswith(_NON_SOURCE_PREFIXES)


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
) -> dict[str, Path]:
    records = read_ledger(ledger_path)
    if not include_smoke:
        records = [r for r in records if not r.get("smoke", False)]
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
    args = parser.parse_args()
    result = main(args.ledger, args.output_dir, include_smoke=args.include_smoke)
    for name, path in result.items():
        print(f"{name}: {path}")
