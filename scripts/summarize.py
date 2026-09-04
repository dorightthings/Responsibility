#!/usr/bin/env python3
"""Summarize completed ResponsibilityModel jobs into horizontal tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from responsibility.training.io import atomic_csv, atomic_json, atomic_text, utc_now  # noqa: E402


METRICS = ("IC", "ICIR", "RankIC", "RankICIR", "AR", "IR")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _records(run_root: Path) -> List[Dict[str, Any]]:
    records = []
    seen = set()
    for path in run_root.rglob("result.json"):
        if path.parent.name == "backtest":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "DONE" or payload.get("model") != "ResponsibilityModel":
            continue
        key = (str(payload.get("dataset")), int(payload.get("seed")))
        if key in seen:
            raise ValueError("duplicate completed dataset/seed under run-root: %s" % (key,))
        seen.add(key)
        metrics = payload.get("metrics", {})
        backtest_path = path.parent / "backtest/result.json"
        backtest = {}
        if backtest_path.is_file():
            backtest_payload = json.loads(backtest_path.read_text(encoding="utf-8"))
            if backtest_payload.get("status") == "DONE":
                backtest = backtest_payload.get("metrics", {})
        row = {
            "dataset": key[0],
            "seed": key[1],
            "mode": payload.get("mode"),
            "run_dir": str(path.parent),
            "IC": _finite(metrics.get("IC")),
            "ICIR": _finite(metrics.get("ICIR")),
            "RankIC": _finite(metrics.get("RankIC")),
            "RankICIR": _finite(metrics.get("RankICIR")),
            "AR": _finite(backtest.get("AR")),
            "IR": _finite(backtest.get("IR")),
        }
        records.append(row)
    return sorted(records, key=lambda row: (row["dataset"], row["seed"]))


def _aggregate(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for row in records:
        grouped[row["dataset"]].append(row)
    result = []
    for dataset, rows in sorted(grouped.items()):
        aggregate: Dict[str, Any] = {
            "dataset": dataset,
            "seeds": ",".join(str(row["seed"]) for row in rows),
            "runs": len(rows),
        }
        for metric in METRICS:
            values = [row[metric] for row in rows if row[metric] is not None]
            aggregate[metric + "_n"] = len(values)
            aggregate[metric + "_mean"] = (
                statistics.fmean(values) if values else None
            )
            aggregate[metric + "_std"] = (
                statistics.pstdev(values) if values else None
            )
        result.append(aggregate)
    return result


def _markdown(rows: List[Dict[str, Any]]) -> str:
    header = ["Dataset", "Seeds", "IC", "ICIR", "RankIC", "RankICIR", "AR", "IR"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        cells = [str(row["dataset"]), str(row["seeds"])]
        for metric in METRICS:
            mean = row[metric + "_mean"]
            std = row[metric + "_std"]
            cells.append(
                "—" if mean is None else "%.6f ± %.6f" % (mean, std)
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "AR/IR use Qlib Top30/Drop30 `excess_return_without_cost`.",
            "Missing AR/IR means `scripts/evaluate.py` has not completed; no value is imputed.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    run_root = args.run_root.expanduser().resolve(strict=True)
    records = _records(run_root)
    if not records:
        raise FileNotFoundError("no completed ResponsibilityModel results under %s" % run_root)
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = run_root / ("summary_" + stamp)
    else:
        output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite summary: %s" % output)
    output.mkdir(parents=True)
    aggregates = _aggregate(records)
    atomic_csv(output / "per_seed.csv", records)
    atomic_csv(output / "summary.csv", aggregates)
    atomic_text(output / "summary.md", _markdown(aggregates))
    result = {
        "status": "DONE",
        "finished_utc": utc_now(),
        "run_root": str(run_root),
        "datasets": [row["dataset"] for row in aggregates],
        "runs": len(records),
        "headline": "Top30/Drop30 excess_return_without_cost",
        "summary": aggregates,
    }
    atomic_json(output / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
