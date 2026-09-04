#!/usr/bin/env python3
"""Run frozen Top30/Drop30 Qlib evaluation for completed formal jobs."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from responsibility.training.backtest import run_backtest  # noqa: E402
from responsibility.training.config import ExperimentConfig  # noqa: E402
from responsibility.training.data import resolve_data_paths  # noqa: E402
from responsibility.training.io import atomic_json, utc_now  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=REPO_ROOT / "data"
    )
    parser.add_argument(
        "--provider",
        type=Path,
        default=None,
        help="override the qlib_provider.path recorded by data/manifest.json",
    )
    parser.add_argument("--allow-other-qlib-version", action="store_true")
    return parser.parse_args(argv)


def _jobs(run_root: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    candidates = [run_root / "result.json"]
    candidates.extend(run_root.rglob("result.json"))
    jobs = []
    seen = set()
    for path in candidates:
        if path in seen or not path.is_file() or path.parent.name == "backtest":
            continue
        seen.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("status") != "DONE" or payload.get("model") != "ResponsibilityModel":
            continue
        jobs.append((path.parent, payload))
    return sorted(jobs, key=lambda item: str(item[0]))


def _config(run_dir: Path) -> ExperimentConfig:
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    value = payload.get("config")
    if not isinstance(value, dict):
        raise ValueError("run_config.json lacks a config snapshot")
    return ExperimentConfig(**value)


def run(args: argparse.Namespace) -> int:
    run_root = args.run_root.expanduser().resolve(strict=True)
    jobs = _jobs(run_root)
    if not jobs:
        raise FileNotFoundError("no completed ResponsibilityModel jobs under %s" % run_root)
    records = []
    failures = 0
    for run_dir, result in jobs:
        if result.get("mode") == "smoke":
            records.append({"run_dir": str(run_dir), "status": "SKIPPED_SMOKE"})
            continue
        output = run_dir / "backtest"
        if (output / "DONE").is_file() and (output / "result.json").is_file():
            records.append({"run_dir": str(run_dir), "status": "ALREADY_DONE"})
            continue
        try:
            config = _config(run_dir)
            data_paths = resolve_data_paths(args.data_root, config.dataset)
            provider = (
                args.provider.expanduser().resolve(strict=True)
                if args.provider is not None
                else data_paths.provider
            )
            evaluation = run_backtest(
                run_dir / "predictions/qlib_signal.pkl",
                provider=provider,
                config=config,
                output_dir=output,
                allow_other_qlib_version=args.allow_other_qlib_version,
            )
            records.append(
                {
                    "run_dir": str(run_dir),
                    "status": "DONE",
                    "metrics": evaluation["metrics"],
                }
            )
        except Exception as exc:
            failures += 1
            records.append(
                {
                    "run_dir": str(run_dir),
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt = {
        "status": "DONE" if failures == 0 else "DONE_WITH_FAILURES",
        "finished_utc": utc_now(),
        "run_root": str(run_root),
        "jobs": records,
    }
    atomic_json(run_root / ("evaluation_attempt_%s.json" % stamp), attempt)
    print(json.dumps(attempt, ensure_ascii=False, indent=2), flush=True)
    return 0 if failures == 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
