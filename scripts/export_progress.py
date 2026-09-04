#!/usr/bin/env python3
"""Export a small, path-free experiment summary that is safe to commit to Git."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "results/experiments"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--machine",
        default=platform.node(),
        help="friendly machine label stored with the summary",
    )
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT
    )
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="config file to copy; repeat for multiple datasets",
    )
    parser.add_argument("--note", default="")
    return parser.parse_args(argv)


def git_value(arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=str(REPOSITORY_ROOT),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def validate_experiment_id(value: str) -> str:
    if not value or value in {".", ".."}:
        raise ValueError("--experiment-id must not be empty")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in value):
        raise ValueError("--experiment-id may contain only letters, digits, '-' and '_'")
    return value


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_new(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError("CSV has no records: %s" % path)
    fields = [name for name in rows[0] if name != "run_dir"]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        experiment_id = validate_experiment_id(args.experiment_id)
        summary_dir = args.summary_dir.expanduser().resolve(strict=True)
        required = {
            "per_seed.csv": summary_dir / "per_seed.csv",
            "summary.csv": summary_dir / "summary.csv",
            "summary.md": summary_dir / "summary.md",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("summary files are missing: " + ", ".join(missing))
        git_commit = git_value(["rev-parse", "HEAD"])
        dirty = bool(git_value(["status", "--porcelain"]))
        output = args.results_root.expanduser().resolve() / experiment_id
        output.mkdir(parents=True, exist_ok=False)

        write_csv_new(output / "per_seed.csv", read_csv(required["per_seed.csv"]))
        shutil.copyfile(str(required["summary.csv"]), str(output / "summary.csv"))
        shutil.copyfile(str(required["summary.md"]), str(output / "summary.md"))
        config_names = set()
        for config in args.config:
            source = config.expanduser().resolve(strict=True)
            if not source.is_file():
                raise FileNotFoundError(source)
            if source.name in config_names:
                raise ValueError("duplicate config filename: %s" % source.name)
            config_names.add(source.name)
            shutil.copyfile(str(source), str(output / source.name))

        metadata: Dict[str, Any] = {
            "schema_version": "responsibility-experiment-progress-v1",
            "experiment_id": experiment_id,
            "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "machine": args.machine,
            "git_commit": git_commit,
            "git_dirty_at_export": dirty,
            "configs": sorted(config_names),
            "note": args.note,
            "large_artifacts_in_git": False,
        }
        with (output / "metadata.json").open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps({"status": "PASS", "output": str(output), **metadata}, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print("ERROR: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
