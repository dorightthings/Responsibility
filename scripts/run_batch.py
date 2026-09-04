#!/usr/bin/env python3
"""Launch the released multi-dataset, multi-seed Responsibility experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from responsibility.training.config import load_experiment_config  # noqa: E402
from responsibility.training.io import atomic_json, utc_now  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=REPO_ROOT / "data"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--provider",
        type=Path,
        default=None,
        help="optional Qlib provider override for the formal backtest",
    )
    parser.add_argument("--cpu-data-cache", action="store_true")
    parser.add_argument("--allow-other-qlib-version", action="store_true")
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="train/predict only; AR/IR can be produced later with evaluate.py",
    )
    return parser.parse_args(argv)


def _command_record(item: MappingLike) -> Dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if key not in {"process", "log_handle"}
    }


MappingLike = Dict[str, Any]


def run(args: argparse.Namespace) -> int:
    if not args.gpus or any(gpu < 0 for gpu in args.gpus):
        raise ValueError("gpus must contain non-negative physical CUDA indices")
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("gpus contain duplicates")
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds contain duplicates")
    if args.jobs_per_gpu <= 0:
        raise ValueError("jobs-per-gpu must be positive")
    configs = []
    for path in args.configs:
        config = load_experiment_config(path)
        configs.append((Path(config.source_path), config))
    datasets = [config.dataset for _, config in configs]
    if len(set(datasets)) != len(datasets):
        raise ValueError("configs contain duplicate dataset identifiers")

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError("refusing to overwrite existing batch: %s" % output_root)
    output_root.mkdir(parents=True)
    log_root = output_root / "logs"
    log_root.mkdir()
    data_root = args.data_root.expanduser().resolve(strict=True)
    slots = [gpu for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
    queue = [
        (config_path, config.dataset, seed)
        for config_path, config in configs
        for seed in args.seeds
    ]
    trainer = REPO_ROOT / "scripts/train.py"
    active: List[MappingLike] = []
    finished: List[MappingLike] = []
    started_utc = utc_now()
    atomic_json(
        output_root / "batch_manifest.json",
        {
            "status": "RUNNING",
            "started_utc": started_utc,
            "configs": [str(path) for path, _ in configs],
            "datasets": datasets,
            "seeds": args.seeds,
            "physical_gpus": args.gpus,
            "jobs_per_gpu": args.jobs_per_gpu,
            "mode": args.mode,
            "data_root": str(data_root),
        },
    )

    while queue or active:
        while queue and len(active) < len(slots):
            used_slots = {int(item["slot"]) for item in active}
            slot = next(index for index in range(len(slots)) if index not in used_slots)
            physical_gpu = slots[slot]
            config_path, dataset, seed = queue.pop(0)
            name = "%s_seed%d" % (dataset, seed)
            job_dir = output_root / "jobs" / name
            log_path = log_root / (name + ".log")
            command = [
                sys.executable,
                str(trainer),
                "--config",
                str(config_path),
                "--data-root",
                str(data_root),
                "--seed",
                str(seed),
                # The child sees only its assigned physical GPU.
                "--gpu",
                "0",
                "--mode",
                args.mode,
                "--output-dir",
                str(job_dir),
            ]
            if args.cpu_data_cache:
                command.append("--cpu-data-cache")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active.append(
                {
                    "name": name,
                    "dataset": dataset,
                    "seed": seed,
                    "physical_gpu": physical_gpu,
                    "visible_gpu": 0,
                    "slot": slot,
                    "pid": process.pid,
                    "command": command,
                    "log": str(log_path),
                    "output": str(job_dir),
                    "process": process,
                    "log_handle": handle,
                    "started_utc": utc_now(),
                }
            )
            print(
                "started %s on physical GPU %d, pid=%d"
                % (name, physical_gpu, process.pid),
                flush=True,
            )

        time.sleep(1)
        remaining = []
        for item in active:
            returncode = item["process"].poll()
            if returncode is None:
                remaining.append(item)
                continue
            item["log_handle"].close()
            record = _command_record(item)
            record["returncode"] = int(returncode)
            record["finished_utc"] = utc_now()
            finished.append(record)
            print("finished %s, returncode=%d" % (item["name"], returncode), flush=True)
        active = remaining
        atomic_json(
            output_root / "batch_progress.json",
            {
                "status": "RUNNING",
                "queued": len(queue),
                "active": [_command_record(item) for item in active],
                "finished": finished,
                "updated_utc": utc_now(),
            },
        )

    failures = sum(record["returncode"] != 0 for record in finished)
    auxiliary_commands = []
    # A smoke run intentionally stops after two epochs and never performs a
    # portfolio backtest.  Formal batches evaluate every successful job.
    if args.mode == "formal" and not args.skip_evaluation:
        evaluate_command = [
            sys.executable,
            str(REPO_ROOT / "scripts/evaluate.py"),
            "--run-root",
            str(output_root),
            "--data-root",
            str(data_root),
        ]
        if args.provider is not None:
            evaluate_command.extend(
                ["--provider", str(args.provider.expanduser().resolve())]
            )
        if args.allow_other_qlib_version:
            evaluate_command.append("--allow-other-qlib-version")
        evaluation_log = log_root / "evaluate.log"
        with evaluation_log.open("w", encoding="utf-8") as handle:
            code = subprocess.run(
                evaluate_command,
                cwd=str(REPO_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        auxiliary_commands.append(
            {
                "name": "evaluate",
                "command": evaluate_command,
                "log": str(evaluation_log),
                "returncode": int(code),
            }
        )
        failures += int(code != 0)

    summary_command = [
        sys.executable,
        str(REPO_ROOT / "scripts/summarize.py"),
        "--run-root",
        str(output_root),
        "--output-dir",
        str(output_root / "summary"),
    ]
    summary_log = log_root / "summarize.log"
    with summary_log.open("w", encoding="utf-8") as handle:
        code = subprocess.run(
            summary_command,
            cwd=str(REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    auxiliary_commands.append(
        {
            "name": "summarize",
            "command": summary_command,
            "log": str(summary_log),
            "returncode": int(code),
        }
    )
    failures += int(code != 0)
    result = {
        "status": "DONE" if failures == 0 else "DONE_WITH_FAILURES",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "mode": args.mode,
        "datasets": datasets,
        "seeds": args.seeds,
        "physical_gpus": args.gpus,
        "jobs": finished,
        "post_commands": auxiliary_commands,
    }
    atomic_json(output_root / "batch_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if failures == 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
