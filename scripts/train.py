#!/usr/bin/env python3
"""Train one ResponsibilityModel seed and write predictions/IC metrics."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from responsibility.training.config import load_experiment_config  # noqa: E402
from responsibility.training.engine import train_one_seed  # noqa: E402
from responsibility.training.io import atomic_json, utc_now  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="directory containing manifest.json and local data assets",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="visible CUDA index; CPU is used when CUDA is unavailable",
    )
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cpu-data-cache",
        action="store_true",
        help="keep the sampler cache on CPU for lower GPU-memory use",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    existed_before = output_dir.exists()
    try:
        config = load_experiment_config(args.config)
        result = train_one_seed(
            config,
            data_root=args.data_root,
            seed=args.seed,
            gpu=args.gpu,
            mode=args.mode,
            output_dir=output_dir,
            cache_on_gpu=not args.cpu_data_cache,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "status": "FAILED",
            "failed_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if not existed_before and output_dir.is_dir():
            atomic_json(output_dir / "FAILED.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
