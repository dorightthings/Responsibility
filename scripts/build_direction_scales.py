#!/usr/bin/env python3
"""Build the four train-only responsibility direction scales."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from responsibility.preprocessing.direction_scale import build_direction_scale  # noqa: E402
from responsibility.preprocessing.responsibility_features import (  # noqa: E402
    DATASET_BENCHMARKS,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_BENCHMARKS))
    parser.add_argument(
        "--sampler-dir",
        required=True,
        type=Path,
        help="directory containing train.pkl",
    )
    parser.add_argument("--feature-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = build_direction_scale(
        dataset=args.dataset,
        sampler_dir=args.sampler_dir,
        feature_root=args.feature_root,
        output=args.output,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
