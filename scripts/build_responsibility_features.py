#!/usr/bin/env python3
"""Build responsibility stock states and rule components from local data."""

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

from responsibility.preprocessing.responsibility_features import (  # noqa: E402
    DATASET_BENCHMARKS,
    DEFAULT_CHUNK_ROWS,
    SEGMENTS,
    build_responsibility_feature_store,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_BENCHMARKS))
    parser.add_argument(
        "--sampler-dir",
        required=True,
        type=Path,
        help="directory containing train.pkl, valid.pkl and test.pkl",
    )
    parser.add_argument("--qlib-provider", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--benchmark",
        default=None,
        help="optional consistency check; it must match the frozen dataset benchmark",
    )
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = build_responsibility_feature_store(
        dataset=args.dataset,
        sampler_dir=args.sampler_dir,
        qlib_provider=args.qlib_provider,
        output_root=args.output_root,
        benchmark=args.benchmark,
        chunk_rows=args.chunk_rows,
    )
    print(
        json.dumps(
            {
                "status": "DONE",
                "dataset": manifest["dataset_id"],
                "output_root": str(args.output_root.expanduser().resolve()),
                "rows": {
                    segment: manifest["segments"][segment]["rows"]
                    for segment in SEGMENTS
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
