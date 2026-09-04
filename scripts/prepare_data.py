#!/usr/bin/env python3
"""Build all frozen Responsibility inputs from one pinned local Qlib provider."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data"
DATASETS = ("short_csi300", "short_csi800_direct")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        type=Path,
        required=True,
        help="local Qlib cn_data provider, normally data/source/qlib_bin",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
    )
    parser.add_argument("--chunk-rows", type=int, default=262_144)
    parser.add_argument(
        "--skip-file-sha256",
        action="store_true",
        help="make the final manifest faster by recording sizes without payload hashes",
    )
    return parser.parse_args(argv)


def run(arguments: Sequence[str]) -> None:
    command = [sys.executable, "-B"] + [str(value) for value in arguments]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(REPOSITORY_ROOT), check=True)


def preflight(data_root: Path, provider: Path, datasets: Sequence[str]) -> None:
    if not provider.is_dir():
        raise NotADirectoryError(provider)
    try:
        provider.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(
            "--provider must be inside --data-root so the finished dataset remains portable"
        ) from exc
    if (data_root / "manifest.json").exists():
        raise FileExistsError(
            "data/manifest.json already exists; use a fresh data-root for a rebuild"
        )
    conflicts: List[Path] = []
    for dataset in datasets:
        conflicts.extend(
            [
                data_root / "samplers" / dataset,
                data_root / "responsibility" / dataset,
                data_root / "direction_scales" / (dataset + ".json"),
            ]
        )
    existing = [str(path) for path in conflicts if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to mix a new build with existing outputs: " + ", ".join(existing)
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.chunk_rows <= 0:
            raise ValueError("--chunk-rows must be positive")
        data_root = args.data_root.expanduser().resolve()
        data_root.mkdir(parents=True, exist_ok=True)
        provider = args.provider.expanduser().resolve(strict=True)
        datasets = tuple(args.datasets)
        if len(datasets) != len(set(datasets)):
            raise ValueError("datasets must not contain duplicates")
        preflight(data_root, provider, datasets)

        run(
            [
                REPOSITORY_ROOT / "scripts/build_samplers.py",
                "--provider",
                provider,
                "--output-root",
                data_root / "samplers",
                "--datasets",
                *datasets,
            ]
        )
        for dataset in datasets:
            sampler_dir = data_root / "samplers" / dataset
            feature_root = data_root / "responsibility" / dataset
            run(
                [
                    REPOSITORY_ROOT / "scripts/build_responsibility_features.py",
                    "--dataset",
                    dataset,
                    "--sampler-dir",
                    sampler_dir,
                    "--qlib-provider",
                    provider,
                    "--output-root",
                    feature_root,
                    "--chunk-rows",
                    str(args.chunk_rows),
                ]
            )
            run(
                [
                    REPOSITORY_ROOT / "scripts/build_direction_scales.py",
                    "--dataset",
                    dataset,
                    "--sampler-dir",
                    sampler_dir,
                    "--feature-root",
                    feature_root,
                    "--output",
                    data_root / "direction_scales" / (dataset + ".json"),
                ]
            )

        manifest_command: List[object] = [
            REPOSITORY_ROOT / "scripts/build_data_manifest.py",
            "--data-root",
            data_root,
            "--provider",
            provider,
            "--datasets",
            *datasets,
        ]
        if args.skip_file_sha256:
            manifest_command.append("--skip-file-sha256")
        run([str(value) for value in manifest_command])

        verify_command: List[object] = [
            REPOSITORY_ROOT / "scripts/verify_data.py",
            "--data-root",
            data_root,
        ]
        if args.skip_file_sha256:
            verify_command.append("--quick")
        for dataset in datasets:
            verify_command.extend(["--dataset", dataset])
        run([str(value) for value in verify_command])
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "data_root": str(data_root),
                    "provider": str(provider),
                    "datasets": list(datasets),
                    "next": "python scripts/train.py --config configs/csi300.yaml "
                    "--data-root data --mode smoke --seed 0 --gpu 0 "
                    "--output-dir outputs/smoke_csi300_seed0",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except subprocess.CalledProcessError as error:
        print(
            "ERROR: data build command failed with exit code %d; existing partial "
            "outputs were preserved" % error.returncode,
            file=sys.stderr,
        )
        return int(error.returncode or 1)
    except Exception as error:
        print("ERROR: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
