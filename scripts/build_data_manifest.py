#!/usr/bin/env python3
"""Create a portable manifest for a locally built Responsibility dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data"
SCHEMA_VERSION = "responsibility-data-manifest-v1"
DATASETS = ("short_csi300", "short_csi800_direct")
SEGMENTS = ("train", "valid", "test")
SOURCE_URL = (
    "https://github.com/chenditc/investment_data/releases/download/"
    "2024-12-07/qlib_bin.tar.gz"
)
SOURCE_ARCHIVE_SHA256 = (
    "5f6ec2448070d73834dcbba27a778462451359334757876815ff2795d2358b93"
)
BUFFER_SIZE = 8 * 1024 * 1024


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--provider",
        type=Path,
        required=True,
        help="Qlib provider used by training/backtest; it must be inside data-root",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
    )
    parser.add_argument(
        "--data-version", default="responsibility-short-built-v1"
    )
    parser.add_argument("--archive-url", default="")
    parser.add_argument("--archive-sha256", default="")
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument(
        "--source-archive-sha256", default=SOURCE_ARCHIVE_SHA256
    )
    parser.add_argument(
        "--skip-file-sha256",
        action="store_true",
        help="record sizes and NPY schemas without hashing every generated file",
    )
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: %s" % path)
    return value


def relative_posix(path: Path, root: Path) -> str:
    resolved = path.expanduser().resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("path must be inside data-root: %s" % resolved) from exc
    pure = PurePosixPath(relative.as_posix())
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("unsafe relative path: %s" % relative)
    return pure.as_posix()


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("refusing to overwrite manifest: %s" % path)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.rename(path)


def known_sampler_hash(dataset_dir: Path, segment: str) -> Optional[str]:
    metadata_path = dataset_dir / "build_metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = read_json(metadata_path)
    try:
        record = metadata["stage_results"][segment]["pickle"]
    except (KeyError, TypeError):
        return None
    if record.get("relative_path") != "%s.pkl" % segment:
        return None
    value = str(record.get("sha256", "")).lower()
    return value if len(value) == 64 else None


def file_record(
    path: Path,
    root: Path,
    include_sha256: bool,
    known_sha256: Optional[str] = None,
    required: bool = True,
) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    record: Dict[str, Any] = {
        "path": relative_posix(path, root),
        "bytes": int(path.stat().st_size),
        "required": bool(required),
    }
    if path.suffix == ".npy":
        array = np.load(str(path), mmap_mode="r", allow_pickle=False)
        record["shape"] = [int(value) for value in array.shape]
        record["dtype"] = str(array.dtype)
    if include_sha256:
        record["sha256"] = known_sha256 or file_sha256(path)
    return record


def provider_anchors(provider: Path, selected: Sequence[str]) -> List[Path]:
    values = [provider / "calendars/day.txt"]
    if "short_csi300" in selected:
        values.append(provider / "instruments/csi300.txt")
    if "short_csi800_direct" in selected:
        values.append(provider / "instruments/csi800.txt")
    values.extend(
        [
            provider / "features/sh000300/close.day.bin",
            provider / "features/sh000906/close.day.bin",
        ]
    )
    for path in values:
        if not path.is_file():
            raise FileNotFoundError("Qlib provider anchor is missing: %s" % path)
    return values


def ensure_provider_manifest(
    data_root: Path,
    provider: Path,
    selected: Sequence[str],
    source_url: str,
    source_sha256: str,
) -> Path:
    manifest_path = provider.parent / "PROVIDER_MANIFEST.json"
    anchors = provider_anchors(provider, selected)
    payload = {
        "schema_version": "responsibility-provider-manifest-v1",
        "purpose": "frozen data construction and Top30/Drop30 backtest",
        "source_url": source_url,
        "source_archive_sha256": source_sha256,
        "provider_path": relative_posix(provider, data_root),
        "path_is_relative_to_data_root": True,
        "control_files": [
            {
                "path": path.relative_to(provider).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
            for path in anchors
        ],
    }
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing != payload:
            raise FileExistsError(
                "provider manifest already exists with different content: %s"
                % manifest_path
            )
    else:
        write_json_new(manifest_path, payload)
    return manifest_path


def build_manifest(args: argparse.Namespace) -> Mapping[str, Any]:
    data_root = args.data_root.expanduser().resolve(strict=True)
    selected = tuple(args.datasets)
    if len(selected) != len(set(selected)):
        raise ValueError("datasets must not contain duplicates")
    provider = args.provider.expanduser().resolve(strict=True)
    provider_relative = relative_posix(provider, data_root)
    provider_manifest = ensure_provider_manifest(
        data_root,
        provider,
        selected,
        args.source_url,
        args.source_archive_sha256,
    )

    records: List[Dict[str, Any]] = []
    datasets: Dict[str, Any] = {}
    include_sha = not args.skip_file_sha256
    for dataset in selected:
        sampler_dir = data_root / "samplers" / dataset
        responsibility_root = data_root / "responsibility" / dataset
        scale_path = data_root / "direction_scales" / (dataset + ".json")
        sampler_paths = {
            segment: sampler_dir / (segment + ".pkl") for segment in SEGMENTS
        }
        datasets[dataset] = {
            "samplers": {
                segment: relative_posix(path, data_root)
                for segment, path in sampler_paths.items()
            },
            "responsibility_root": relative_posix(
                responsibility_root, data_root
            ),
            "direction_scale": relative_posix(scale_path, data_root),
        }
        for segment, path in sampler_paths.items():
            records.append(
                file_record(
                    path,
                    data_root,
                    include_sha,
                    known_sampler_hash(sampler_dir, segment),
                )
            )
        for optional in (sampler_dir / "build_metadata.json", sampler_dir / "DONE"):
            if optional.is_file():
                records.append(
                    file_record(optional, data_root, include_sha, required=False)
                )

        sidecar_manifest = responsibility_root / "manifest.json"
        records.append(file_record(sidecar_manifest, data_root, include_sha))
        for segment in SEGMENTS:
            for name in ("stock_state.npy", "rule_components.npy"):
                records.append(
                    file_record(
                        responsibility_root / segment / name,
                        data_root,
                        include_sha,
                    )
                )
        records.append(file_record(scale_path, data_root, include_sha))

    anchors = provider_anchors(provider, selected)
    records.append(file_record(provider_manifest, data_root, include_sha))
    records.extend(file_record(path, data_root, include_sha) for path in anchors)
    records.sort(key=lambda value: value["path"])
    return {
        "schema_version": SCHEMA_VERSION,
        "data_version": args.data_version,
        "archive_url": args.archive_url,
        "archive_sha256": args.archive_sha256,
        "datasets": datasets,
        "qlib_provider": {
            "path": provider_relative,
            "manifest": relative_posix(provider_manifest, data_root),
        },
        "files": records,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        output = args.data_root.expanduser().resolve() / "manifest.json"
        if output.exists():
            raise FileExistsError("refusing to overwrite manifest: %s" % output)
        payload = build_manifest(args)
        write_json_new(output, payload)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "manifest": str(output),
                    "datasets": list(args.datasets),
                    "files": len(payload["files"]),
                    "file_sha256_recorded": not args.skip_file_sha256,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as error:
        print("ERROR: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
