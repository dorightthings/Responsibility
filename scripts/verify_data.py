#!/usr/bin/env python3
"""Validate Responsibility data layout, sizes, shapes and optional hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data"
SCHEMA_VERSION = "responsibility-data-manifest-v1"
SEGMENTS = ("train", "valid", "test")
EXPECTED_DATASETS = ("short_csi300", "short_csi800_direct")
BUFFER_SIZE = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="default: <data-root>/manifest.json",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=EXPECTED_DATASETS,
        help="validate one dataset; repeat for more than one (default: both)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip SHA-256 computation, but still check paths, sizes and NPY shapes",
    )
    return parser.parse_args()


def relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("%s must be a non-empty POSIX relative path" % label)
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("unsafe %s: %r" % (label, value))
    return Path(*pure.parts)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampler_path(value: Any, label: str) -> Path:
    if isinstance(value, str):
        return relative_path(value, label)
    if isinstance(value, Mapping):
        return relative_path(value.get("path"), label + ".path")
    raise ValueError("%s must be a path string or an object with path" % label)


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            "data manifest is missing: %s (see data/README.md)" % path
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "manifest schema_version must be %r" % SCHEMA_VERSION
        )
    if not isinstance(payload.get("datasets"), dict):
        raise ValueError("manifest.datasets must be an object")
    return payload


def dataset_required_paths(
    manifest: Mapping[str, Any], datasets: Sequence[str]
) -> Set[Path]:
    result: Set[Path] = set()
    table = manifest["datasets"]
    for dataset in datasets:
        record = table.get(dataset)
        if not isinstance(record, Mapping):
            raise ValueError("manifest.datasets lacks %s" % dataset)
        samplers = record.get("samplers")
        if not isinstance(samplers, Mapping):
            raise ValueError("datasets.%s.samplers must be an object" % dataset)
        for segment in SEGMENTS:
            if segment not in samplers:
                raise ValueError(
                    "datasets.%s.samplers lacks %s" % (dataset, segment)
                )
            result.add(
                sampler_path(
                    samplers[segment],
                    "datasets.%s.samplers.%s" % (dataset, segment),
                )
            )
        responsibility = relative_path(
            record.get("responsibility_root"),
            "datasets.%s.responsibility_root" % dataset,
        )
        result.add(responsibility / "manifest.json")
        for segment in SEGMENTS:
            result.add(responsibility / segment / "stock_state.npy")
            result.add(responsibility / segment / "rule_components.npy")
        result.add(
            relative_path(
                record.get("direction_scale"),
                "datasets.%s.direction_scale" % dataset,
            )
        )
    return result


def provider_required_paths(manifest: Mapping[str, Any]) -> Tuple[Set[Path], Path]:
    provider = manifest.get("qlib_provider")
    if not isinstance(provider, Mapping):
        raise ValueError("manifest.qlib_provider must be an object")
    root = relative_path(provider.get("path"), "qlib_provider.path")
    provider_manifest = relative_path(
        provider.get("manifest"), "qlib_provider.manifest"
    )
    anchors = {
        provider_manifest,
        root / "calendars/day.txt",
        root / "instruments/csi300.txt",
        root / "instruments/csi800.txt",
        root / "features/sh000300/close.day.bin",
        root / "features/sh000906/close.day.bin",
    }
    return anchors, root


def inventory_records(manifest: Mapping[str, Any]) -> Dict[Path, Mapping[str, Any]]:
    raw = manifest.get("files", [])
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError("manifest.files must be a list")
    records: Dict[Path, Mapping[str, Any]] = {}
    for index, record in enumerate(raw):
        if not isinstance(record, Mapping):
            raise ValueError("manifest.files[%d] must be an object" % index)
        path = relative_path(record.get("path"), "files[%d].path" % index)
        if path in records:
            raise ValueError("duplicate manifest file record: %s" % path)
        records[path] = record
    return records


def validate_record(
    data_root: Path,
    relative: Path,
    record: Optional[Mapping[str, Any]],
    quick: bool,
) -> Tuple[int, bool]:
    path = data_root / relative
    kind = "file" if record is None else str(record.get("kind", "file"))
    if kind == "directory":
        if not path.is_dir():
            raise FileNotFoundError("required directory is missing: %s" % path)
        return 0, False
    if kind != "file":
        raise ValueError("unsupported kind %r for %s" % (kind, relative))
    if not path.is_file():
        raise FileNotFoundError("required file is missing: %s" % path)

    size = path.stat().st_size
    if record is not None and record.get("bytes") is not None:
        expected_size = record["bytes"]
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
            raise ValueError("invalid bytes field for %s" % relative)
        if size != expected_size:
            raise ValueError(
                "size mismatch for %s: expected %d, got %d"
                % (relative, expected_size, size)
            )

    if record is not None and record.get("shape") is not None:
        expected_shape = record["shape"]
        if not isinstance(expected_shape, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in expected_shape
        ):
            raise ValueError("invalid shape field for %s" % relative)
        if path.suffix != ".npy":
            raise ValueError("shape checking is supported only for .npy: %s" % relative)
        array = np.load(str(path), mmap_mode="r", allow_pickle=False)
        if list(array.shape) != expected_shape:
            raise ValueError(
                "shape mismatch for %s: expected %s, got %s"
                % (relative, expected_shape, list(array.shape))
            )
        expected_dtype = record.get("dtype")
        if expected_dtype is not None and str(array.dtype) != str(expected_dtype):
            raise ValueError(
                "dtype mismatch for %s: expected %s, got %s"
                % (relative, expected_dtype, array.dtype)
            )

    digest_checked = False
    if record is not None and record.get("sha256") not in (None, ""):
        expected_digest = str(record["sha256"]).lower()
        if len(expected_digest) != 64 or any(
            character not in "0123456789abcdef" for character in expected_digest
        ):
            raise ValueError("invalid sha256 field for %s" % relative)
        if not quick:
            actual = file_sha256(path)
            if actual != expected_digest:
                raise ValueError(
                    "SHA-256 mismatch for %s: expected %s, got %s"
                    % (relative, expected_digest, actual)
                )
            digest_checked = True
    return size, digest_checked


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else data_root / "manifest.json"
    )
    manifest = load_manifest(manifest_path)
    datasets = tuple(args.dataset or EXPECTED_DATASETS)
    required = dataset_required_paths(manifest, datasets)
    provider_anchors, provider_root = provider_required_paths(manifest)
    required.update(provider_anchors)
    records = inventory_records(manifest)

    errors: List[str] = []
    checked_files = 0
    checked_bytes = 0
    checked_hashes = 0
    paths_to_check = set(required)
    paths_to_check.update(
        path
        for path, record in records.items()
        if bool(record.get("required", True)) or (data_root / path).exists()
    )
    for relative in sorted(paths_to_check, key=str):
        try:
            size, digest_checked = validate_record(
                data_root, relative, records.get(relative), args.quick
            )
            if (data_root / relative).is_file():
                checked_files += 1
                checked_bytes += size
            checked_hashes += int(digest_checked)
        except Exception as error:
            errors.append(str(error))

    if not (data_root / provider_root).is_dir():
        errors.append("Qlib provider root is missing: %s" % (data_root / provider_root))

    summary = {
        "status": "PASS" if not errors else "FAIL",
        "schema_version": manifest.get("schema_version"),
        "data_version": manifest.get("data_version"),
        "datasets": list(datasets),
        "data_root": str(data_root),
        "manifest": str(manifest_path),
        "quick": bool(args.quick),
        "checked_files": checked_files,
        "checked_bytes": checked_bytes,
        "checked_hashes": checked_hashes,
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("ERROR: %s" % error, file=sys.stderr)
        raise SystemExit(1)
