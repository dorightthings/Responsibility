#!/usr/bin/env python3
"""Pack one local Responsibility data tree for transfer through private storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data"
BUFFER_SIZE = 8 * 1024 * 1024


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new .tar.gz path, for example ../责任机制股票预测_实验数据包_v2.tar.gz",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="skip the quick layout check before creating the archive",
    )
    parser.add_argument(
        "--compresslevel",
        type=int,
        default=3,
        choices=range(1, 10),
        help="gzip compression level (default: 3 for faster private transfer packing)",
    )
    return parser.parse_args(argv)


def read_manifest(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("data manifest must be a JSON object")
    if value.get("schema_version") != "responsibility-data-manifest-v1":
        raise ValueError("unsupported data manifest schema")
    return value


def relative_path(value: Any) -> Path:
    raw = value.get("path") if isinstance(value, Mapping) else value
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError("invalid relative path: %r" % raw)
    path = Path(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("unsafe relative path: %s" % path)
    return path


def transfer_roots(manifest: Mapping[str, Any]) -> Set[Path]:
    result: Set[Path] = {Path("manifest.json")}
    datasets = manifest.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        raise ValueError("manifest.datasets must be a non-empty object")
    for dataset, value in datasets.items():
        if not isinstance(value, Mapping):
            raise ValueError("invalid dataset record: %s" % dataset)
        samplers = value.get("samplers")
        if not isinstance(samplers, Mapping):
            raise ValueError("dataset %s lacks samplers" % dataset)
        for record in samplers.values():
            result.add(relative_path(record))
        result.add(relative_path(value.get("responsibility_root")))
        result.add(relative_path(value.get("direction_scale")))
    provider = manifest.get("qlib_provider")
    if not isinstance(provider, Mapping):
        raise ValueError("manifest.qlib_provider must be an object")
    result.add(relative_path(provider.get("path")))
    result.add(relative_path(provider.get("manifest")))
    return result


def expand_files(data_root: Path, roots: Iterable[Path]) -> List[Path]:
    files: Set[Path] = set()
    for relative in roots:
        source = data_root / relative
        if source.is_symlink():
            raise ValueError("symbolic links are not allowed: %s" % source)
        if source.is_file():
            files.add(source)
            continue
        if not source.is_dir():
            raise FileNotFoundError(source)
        for path in source.rglob("*"):
            if path.is_symlink():
                raise ValueError("symbolic links are not allowed: %s" % path)
            if path.is_file():
                files.add(path)
    return sorted(files, key=lambda path: path.relative_to(data_root).as_posix())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Remove machine-specific ownership and timestamp metadata."""

    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def validate(data_root: Path) -> None:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/verify_data.py"),
        "--data-root",
        str(data_root),
        "--quick",
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError("data validation failed; archive was not created")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        data_root = args.data_root.expanduser().resolve(strict=True)
        output = args.output.expanduser().resolve()
        if output.suffixes[-2:] != [".tar", ".gz"]:
            raise ValueError("--output must end in .tar.gz")
        partial = output.with_suffix(output.suffix + ".partial")
        sidecar = output.with_name(output.name + ".sha256")
        if output.exists() or partial.exists() or sidecar.exists():
            raise FileExistsError("refusing to overwrite archive: %s" % output)
        try:
            output.relative_to(data_root)
        except ValueError:
            pass
        else:
            raise ValueError("archive output must be outside data-root")
        if not args.skip_validation:
            validate(data_root)
        manifest = read_manifest(data_root / "manifest.json")
        files = expand_files(data_root, transfer_roots(manifest))
        if not files:
            raise ValueError("no data files selected")
        output.parent.mkdir(parents=True, exist_ok=True)
        payload_bytes = sum(path.stat().st_size for path in files)
        with tarfile.open(
            str(partial), mode="w:gz", compresslevel=args.compresslevel
        ) as archive:
            for path in files:
                archive.add(
                    str(path),
                    arcname=path.relative_to(data_root).as_posix(),
                    recursive=False,
                    filter=portable_tar_info,
                )
        os.replace(str(partial), str(output))
        digest = file_sha256(output)
        with sidecar.open("x", encoding="utf-8") as handle:
            handle.write("%s  %s\n" % (digest, output.name))
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "archive": str(output),
                    "sha256": digest,
                    "sha256_file": str(sidecar),
                    "files": len(files),
                    "payload_bytes": payload_bytes,
                    "archive_bytes": int(output.stat().st_size),
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
