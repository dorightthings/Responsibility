#!/usr/bin/env python3
"""Download and safely extract one Responsibility data archive.

The repository deliberately does not contain a fixed data URL.  Supply a
versioned HTTP(S) or file URL and, whenever available, its SHA-256 digest.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = REPOSITORY_ROOT / "data"
BUFFER_SIZE = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="HTTP(S) or file URL")
    parser.add_argument(
        "--sha256",
        default=None,
        help="optional expected SHA-256 of the downloaded archive",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="extraction root (default: <repository>/data)",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="local archive name when it cannot be inferred from the URL",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="download and verify the archive without extracting it",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="retain the downloaded archive under data/.downloads/",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def normalize_sha256(value: Optional[str]) -> Optional[str]:
    if value is None or not value.strip():
        return None
    digest = value.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("--sha256 must contain exactly 64 hexadecimal characters")
    return digest


def infer_filename(url: str, override: Optional[str]) -> str:
    candidate = override or Path(urllib.parse.urlparse(url).path).name
    if not candidate or candidate in {".", ".."}:
        raise ValueError("cannot infer archive name; pass --filename")
    if Path(candidate).name != candidate or "/" in candidate or "\\" in candidate:
        raise ValueError("--filename must be a plain filename")
    return candidate


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, expected_sha256: Optional[str], timeout: float) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file():
            raise FileExistsError(target)
        existing = file_sha256(target)
        if expected_sha256 is not None and existing == expected_sha256:
            print("Using already downloaded archive: %s" % target)
            return existing
        raise FileExistsError(
            "download target already exists and cannot be safely reused: %s" % target
        )

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Responsibility-data-downloader/0.1"},
    )
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=target.name + ".", suffix=".part", dir=str(target.parent), delete=False
        ) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                shutil.copyfileobj(response, output, length=BUFFER_SIZE)
        actual = file_sha256(temporary)
        if expected_sha256 is not None and actual != expected_sha256:
            raise ValueError(
                "archive SHA-256 mismatch: expected %s, got %s"
                % (expected_sha256, actual)
            )
        os.replace(str(temporary), str(target))
        temporary = None
        return actual
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def safe_relative_path(name: str) -> Path:
    if not name or "\\" in name:
        raise ValueError("unsafe archive member: %r" % name)
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("unsafe archive member: %r" % name)
    return Path(*pure.parts)


def zip_members(archive: zipfile.ZipFile) -> Iterable[Tuple[zipfile.ZipInfo, Path]]:
    seen = set()
    for member in archive.infolist():
        relative = safe_relative_path(member.filename.rstrip("/"))
        if relative in seen:
            raise ValueError("duplicate archive member: %s" % relative)
        seen.add(relative)
        unix_mode = member.external_attr >> 16
        if stat.S_ISLNK(unix_mode):
            raise ValueError("symbolic links are not allowed in data archives")
        yield member, relative


def extract_zip(archive_path: Path, staging: Path) -> None:
    with zipfile.ZipFile(str(archive_path), "r") as archive:
        members = list(zip_members(archive))
        for member, relative in members:
            target = staging / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=BUFFER_SIZE)


def tar_members(archive: tarfile.TarFile) -> Iterable[Tuple[tarfile.TarInfo, Path]]:
    seen = set()
    for member in archive.getmembers():
        relative = safe_relative_path(member.name.rstrip("/"))
        if relative in seen:
            raise ValueError("duplicate archive member: %s" % relative)
        seen.add(relative)
        if member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise ValueError("links and special files are not allowed in data archives")
        if not member.isdir() and not member.isfile():
            raise ValueError("unsupported archive member: %s" % member.name)
        yield member, relative


def extract_tar(archive_path: Path, staging: Path) -> None:
    with tarfile.open(str(archive_path), "r:*") as archive:
        members = list(tar_members(archive))
        for member, relative in members:
            target = staging / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("cannot read archive member: %s" % member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=BUFFER_SIZE)


def preflight_merge(staging: Path, destination: Path) -> None:
    for source in staging.rglob("*"):
        if source.is_symlink():
            raise ValueError("staging tree unexpectedly contains a symbolic link")
        relative = source.relative_to(staging)
        target = destination / relative
        if source.is_file() and target.exists():
            raise FileExistsError("refusing to overwrite existing data file: %s" % target)
        if source.is_dir() and target.exists() and not target.is_dir():
            raise FileExistsError("directory/file conflict while extracting: %s" % target)


def merge_staging(staging: Path, destination: Path) -> int:
    preflight_merge(staging, destination)
    directories = sorted(
        (path for path in staging.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
    )
    for source in directories:
        (destination / source.relative_to(staging)).mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in staging.rglob("*") if path.is_file())
    for source in files:
        target = destination / source.relative_to(staging)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(source), str(target))
    return len(files)


def extract_archive(archive_path: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".responsibility-extract-", dir=str(destination.parent)
    ) as temporary:
        staging = Path(temporary)
        lower = archive_path.name.lower()
        if lower.endswith(".zip"):
            extract_zip(archive_path, staging)
        elif lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
            extract_tar(archive_path, staging)
        else:
            raise ValueError(
                "unsupported archive type; use zip, tar, tar.gz, tar.bz2 or tar.xz"
            )
        return merge_staging(staging, destination)


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    expected = normalize_sha256(args.sha256)
    destination = args.destination.expanduser().resolve()
    filename = infer_filename(args.url, args.filename)
    archive_path = destination / ".downloads" / filename

    actual = download(args.url, archive_path, expected, args.timeout)
    print("Archive: %s" % archive_path)
    print("SHA-256: %s" % actual)

    if args.no_extract:
        print("Download completed; extraction was skipped.")
        return 0

    file_count = extract_archive(archive_path, destination)
    print("Extracted %d files into %s" % (file_count, destination))
    if not args.keep_archive:
        archive_path.unlink()
    print("Next: python scripts/verify_data.py --data-root %s" % destination)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print("ERROR: %s" % error, file=sys.stderr)
        raise SystemExit(1)
