"""Shared, dependency-free Forward Record backup helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST_NAME = "manifest.json"
ARCHIVE_ROOT = "forward_record"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_name(name: str) -> bool:
    pure = PurePosixPath(name)
    return (
        bool(name)
        and not pure.is_absolute()
        and ".." not in pure.parts
        and "" not in pure.parts
    )


def inventory(source: Path) -> list[dict[str, Any]]:
    source = source.resolve()
    records: list[dict[str, Any]] = []
    if not source.is_dir():
        raise FileNotFoundError(f"Forward Record directory does not exist: {source}")

    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Refusing to back up symbolic link: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        archive_path = f"{ARCHIVE_ROOT}/{relative}"
        records.append({
            "path": archive_path,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return records


def create_backup(
    source: Path,
    output_dir: Path,
    *,
    commit: str = "unknown",
) -> tuple[Path, dict[str, Any]]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    if output_dir == source or source in output_dir.parents:
        raise ValueError("Backup output directory must be outside Forward Record storage.")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = output_dir / f"neuraltrend-forward-record-{timestamp}.tar.gz"
    counter = 1
    while archive.exists():
        archive = output_dir / (
            f"neuraltrend-forward-record-{timestamp}-{counter}.tar.gz"
        )
        counter += 1

    records = inventory(source)
    manifest = {
        "format": "neuraltrend-forward-record-backup-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit[:40] if commit else "unknown",
        "file_count": len(records),
        "total_bytes": sum(item["size"] for item in records),
        "files": records,
    }

    with tempfile.TemporaryDirectory(prefix="neuraltrend-forward-backup-") as temp:
        temp_root = Path(temp)
        manifest_path = temp_root / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        with tarfile.open(archive, "w:gz") as tar:
            tar.add(manifest_path, arcname=MANIFEST_NAME, recursive=False)
            for record in records:
                relative = PurePosixPath(record["path"]).relative_to(ARCHIVE_ROOT)
                source_path = source.joinpath(*relative.parts)
                tar.add(source_path, arcname=record["path"], recursive=False)

    manifest["archive_sha256"] = sha256_file(archive)
    return archive, manifest


def verify_backup(archive: Path) -> dict[str, Any]:
    archive = archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Backup archive not found: {archive}")

    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        by_name = {member.name: member for member in members}

        for member in members:
            if not _safe_member_name(member.name):
                raise ValueError(f"Unsafe archive path: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Unsupported archive member type: {member.name}")

        manifest_member = by_name.get(MANIFEST_NAME)
        if manifest_member is None or not manifest_member.isfile():
            raise ValueError("Backup manifest is missing.")
        handle = tar.extractfile(manifest_member)
        if handle is None:
            raise ValueError("Backup manifest cannot be read.")
        manifest = json.loads(handle.read().decode("utf-8"))

        if manifest.get("format") != "neuraltrend-forward-record-backup-v1":
            raise ValueError("Unsupported Forward Record backup format.")

        expected_paths = {MANIFEST_NAME}
        total_bytes = 0
        for record in manifest.get("files", []):
            name = str(record.get("path", ""))
            if not name.startswith(f"{ARCHIVE_ROOT}/") or not _safe_member_name(name):
                raise ValueError(f"Invalid file path in manifest: {name}")
            member = by_name.get(name)
            if member is None or not member.isfile():
                raise ValueError(f"Backup file is missing: {name}")
            extracted = tar.extractfile(member)
            if extracted is None:
                raise ValueError(f"Backup file cannot be read: {name}")
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            if size != int(record["size"]):
                raise ValueError(f"Size mismatch for {name}")
            if digest.hexdigest() != record["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {name}")
            expected_paths.add(name)
            total_bytes += size

        actual_files = {member.name for member in members if member.isfile()}
        unexpected = sorted(actual_files - expected_paths)
        if unexpected:
            raise ValueError(
                "Unexpected file(s) in archive: " + ", ".join(unexpected)
            )

        if len(manifest.get("files", [])) != int(manifest.get("file_count", -1)):
            raise ValueError("Manifest file count is inconsistent.")
        if total_bytes != int(manifest.get("total_bytes", -1)):
            raise ValueError("Manifest total byte count is inconsistent.")

    manifest["archive_sha256"] = sha256_file(archive)
    return manifest


def extract_verified_backup(archive: Path, destination: Path) -> dict[str, Any]:
    manifest = verify_backup(archive)
    destination.mkdir(parents=True, exist_ok=False)

    with tarfile.open(archive, "r:gz") as tar:
        for record in manifest["files"]:
            member = tar.getmember(record["path"])
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Backup file cannot be read: {record['path']}")
            relative = PurePosixPath(record["path"]).relative_to(ARCHIVE_ROOT)
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                shutil.copyfileobj(source, output)

    return manifest
