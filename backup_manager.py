"""Secure backup creation, verification, listing, retention and deletion helpers."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from tools.forward_record_backup_lib import create_backup, verify_backup

ALLOWED_SUFFIXES = (".dump", ".tar.gz", ".sha256")
DATABASE_PREFIX = "neuraltrend-postgres-"
FORWARD_PREFIX = "neuraltrend-forward-record-"


@dataclass(frozen=True)
class BackupFile:
    name: str
    kind: str
    size: int
    modified_at: datetime
    checksum_available: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksum(path: Path) -> Path:
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    return sidecar


def _postgres_env(database_url: str) -> dict[str, str]:
    parsed = urlparse(database_url)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise ValueError("DATABASE_URL must be a PostgreSQL URL.")
    if not parsed.hostname or not parsed.path.strip("/"):
        raise ValueError("DATABASE_URL is incomplete.")
    env = os.environ.copy()
    env.update({
        "PGHOST": parsed.hostname,
        "PGPORT": str(parsed.port or 5432),
        "PGDATABASE": unquote(parsed.path.lstrip("/")),
        "PGUSER": unquote(parsed.username or ""),
        "PGPASSWORD": unquote(parsed.password or ""),
    })
    query = parse_qs(parsed.query)
    env["PGSSLMODE"] = query.get("sslmode", [env.get("PGSSLMODE", "prefer")])[0]
    return env


def ensure_backup_dir(directory: Path) -> Path:
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        directory.chmod(0o700)
    except OSError:
        pass
    return directory


def create_database_backup(directory: Path, database_url: str) -> Path:
    directory = ensure_backup_dir(directory)
    pg_dump = shutil.which("pg_dump")
    pg_restore = shutil.which("pg_restore")
    if not pg_dump or not pg_restore:
        raise RuntimeError("pg_dump and pg_restore must be installed on the service.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = directory / f"{DATABASE_PREFIX}{timestamp}.dump"
    completed = subprocess.run(
        [pg_dump, "--format=custom", "--compress=9", "--no-owner", "--no-acl", "--file", str(output)],
        env=_postgres_env(database_url), text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        output.unlink(missing_ok=True)
        raise RuntimeError("PostgreSQL backup command failed. Check the Render logs for details.")
    if not output.is_file() or output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError("pg_dump produced an empty backup.")
    verify_database_backup(output)
    _write_checksum(output)
    return output


def verify_database_backup(path: Path) -> dict[str, object]:
    pg_restore = shutil.which("pg_restore")
    if not pg_restore:
        raise RuntimeError("pg_restore is not installed.")
    result = subprocess.run([pg_restore, "--list", str(path)], text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError("Database backup verification failed: " + (result.stderr.strip() or "unknown error"))
    entries = [line for line in result.stdout.splitlines() if line and not line.startswith(";")]
    return {"catalog_entries": len(entries), "sha256": sha256_file(path)}


def create_forward_backup(directory: Path, source_dir: Path, commit: str) -> Path:
    directory = ensure_backup_dir(directory)
    archive, _ = create_backup(source_dir, directory, commit=commit)
    verify_backup(archive)
    _write_checksum(archive)
    return archive


def verify_forward_backup(path: Path) -> dict[str, object]:
    return verify_backup(path)


def classify_name(name: str) -> str | None:
    if name.startswith(DATABASE_PREFIX) and name.endswith(".dump"):
        return "database"
    if name.startswith(FORWARD_PREFIX) and name.endswith(".tar.gz"):
        return "forward_record"
    return None


def resolve_managed_file(directory: Path, name: str, *, allow_checksum: bool = True) -> Path:
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Invalid backup filename.")
    base_name = name[:-7] if name.endswith(".sha256") else name
    if name.endswith(".sha256") and not allow_checksum:
        raise ValueError("Checksum file is not allowed here.")
    if classify_name(base_name) is None:
        raise ValueError("Unrecognized backup filename.")
    directory = ensure_backup_dir(directory)
    path = (directory / name).resolve()
    if path.parent != directory:
        raise ValueError("Invalid backup path.")
    return path


def list_backups(directory: Path) -> list[BackupFile]:
    directory = ensure_backup_dir(directory)
    results: list[BackupFile] = []
    for path in directory.iterdir():
        if not path.is_file() or classify_name(path.name) is None:
            continue
        stat = path.stat()
        results.append(BackupFile(
            name=path.name,
            kind=classify_name(path.name) or "unknown",
            size=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc),
            checksum_available=Path(str(path) + ".sha256").is_file(),
        ))
    return sorted(results, key=lambda item: item.modified_at, reverse=True)


def enforce_retention(directory: Path, keep_per_kind: int) -> list[str]:
    keep_per_kind = max(1, min(int(keep_per_kind), 100))
    removed: list[str] = []
    for kind in ("database", "forward_record"):
        matching = [item for item in list_backups(directory) if item.kind == kind]
        for item in matching[keep_per_kind:]:
            path = resolve_managed_file(directory, item.name)
            sidecar = Path(str(path) + ".sha256")
            path.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            removed.append(item.name)
    return removed


def delete_backup(directory: Path, name: str) -> None:
    path = resolve_managed_file(directory, name, allow_checksum=False)
    if not path.is_file():
        raise FileNotFoundError(name)
    sidecar = Path(str(path) + ".sha256")
    path.unlink()
    sidecar.unlink(missing_ok=True)
