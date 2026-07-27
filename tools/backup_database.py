#!/usr/bin/env python3
"""Create a verified PostgreSQL custom-format backup without printing secrets."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def postgres_env(database_url: str) -> dict[str, str]:
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
    if query.get("sslmode"):
        env["PGSSLMODE"] = query["sslmode"][0]
    else:
        env.setdefault("PGSSLMODE", "prefer")
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="/tmp/neuraltrend-backups",
        help="Directory that will receive the .dump file.",
    )
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        print("ERROR DATABASE_URL is not configured.")
        return 2

    pg_dump = shutil.which("pg_dump")
    pg_restore = shutil.which("pg_restore")
    if not pg_dump:
        print("ERROR pg_dump is not installed in this environment.")
        return 2

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_dir / f"neuraltrend-postgres-{timestamp}.dump"

    env = postgres_env(database_url)
    command = [
        pg_dump,
        "--format=custom",
        "--compress=9",
        "--no-owner",
        "--no-acl",
        "--file",
        str(output),
    ]

    print("Creating PostgreSQL backup...")
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        output.unlink(missing_ok=True)
        print("ERROR pg_dump failed.")
        if completed.stderr:
            # pg_dump errors should not contain the password because the URL is
            # not passed on the command line.
            print(completed.stderr.strip())
        return completed.returncode

    if not output.is_file() or output.stat().st_size == 0:
        print("ERROR pg_dump produced an empty backup.")
        return 1

    if pg_restore:
        verify = subprocess.run(
            [pg_restore, "--list", str(output)],
            text=True,
            capture_output=True,
            check=False,
        )
        if verify.returncode:
            print("ERROR pg_restore could not read the new backup.")
            print(verify.stderr.strip())
            return verify.returncode
        print("PASS pg_restore verified the archive catalog.")
    else:
        print("WARNING pg_restore is unavailable; archive catalog was not verified.")

    print(f"PASS archive={output}")
    print(f"PASS bytes={output.stat().st_size}")
    print(f"PASS sha256={sha256_file(output)}")
    print(
        "IMPORTANT Copy this backup off Render. Test restoration into a separate "
        "temporary database before relying on it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
