#!/usr/bin/env python3
"""Verify that pg_restore can read a PostgreSQL custom-format backup."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    args = parser.parse_args()

    archive = Path(args.archive).resolve()
    if not archive.is_file() or archive.stat().st_size == 0:
        print("ERROR backup file is missing or empty.")
        return 2

    pg_restore = shutil.which("pg_restore")
    if not pg_restore:
        print("ERROR pg_restore is not installed.")
        return 2

    result = subprocess.run(
        [pg_restore, "--list", str(archive)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        print("ERROR backup catalog verification failed.")
        print(result.stderr.strip())
        return result.returncode

    catalog_lines = [line for line in result.stdout.splitlines() if line and not line.startswith(";")]
    print(f"PASS catalog_entries={len(catalog_lines)}")
    print(f"PASS bytes={archive.stat().st_size}")
    print(f"PASS sha256={sha256_file(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
