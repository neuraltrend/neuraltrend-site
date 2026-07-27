#!/usr/bin/env python3
"""Safely restore approved Forward Record files from a verified archive.

Dry-run is the default. Applying a restore requires both --apply and the exact
confirmation phrase. The current live directory is backed up first.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import FORWARD_RECORD_STORAGE_DIR  # noqa: E402
from tools.forward_record_backup_lib import (  # noqa: E402
    create_backup,
    extract_verified_backup,
    verify_backup,
)

CONFIRMATION = "RESTORE-FORWARD-RECORD"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument(
        "--safety-backup-dir",
        default="/tmp/neuraltrend-backups",
    )
    args = parser.parse_args()

    archive = Path(args.archive)
    manifest = verify_backup(archive)
    destination = Path(FORWARD_RECORD_STORAGE_DIR).resolve()

    print(f"Verified archive: files={manifest['file_count']} bytes={manifest['total_bytes']}")
    print(f"Restore destination: {destination}")

    if not args.apply:
        print("DRY RUN ONLY: no files changed.")
        print(
            f"To apply, repeat with --apply --confirmation {CONFIRMATION}"
        )
        return 0

    if args.confirmation != CONFIRMATION:
        print("ERROR exact confirmation phrase is required.")
        return 2

    safety_archive, _ = create_backup(
        destination,
        Path(args.safety_backup_dir),
        commit=os.environ.get("RENDER_GIT_COMMIT", "unknown"),
    )
    print(f"Safety backup created: {safety_archive}")

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = Path(tempfile.mkdtemp(prefix=".forward-restore-", dir=parent))
    extracted = staging / "new"
    old = parent / f"{destination.name}.pre-restore-{timestamp}"

    try:
        extract_verified_backup(archive, extracted)
        if old.exists():
            raise FileExistsError(f"Safety directory already exists: {old}")
        destination.rename(old)
        extracted.rename(destination)
        shutil.rmtree(old)
        print("PASS Forward Record files restored.")
        print("NEXT Run python tools/operational_check.py")
        return 0
    except Exception:
        if not destination.exists() and old.exists():
            old.rename(destination)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
