#!/usr/bin/env python3
"""Create and verify a portable backup of approved Forward Record files."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import FORWARD_RECORD_STORAGE_DIR  # noqa: E402
from tools.forward_record_backup_lib import (  # noqa: E402
    create_backup,
    verify_backup,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="/tmp/neuraltrend-backups",
        help="Directory outside Forward Record storage.",
    )
    args = parser.parse_args()

    archive, created = create_backup(
        Path(FORWARD_RECORD_STORAGE_DIR),
        Path(args.output_dir),
        commit=os.environ.get("RENDER_GIT_COMMIT", "unknown"),
    )
    verified = verify_backup(archive)
    print(f"PASS archive={archive}")
    print(f"PASS files={verified['file_count']} bytes={verified['total_bytes']}")
    print(f"PASS sha256={verified['archive_sha256']}")
    print(
        "IMPORTANT Copy this archive off the Render service/disk. "
        "A backup stored only beside the live data is not disaster recovery."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
