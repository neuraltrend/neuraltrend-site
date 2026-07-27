#!/usr/bin/env python3
"""Verify the manifest, paths, sizes and SHA-256 hashes in a backup."""

from __future__ import annotations

import argparse
from pathlib import Path

from forward_record_backup_lib import verify_backup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    args = parser.parse_args()

    manifest = verify_backup(Path(args.archive))
    print(f"PASS format={manifest['format']}")
    print(f"PASS files={manifest['file_count']} bytes={manifest['total_bytes']}")
    print(f"PASS sha256={manifest['archive_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
