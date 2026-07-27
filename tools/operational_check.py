#!/usr/bin/env python3
"""Print a read-only operational summary from the real NeuralTrend environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app, build_operational_status  # noqa: E402


def human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} TiB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero for warnings as well as errors.",
    )
    args = parser.parse_args()

    with app.app_context():
        status = build_operational_status()

    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print(f"STATUS {status['status'].upper()}")
        print(f"Generated: {status['generated_at']}")
        print(
            "Database: "
            f"{status['database']['status']} "
            f"users={status['database'].get('counts', {}).get('users', 'unknown')} "
            f"active_pro={status['database'].get('counts', {}).get('active_pro_users', 'unknown')}"
        )
        print(
            "Processing: "
            f"failed_webhooks_24h={status['processing'].get('failed_webhooks_24h', 'unknown')} "
            f"stale_webhooks={status['processing'].get('stale_webhooks', 'unknown')} "
            f"failed_alerts_24h={status['processing'].get('failed_alerts_24h', 'unknown')} "
            f"stale_alerts={status['processing'].get('stale_alerts', 'unknown')}"
        )
        print(
            "Storage: "
            f"{status['storage']['status']} "
            f"files={status['storage'].get('file_count', 'unknown')} "
            f"forward_bytes={human_bytes(status['storage'].get('used_by_forward_record_bytes'))} "
            f"disk_free={human_bytes(status['storage'].get('disk_free_bytes'))} "
            f"disk_free_percent={status['storage'].get('disk_free_percent', 'unknown')}"
        )
        print(
            "Market data: "
            f"{status['market_data']['status']} "
            f"btc_latest={status['market_data'].get('btc_latest_date', 'unknown')} "
            f"age_days={status['market_data'].get('age_days', 'unknown')}"
        )
        for message in status["warnings"]:
            print(f"WARNING {message}")
        for message in status["errors"]:
            print(f"ERROR   {message}")

    if status["status"] == "error":
        return 1
    if args.strict and status["status"] == "warning":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
