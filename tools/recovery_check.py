#!/usr/bin/env python3
"""Verify internal dependencies and public routes after rollback or recovery."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app, build_operational_status  # noqa: E402

PATHS = ("/healthz", "/", "/dashboard", "/performance", "/subscription")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_url",
        nargs="?",
        default="https://neuraltrend.org",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    failures = 0
    with app.app_context():
        operational = build_operational_status()
    print(f"{'PASS' if operational['status'] != 'error' else 'FAIL'} internal_status={operational['status']}")
    if operational["status"] == "error":
        failures += 1

    base = args.base_url.rstrip("/") + "/"
    session = requests.Session()
    session.headers["User-Agent"] = "NeuralTrend-Recovery-Check/1.0"

    for path in PATHS:
        url = urljoin(base, path.lstrip("/"))
        try:
            response = session.get(url, timeout=args.timeout, allow_redirects=True)
            ok = response.status_code == 200
            if path == "/healthz" and ok:
                ok = response.json().get("status") == "ok"
            if path != "/healthz" and ok:
                ok = bool(response.headers.get("X-Request-ID"))
            print(f"{'PASS' if ok else 'FAIL'} {response.status_code:3d} {url}")
            failures += 0 if ok else 1
        except Exception as exc:
            failures += 1
            print(f"FAIL --- {url}: {exc}")

    print(f"Recovery check complete: failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
