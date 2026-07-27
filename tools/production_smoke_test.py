#!/usr/bin/env python3
"""Read-only production smoke test for a deployed NeuralTrend site.

This script performs only GET requests. It never logs in, sends email, opens
Stripe Checkout, publishes Forward Record data, or changes user information.
"""

from __future__ import annotations

import argparse
import sys
from urllib.parse import urljoin

import requests

DEFAULT_PATHS = (
    "/healthz",
    "/",
    "/dashboard",
    "/subscription",
    "/methodology",
    "/performance",
    "/privacy",
    "/terms",
    "/risk-disclaimer",
    "/refund-policy",
    "/robots.txt",
    "/sitemap.xml",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_url",
        nargs="?",
        default="https://neuraltrend.org",
        help="Deployment root, for example https://neuraltrend.org",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--include-summary",
        action="store_true",
        help="Also exercise the heavier 1Y Signal Overview summary endpoint.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/") + "/"
    paths = list(DEFAULT_PATHS)
    if args.include_summary:
        paths.append("/signals/summary?duration=1y")

    failures = 0
    session = requests.Session()
    session.headers["User-Agent"] = "NeuralTrend-Production-Smoke-Test/1.0"

    for path in paths:
        url = urljoin(base_url, path.lstrip("/"))
        try:
            response = session.get(
                url,
                timeout=args.timeout,
                allow_redirects=True,
            )
            status_ok = response.status_code == 200
            if path == "/healthz" and status_ok:
                payload = response.json()
                status_ok = payload.get("status") == "ok"
            label = "PASS" if status_ok else "FAIL"
            print(f"{label} {response.status_code:3d} {url}")
            failures += 0 if status_ok else 1
        except Exception as exc:
            failures += 1
            print(f"FAIL --- {url}: {exc}")

    print(
        f"Smoke test complete: checks={len(paths)} failures={failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
