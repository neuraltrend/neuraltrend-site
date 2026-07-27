#!/usr/bin/env python3
"""Run safe configuration, database and storage checks before launch.

The command does not contact Stripe or send email. It executes a database
SELECT and creates then removes a tiny write-test file in Forward Record
storage.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from sqlalchemy import inspect, text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import (  # noqa: E402
    ADMIN_ONLY_TICKERS,
    FORWARD_RECORD_STORAGE_DIR,
    FORWARD_RECORD_STORAGE_EXPLICIT,
    STRIPE_BILLING_PORTAL_CONFIGURATION_ID,
    STRIPE_PRO_ANNUAL_PRICE_ID,
    STRIPE_PRO_MONTHLY_PRICE_ID,
    STRIPE_WEBHOOK_SECRET,
    app,
    get_epoch_csv_path,
)
from extensions import db  # noqa: E402

REQUIRED_TABLES = {
    "users",
    "stripe_webhook_events",
    "live_simulations",
    "live_simulation_trades",
    "live_simulation_equity",
    "watchlist_items",
    "signal_alert_deliveries",
    "forward_record_assets",
    "forward_publication_batches",
}


def configured(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for name in (
        "SECRET_KEY",
        "DATABASE_URL",
        "BASE_URL",
        "EMAIL_USER",
        "EMAIL_PASS",
        "ADMIN_EMAILS",
        "STRIPE_SECRET_KEY",
    ):
        if not configured(name):
            errors.append(f"Missing required environment variable: {name}")

    if not STRIPE_PRO_MONTHLY_PRICE_ID:
        errors.append("Monthly Stripe Pro Price ID is not configured.")
    if not STRIPE_PRO_ANNUAL_PRICE_ID:
        errors.append("Annual Stripe Pro Price ID is not configured.")
    if (
        STRIPE_PRO_MONTHLY_PRICE_ID
        and STRIPE_PRO_ANNUAL_PRICE_ID
        and STRIPE_PRO_MONTHLY_PRICE_ID == STRIPE_PRO_ANNUAL_PRICE_ID
    ):
        errors.append("Monthly and annual Stripe Price IDs must be different.")
    if not STRIPE_WEBHOOK_SECRET:
        errors.append("STRIPE_WEBHOOK_SECRET is not configured.")
    if not STRIPE_BILLING_PORTAL_CONFIGURATION_ID:
        warnings.append(
            "Stripe Customer Portal plan switching is not configured."
        )

    base_url = os.environ.get("BASE_URL", "")
    if base_url and not base_url.startswith("https://"):
        warnings.append("BASE_URL is not HTTPS.")

    if not configured("REDIS_URL"):
        warnings.append(
            "REDIS_URL is not configured; rate limits may not be shared "
            "across multiple web workers."
        )

    log_format = os.environ.get("NEURALTREND_LOG_FORMAT", "text").strip().lower()
    if log_format not in {"text", "json"}:
        errors.append(
            "NEURALTREND_LOG_FORMAT must be either 'text' or 'json'."
        )

    if not FORWARD_RECORD_STORAGE_EXPLICIT:
        errors.append(
            "FORWARD_RECORD_STORAGE_DIR is not explicitly configured."
        )

    storage = Path(FORWARD_RECORD_STORAGE_DIR)
    try:
        storage.mkdir(parents=True, exist_ok=True)
        test_path = storage / ".neuraltrend-prelaunch-write-test"
        test_path.write_text("ok", encoding="utf-8")
        test_path.unlink()
        usage = shutil.disk_usage(storage)
        free_percent = (usage.free / usage.total * 100.0) if usage.total else 0.0
        if usage.free < 512 * 1024 * 1024 or free_percent < 10.0:
            warnings.append("Forward Record persistent storage is low on free space.")
    except Exception as exc:
        errors.append(f"Forward Record storage is not writable: {exc}")

    with app.app_context():
        try:
            db.session.execute(text("SELECT 1")).scalar()
        except Exception as exc:
            db.session.rollback()
            errors.append(f"Database connectivity check failed: {exc}")
        else:
            table_names = set(inspect(db.engine).get_table_names())
            missing_tables = sorted(REQUIRED_TABLES - table_names)
            if missing_tables:
                errors.append(
                    "Database is missing required tables: "
                    + ", ".join(missing_tables)
                )

    try:
        get_epoch_csv_path("BTC-USD")
    except Exception as exc:
        errors.append(f"BTC-USD market CSV check failed: {exc}")

    if not ADMIN_ONLY_TICKERS:
        warnings.append("No static admin-only tickers are configured.")

    for message in warnings:
        print(f"WARNING {message}")
    for message in errors:
        print(f"ERROR   {message}")

    if errors:
        print(
            f"Prelaunch check failed: errors={len(errors)} "
            f"warnings={len(warnings)}"
        )
        return 1

    print(f"Prelaunch check passed: warnings={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
