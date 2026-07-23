#!/usr/bin/env python3
"""Send opt-in NeuralTrend email alerts for newly published signal changes.

Run this from a Render Cron Job after the market CSV deployment/update finishes:

    python tools/send_signal_change_alerts.py

Use --dry-run to inspect pending transitions without sending email or updating
watchlist baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask_mail import Message  # noqa: E402

from app import (  # noqa: E402
    BASE_URL,
    app,
    can_view_full_signals_for_ticker,
    generate_signal_alert_unsubscribe_token,
    is_paid_user,
    load_epoch_csv_for_ticker,
    mail,
    normalize_ticker,
    signal_label,
)
from extensions import db  # noqa: E402
from models import SignalAlertDelivery, User, WatchlistItem  # noqa: E402

PROCESSING_STALE_AFTER = timedelta(minutes=30)


def make_event_key(user_id, ticker, signal_date, previous_signal, current_signal):
    raw = f"{user_id}:{ticker}:{signal_date.isoformat()}:{previous_signal}:{current_signal}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def claim_delivery(item, signal_date, previous_signal, current_signal):
    event_key = make_event_key(
        item.user_id,
        item.ticker,
        signal_date,
        previous_signal,
        current_signal,
    )
    now = datetime.utcnow()

    delivery = (
        SignalAlertDelivery.query
        .filter_by(event_key=event_key)
        .with_for_update()
        .first()
    )

    if delivery and delivery.processing_status == "sent":
        return delivery, "already_sent"

    if (
        delivery
        and delivery.processing_status == "processing"
        and delivery.processing_started_at
        and now - delivery.processing_started_at < PROCESSING_STALE_AFTER
    ):
        return delivery, "busy"

    if delivery is None:
        delivery = SignalAlertDelivery(
            user_id=item.user_id,
            watchlist_item_id=item.id,
            ticker=item.ticker,
            signal_date=signal_date,
            previous_signal=previous_signal,
            current_signal=current_signal,
            event_key=event_key,
            processing_status="processing",
            processing_started_at=now,
            attempted_at=now,
        )
        db.session.add(delivery)
    else:
        delivery.processing_status = "processing"
        delivery.processing_started_at = now
        delivery.attempted_at = now
        delivery.error_message = None

    db.session.commit()
    return delivery, "claimed"


def build_alert_message(user, ticker, signal_date, previous_signal, current_signal, close_price):
    previous_label = signal_label(previous_signal)
    current_label = signal_label(current_signal)
    unsubscribe_token = generate_signal_alert_unsubscribe_token(user)
    unsubscribe_url = f"{BASE_URL}/signal-alerts/unsubscribe/{unsubscribe_token}"
    dashboard_url = f"{BASE_URL}/#products"
    methodology_url = f"{BASE_URL}/methodology"

    message = Message(
        subject=f"NeuralTrend signal change: {ticker} {current_label}",
        sender=app.config["MAIL_USERNAME"],
        recipients=[user.email],
    )

    message.body = f"""Hi,

The published NeuralTrend signal for {ticker} changed:

{previous_label} -> {current_label}
Signal date: {signal_date.isoformat()}
Reference closing price: ${close_price:,.8f}

This daily signal was generated from completed market data and is not a real-time execution alert, brokerage instruction, or financial advice.

Open NeuralTrend:
{dashboard_url}

Review methodology and simulation assumptions:
{methodology_url}

Disable all signal-change emails while keeping your watchlist:
{unsubscribe_url}

NeuralTrend
"""
    return message


def process_item(item_id, dry_run=False, remaining_limit=200):
    item = db.session.get(WatchlistItem, item_id)
    if not item or not item.email_alert_enabled:
        return 0, 0

    user = db.session.get(User, item.user_id)
    if (
        not user
        or not user.is_verified
        or not is_paid_user(user)
        or not can_view_full_signals_for_ticker(user, item.ticker)
    ):
        if not dry_run:
            item.email_alert_enabled = False
            db.session.commit()
        return 0, 0

    ticker = normalize_ticker(item.ticker)
    market_data = load_epoch_csv_for_ticker(ticker)
    if market_data.empty:
        return 0, 1

    latest_date = market_data.index.max().date()
    latest_signal = int(market_data.iloc[-1]["epoch_signal"])

    if item.last_observed_signal is None or item.last_observed_signal_date is None:
        if dry_run:
            print(f"BASELINE {user.email} {ticker}: {signal_label(latest_signal)} on {latest_date}")
        else:
            locked_item = (
                WatchlistItem.query
                .filter_by(id=item.id)
                .with_for_update()
                .first()
            )
            if locked_item and locked_item.email_alert_enabled:
                locked_item.last_observed_signal = latest_signal
                locked_item.last_observed_signal_date = latest_date
                db.session.commit()
        return 0, 0

    new_rows = market_data[market_data.index.date > item.last_observed_signal_date]
    if new_rows.empty:
        return 0, 0

    previous_signal = int(item.last_observed_signal)
    last_observed_date = item.last_observed_signal_date
    sent_count = 0
    failure_count = 0

    for date_index, row in new_rows.iterrows():
        if sent_count >= remaining_limit:
            break

        signal_date = date_index.date()
        current_signal = int(row["epoch_signal"])
        close_price = float(row["Close"])

        if current_signal not in {-1, 0, 1} or not math.isfinite(close_price) or close_price <= 0:
            print(f"INVALID {ticker} row on {signal_date}", file=sys.stderr)
            failure_count += 1
            break

        if current_signal == previous_signal:
            last_observed_date = signal_date
            continue

        if dry_run:
            print(
                f"SEND {user.email} {ticker} {signal_date}: "
                f"{signal_label(previous_signal)} -> {signal_label(current_signal)}"
            )
            previous_signal = current_signal
            last_observed_date = signal_date
            sent_count += 1
            continue

        # Claim the deterministic transition before contacting SMTP. A second
        # cron worker will see the unique event and not send it concurrently.
        locked_item = (
            WatchlistItem.query
            .filter_by(id=item.id)
            .with_for_update()
            .first()
        )
        if not locked_item or not locked_item.email_alert_enabled:
            db.session.rollback()
            break

        delivery, claim_status = claim_delivery(
            locked_item,
            signal_date,
            previous_signal,
            current_signal,
        )

        if claim_status == "busy":
            db.session.rollback()
            break

        if claim_status == "already_sent":
            locked_item = (
                WatchlistItem.query
                .filter_by(id=item.id)
                .with_for_update()
                .first()
            )
            if locked_item:
                locked_item.last_observed_signal = current_signal
                locked_item.last_observed_signal_date = signal_date
                db.session.commit()
            previous_signal = current_signal
            last_observed_date = signal_date
            continue

        try:
            mail.send(build_alert_message(
                user,
                ticker,
                signal_date,
                previous_signal,
                current_signal,
                close_price,
            ))
        except Exception as error:
            db.session.rollback()
            delivery = (
                SignalAlertDelivery.query
                .filter_by(id=delivery.id)
                .with_for_update()
                .first()
            )
            if delivery:
                delivery.processing_status = "failed"
                delivery.error_message = str(error)[:1000]
                delivery.attempted_at = datetime.utcnow()
                db.session.commit()
            print(
                f"FAILED {user.email} {ticker} {signal_date}: {error}",
                file=sys.stderr,
            )
            failure_count += 1
            break

        db.session.rollback()
        delivery = (
            SignalAlertDelivery.query
            .filter_by(id=delivery.id)
            .with_for_update()
            .first()
        )
        locked_item = (
            WatchlistItem.query
            .filter_by(id=item.id)
            .with_for_update()
            .first()
        )

        if delivery:
            delivery.processing_status = "sent"
            delivery.sent_at = datetime.utcnow()
            delivery.attempted_at = datetime.utcnow()
            delivery.error_message = None

        if locked_item:
            locked_item.last_observed_signal = current_signal
            locked_item.last_observed_signal_date = signal_date

        db.session.commit()

        print(
            f"SENT {user.email} {ticker} {signal_date}: "
            f"{signal_label(previous_signal)} -> {signal_label(current_signal)}"
        )
        previous_signal = current_signal
        last_observed_date = signal_date
        sent_count += 1

    # Advance through no-change rows only after all earlier transitions were
    # handled. This avoids skipping a failed signal-change email.
    if not dry_run and failure_count == 0 and last_observed_date:
        locked_item = (
            WatchlistItem.query
            .filter_by(id=item.id)
            .with_for_update()
            .first()
        )
        if locked_item and locked_item.email_alert_enabled:
            locked_item.last_observed_signal = previous_signal
            locked_item.last_observed_signal_date = last_observed_date
            db.session.commit()

    return sent_count, failure_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    if args.limit < 1 or args.limit > 1000:
        parser.error("--limit must be between 1 and 1000")

    if not args.dry_run and not os.environ.get("EMAIL_USER"):
        print("EMAIL_USER is not configured.", file=sys.stderr)
        return 2

    total_sent = 0
    total_failures = 0

    with app.app_context():
        item_ids = [
            item_id
            for (item_id,) in (
                db.session.query(WatchlistItem.id)
                .filter(WatchlistItem.email_alert_enabled.is_(True))
                .order_by(WatchlistItem.id.asc())
                .all()
            )
        ]

        for item_id in item_ids:
            if total_sent >= args.limit:
                break

            try:
                sent, failures = process_item(
                    item_id,
                    dry_run=args.dry_run,
                    remaining_limit=args.limit - total_sent,
                )
            except Exception as error:
                db.session.rollback()
                print(f"FAILED watchlist_item_id={item_id}: {error}", file=sys.stderr)
                sent, failures = 0, 1

            total_sent += sent
            total_failures += failures

    print(
        f"Signal alert run complete: sent={total_sent} failures={total_failures} "
        f"dry_run={args.dry_run}"
    )
    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
