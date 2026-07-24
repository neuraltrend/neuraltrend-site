#!/usr/bin/env python3
"""Preview or manually dispatch opt-in NeuralTrend signal-change alerts.

Nothing is sent unless an administrator explicitly chooses ``--send`` or uses
NeuralTrend's admin dispatch page. A command with no mode exits without sending.

The dispatcher:
- processes each watched asset independently from its current published CSV;
- combines delayed multi-row updates into one catch-up email per asset/user;
- detects changed signals on already observed dates as revisions;
- ignores consecutive identical signals;
- checks subscription/access and the user's alert preference at dispatch time;
- deduplicates delivery across retries and concurrent dispatch attempts.

Use ``--preview`` before ``--send``. Manual dispatch uses the current stable
read of each file immediately; it does not impose a 60-minute quiet period.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import sys
from contextlib import nullcontext
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
    get_epoch_csv_path,
    is_paid_user,
    load_epoch_csv_for_ticker,
    mail,
    make_signal_row_fingerprint,
    normalize_ticker,
    signal_label,
)
from extensions import db  # noqa: E402
from models import SignalAlertDelivery, User, WatchlistItem  # noqa: E402

PROCESSING_STALE_AFTER = timedelta(minutes=30)
DEFAULT_STABILITY_MINUTES = 0
MAX_TRANSITIONS_IN_EMAIL = 20


def utcnow():
    return datetime.utcnow()


def parse_stability_minutes(cli_value=None):
    """Validate an optional explicit quiet period.

    Manual dispatch defaults to zero minutes. The old environment-based
    60-minute scheduler setting is intentionally ignored so an administrator's
    approved dispatch is not delayed unexpectedly.
    """
    raw = DEFAULT_STABILITY_MINUTES if cli_value is None else cli_value

    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError("Stability minutes must be a whole number.")

    if value < 0 or value > 1440:
        raise ValueError("Signal-alert stability minutes must be between 0 and 1440.")

    return value


def load_stable_market_snapshot(ticker, stability_minutes, market_cache):
    """Load one CSV only after its file timestamp has remained quiet.

    The before/after stat check also avoids processing a file that changed while
    pandas was reading it. Results are cached for the duration of one worker run
    so many users watching the same asset share one consistent snapshot.
    """
    ticker = normalize_ticker(ticker)
    if ticker in market_cache:
        return market_cache[ticker]

    try:
        csv_path = get_epoch_csv_path(ticker)
        stat_before = os.stat(csv_path)
    except (OSError, ValueError, FileNotFoundError) as error:
        result = {
            "status": "error",
            "error": str(error),
            "data": None,
            "mtime_utc": None,
            "source_identity": f"error:{error}",
        }
        market_cache[ticker] = result
        return result

    modified_at = datetime.utcfromtimestamp(stat_before.st_mtime)
    age = utcnow() - modified_at
    quiet_period = timedelta(minutes=stability_minutes)

    if age < quiet_period:
        wait_seconds = max(0, int((quiet_period - age).total_seconds()))
        result = {
            "status": "waiting",
            "data": None,
            "mtime_utc": modified_at,
            "wait_seconds": wait_seconds,
            "source_identity": (
                f"{csv_path}:{stat_before.st_mtime_ns}:{stat_before.st_size}:waiting"
            ),
        }
        market_cache[ticker] = result
        return result

    try:
        market_data = load_epoch_csv_for_ticker(ticker).sort_index()
        stat_after = os.stat(csv_path)
    except Exception as error:
        result = {
            "status": "error",
            "error": str(error),
            "data": None,
            "mtime_utc": modified_at,
            "source_identity": (
                f"{csv_path}:{stat_before.st_mtime_ns}:{stat_before.st_size}:error"
            ),
        }
        market_cache[ticker] = result
        return result

    if (
        stat_before.st_mtime_ns != stat_after.st_mtime_ns
        or stat_before.st_size != stat_after.st_size
    ):
        result = {
            "status": "waiting",
            "data": None,
            "mtime_utc": datetime.utcfromtimestamp(stat_after.st_mtime),
            "wait_seconds": stability_minutes * 60,
            "reason": "changed_during_read",
            "source_identity": (
                f"{csv_path}:{stat_after.st_mtime_ns}:{stat_after.st_size}:changed"
            ),
        }
        market_cache[ticker] = result
        return result

    result = {
        "status": "stable",
        "data": market_data,
        "mtime_utc": datetime.utcfromtimestamp(stat_after.st_mtime),
        "wait_seconds": 0,
        "source_identity": (
            f"{csv_path}:{stat_after.st_mtime_ns}:{stat_after.st_size}:stable"
        ),
    }
    market_cache[ticker] = result
    return result


def row_for_date(market_data, signal_date):
    matching = market_data[market_data.index.date == signal_date]
    if matching.empty:
        return None
    return matching.iloc[-1]


def validate_row(ticker, signal_date, row):
    current_signal = int(row["epoch_signal"])
    close_price = float(row["Close"])

    if current_signal not in {-1, 0, 1}:
        raise ValueError(f"Invalid {ticker} signal on {signal_date}.")
    if not math.isfinite(close_price) or close_price <= 0:
        raise ValueError(f"Invalid {ticker} closing price on {signal_date}.")

    fingerprint = make_signal_row_fingerprint(
        ticker,
        signal_date,
        current_signal,
        close_price,
    )
    return current_signal, close_price, fingerprint


def canonical_event_summary(event):
    return json.dumps(
        {
            "event_type": event["event_type"],
            "observation_start_date": event["observation_start_date"].isoformat(),
            "observation_end_date": event["observation_end_date"].isoformat(),
            "initial_signal": event["initial_signal"],
            "final_signal": event["final_signal"],
            "final_close_price": format(event["final_close_price"], ".12g"),
            "final_fingerprint": event["final_fingerprint"],
            "transitions": [
                {
                    "kind": transition["kind"],
                    "date": transition["date"].isoformat(),
                    "previous_signal": transition["previous_signal"],
                    "current_signal": transition["current_signal"],
                    "close_price": format(transition["close_price"], ".12g"),
                }
                for transition in event["transitions"]
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def make_event_key(item, event):
    raw = (
        f"{item.user_id}:{item.ticker}:"
        f"{canonical_event_summary(event)}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_pending_event(item, market_data):
    """Return baseline advancement and, when needed, one email event.

    Multiple newly dated rows are intentionally consolidated into one catch-up
    event. This avoids sending stale BUY/SELL emails one by one after a delayed
    data refresh while still disclosing every actual transition in the digest.
    """
    ticker = normalize_ticker(item.ticker)

    if market_data.empty:
        raise ValueError("Market data has no valid rows for this asset.")

    latest_index = market_data.index.max()
    latest_date = latest_index.date()
    latest_row = market_data.loc[latest_index]
    if hasattr(latest_row, "iloc") and getattr(latest_row, "ndim", 1) > 1:
        latest_row = latest_row.iloc[-1]
    latest_signal, latest_close, latest_fingerprint = validate_row(
        ticker,
        latest_date,
        latest_row,
    )

    if item.last_observed_signal is None or item.last_observed_signal_date is None:
        return {
            "baseline_only": True,
            "final_signal": latest_signal,
            "final_date": latest_date,
            "final_fingerprint": latest_fingerprint,
            "event": None,
        }

    initial_signal = int(item.last_observed_signal)
    initial_date = item.last_observed_signal_date
    previous_signal = initial_signal
    current_baseline_fingerprint = item.last_observed_row_fingerprint
    final_date = initial_date
    final_signal = initial_signal
    final_fingerprint = current_baseline_fingerprint
    transitions = []
    revision_detected = False

    baseline_row = row_for_date(market_data, initial_date)
    if baseline_row is not None:
        baseline_signal, baseline_close, baseline_fingerprint = validate_row(
            ticker,
            initial_date,
            baseline_row,
        )

        # Existing Step 4 users have a NULL fingerprint until this upgraded
        # worker sees them once. Establishing it is not itself an alert.
        if current_baseline_fingerprint is None:
            # Existing Step 4 rows do not yet have a fingerprint, but their
            # stored signal/date are still enough to identify a signal revision.
            if baseline_signal != initial_signal:
                revision_detected = True
                transitions.append({
                    "kind": "revision",
                    "date": initial_date,
                    "previous_signal": initial_signal,
                    "current_signal": baseline_signal,
                    "close_price": baseline_close,
                })
                previous_signal = baseline_signal
                final_signal = baseline_signal
            current_baseline_fingerprint = baseline_fingerprint
        elif baseline_fingerprint != current_baseline_fingerprint:
            revision_detected = True
            if baseline_signal != initial_signal:
                transitions.append({
                    "kind": "revision",
                    "date": initial_date,
                    "previous_signal": initial_signal,
                    "current_signal": baseline_signal,
                    "close_price": baseline_close,
                })
            # A close-only correction updates the fingerprint silently. A signal
            # correction becomes a specially labelled revision event.
            previous_signal = baseline_signal
            final_signal = baseline_signal

        final_fingerprint = baseline_fingerprint
    else:
        print(
            f"WARNING {ticker}: previously observed date {initial_date} "
            "is no longer present; same-date revision comparison was skipped.",
            file=sys.stderr,
        )

    new_rows = market_data[market_data.index.date > initial_date]
    new_row_count = len(new_rows)

    for date_index, row in new_rows.iterrows():
        signal_date = date_index.date()
        current_signal, close_price, row_fingerprint = validate_row(
            ticker,
            signal_date,
            row,
        )

        if current_signal != previous_signal:
            transitions.append({
                "kind": "change",
                "date": signal_date,
                "previous_signal": previous_signal,
                "current_signal": current_signal,
                "close_price": close_price,
            })

        previous_signal = current_signal
        final_signal = current_signal
        final_date = signal_date
        final_fingerprint = row_fingerprint
        latest_close = close_price

    # No later rows: use the baseline row's current close for a revision email
    # or for a silent fingerprint refresh.
    if new_row_count == 0 and baseline_row is not None:
        final_date = initial_date
        final_signal, latest_close, final_fingerprint = validate_row(
            ticker,
            initial_date,
            baseline_row,
        )

    advancement = {
        "baseline_only": False,
        "final_signal": final_signal,
        "final_date": final_date,
        "final_fingerprint": final_fingerprint,
        "event": None,
    }

    if not transitions:
        return advancement

    if (
        len(transitions) == 1
        and transitions[0]["kind"] == "revision"
        and new_row_count == 0
    ):
        event_type = "revision"
    elif (
        len(transitions) == 1
        and transitions[0]["kind"] == "change"
        and new_row_count == 1
        and not revision_detected
    ):
        event_type = "change"
    else:
        event_type = "catch_up"

    if revision_detected:
        observation_start_date = initial_date
    elif new_row_count:
        observation_start_date = new_rows.index.min().date()
    else:
        observation_start_date = transitions[0]["date"]

    event = {
        "event_type": event_type,
        "observation_start_date": observation_start_date,
        "observation_end_date": final_date,
        "initial_signal": initial_signal,
        "final_signal": final_signal,
        "final_close_price": latest_close,
        "final_fingerprint": final_fingerprint,
        "transitions": transitions,
        "new_row_count": new_row_count,
        "includes_revision": any(t["kind"] == "revision" for t in transitions),
    }
    advancement["event"] = event
    return advancement


def claim_delivery(item, event):
    event_key = make_event_key(item, event)
    now = utcnow()

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

    summary = canonical_event_summary(event)

    if delivery is None:
        delivery = SignalAlertDelivery(
            user_id=item.user_id,
            watchlist_item_id=item.id,
            ticker=item.ticker,
            signal_date=event["observation_end_date"],
            previous_signal=event["initial_signal"],
            current_signal=event["final_signal"],
            event_type=event["event_type"],
            observation_start_date=event["observation_start_date"],
            observation_end_date=event["observation_end_date"],
            change_count=len(event["transitions"]),
            source_row_fingerprint=event["final_fingerprint"],
            event_summary=summary,
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
        delivery.event_summary = summary
        delivery.source_row_fingerprint = event["final_fingerprint"]

    db.session.commit()
    return delivery, "claimed"


def format_transition_line(transition):
    prefix = "REVISED" if transition["kind"] == "revision" else transition["date"].isoformat()
    if transition["kind"] == "revision":
        prefix = f"{transition['date'].isoformat()} (revision)"
    return (
        f"{prefix}: {signal_label(transition['previous_signal'])} -> "
        f"{signal_label(transition['current_signal'])} "
        f"at ${transition['close_price']:,.8f}"
    )


def build_alert_message(user, ticker, event):
    unsubscribe_token = generate_signal_alert_unsubscribe_token(user)
    unsubscribe_url = f"{BASE_URL}/signal-alerts/unsubscribe/{unsubscribe_token}"
    dashboard_url = f"{BASE_URL}/#products"
    methodology_url = f"{BASE_URL}/methodology"
    performance_url = f"{BASE_URL}/performance?ticker={ticker}"
    final_label = signal_label(event["final_signal"])

    if event["event_type"] == "change":
        transition = event["transitions"][0]
        subject = f"NeuralTrend signal change: {ticker} {final_label}"
        heading = f"The published NeuralTrend signal for {ticker} changed:"
        detail_block = (
            f"{signal_label(transition['previous_signal'])} -> {final_label}\n"
            f"Signal date: {transition['date'].isoformat()}\n"
            f"Reference closing price: ${transition['close_price']:,.8f}"
        )
    elif event["event_type"] == "revision":
        transition = event["transitions"][0]
        subject = f"NeuralTrend signal revision: {ticker} {final_label}"
        heading = (
            f"A previously published NeuralTrend signal for {ticker} was revised:"
        )
        detail_block = (
            f"Signal date: {transition['date'].isoformat()}\n"
            f"{signal_label(transition['previous_signal'])} -> {final_label}\n"
            f"Reference closing price: ${transition['close_price']:,.8f}\n\n"
            "This is a revision to an already observed signal date, not a new "
            "next-day transition."
        )
    else:
        subject = (
            f"NeuralTrend catch-up: {ticker} now {final_label} "
            f"({len(event['transitions'])} change"
            f"{'s' if len(event['transitions']) != 1 else ''})"
        )
        heading = (
            f"NeuralTrend published multiple new or revised rows for {ticker}. "
            "They are combined here so older changes are not sent as separate "
            "real-time-looking emails."
        )
        displayed = event["transitions"][:MAX_TRANSITIONS_IN_EMAIL]
        transition_lines = "\n".join(format_transition_line(t) for t in displayed)
        if len(event["transitions"]) > len(displayed):
            transition_lines += (
                f"\n... plus {len(event['transitions']) - len(displayed)} "
                "additional changes."
            )
        detail_block = (
            f"Published data range: {event['observation_start_date'].isoformat()} "
            f"through {event['observation_end_date'].isoformat()}\n"
            f"Current published signal: {final_label}\n"
            f"Latest reference closing price: ${event['final_close_price']:,.8f}\n\n"
            f"Signal path:\n{transition_lines}"
        )

    message = Message(
        subject=subject,
        sender=app.config["MAIL_USERNAME"],
        recipients=[user.email],
    )

    message.body = f"""Hi,

{heading}

{detail_block}

This alert was released after an administrator approved the currently published data snapshot. This daily signal information was generated from completed market data. Delayed/catch-up entries describe their original signal dates and should not be interpreted as real-time execution alerts, brokerage instructions, or financial advice.

Open NeuralTrend:
{dashboard_url}

View the prospective Forward Record:
{performance_url}

Review methodology and simulation assumptions:
{methodology_url}

Disable all signal-change emails while keeping your watchlist:
{unsubscribe_url}

NeuralTrend
"""
    return message


def baseline_matches(item, original):
    return (
        item.last_observed_signal == original["signal"]
        and item.last_observed_signal_date == original["date"]
        and item.last_observed_row_fingerprint == original["fingerprint"]
    )


def advance_baseline(item_id, advancement, original=None):
    locked_item = (
        WatchlistItem.query
        .filter_by(id=item_id)
        .with_for_update()
        .first()
    )
    if not locked_item or not locked_item.email_alert_enabled:
        db.session.rollback()
        return False

    if original is not None and not baseline_matches(locked_item, original):
        db.session.rollback()
        return False

    locked_item.last_observed_signal = advancement["final_signal"]
    locked_item.last_observed_signal_date = advancement["final_date"]
    locked_item.last_observed_row_fingerprint = advancement["final_fingerprint"]
    db.session.commit()
    return True


def process_item(
    item_id,
    *,
    dry_run=False,
    stability_minutes=DEFAULT_STABILITY_MINUTES,
    market_cache=None,
    mail_connection=None,
    approved_state=None,
):
    if market_cache is None:
        market_cache = {}

    item = db.session.get(WatchlistItem, item_id)
    if not item or not item.email_alert_enabled:
        return 0, 0, 0

    user = db.session.get(User, item.user_id)

    if approved_state is not None:
        current_state = {
            "item_id": item.id,
            "user_id": item.user_id,
            "ticker": item.ticker,
            "email_alert_enabled": bool(item.email_alert_enabled),
            "last_observed_signal": item.last_observed_signal,
            "last_observed_signal_date": (
                item.last_observed_signal_date.isoformat()
                if item.last_observed_signal_date else None
            ),
            "last_observed_row_fingerprint": item.last_observed_row_fingerprint,
            "user_email": user.email if user else None,
            "user_is_verified": bool(user.is_verified) if user else False,
            "user_subscription_type": user.subscription_type if user else None,
            "user_subscription_status": user.subscription_status if user else None,
        }
        if current_state != approved_state:
            print(
                f"SKIP-CHANGED watchlist_item_id={item_id}: "
                "state changed after approval; preview again."
            )
            return 0, 0, 0

    if (
        not user
        or not user.is_verified
        or not is_paid_user(user)
        or not can_view_full_signals_for_ticker(user, item.ticker)
    ):
        if not dry_run:
            item.email_alert_enabled = False
            db.session.commit()
        return 0, 0, 0

    ticker = normalize_ticker(item.ticker)
    snapshot = load_stable_market_snapshot(
        ticker,
        stability_minutes,
        market_cache,
    )

    if snapshot["status"] == "waiting":
        print(
            f"WAIT {ticker}: source file has not been stable for "
            f"{stability_minutes} minute(s)."
        )
        return 0, 0, 1

    if snapshot["status"] == "error":
        print(f"FAILED {ticker}: {snapshot['error']}", file=sys.stderr)
        return 0, 1, 0

    market_data = snapshot["data"]
    try:
        advancement = build_pending_event(item, market_data)
    except Exception as error:
        print(f"FAILED {ticker}: {error}", file=sys.stderr)
        return 0, 1, 0

    original = {
        "signal": item.last_observed_signal,
        "date": item.last_observed_signal_date,
        "fingerprint": item.last_observed_row_fingerprint,
    }
    event = advancement["event"]

    if event is None:
        if dry_run:
            if advancement["baseline_only"]:
                print(
                    f"BASELINE {user.email} {ticker}: "
                    f"{signal_label(advancement['final_signal'])} "
                    f"on {advancement['final_date']}"
                )
            elif (
                advancement["final_date"] != original["date"]
                or advancement["final_fingerprint"] != original["fingerprint"]
            ):
                print(
                    f"ADVANCE {user.email} {ticker}: no signal change; "
                    f"observed through {advancement['final_date']}"
                )
            return 0, 0, 0

        advance_baseline(item.id, advancement, original=original)
        return 0, 0, 0

    if dry_run:
        print(
            f"SEND-{event['event_type'].upper()} {user.email} {ticker}: "
            f"{event['observation_start_date']}..{event['observation_end_date']} "
            f"{signal_label(event['initial_signal'])} -> "
            f"{signal_label(event['final_signal'])}; "
            f"changes={len(event['transitions'])}"
        )
        for transition in event["transitions"]:
            print(f"  {format_transition_line(transition)}")
        return 1, 0, 0

    locked_item = (
        WatchlistItem.query
        .filter_by(id=item.id)
        .with_for_update()
        .first()
    )
    if not locked_item or not locked_item.email_alert_enabled:
        db.session.rollback()
        return 0, 0, 0
    if not baseline_matches(locked_item, original):
        db.session.rollback()
        return 0, 0, 0

    delivery, claim_status = claim_delivery(locked_item, event)

    if claim_status == "busy":
        db.session.rollback()
        return 0, 0, 0

    if claim_status == "already_sent":
        advance_baseline(item.id, advancement, original=original)
        return 0, 0, 0

    try:
        message = build_alert_message(user, ticker, event)
        if mail_connection is not None:
            mail_connection.send(message)
        else:
            mail.send(message)
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
            delivery.attempted_at = utcnow()
            db.session.commit()
        print(f"FAILED {user.email} {ticker}: {error}", file=sys.stderr)
        return 0, 1, 0

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
        delivery.sent_at = utcnow()
        delivery.attempted_at = utcnow()
        delivery.error_message = None

    if (
        locked_item
        and locked_item.email_alert_enabled
        and baseline_matches(locked_item, original)
    ):
        locked_item.last_observed_signal = advancement["final_signal"]
        locked_item.last_observed_signal_date = advancement["final_date"]
        locked_item.last_observed_row_fingerprint = advancement["final_fingerprint"]

    db.session.commit()

    print(
        f"SENT-{event['event_type'].upper()} {user.email} {ticker}: "
        f"through {advancement['final_date']} changes={len(event['transitions'])}"
    )
    return 1, 0, 0


def run_dispatch(
    *,
    dry_run,
    limit=200,
    stability_minutes=DEFAULT_STABILITY_MINUTES,
    ticker=None,
    expected_approval_signature=None,
):
    """Preview or dispatch eligible alerts inside an existing app context.

    Returns a structured summary while retaining the human-readable console
    output used by the admin page and CLI. Preview mode never sends email and
    never advances user checkpoints.
    """
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000")

    stability_minutes = parse_stability_minutes(stability_minutes)
    clean_ticker = normalize_ticker(ticker) if ticker else None

    if not dry_run and not os.environ.get("EMAIL_USER"):
        raise RuntimeError("EMAIL_USER is not configured.")

    total_sent = 0
    total_failures = 0
    total_waiting = 0
    processed_items = 0
    market_cache = {}

    query = (
        db.session.query(WatchlistItem, User)
        .join(User, User.id == WatchlistItem.user_id)
        .filter(WatchlistItem.email_alert_enabled.is_(True))
        .order_by(WatchlistItem.id.asc())
    )
    if clean_ticker:
        query = query.filter(WatchlistItem.ticker == clean_ticker)

    records = query.all()
    approved_states = {}
    for item, user in records:
        approved_states[item.id] = {
            "item_id": item.id,
            "user_id": item.user_id,
            "ticker": item.ticker,
            "email_alert_enabled": bool(item.email_alert_enabled),
            "last_observed_signal": item.last_observed_signal,
            "last_observed_signal_date": (
                item.last_observed_signal_date.isoformat()
                if item.last_observed_signal_date else None
            ),
            "last_observed_row_fingerprint": item.last_observed_row_fingerprint,
            "user_email": user.email,
            "user_is_verified": bool(user.is_verified),
            "user_subscription_type": user.subscription_type,
            "user_subscription_status": user.subscription_status,
        }

    item_ids = [item.id for item, _user in records]
    available_items = len(item_ids)

    # Freeze one in-memory snapshot per involved asset before any email is sent.
    # This makes the reviewed batch deterministic even if a file is replaced
    # immediately after the administrator presses Send.
    for source_ticker in sorted({item.ticker for item, _user in records}):
        load_stable_market_snapshot(
            source_ticker,
            stability_minutes,
            market_cache,
        )

    approval_payload = {
        "items": [approved_states[item_id] for item_id in item_ids],
        "sources": [
            {
                "ticker": source_ticker,
                "status": snapshot.get("status"),
                "identity": snapshot.get("source_identity"),
            }
            for source_ticker, snapshot in sorted(market_cache.items())
        ],
    }
    approval_signature = hashlib.sha256(
        json.dumps(
            approval_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    if (
        expected_approval_signature is not None
        and not hmac.compare_digest(
            str(expected_approval_signature),
            approval_signature,
        )
    ):
        print(
            "ABORT-CHANGED: published files or alert recipients changed "
            "after preview; preview again before sending.",
            file=sys.stderr,
        )
        return {
            "mode": "send",
            "pending_or_sent": 0,
            "failures": 0,
            "waiting": 0,
            "processed_items": 0,
            "available_items": available_items,
            "limit_reached": False,
            "stability_minutes": stability_minutes,
            "ticker": clean_ticker,
            "approval_signature": approval_signature,
            "approval_mismatch": True,
        }

    connection_context = (
        nullcontext(None) if dry_run or not item_ids else mail.connect()
    )
    with connection_context as mail_connection:
        for item_id in item_ids:
            if total_sent >= limit:
                break

            processed_items += 1
            try:
                sent, failures, waiting = process_item(
                    item_id,
                    dry_run=dry_run,
                    stability_minutes=stability_minutes,
                    market_cache=market_cache,
                    mail_connection=mail_connection,
                    approved_state=approved_states.get(item_id),
                )
            except Exception as error:
                db.session.rollback()
                print(f"FAILED watchlist_item_id={item_id}: {error}", file=sys.stderr)
                sent, failures, waiting = 0, 1, 0

            total_sent += sent
            total_failures += failures
            total_waiting += waiting

    mode = "preview" if dry_run else "send"
    print(
        f"Signal alert {mode} complete: pending_or_sent={total_sent} "
        f"failures={total_failures} waiting={total_waiting} "
        f"processed_items={processed_items} stability_minutes={stability_minutes}"
    )

    return {
        "mode": mode,
        "pending_or_sent": total_sent,
        "failures": total_failures,
        "waiting": total_waiting,
        "processed_items": processed_items,
        "available_items": available_items,
        "limit_reached": processed_items < available_items,
        "stability_minutes": stability_minutes,
        "ticker": clean_ticker,
        "approval_signature": approval_signature,
        "approval_mismatch": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Preview or manually send NeuralTrend signal-change alerts. "
            "A mode is required; running the old no-argument Cron command "
            "cannot send email."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preview",
        action="store_true",
        help="Show pending actions without sending or advancing checkpoints.",
    )
    mode.add_argument(
        "--send",
        action="store_true",
        help="Manually approve and send all currently eligible alerts.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Legacy alias for --preview.",
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Optionally preview/send only one canonical ticker, such as BTC-USD.",
    )
    parser.add_argument(
        "--stability-minutes",
        type=int,
        default=DEFAULT_STABILITY_MINUTES,
        help=(
            "Optional explicit quiet period. Manual dispatch defaults to 0. "
            "Normally leave this unchanged."
        ),
    )
    args = parser.parse_args()

    dry_run = bool(args.preview or args.dry_run)

    try:
        stability_minutes = parse_stability_minutes(args.stability_minutes)
        with app.app_context():
            summary = run_dispatch(
                dry_run=dry_run,
                limit=args.limit,
                stability_minutes=stability_minutes,
                ticker=args.ticker,
            )
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))

    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
