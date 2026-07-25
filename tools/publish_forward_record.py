"""Preview and publish NeuralTrend forward-record snapshots.

Two fully separated record modes are supported:

* ``sandbox`` is private, admin-only, resettable, and intended for pre-launch
  testing while models and data workflows are still changing.
* ``public`` is customer-facing and append-only. Each supported asset enters
  prospectively through an explicit admin-approved first publication and keeps
  its own public start date.

Neither mode imports history that predates its own first approved publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import (  # noqa: E402
    ADMIN_ONLY_TICKERS,
    SUPPORTED_TICKERS,
    app,
    build_data_freshness_metadata,
    get_epoch_csv_path,
    make_signal_row_fingerprint,
    normalize_ticker,
    signal_label,
)
from extensions import db  # noqa: E402
from models import (  # noqa: E402
    ForwardMarketObservation,
    ForwardPublicationBatch,
    ForwardSignalPublication,
    User,
)

PUBLICATION_TYPES = {"initial", "regular", "delayed", "revision", "correction"}
RECORD_MODES = {"sandbox", "public"}


def utcnow() -> datetime:
    return datetime.utcnow()


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def public_record_enabled() -> bool:
    return (
        env_flag("FORWARD_RECORD_PUBLIC_ENABLED", default=False)
        or ForwardPublicationBatch.query.filter_by(record_mode="public").first()
        is not None
    )


def normalize_record_mode(value: str | None) -> str:
    mode = (value or "sandbox").strip().lower()
    if mode not in RECORD_MODES:
        raise ValueError("Record mode must be sandbox or public.")
    return mode


def require_public_record_enabled(record_mode: str) -> None:
    if normalize_record_mode(record_mode) == "public" and not public_record_enabled():
        raise PermissionError(
            "Public Forward Record publication is disabled. Set "
            "FORWARD_RECORD_PUBLIC_ENABLED=true only when NeuralTrend is ready "
            "to begin its customer-facing record."
        )


def public_tickers() -> list[str]:
    """Return every customer-facing asset supported by the main product."""
    seen = set()
    ordered = []
    for raw in SUPPORTED_TICKERS:
        ticker = normalize_ticker(raw)
        if not ticker or ticker in ADMIN_ONLY_TICKERS or ticker in seen:
            continue
        seen.add(ticker)
        ordered.append(ticker)
    return ordered


def tracked_public_tickers() -> list[str]:
    """Return only assets that have explicitly entered the public record."""
    return [
        row[0]
        for row in (
            db.session.query(ForwardSignalPublication.ticker)
            .join(
                ForwardPublicationBatch,
                ForwardPublicationBatch.id == ForwardSignalPublication.batch_id,
            )
            .filter(ForwardPublicationBatch.record_mode == "public")
            .distinct()
            .order_by(ForwardSignalPublication.ticker.asc())
            .all()
        )
    ]


def finite_positive(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric.") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be positive and finite.")
    return number


def market_row_fingerprint(ticker: str, market_date: date, close_price: float) -> str:
    raw = (
        f"{normalize_ticker(ticker)}|{market_date.isoformat()}|"
        f"{format(float(close_price), '.12g')}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def publication_query_for_mode(record_mode: str):
    mode = normalize_record_mode(record_mode)
    return (
        ForwardSignalPublication.query
        .join(
            ForwardPublicationBatch,
            ForwardPublicationBatch.id == ForwardSignalPublication.batch_id,
        )
        .filter(ForwardPublicationBatch.record_mode == mode)
    )


def latest_publication_for_ticker(ticker: str, record_mode: str):
    return (
        publication_query_for_mode(record_mode)
        .filter(ForwardSignalPublication.ticker == normalize_ticker(ticker))
        .order_by(
            ForwardSignalPublication.published_at.desc(),
            ForwardSignalPublication.id.desc(),
        )
        .first()
    )


def first_publication_for_ticker(ticker: str, record_mode: str):
    return (
        publication_query_for_mode(record_mode)
        .filter(ForwardSignalPublication.ticker == normalize_ticker(ticker))
        .order_by(
            ForwardSignalPublication.published_at.asc(),
            ForwardSignalPublication.id.asc(),
        )
        .first()
    )


def load_forward_source_data(ticker: str) -> pd.DataFrame:
    """Load only the columns required for prospective publication/performance."""
    clean = normalize_ticker(ticker)
    csv_path = get_epoch_csv_path(clean)
    data = pd.read_csv(
        csv_path,
        usecols=["Date", "Close", "epoch_signal"],
        parse_dates=["Date"],
    )

    for column in ("Close", "epoch_signal"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["Date", "Close", "epoch_signal"]).copy()
    finite_mask = data["Close"].map(math.isfinite) & data["epoch_signal"].map(math.isfinite)
    valid_mask = (
        finite_mask
        & (data["Close"] > 0)
        & data["epoch_signal"].isin([-1, 0, 1])
    )
    data = data[valid_mask].copy()
    data["epoch_signal"] = data["epoch_signal"].astype(int)
    data = data.sort_values("Date")
    data = data.drop_duplicates(subset=["Date"], keep="last")
    data.set_index("Date", inplace=True)

    if data.empty:
        raise ValueError("Market-data file contains no valid forward-record rows.")
    return data


def load_latest_source_snapshot(ticker: str) -> dict[str, Any]:
    clean = normalize_ticker(ticker)
    path = get_epoch_csv_path(clean)
    data = load_forward_source_data(clean).sort_index()

    latest_index = data.index.max()
    latest_row = data.loc[latest_index]
    if getattr(latest_row, "ndim", 1) > 1:
        latest_row = latest_row.iloc[-1]

    signal = int(latest_row["epoch_signal"])
    if signal not in {-1, 0, 1}:
        raise ValueError("Latest signal must be BUY, HOLD, or SELL.")

    close_price = finite_positive(latest_row["Close"], "Latest close")
    source_date = latest_index.date()
    source_fingerprint = make_signal_row_fingerprint(
        clean,
        source_date,
        signal,
        close_price,
    )
    stat = os.stat(path)

    return {
        "ticker": clean,
        "path": path,
        "file_identity": f"{stat.st_mtime_ns}:{stat.st_size}",
        "data": data,
        "source_data_date": source_date,
        "signal": signal,
        "reference_close": close_price,
        "source_row_fingerprint": source_fingerprint,
        "freshness": build_data_freshness_metadata(clean, source_date),
    }


def classify_candidate(
    snapshot: dict[str, Any],
    previous,
) -> tuple[str | None, int, int | None]:
    """Return publication type, newly observed row count, and superseded id."""
    if previous is None:
        return "initial", 0, None

    source_date = snapshot["source_data_date"]
    if source_date < previous.source_data_date:
        raise ValueError(
            f"Latest source date {source_date} is older than the already recorded "
            f"date {previous.source_data_date}."
        )

    if source_date == previous.source_data_date:
        if snapshot["source_row_fingerprint"] == previous.source_row_fingerprint:
            return None, 0, None
        if int(snapshot["signal"]) != int(previous.signal):
            return "revision", 0, previous.id
        return "correction", 0, previous.id

    new_rows = snapshot["data"][
        snapshot["data"].index.date > previous.source_data_date
    ]
    new_row_count = len(new_rows)
    if new_row_count < 1:
        raise ValueError("A newer source date was found without a corresponding new row.")

    publication_type = "delayed" if new_row_count > 1 else "regular"
    return publication_type, new_row_count, None


def build_market_capture_plan(
    ticker: str,
    snapshot: dict[str, Any],
    first_publication,
    record_mode: str,
) -> tuple[list[dict], list[str]]:
    """Plan immutable closes after this mode's first approved publication."""
    if first_publication is None:
        return [], []

    mode = normalize_record_mode(record_mode)
    clean = normalize_ticker(ticker)
    data = snapshot["data"]
    start_after = first_publication.source_data_date
    eligible = data[data.index.date > start_after]
    capture_rows = []
    preserved_differences = []

    existing = {
        row.market_date: row
        for row in ForwardMarketObservation.query.filter_by(
            record_mode=mode,
            ticker=clean,
        ).all()
    }

    for index, raw_row in eligible.iterrows():
        market_date = index.date()
        close_price = finite_positive(raw_row["Close"], f"{clean} close on {market_date}")
        fingerprint = market_row_fingerprint(clean, market_date, close_price)

        prior = existing.get(market_date)
        if prior is not None:
            if prior.source_row_fingerprint != fingerprint:
                preserved_differences.append(
                    f"{clean} {market_date}: upstream close differs from the preserved "
                    f"{mode} snapshot; the original snapshot remains in use."
                )
            continue

        capture_rows.append({
            "ticker": clean,
            "market_date": market_date,
            "close_price": close_price,
            "source_row_fingerprint": fingerprint,
        })

    return capture_rows, preserved_differences


def _serializable_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": candidate["ticker"],
        "source_data_date": candidate["source_data_date"].isoformat(),
        "signal": candidate["signal"],
        "reference_close": format(candidate["reference_close"], ".12g"),
        "source_row_fingerprint": candidate["source_row_fingerprint"],
        "publication_type": candidate["publication_type"],
        "new_row_count": candidate["new_row_count"],
        "supersedes_publication_id": candidate["supersedes_publication_id"],
        "file_identity": candidate["file_identity"],
        "market_rows": [
            {
                "market_date": row["market_date"].isoformat(),
                "close_price": format(row["close_price"], ".12g"),
                "source_row_fingerprint": row["source_row_fingerprint"],
            }
            for row in candidate["market_rows"]
        ],
    }


def build_publication_preview(
    ticker: str | None = None,
    *,
    record_mode: str = "sandbox",
) -> dict[str, Any]:
    mode = normalize_record_mode(record_mode)
    require_public_record_enabled(mode)

    clean_filter = normalize_ticker(ticker) if ticker else None
    supported = public_tickers()
    if clean_filter and clean_filter not in supported:
        raise ValueError("Forward Record publication supports public assets only.")

    if mode == "public":
        tracked = tracked_public_tickers()
        if clean_filter:
            # An explicit ticker either updates an already tracked asset or
            # prospectively enrolls a new supported asset from this moment.
            tickers = [clean_filter]
        elif tracked:
            # Blank means "update the existing public universe". It never
            # auto-enrolls every supported asset merely because a CSV exists.
            tickers = tracked
        else:
            raise ValueError(
                "Choose one supported ticker for the first public Forward Record "
                "publication. Assets enter the public record individually."
            )
    else:
        tickers = [clean_filter] if clean_filter else supported

    candidates = []
    errors = []
    warnings = []
    unchanged = 0
    checked = 0

    for source_ticker in tickers:
        checked += 1
        try:
            snapshot = load_latest_source_snapshot(source_ticker)
            previous = latest_publication_for_ticker(source_ticker, mode)
            first = first_publication_for_ticker(source_ticker, mode)
            publication_type, new_row_count, supersedes_id = classify_candidate(
                snapshot,
                previous,
            )
            market_rows, row_warnings = build_market_capture_plan(
                source_ticker,
                snapshot,
                first,
                mode,
            )
            warnings.extend(row_warnings)

            if publication_type is None:
                unchanged += 1
                continue

            candidates.append({
                "ticker": source_ticker,
                "source_data_date": snapshot["source_data_date"],
                "signal": snapshot["signal"],
                "signal_label": signal_label(snapshot["signal"]),
                "reference_close": snapshot["reference_close"],
                "source_row_fingerprint": snapshot["source_row_fingerprint"],
                "publication_type": publication_type,
                "new_row_count": new_row_count,
                "supersedes_publication_id": supersedes_id,
                "file_identity": snapshot["file_identity"],
                "freshness_label": snapshot["freshness"].get("freshness_label", "Unknown"),
                "market_rows": market_rows,
            })
        except Exception as error:
            errors.append(f"{source_ticker}: {error}")

    candidates.sort(key=lambda item: item["ticker"])
    signature_payload = {
        "record_mode": mode,
        "ticker_filter": clean_filter,
        "candidates": [_serializable_candidate(item) for item in candidates],
        "errors": errors,
        "warnings": warnings,
        "unchanged": unchanged,
    }
    approval_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    type_counts = {
        kind: sum(1 for item in candidates if item["publication_type"] == kind)
        for kind in PUBLICATION_TYPES
    }
    market_row_count = sum(len(item["market_rows"]) for item in candidates)

    return {
        "record_mode": mode,
        "ticker": clean_filter,
        "checked_assets": checked,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "unchanged_count": unchanged,
        "errors": errors,
        "error_count": len(errors),
        "warnings": warnings,
        "warning_count": len(warnings),
        "type_counts": type_counts,
        "market_row_count": market_row_count,
        "approval_signature": approval_signature,
    }


def print_preview(preview: dict[str, Any]) -> None:
    mode_label = preview["record_mode"].upper()
    for candidate in preview["candidates"]:
        print(
            f"{mode_label}-PUBLISH-{candidate['publication_type'].upper()} "
            f"{candidate['ticker']}: {candidate['signal_label']} "
            f"data_through={candidate['source_data_date']} "
            f"market_rows={len(candidate['market_rows'])}"
        )
    for warning in preview["warnings"]:
        print(f"{mode_label}-PRESERVED {warning}")
    for error in preview["errors"]:
        print(f"FAILED {mode_label} {error}", file=sys.stderr)
    print(
        f"{mode_label} Forward Record preview complete: "
        f"pending={preview['candidate_count']} unchanged={preview['unchanged_count']} "
        f"market_rows={preview['market_row_count']} errors={preview['error_count']} "
        f"preserved_differences={preview['warning_count']}"
    )


def make_publication_key(
    batch_digest: str,
    candidate: dict[str, Any],
    record_mode: str,
) -> str:
    raw = (
        f"{normalize_record_mode(record_mode)}|{batch_digest}|{candidate['ticker']}|"
        f"{candidate['source_data_date'].isoformat()}|"
        f"{candidate['source_row_fingerprint']}|{candidate['publication_type']}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def publish_forward_batch(
    *,
    admin_user_id: int,
    expected_approval_signature: str,
    ticker: str | None = None,
    record_mode: str = "sandbox",
) -> dict[str, Any]:
    mode = normalize_record_mode(record_mode)
    require_public_record_enabled(mode)
    preview = build_publication_preview(ticker=ticker, record_mode=mode)

    if not expected_approval_signature or preview["approval_signature"] != expected_approval_signature:
        return {
            **preview,
            "published": 0,
            "approval_mismatch": True,
            "batch_id": None,
        }
    if preview["errors"]:
        raise RuntimeError(f"{mode.title()} Forward Record preview contains source errors.")
    if not preview["candidates"]:
        return {
            **preview,
            "published": 0,
            "approval_mismatch": False,
            "batch_id": None,
        }

    admin = db.session.get(User, int(admin_user_id))
    if admin is None:
        raise ValueError("Admin user was not found.")

    published_at = utcnow()
    batch = ForwardPublicationBatch(
        published_by_user_id=admin.id,
        published_at=published_at,
        record_mode=mode,
        publication_count=len(preview["candidates"]),
        revision_count=(
            preview["type_counts"].get("revision", 0)
            + preview["type_counts"].get("correction", 0)
        ),
        delayed_count=preview["type_counts"].get("delayed", 0),
        source_digest=preview["approval_signature"],
    )
    db.session.add(batch)
    db.session.flush()

    for candidate in preview["candidates"]:
        publication = ForwardSignalPublication(
            batch_id=batch.id,
            ticker=candidate["ticker"],
            source_data_date=candidate["source_data_date"],
            signal=candidate["signal"],
            reference_close=candidate["reference_close"],
            published_at=published_at,
            publication_type=candidate["publication_type"],
            source_row_fingerprint=candidate["source_row_fingerprint"],
            publication_key=make_publication_key(
                preview["approval_signature"],
                candidate,
                mode,
            ),
            supersedes_publication_id=candidate["supersedes_publication_id"],
        )
        db.session.add(publication)

        for row in candidate["market_rows"]:
            db.session.add(ForwardMarketObservation(
                record_mode=mode,
                ticker=candidate["ticker"],
                market_date=row["market_date"],
                close_price=row["close_price"],
                source_row_fingerprint=row["source_row_fingerprint"],
                captured_in_batch_id=batch.id,
                captured_at=published_at,
            ))

    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise

    mode_label = mode.upper()
    for candidate in preview["candidates"]:
        print(
            f"{mode_label}-PUBLISHED-{candidate['publication_type'].upper()} "
            f"{candidate['ticker']}: {candidate['signal_label']} "
            f"data_through={candidate['source_data_date']}"
        )
    print(
        f"{mode_label} Forward Record batch published: batch_id={batch.id} "
        f"publications={len(preview['candidates'])} "
        f"market_rows={preview['market_row_count']}"
    )

    return {
        **preview,
        "published": len(preview["candidates"]),
        "approval_mismatch": False,
        "batch_id": batch.id,
    }


def reset_sandbox_record(*, admin_user_id: int) -> dict[str, int]:
    """Delete only private sandbox rows so pre-launch testing can start over."""
    admin = db.session.get(User, int(admin_user_id))
    if admin is None:
        raise ValueError("Admin user was not found.")

    sandbox_batches = (
        ForwardPublicationBatch.query
        .filter_by(record_mode="sandbox")
        .order_by(ForwardPublicationBatch.id.asc())
        .all()
    )
    batch_ids = [batch.id for batch in sandbox_batches]
    if not batch_ids:
        return {"batches": 0, "publications": 0, "observations": 0}

    observation_count = (
        ForwardMarketObservation.query
        .filter_by(record_mode="sandbox")
        .delete(synchronize_session=False)
    )
    publication_count = (
        ForwardSignalPublication.query
        .filter(ForwardSignalPublication.batch_id.in_(batch_ids))
        .delete(synchronize_session=False)
    )
    batch_count = (
        ForwardPublicationBatch.query
        .filter(ForwardPublicationBatch.id.in_(batch_ids))
        .delete(synchronize_session=False)
    )
    db.session.commit()

    print(
        "SANDBOX Forward Record reset: "
        f"batches={batch_count} publications={publication_count} "
        f"observations={observation_count} admin_user_id={admin.id}"
    )
    return {
        "batches": int(batch_count or 0),
        "publications": int(publication_count or 0),
        "observations": int(observation_count or 0),
    }


def resolve_admin(email: str | None) -> User:
    raw = (email or "").strip().lower()
    if not raw:
        configured = [
            item.strip().lower()
            for item in os.environ.get("ADMIN_EMAILS", "").split(",")
            if item.strip()
        ]
        if len(configured) != 1:
            raise ValueError("Pass --admin-email when ADMIN_EMAILS has zero or multiple entries.")
        raw = configured[0]
    user = User.query.filter(User.email == raw).first()
    if user is None:
        raise ValueError("Configured admin account was not found in the database.")
    return user


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preview, publish, or reset a NeuralTrend Forward Record mode."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preview", action="store_true")
    action.add_argument("--publish", action="store_true")
    action.add_argument("--reset-sandbox", action="store_true")
    parser.add_argument("--record-mode", choices=sorted(RECORD_MODES), default="sandbox")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--admin-email", default=None)
    parser.add_argument(
        "--approval-signature",
        default=None,
        help="Required with --publish; copy it from a fresh matching preview.",
    )
    parser.add_argument(
        "--confirm-public-launch",
        action="store_true",
        help="Additional irreversible acknowledgement required for public publication.",
    )
    parser.add_argument(
        "--confirm-reset-sandbox",
        action="store_true",
        help="Required with --reset-sandbox.",
    )
    args = parser.parse_args()

    with app.app_context():
        admin = resolve_admin(args.admin_email) if (args.publish or args.reset_sandbox) else None

        if args.reset_sandbox:
            if args.record_mode != "sandbox":
                parser.error("--reset-sandbox can only be used with --record-mode sandbox.")
            if not args.confirm_reset_sandbox:
                parser.error("--reset-sandbox requires --confirm-reset-sandbox.")
            reset_sandbox_record(admin_user_id=admin.id)
            return 0

        if args.record_mode == "public" and args.publish and not args.confirm_public_launch:
            parser.error("Public publication requires --confirm-public-launch.")

        preview = build_publication_preview(
            ticker=args.ticker,
            record_mode=args.record_mode,
        )
        print_preview(preview)
        print(f"Approval signature: {preview['approval_signature']}")
        if args.preview:
            return 1 if preview["errors"] else 0
        if not args.approval_signature:
            parser.error("--publish requires --approval-signature from a fresh preview.")

        result = publish_forward_batch(
            admin_user_id=admin.id,
            expected_approval_signature=args.approval_signature,
            ticker=args.ticker,
            record_mode=args.record_mode,
        )
        if result["approval_mismatch"]:
            print("ABORT-CHANGED: source data changed after preview.", file=sys.stderr)
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
