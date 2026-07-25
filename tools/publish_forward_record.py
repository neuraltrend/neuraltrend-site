"""Admin-approved CSV-backed NeuralTrend Forward Record workflow.

Working model CSVs never become customer-facing automatically. Preview compares
those files with a compact approved CSV containing only Date, Close and
``epoch_signal``. Approval atomically updates the approved files and a small
PostgreSQL metadata/audit ledger.

Sandbox and public modes are independent. Sandbox files are private and
resettable. Public assets enter one at a time, then blank batch approvals update
all already-enrolled active assets.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import uuid
from datetime import date, datetime, timedelta
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
    get_epoch_csv_path,
    get_forward_record_csv_path,
    normalize_ticker,
    signal_label,
)
from extensions import db  # noqa: E402
from models import (  # noqa: E402
    ForwardPublicationBatch,
    ForwardRecordAsset,
    User,
    WatchlistItem,
)

RECORD_MODES = {"sandbox", "public"}
APPROVED_COLUMNS = ["Date", "Close", "epoch_signal"]


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
        or ForwardRecordAsset.query.filter_by(record_mode="public").first()
        is not None
    )


def normalize_record_mode(value: str | None) -> str:
    mode = str(value or "sandbox").strip().lower()
    if mode not in RECORD_MODES:
        raise ValueError("Record mode must be sandbox or public.")
    return mode


def require_public_record_enabled(record_mode: str) -> None:
    if normalize_record_mode(record_mode) == "public" and not public_record_enabled():
        raise PermissionError(
            "Public Forward Record publication is disabled. Set "
            "FORWARD_RECORD_PUBLIC_ENABLED=true only when NeuralTrend is ready "
            "to begin customer-facing tracking."
        )


def public_tickers() -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in SUPPORTED_TICKERS:
        ticker = normalize_ticker(raw)
        if not ticker or ticker in ADMIN_ONLY_TICKERS or ticker in seen:
            continue
        seen.add(ticker)
        ordered.append(ticker)
    return ordered


def tracked_asset(ticker: str, record_mode: str) -> ForwardRecordAsset | None:
    return ForwardRecordAsset.query.filter_by(
        record_mode=normalize_record_mode(record_mode),
        ticker=normalize_ticker(ticker),
    ).first()


def tracked_active_tickers(record_mode: str) -> list[str]:
    return [
        row.ticker
        for row in (
            ForwardRecordAsset.query
            .filter_by(record_mode=normalize_record_mode(record_mode), status="active")
            .order_by(ForwardRecordAsset.ticker.asc())
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


def _normalize_frame(data: pd.DataFrame, *, source_label: str) -> pd.DataFrame:
    missing = [column for column in APPROVED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            f"{source_label} is missing required column(s): {', '.join(missing)}."
        )

    frame = data[APPROVED_COLUMNS].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame["epoch_signal"] = pd.to_numeric(frame["epoch_signal"], errors="coerce")
    frame = frame.dropna(subset=APPROVED_COLUMNS).copy()

    finite_close = frame["Close"].map(math.isfinite)
    finite_signal = frame["epoch_signal"].map(math.isfinite)
    frame = frame[
        finite_close
        & finite_signal
        & (frame["Close"] > 0)
        & frame["epoch_signal"].isin([-1, 0, 1])
    ].copy()
    frame["epoch_signal"] = frame["epoch_signal"].astype(int)
    frame = frame.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    frame.reset_index(drop=True, inplace=True)

    if source_label == "working CSV" and frame.empty:
        raise ValueError("Working market-data CSV contains no valid Forward Record rows.")
    return frame


def load_working_data(ticker: str) -> tuple[pd.DataFrame, str]:
    clean = normalize_ticker(ticker)
    path = get_epoch_csv_path(clean)
    raw = pd.read_csv(path, usecols=APPROVED_COLUMNS)
    frame = _normalize_frame(raw, source_label="working CSV")
    stat = os.stat(path)
    identity = f"{stat.st_mtime_ns}:{stat.st_size}"
    return frame, identity


def load_approved_data(ticker: str, record_mode: str) -> pd.DataFrame:
    path = get_forward_record_csv_path(
        ticker,
        normalize_record_mode(record_mode),
        must_exist=True,
    )
    raw = pd.read_csv(path, usecols=APPROVED_COLUMNS)
    return _normalize_frame(raw, source_label="approved Forward Record CSV")


def canonical_csv_bytes(frame: pd.DataFrame) -> bytes:
    normalized = _normalize_frame(
        frame if not frame.empty else pd.DataFrame(columns=APPROVED_COLUMNS),
        source_label="approved Forward Record CSV",
    )
    output = io.StringIO()
    normalized.to_csv(
        output,
        index=False,
        columns=APPROVED_COLUMNS,
        date_format="%Y-%m-%d",
        float_format="%.12g",
        lineterminator="\n",
    )
    return output.getvalue().encode("utf-8")


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def frame_digest(frame: pd.DataFrame) -> str:
    return digest_bytes(canonical_csv_bytes(frame))


def _date_map(frame: pd.DataFrame) -> dict[date, tuple[str, int]]:
    return {
        row.Date.date(): (format(float(row.Close), ".12g"), int(row.epoch_signal))
        for row in frame.itertuples(index=False)
    }


def verify_approved_history(
    ticker: str,
    source: pd.DataFrame,
    approved: pd.DataFrame,
    asset: ForwardRecordAsset,
) -> str:
    """Reject deletion or rewriting of any already approved row."""
    actual_digest = frame_digest(approved)
    if asset.last_approved_digest and actual_digest != asset.last_approved_digest:
        raise ValueError(
            f"{ticker}: approved Forward Record CSV changed outside the approval "
            "workflow. Restore the approved file before publishing."
        )

    source_rows = _date_map(source)
    for market_date, expected in _date_map(approved).items():
        current = source_rows.get(market_date)
        if current is None:
            raise ValueError(
                f"{ticker}: working CSV no longer contains approved date {market_date}. "
                "Previously approved history cannot be deleted."
            )
        if current != expected:
            raise ValueError(
                f"{ticker}: working CSV changed previously approved Date/Close/signal "
                f"history on {market_date}. Restore that row before publishing."
            )
    return actual_digest


def _frame_on_or_after(frame: pd.DataFrame, start_date: date) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame[frame["Date"].dt.date >= start_date].copy()


def _frame_after(frame: pd.DataFrame, after_date: date | None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if after_date is None:
        return frame.copy()
    return frame[frame["Date"].dt.date > after_date].copy()


def _candidate_serializable(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": candidate["ticker"],
        "classification": candidate["classification"],
        "start_date": candidate["start_date"].isoformat(),
        "previous_through_date": (
            candidate["previous_through_date"].isoformat()
            if candidate["previous_through_date"] else None
        ),
        "approved_through_date": (
            candidate["approved_through_date"].isoformat()
            if candidate["approved_through_date"] else None
        ),
        "new_row_count": candidate["new_row_count"],
        "latest_signal": candidate["latest_signal"],
        "source_identity": candidate["source_identity"],
        "approved_digest_before": candidate["approved_digest_before"],
        "approved_digest_after": candidate["approved_digest_after"],
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
        raise ValueError("Forward Record publication supports customer-facing assets only.")

    tracked = tracked_active_tickers(mode)
    if clean_filter:
        existing = tracked_asset(clean_filter, mode)
        if existing and existing.status != "active":
            raise ValueError(
                f"{clean_filter} is {existing.status} and cannot receive new approved rows."
            )
        tickers = [clean_filter]
    elif tracked:
        tickers = tracked
    else:
        raise ValueError(
            f"Choose one supported ticker for the first {mode} Forward Record asset."
        )

    preview_date = utcnow().date()
    candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    unchanged: list[str] = []

    for clean in tickers:
        try:
            source, source_identity = load_working_data(clean)
            asset = tracked_asset(clean, mode)

            if asset is None:
                start_date = preview_date
                eligible = _frame_on_or_after(source, start_date)
                approved_before = pd.DataFrame(columns=APPROVED_COLUMNS)
                approved_digest_before = None
                new_rows = eligible
                classification = "initial"
                previous_through = None
            else:
                if asset.status != "active":
                    raise ValueError(f"{clean} is not active in the {mode} record.")
                start_date = asset.start_date
                approved_before = load_approved_data(clean, mode)
                approved_digest_before = verify_approved_history(
                    clean,
                    source,
                    approved_before,
                    asset,
                )
                previous_through = asset.approved_through_date
                new_rows = _frame_after(
                    _frame_on_or_after(source, start_date),
                    previous_through,
                )
                if new_rows.empty:
                    unchanged.append(clean)
                    continue
                classification = "delayed" if len(new_rows) > 1 else "append"

            approved_after = pd.concat(
                [approved_before, new_rows],
                ignore_index=True,
            )
            approved_after = _normalize_frame(
                approved_after,
                source_label="approved Forward Record CSV",
            )
            approved_bytes = canonical_csv_bytes(approved_after)
            approved_through = (
                approved_after["Date"].max().date()
                if not approved_after.empty else None
            )
            latest_signal = (
                int(approved_after.iloc[-1]["epoch_signal"])
                if not approved_after.empty else None
            )

            candidates.append({
                "ticker": clean,
                "asset_id": asset.id if asset else None,
                "classification": classification,
                "start_date": start_date,
                "previous_through_date": previous_through,
                "approved_through_date": approved_through,
                "new_row_count": len(new_rows),
                "latest_signal": latest_signal,
                "source_identity": source_identity,
                "approved_digest_before": approved_digest_before,
                "approved_digest_after": digest_bytes(approved_bytes),
                "approved_bytes": approved_bytes,
            })
        except Exception as exc:
            errors.append(f"{clean}: {exc}")

    signature_payload = {
        "record_mode": mode,
        "ticker_filter": clean_filter,
        "candidates": [_candidate_serializable(item) for item in candidates],
        "errors": errors,
        "unchanged": unchanged,
    }
    approval_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return {
        "record_mode": mode,
        "ticker_filter": clean_filter,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "new_row_count": sum(item["new_row_count"] for item in candidates),
        "delayed_count": sum(
            1 for item in candidates if item["classification"] == "delayed"
        ),
        "unchanged": unchanged,
        "unchanged_count": len(unchanged),
        "errors": errors,
        "error_count": len(errors),
        "approval_signature": approval_signature,
        "published": 0,
    }


def print_preview(preview: dict[str, Any]) -> None:
    prefix = preview["record_mode"].upper()
    for candidate in preview["candidates"]:
        through = candidate["approved_through_date"] or "waiting-for-first-row"
        latest = (
            signal_label(candidate["latest_signal"])
            if candidate["latest_signal"] is not None else "—"
        )
        print(
            f"{prefix}-PUBLISH-{candidate['classification'].upper()} "
            f"{candidate['ticker']}: rows={candidate['new_row_count']} "
            f"start={candidate['start_date']} through={through} latest={latest}"
        )
    for ticker in preview["unchanged"]:
        print(f"{prefix}-UNCHANGED {ticker}: no newly dated approved rows.")
    for error in preview["errors"]:
        print(f"FAILED {error}")
    print(
        f"{prefix} preview complete: assets={preview['candidate_count']} "
        f"new_rows={preview['new_row_count']} unchanged={preview['unchanged_count']} "
        f"errors={preview['error_count']}"
    )


def _safe_remove(path: str | Path) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def publish_forward_batch(
    *,
    admin_user_id: int,
    expected_approval_signature: str,
    ticker: str | None = None,
    record_mode: str = "sandbox",
) -> dict[str, Any]:
    mode = normalize_record_mode(record_mode)
    require_public_record_enabled(mode)

    admin = db.session.get(User, int(admin_user_id))
    if admin is None:
        raise ValueError("Admin user was not found.")

    # Serialize publication actions for this administrator account.
    db.session.query(User.id).filter(User.id == admin.id).with_for_update().one()
    preview = build_publication_preview(ticker=ticker, record_mode=mode)

    if preview["error_count"]:
        db.session.rollback()
        raise ValueError("Resolve every preview error before publication.")
    if (
        not expected_approval_signature
        or preview["approval_signature"] != expected_approval_signature
    ):
        db.session.rollback()
        return {**preview, "approval_mismatch": True, "published": 0}
    if not preview["candidates"]:
        db.session.rollback()
        print(f"{mode.upper()} publication: no pending approved rows.")
        return {**preview, "approval_mismatch": False, "published": 0}

    staged: list[dict[str, Any]] = []
    replaced: list[dict[str, Any]] = []
    published_at = utcnow()

    try:
        for candidate in preview["candidates"]:
            target = Path(get_forward_record_csv_path(candidate["ticker"], mode))
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{target.stem}.",
                suffix=".tmp",
                dir=str(target.parent),
            )
            with os.fdopen(fd, "wb") as handle:
                handle.write(candidate["approved_bytes"])
                handle.flush()
                os.fsync(handle.fileno())
            staged.append({"candidate": candidate, "target": target, "temp": Path(temp_name)})

        # Replace approved files first while keeping same-directory backups. If
        # the database commit fails, every file is restored before returning.
        for item in staged:
            target: Path = item["target"]
            backup = target.with_name(f".{target.name}.{uuid.uuid4().hex}.bak")
            had_original = target.exists()
            if had_original:
                os.replace(target, backup)
            os.replace(item["temp"], target)
            replaced.append({
                "target": target,
                "backup": backup,
                "had_original": had_original,
            })

        batch = ForwardPublicationBatch(
            published_by_user_id=admin.id,
            published_at=published_at,
            record_mode=mode,
            publication_count=len(preview["candidates"]),
            revision_count=0,
            delayed_count=preview["delayed_count"],
            source_digest=preview["approval_signature"],
        )
        db.session.add(batch)

        for candidate in preview["candidates"]:
            asset = tracked_asset(candidate["ticker"], mode)
            if asset is None:
                asset = ForwardRecordAsset(
                    record_mode=mode,
                    ticker=candidate["ticker"],
                    status="active",
                    start_date=candidate["start_date"],
                    enrolled_at=published_at,
                    created_at=published_at,
                    updated_at=published_at,
                )
                db.session.add(asset)
            elif asset.status != "active":
                raise RuntimeError(
                    f"{candidate['ticker']} is {asset.status} and cannot be updated."
                )

            asset.approved_through_date = candidate["approved_through_date"]
            asset.last_approved_at = published_at
            asset.last_approved_digest = candidate["approved_digest_after"]
            asset.updated_at = published_at

        db.session.commit()
    except Exception:
        db.session.rollback()
        for item in reversed(replaced):
            _safe_remove(item["target"])
            if item["had_original"] and item["backup"].exists():
                os.replace(item["backup"], item["target"])
        for item in staged:
            _safe_remove(item["temp"])
        raise
    else:
        for item in replaced:
            _safe_remove(item["backup"])
        for item in staged:
            _safe_remove(item["temp"])

    for candidate in preview["candidates"]:
        latest = (
            signal_label(candidate["latest_signal"])
            if candidate["latest_signal"] is not None else "—"
        )
        print(
            f"{mode.upper()}-PUBLISHED-{candidate['classification'].upper()} "
            f"{candidate['ticker']}: rows={candidate['new_row_count']} "
            f"through={candidate['approved_through_date'] or 'waiting'} latest={latest}"
        )
    print(
        f"{mode.upper()} approved-file batch published: batch_id={batch.id} "
        f"assets={len(preview['candidates'])} new_rows={preview['new_row_count']}"
    )
    return {
        **preview,
        "approval_mismatch": False,
        "published": len(preview["candidates"]),
        "batch_id": batch.id,
    }


def reset_sandbox_record(*, admin_user_id: int) -> dict[str, int]:
    admin = db.session.get(User, int(admin_user_id))
    if admin is None:
        raise ValueError("Admin user was not found.")

    sandbox_assets = ForwardRecordAsset.query.filter_by(record_mode="sandbox").all()
    removed_files = 0
    for asset in sandbox_assets:
        path = get_forward_record_csv_path(asset.ticker, "sandbox")
        if os.path.isfile(path):
            os.remove(path)
            removed_files += 1

    asset_count = ForwardRecordAsset.query.filter_by(record_mode="sandbox").delete(
        synchronize_session=False
    )
    batch_count = ForwardPublicationBatch.query.filter_by(record_mode="sandbox").delete(
        synchronize_session=False
    )
    db.session.commit()

    print(
        "SANDBOX approved-file record reset: "
        f"assets={asset_count} batches={batch_count} files={removed_files} "
        f"admin_user_id={admin.id}"
    )
    return {
        "assets": int(asset_count or 0),
        "batches": int(batch_count or 0),
        "files": int(removed_files),
    }


def normalize_retirement_reason(value: str | None) -> str:
    reason = " ".join(str(value or "").split())
    if len(reason) < 5:
        raise ValueError("Provide a public retirement reason of at least 5 characters.")
    if len(reason) > 300:
        raise ValueError("The public retirement reason must be 300 characters or fewer.")
    return reason


def normalize_removal_date(value: str | date | None, *, retirement_date: date) -> date:
    if isinstance(value, date):
        parsed = value
    else:
        raw = str(value or "").strip()
        if not raw:
            parsed = retirement_date + timedelta(days=365)
        else:
            try:
                parsed = date.fromisoformat(raw)
            except ValueError as exc:
                raise ValueError("Removal date must use YYYY-MM-DD format.") from exc
    if parsed < retirement_date:
        raise ValueError("Removal date cannot be earlier than the retirement date.")
    return parsed


def build_retirement_preview(
    ticker: str,
    reason: str | None,
    removal_after_date: str | date | None = None,
) -> dict[str, Any]:
    require_public_record_enabled("public")
    clean = normalize_ticker(ticker)
    public_reason = normalize_retirement_reason(reason)
    asset = tracked_asset(clean, "public")
    if asset is None:
        raise ValueError(f"{clean} has not entered the public Forward Record.")
    if asset.status != "active":
        raise ValueError(f"{clean} is already {asset.status}.")

    retirement_date = utcnow().date()
    removal_date = normalize_removal_date(
        removal_after_date,
        retirement_date=retirement_date,
    )
    path = get_forward_record_csv_path(clean, "public", must_exist=True)
    approved = load_approved_data(clean, "public")
    digest = frame_digest(approved)
    if asset.last_approved_digest and digest != asset.last_approved_digest:
        raise ValueError("Approved public CSV changed outside the approval workflow.")

    enabled_alert_count = WatchlistItem.query.filter_by(
        ticker=clean,
        email_alert_enabled=True,
    ).count()
    payload = {
        "asset_id": asset.id,
        "ticker": clean,
        "status": asset.status,
        "start_date": asset.start_date.isoformat(),
        "approved_through_date": (
            asset.approved_through_date.isoformat()
            if asset.approved_through_date else None
        ),
        "approved_digest": digest,
        "file_size": os.path.getsize(path),
        "reason": public_reason,
        "retirement_date": retirement_date.isoformat(),
        "removal_after_date": removal_date.isoformat(),
        "enabled_alert_count": enabled_alert_count,
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **payload,
        "retirement_reason": public_reason,
        "approval_signature": signature,
    }


def print_retirement_preview(preview: dict[str, Any]) -> None:
    print(
        f"RETIRE-PREVIEW {preview['ticker']}: tracking_end={preview['retirement_date']} "
        f"approved_through={preview['approved_through_date'] or 'no rows yet'} "
        f"record_available_until={preview['removal_after_date']} "
        f"alerts_to_disable={preview['enabled_alert_count']}"
    )
    print(f"Public reason: {preview['retirement_reason']}")


def retire_public_asset(
    *,
    admin_user_id: int,
    ticker: str,
    reason: str,
    removal_after_date: str | date | None,
    expected_approval_signature: str,
) -> dict[str, Any]:
    admin = db.session.get(User, int(admin_user_id))
    if admin is None:
        raise ValueError("Admin user was not found.")

    clean = normalize_ticker(ticker)
    asset = (
        ForwardRecordAsset.query
        .filter_by(record_mode="public", ticker=clean)
        .with_for_update()
        .first()
    )
    if asset is None or asset.status != "active":
        db.session.rollback()
        return {
            "ticker": clean,
            "retired": False,
            "approval_mismatch": True,
            "disabled_alert_count": 0,
        }

    preview = build_retirement_preview(clean, reason, removal_after_date)
    if (
        not expected_approval_signature
        or preview["approval_signature"] != expected_approval_signature
    ):
        db.session.rollback()
        return {
            **preview,
            "retired": False,
            "approval_mismatch": True,
            "disabled_alert_count": 0,
        }

    now = utcnow()
    asset.status = "retired"
    asset.retired_at = now
    asset.retirement_date = date.fromisoformat(preview["retirement_date"])
    asset.removal_after_date = date.fromisoformat(preview["removal_after_date"])
    asset.retired_by_user_id = admin.id
    asset.retirement_reason = preview["retirement_reason"]
    asset.updated_at = now

    disabled = WatchlistItem.query.filter_by(
        ticker=clean,
        email_alert_enabled=True,
    ).update(
        {WatchlistItem.email_alert_enabled: False},
        synchronize_session=False,
    )
    db.session.commit()
    print(
        f"RETIRED-PUBLIC {clean}: tracking_end={asset.retirement_date} "
        f"record_available_until={asset.removal_after_date} "
        f"alerts_disabled={disabled}"
    )
    return {
        **preview,
        "retired": True,
        "approval_mismatch": False,
        "disabled_alert_count": int(disabled or 0),
    }


def build_removal_preview(ticker: str) -> dict[str, Any]:
    clean = normalize_ticker(ticker)
    asset = tracked_asset(clean, "public")
    if asset is None:
        raise ValueError(f"{clean} has not entered the public Forward Record.")
    if asset.status != "retired":
        raise ValueError(f"{clean} must be retired before its public data can be removed.")

    path = get_forward_record_csv_path(clean, "public")
    file_exists = os.path.isfile(path)
    digest = None
    file_size = 0
    if file_exists:
        approved = load_approved_data(clean, "public")
        digest = frame_digest(approved)
        file_size = os.path.getsize(path)
        if asset.last_approved_digest and digest != asset.last_approved_digest:
            raise ValueError("Approved public CSV changed outside the approval workflow.")

    today = utcnow().date()
    early = bool(asset.removal_after_date and today < asset.removal_after_date)
    payload = {
        "asset_id": asset.id,
        "ticker": clean,
        "status": asset.status,
        "retirement_date": asset.retirement_date.isoformat(),
        "removal_after_date": asset.removal_after_date.isoformat(),
        "today": today.isoformat(),
        "early_removal": early,
        "file_exists": file_exists,
        "file_digest": digest,
        "file_size": file_size,
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**payload, "approval_signature": signature}


def print_removal_preview(preview: dict[str, Any]) -> None:
    timing = "EARLY" if preview["early_removal"] else "SCHEDULED"
    print(
        f"REMOVE-PREVIEW-{timing} {preview['ticker']}: "
        f"announced_removal={preview['removal_after_date']} "
        f"file_exists={preview['file_exists']} bytes={preview['file_size']}"
    )
    print(
        "Removal deletes the approved public CSV and removes the asset from the "
        "customer performance selector. A tiny PostgreSQL lifecycle record remains."
    )


def remove_retired_public_data(
    *,
    admin_user_id: int,
    ticker: str,
    expected_approval_signature: str,
) -> dict[str, Any]:
    admin = db.session.get(User, int(admin_user_id))
    if admin is None:
        raise ValueError("Admin user was not found.")

    clean = normalize_ticker(ticker)
    asset = (
        ForwardRecordAsset.query
        .filter_by(record_mode="public", ticker=clean)
        .with_for_update()
        .first()
    )
    if asset is None or asset.status != "retired":
        db.session.rollback()
        return {
            "ticker": clean,
            "removed": False,
            "approval_mismatch": True,
        }

    preview = build_removal_preview(clean)
    if (
        not expected_approval_signature
        or preview["approval_signature"] != expected_approval_signature
    ):
        db.session.rollback()
        return {**preview, "removed": False, "approval_mismatch": True}

    path = Path(get_forward_record_csv_path(clean, "public"))
    backup = path.with_name(f".{path.name}.{uuid.uuid4().hex}.remove-bak")
    had_file = path.exists()
    if had_file:
        os.replace(path, backup)

    try:
        now = utcnow()
        asset.status = "removed"
        asset.removed_at = now
        asset.removed_by_user_id = admin.id
        asset.updated_at = now
        db.session.commit()
    except Exception:
        db.session.rollback()
        if had_file and backup.exists():
            os.replace(backup, path)
        raise
    else:
        _safe_remove(backup)

    print(
        f"REMOVED-PUBLIC-DATA {clean}: approved_csv_deleted={had_file} "
        f"removed_at={asset.removed_at.isoformat()}"
    )
    return {
        **preview,
        "removed": True,
        "approval_mismatch": False,
        "approved_csv_deleted": had_file,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(RECORD_MODES), default="sandbox")
    parser.add_argument("--ticker")
    parser.add_argument("--admin-user-id", type=int)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preview", action="store_true")
    group.add_argument("--publish", action="store_true")
    group.add_argument("--reset-sandbox", action="store_true")
    group.add_argument("--retire", action="store_true")
    group.add_argument("--remove-retired-data", action="store_true")
    parser.add_argument("--approval-signature")
    parser.add_argument("--reason")
    parser.add_argument("--removal-after")
    args = parser.parse_args()

    with app.app_context():
        if args.preview:
            preview = build_publication_preview(args.ticker, record_mode=args.mode)
            print_preview(preview)
            return 1 if preview["error_count"] else 0
        if args.publish:
            if not args.admin_user_id or not args.approval_signature:
                parser.error("--publish requires --admin-user-id and --approval-signature")
            publish_forward_batch(
                admin_user_id=args.admin_user_id,
                expected_approval_signature=args.approval_signature,
                ticker=args.ticker,
                record_mode=args.mode,
            )
            return 0
        if args.reset_sandbox:
            if not args.admin_user_id:
                parser.error("--reset-sandbox requires --admin-user-id")
            reset_sandbox_record(admin_user_id=args.admin_user_id)
            return 0
        if args.retire:
            if not all([args.admin_user_id, args.ticker, args.reason, args.approval_signature]):
                parser.error(
                    "--retire requires --admin-user-id, --ticker, --reason and "
                    "--approval-signature"
                )
            retire_public_asset(
                admin_user_id=args.admin_user_id,
                ticker=args.ticker,
                reason=args.reason,
                removal_after_date=args.removal_after,
                expected_approval_signature=args.approval_signature,
            )
            return 0
        if args.remove_retired_data:
            if not all([args.admin_user_id, args.ticker, args.approval_signature]):
                parser.error(
                    "--remove-retired-data requires --admin-user-id, --ticker and "
                    "--approval-signature"
                )
            remove_retired_public_data(
                admin_user_id=args.admin_user_id,
                ticker=args.ticker,
                expected_approval_signature=args.approval_signature,
            )
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
