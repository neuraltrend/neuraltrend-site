from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from extensions import db
from models import ForwardRecordAsset, WatchlistItem
from tools import publish_forward_record as forward


def write_working_csv(path: Path, rows):
    pd.DataFrame(rows, columns=["Date", "Close", "epoch_signal"]).to_csv(
        path, index=False
    )


def test_sandbox_publish_appends_only_after_admin_approval(
    app,
    create_user,
    monkeypatch,
    tmp_path,
):
    admin_id = create_user(email="admin@example.com")
    working = tmp_path / "epoch_BTC.csv"
    today = date.today()
    write_working_csv(working, [(today.isoformat(), 100, 1)])
    monkeypatch.setattr(forward, "get_epoch_csv_path", lambda ticker: str(working))

    with app.app_context():
        preview = forward.build_publication_preview("BTC-USD", record_mode="sandbox")
        approved_path = Path(
            forward.get_forward_record_csv_path("BTC-USD", "sandbox")
        )
        assert not approved_path.exists()

        result = forward.publish_forward_batch(
            admin_user_id=admin_id,
            expected_approval_signature=preview["approval_signature"],
            ticker="BTC-USD",
            record_mode="sandbox",
        )
        assert result["published"] == 1
        assert approved_path.exists()
        assert list(pd.read_csv(approved_path).columns) == [
            "Date",
            "Close",
            "epoch_signal",
        ]

        next_day = today + timedelta(days=1)
        write_working_csv(
            working,
            [
                (today.isoformat(), 100, 1),
                (next_day.isoformat(), 105, 0),
            ],
        )
        preview = forward.build_publication_preview("BTC-USD", record_mode="sandbox")
        assert preview["candidates"][0]["classification"] == "append"
        forward.publish_forward_batch(
            admin_user_id=admin_id,
            expected_approval_signature=preview["approval_signature"],
            ticker="BTC-USD",
            record_mode="sandbox",
        )
        assert len(pd.read_csv(approved_path)) == 2


def test_working_history_changes_do_not_rewrite_approved_record(
    app,
    create_user,
    monkeypatch,
    tmp_path,
):
    admin_id = create_user(email="admin@example.com")
    working = tmp_path / "epoch_BTC.csv"
    today = date.today()
    next_day = today + timedelta(days=1)
    write_working_csv(working, [(today.isoformat(), 100, 1)])
    monkeypatch.setattr(forward, "get_epoch_csv_path", lambda ticker: str(working))

    with app.app_context():
        preview = forward.build_publication_preview("BTC-USD", record_mode="sandbox")
        forward.publish_forward_batch(
            admin_user_id=admin_id,
            expected_approval_signature=preview["approval_signature"],
            ticker="BTC-USD",
            record_mode="sandbox",
        )
        approved_path = Path(
            forward.get_forward_record_csv_path("BTC-USD", "sandbox")
        )

        # The working CSV may revise an old row used by Signal Overview. The
        # approved Forward Record keeps its original immutable value and only
        # the genuinely new date becomes a publication candidate.
        write_working_csv(
            working,
            [
                (today.isoformat(), 999, -1),
                (next_day.isoformat(), 105, 0),
            ],
        )
        preview = forward.build_publication_preview("BTC-USD", record_mode="sandbox")
        assert preview["error_count"] == 0
        assert preview["new_row_count"] == 1
        forward.publish_forward_batch(
            admin_user_id=admin_id,
            expected_approval_signature=preview["approval_signature"],
            ticker="BTC-USD",
            record_mode="sandbox",
        )

        approved = pd.read_csv(approved_path)
        assert len(approved) == 2
        assert float(approved.iloc[0]["Close"]) == 100
        assert int(approved.iloc[0]["epoch_signal"]) == 1
        assert float(approved.iloc[1]["Close"]) == 105


def test_approved_file_tampering_is_still_blocked(
    app,
    create_user,
    monkeypatch,
    tmp_path,
):
    admin_id = create_user(email="admin@example.com")
    working = tmp_path / "epoch_BTC.csv"
    today = date.today()
    write_working_csv(working, [(today.isoformat(), 100, 1)])
    monkeypatch.setattr(forward, "get_epoch_csv_path", lambda ticker: str(working))

    with app.app_context():
        preview = forward.build_publication_preview("BTC-USD", record_mode="sandbox")
        forward.publish_forward_batch(
            admin_user_id=admin_id,
            expected_approval_signature=preview["approval_signature"],
            ticker="BTC-USD",
            record_mode="sandbox",
        )
        approved_path = Path(
            forward.get_forward_record_csv_path("BTC-USD", "sandbox")
        )
        write_working_csv(approved_path, [(today.isoformat(), 100, -1)])

        preview = forward.build_publication_preview("BTC-USD", record_mode="sandbox")
        assert preview["error_count"] == 1
        assert "changed outside the approval workflow" in preview["errors"][0]


def test_blank_batch_updates_only_already_enrolled_assets(
    app,
    create_user,
    monkeypatch,
    tmp_path,
):
    admin_id = create_user(email="admin@example.com")
    today = date.today()
    btc = tmp_path / "epoch_BTC.csv"
    eth = tmp_path / "epoch_ETH.csv"
    write_working_csv(btc, [(today.isoformat(), 100, 1)])
    write_working_csv(eth, [(today.isoformat(), 200, -1)])
    paths = {"BTC-USD": btc, "ETH-USD": eth}
    monkeypatch.setattr(
        forward,
        "get_epoch_csv_path",
        lambda ticker: str(paths[ticker]),
    )

    with app.app_context():
        preview = forward.build_publication_preview("BTC-USD", record_mode="sandbox")
        forward.publish_forward_batch(
            admin_user_id=admin_id,
            expected_approval_signature=preview["approval_signature"],
            ticker="BTC-USD",
            record_mode="sandbox",
        )
        blank = forward.build_publication_preview(record_mode="sandbox")
        assert "ETH-USD" not in blank["unchanged"]
        assert all(c["ticker"] == "BTC-USD" for c in blank["candidates"])
        assert ForwardRecordAsset.query.filter_by(
            record_mode="sandbox", ticker="ETH-USD"
        ).first() is None


def test_retirement_disables_alerts_and_removal_deletes_public_csv(
    app,
    create_user,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("FORWARD_RECORD_PUBLIC_ENABLED", "true")
    admin_id = create_user(email="admin@example.com")
    pro_id = create_user(
        email="pro@example.com",
        subscription_type="pro",
        subscription_status="active",
    )
    working = tmp_path / "epoch_BTC.csv"
    today = date.today()
    write_working_csv(working, [(today.isoformat(), 100, 1)])
    monkeypatch.setattr(forward, "get_epoch_csv_path", lambda ticker: str(working))

    with app.app_context():
        preview = forward.build_publication_preview("BTC-USD", record_mode="public")
        forward.publish_forward_batch(
            admin_user_id=admin_id,
            expected_approval_signature=preview["approval_signature"],
            ticker="BTC-USD",
            record_mode="public",
        )
        item = WatchlistItem(
            user_id=pro_id,
            ticker="BTC-USD",
            email_alert_enabled=True,
            last_observed_signal=1,
            last_observed_signal_date=today,
        )
        db.session.add(item)
        db.session.commit()

        retire_preview = forward.build_retirement_preview(
            "BTC-USD",
            "Reliable market data is no longer available.",
            today + timedelta(days=30),
        )
        retired = forward.retire_public_asset(
            admin_user_id=admin_id,
            ticker="BTC-USD",
            reason=retire_preview["retirement_reason"],
            removal_after_date=retire_preview["removal_after_date"],
            expected_approval_signature=retire_preview["approval_signature"],
        )
        assert retired["retired"] is True
        assert db.session.get(WatchlistItem, item.id).email_alert_enabled is False

        remove_preview = forward.build_removal_preview("BTC-USD")
        approved_path = Path(
            forward.get_forward_record_csv_path("BTC-USD", "public")
        )
        assert approved_path.exists()
        removed = forward.remove_retired_public_data(
            admin_user_id=admin_id,
            ticker="BTC-USD",
            expected_approval_signature=remove_preview["approval_signature"],
        )
        assert removed["removed"] is True
        assert not approved_path.exists()
        asset = ForwardRecordAsset.query.filter_by(
            record_mode="public", ticker="BTC-USD"
        ).one()
        assert asset.status == "removed"
