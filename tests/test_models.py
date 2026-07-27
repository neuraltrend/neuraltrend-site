from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.exc import IntegrityError

from extensions import db
from models import ForwardRecordAsset, User, WatchlistItem


def test_watchlist_uniqueness(app, create_user):
    user_id = create_user()
    with app.app_context():
        db.session.add_all(
            [
                WatchlistItem(user_id=user_id, ticker="BTC-USD"),
                WatchlistItem(user_id=user_id, ticker="BTC-USD"),
            ]
        )
        with pytest.raises(IntegrityError):
            db.session.commit()
        db.session.rollback()


def test_forward_record_lifecycle_constraint_accepts_active_asset(app):
    with app.app_context():
        asset = ForwardRecordAsset(
            record_mode="sandbox",
            ticker="BTC-USD",
            status="active",
            start_date=date.today(),
        )
        db.session.add(asset)
        db.session.commit()
        assert asset.id is not None
