from __future__ import annotations

from datetime import date

import app as application


def signal_snapshot(ticker):
    return {
        "signal": 1,
        "signal_date": date(2026, 7, 25),
        "close_price": 100.0,
        "row_fingerprint": "f" * 64,
    }


def test_free_user_can_add_and_remove_free_asset(
    authenticated_client,
    monkeypatch,
):
    monkeypatch.setattr(application, "get_latest_signal_snapshot", signal_snapshot)
    added = authenticated_client.post("/watchlist", json={"ticker": "BTC-USD"})
    assert added.status_code == 201
    assert added.get_json()["item"]["ticker"] == "BTC-USD"

    removed = authenticated_client.delete("/watchlist/BTC-USD")
    assert removed.status_code == 200
    assert authenticated_client.get("/watchlist").get_json()["items"] == []


def test_free_user_cannot_add_pro_asset(authenticated_client):
    response = authenticated_client.post("/watchlist", json={"ticker": "NVDA"})
    assert response.status_code == 403
    assert response.get_json()["upgrade_required"] is True


def test_pro_user_can_add_pro_asset_and_enable_alerts(pro_client, monkeypatch):
    monkeypatch.setattr(application, "get_latest_signal_snapshot", signal_snapshot)
    added = pro_client.post("/watchlist", json={"ticker": "NVDA"})
    assert added.status_code == 201

    enabled = pro_client.patch(
        "/watchlist/NVDA/alerts",
        json={"enabled": True},
    )
    assert enabled.status_code == 200
    assert enabled.get_json()["item"]["email_alert_enabled"] is True


def test_free_user_cannot_enable_email_alerts(authenticated_client, monkeypatch):
    monkeypatch.setattr(application, "get_latest_signal_snapshot", signal_snapshot)
    assert authenticated_client.post(
        "/watchlist", json={"ticker": "BTC-USD"}
    ).status_code == 201
    response = authenticated_client.patch(
        "/watchlist/BTC-USD/alerts",
        json={"enabled": True},
    )
    assert response.status_code == 403
