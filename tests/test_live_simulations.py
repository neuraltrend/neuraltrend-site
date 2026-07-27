from __future__ import annotations

from extensions import db
from models import LiveSimulation

import app as application


def test_create_and_list_live_simulation(
    authenticated_client,
    monkeypatch,
    sample_market_frame,
):
    monkeypatch.setattr(
        application,
        "load_epoch_csv_for_ticker",
        lambda ticker: sample_market_frame.copy(),
    )

    def initialize(simulation, user):
        simulation.last_processed_date = simulation.start_date
        db.session.commit()
        return simulation

    monkeypatch.setattr(
        application,
        "update_live_simulation_from_csv",
        initialize,
    )

    created = authenticated_client.post(
        "/live-simulations",
        json={
            "ticker": "BTC-USD",
            "name": "BTC test",
            "initial_cash": 10000,
            "position_size_pct": 100,
        },
    )
    assert created.status_code == 201
    payload = created.get_json()["simulation"]
    assert payload["ticker"] == "BTC-USD"
    assert payload["status"] == "active"

    monkeypatch.setattr(
        application,
        "update_live_simulation_from_csv",
        lambda simulation, user: simulation,
    )
    listed = authenticated_client.get("/live-simulations")
    assert listed.status_code == 200
    assert listed.get_json()["count"] == 1


def test_live_simulation_rejects_pro_asset_for_free_user(authenticated_client):
    response = authenticated_client.post(
        "/live-simulations",
        json={
            "ticker": "NVDA",
            "initial_cash": 10000,
            "position_size_pct": 100,
        },
    )
    assert response.status_code == 403
    assert response.get_json()["upgrade_required"] is True
