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
    listed_payload = listed.get_json()
    assert listed_payload["count"] == 1
    assert listed_payload["portfolio_total"]["simulation_count"] == 1
    assert listed_payload["portfolio_total"]["initial_cash"] == 10000
    assert listed_payload["portfolio_total"]["latest_strategy_value"] == 10000

    portfolio_response = authenticated_client.get("/live-simulations/portfolio")
    assert portfolio_response.status_code == 200
    portfolio = portfolio_response.get_json()["portfolio"]
    assert portfolio["is_portfolio_total"] is True
    assert portfolio["simulation_count"] == 1
    assert portfolio["strategy_curve"] == [10000]
    assert portfolio["benchmark_curve"] == [10000]


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

def test_live_simulation_default_name_includes_ticker_cash_and_position(
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
            "initial_cash": "10,000 $",
            "position_size_pct": 50,
        },
    )

    assert created.status_code == 201
    payload = created.get_json()["simulation"]
    assert payload["name"] == "BTC-USD_10000_50%"

