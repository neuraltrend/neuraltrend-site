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
    assert "freshness_counts" in listed_payload["portfolio_total"]
    assert sum(listed_payload["portfolio_total"]["freshness_counts"].values()) == 1

    portfolio_response = authenticated_client.get("/live-simulations/portfolio")
    assert portfolio_response.status_code == 200
    portfolio = portfolio_response.get_json()["portfolio"]
    assert portfolio["is_portfolio_total"] is True
    assert portfolio["simulation_count"] == 1
    assert portfolio["strategy_curve"] == [10000]
    assert portfolio["benchmark_curve"] == [10000]
    assert "freshness_counts" in portfolio
    assert sum(portfolio["freshness_counts"].values()) == 1


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



def test_live_simulation_portfolio_can_filter_by_simulation_ids(
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

    first = authenticated_client.post(
        "/live-simulations",
        json={
            "ticker": "BTC-USD",
            "name": "BTC filtered total",
            "initial_cash": 10000,
            "position_size_pct": 100,
        },
    )
    second = authenticated_client.post(
        "/live-simulations",
        json={
            "ticker": "ETH-USD",
            "name": "ETH filtered total",
            "initial_cash": 20000,
            "position_size_pct": 100,
        },
    )

    assert first.status_code == 201
    assert second.status_code == 201

    first_id = first.get_json()["simulation"]["id"]
    second_id = second.get_json()["simulation"]["id"]

    monkeypatch.setattr(
        application,
        "update_live_simulation_from_csv",
        lambda simulation, user: simulation,
    )

    filtered = authenticated_client.get(
        f"/live-simulations/portfolio?ids={first_id}&skip_refresh=1"
    )
    assert filtered.status_code == 200
    filtered_portfolio = filtered.get_json()["portfolio"]
    assert filtered_portfolio["simulation_count"] == 1
    assert filtered_portfolio["initial_cash"] == 10000

    both = authenticated_client.get(
        f"/live-simulations/portfolio?ids={first_id},{second_id}&skip_refresh=1"
    )
    assert both.status_code == 200
    both_portfolio = both.get_json()["portfolio"]
    assert both_portfolio["simulation_count"] == 2
    assert both_portfolio["initial_cash"] == 30000

    missing = authenticated_client.get(
        "/live-simulations/portfolio?ids=999999&skip_refresh=1"
    )
    assert missing.status_code == 404


def test_live_simulation_detail_skip_refresh_avoids_redundant_update(
    authenticated_client, monkeypatch, sample_market_frame
):
    monkeypatch.setattr(application, "load_epoch_csv_for_ticker", lambda ticker: sample_market_frame.copy())

    def initialize(simulation, user):
        simulation.last_processed_date = simulation.start_date
        db.session.commit()
        return simulation

    monkeypatch.setattr(application, "update_live_simulation_from_csv", initialize)
    created = authenticated_client.post(
        "/live-simulations",
        json={"ticker": "BTC-USD", "name": "skip refresh detail", "initial_cash": 10000, "position_size_pct": 100},
    )
    assert created.status_code == 201
    sim_id = created.get_json()["simulation"]["id"]

    def should_not_run(*args, **kwargs):
        raise AssertionError("detail refresh should have been skipped")

    monkeypatch.setattr(application, "update_live_simulation_from_csv", should_not_run)
    detail = authenticated_client.get(f"/live-simulations/{sim_id}?skip_refresh=1")
    assert detail.status_code == 200
    assert detail.get_json()["simulation"]["id"] == sim_id


def test_live_simulation_curve_only_endpoint_avoids_summary_refresh(authenticated_client, app_module, monkeypatch):
    """The initial chart request should be a lean curve/trade read, not a second full summary refresh."""
    application = app_module

    simulation = application.LiveSimulation.query.filter_by(user_id=1).first()
    if simulation is None:
        pytest.skip("Fixture has no live simulation to inspect.")

    def fail_summary(*args, **kwargs):
        raise AssertionError("curve_only must not rebuild live_simulation_summary")

    monkeypatch.setattr(application, "live_simulation_summary", fail_summary)

    response = authenticated_client.get(
        f"/live-simulations/{simulation.id}?curve_only=1"
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert "curve" in payload
    assert "dates" in payload["curve"]
    assert "strategy_curve" in payload["curve"]
    assert "benchmark_curve" in payload["curve"]


def test_live_simulation_portfolio_accepts_more_than_100_requested_ids(
    authenticated_client,
    monkeypatch,
    sample_market_frame,
):
    """Large filtered accounts must not fail at the old 100-ID ceiling."""
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
            "name": "Large filtered portfolio test",
            "initial_cash": 10000,
            "position_size_pct": 100,
        },
    )
    assert created.status_code == 201
    owned_id = created.get_json()["simulation"]["id"]

    monkeypatch.setattr(
        application,
        "update_live_simulation_from_csv",
        lambda simulation, user: simulation,
    )

    # Include >100 IDs.  Only the owned simulation should match the
    # ownership-scoped query, but the request itself must no longer be
    # rejected simply because the ID list exceeds 100 entries.
    requested_ids = [owned_id] + list(range(100000, 100150))
    response = authenticated_client.get(
        "/live-simulations/portfolio?skip_refresh=1&ids="
        + ",".join(str(value) for value in requested_ids)
    )

    assert response.status_code == 200
    portfolio = response.get_json()["portfolio"]
    assert portfolio["simulation_count"] == 1
    assert portfolio["initial_cash"] == 10000
