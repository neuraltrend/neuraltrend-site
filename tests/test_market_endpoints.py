from __future__ import annotations

from datetime import datetime, timedelta

import app as application


REQUIRED_RESULT_FIELDS = {
    "buy_hold_period_return",
    "strategy_period_return",
    "return_spread",
    "buy_hold_growth_factor",
    "strategy_growth_factor",
    "strategy_max_drawdown",
    "buy_hold_max_drawdown",
    "strategy_annualized_volatility",
    "buy_hold_annualized_volatility",
    "strategy_market_exposure",
    "executed_trade_count",
}


def test_equity_endpoint_returns_standardized_metrics(
    client,
    monkeypatch,
    sample_market_frame,
):
    monkeypatch.setattr(
        application,
        "load_epoch_csv_for_ticker",
        lambda ticker: sample_market_frame.copy(),
    )
    response = client.post(
        "/equity",
        data={"ticker": "BTC-USD", "duration": "max"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert REQUIRED_RESULT_FIELDS.issubset(payload)
    assert payload["valuation_policy"] == "mark_to_market"
    assert len(payload["dates"]) == len(sample_market_frame)


def test_backtest_endpoint_returns_standardized_metrics(
    client,
    monkeypatch,
    sample_market_frame,
):
    monkeypatch.setattr(
        application,
        "load_epoch_csv_for_ticker",
        lambda ticker: sample_market_frame.copy(),
    )
    start = sample_market_frame.index.min().date().isoformat()
    response = client.post(
        "/backtest",
        data={
            "ticker": "BTC-USD",
            "cash": "10000",
            "start": start,
            "duration": "1y",
            "dca_pct": "100_pct",
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert REQUIRED_RESULT_FIELDS.issubset(payload)
    assert payload["position_size_pct"] == 100
    assert payload["executed_trade_count"] == (
        len(payload["executed_buy_dates"]) + len(payload["executed_sell_dates"])
    )


def test_backtest_rejects_future_start_date(client):
    future = (datetime.utcnow().date() + timedelta(days=1)).isoformat()
    response = client.post(
        "/backtest",
        data={
            "ticker": "BTC-USD",
            "cash": "10000",
            "start": future,
            "duration": "1y",
            "dca_pct": "100_pct",
        },
    )
    assert response.status_code == 400
    assert "future" in response.get_json()["error"].lower()


def test_summary_masking_hides_pro_signals_from_anonymous_user(app):
    row = {
        "ticker": "NVDA",
        "today_signal": 1,
        "yesterday_signal": 0,
        "last_week_signal": -1,
        "last_month_signal": 1,
        "buy_hold_period_return": 0.2,
        "strategy_period_return": 0.3,
        "return_spread": 0.1,
        "coverage_end": "2026-07-25",
    }
    with app.app_context():
        safe = application.mask_signal_summary_row_for_user(row, None)
    assert safe["signals_locked"] is True
    assert safe["today_signal"] is None
    assert safe["strategy_period_return"] == 0.3
