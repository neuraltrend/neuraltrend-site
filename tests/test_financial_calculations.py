from __future__ import annotations

import math

import pytest

import app as application


def test_duration_parser_and_max_horizon():
    assert application.duration_to_days("1d") == 1
    assert application.duration_to_days("1y") == 365
    assert application.duration_to_days("max") is None
    with pytest.raises(ValueError):
        application.duration_to_days("forever")


def test_maximum_drawdown():
    result = application.calculate_max_drawdown([100, 120, 90, 95])
    assert result == pytest.approx(-0.25)


def test_volatility_and_sharpe_are_finite():
    curve = [100, 102, 101, 105, 104]
    volatility = application.calculate_annualized_volatility(curve, "BTC-USD")
    sharpe = application.calculate_sharpe_from_equity_curve(curve, "BTC-USD")
    assert math.isfinite(volatility) and volatility >= 0
    assert math.isfinite(sharpe)


def test_buy_and_hold_preserves_residual_cash_for_whole_stock_shares():
    quantity, cash, curve = application.build_buy_and_hold_benchmark(
        initial_cash=1000,
        prices=[333, 350],
        ticker="AAPL",
        transaction_cost_rate=0.001,
        enforce_whole_stock_shares=True,
    )
    assert quantity == 3
    assert cash > 0
    assert curve[0] == pytest.approx(quantity * 333 + cash)


def test_fractional_crypto_benchmark_invests_full_notional_less_cost():
    quantity, cash, curve = application.build_buy_and_hold_benchmark(
        initial_cash=1000,
        prices=[100, 110],
        ticker="BTC-USD",
        transaction_cost_rate=0.01,
        enforce_whole_stock_shares=False,
    )
    assert quantity > 0
    assert cash == pytest.approx(0.0, abs=1e-8)
    assert curve[-1] > curve[0]
