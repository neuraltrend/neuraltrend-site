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


def test_signal_stat_horizon_selection_and_missing_file_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(application, "DATA_DIR", str(tmp_path))

    (tmp_path / "stat_BTC.csv").write_text(
        "window,buy_and_hold_return,strategy_return,alpha,buy_and_hold_prob,strategy_prob,alpha_prob\n"
        "1,1,1.00,1.00,0.5,0.20,0.40\n"
        "7,1,1.11,1.08,0.5,0.49,0.62\n"
        "10,1,1.22,1.15,0.5,0.55,0.70\n",
        encoding="utf-8",
    )

    one_week = application.get_signal_stat_for_horizon("BTC-USD", 7)
    assert one_week["stat_window"] == 7
    assert one_week["alpha"] == pytest.approx(1.08)
    assert one_week["alpha_prob"] == pytest.approx(0.62)
    assert one_week["strategy_avg_return"] == pytest.approx(1.11)
    assert one_week["strategy_profit_prob"] == pytest.approx(0.49)
    assert one_week["recommended_days"] == 10

    longer_than_available = application.get_signal_stat_for_horizon("BTC-USD", 30)
    assert longer_than_available["stat_window"] == 10
    assert longer_than_available["alpha"] == pytest.approx(1.15)
    assert longer_than_available["strategy_avg_return"] == pytest.approx(1.22)
    assert longer_than_available["strategy_profit_prob"] == pytest.approx(0.55)
    assert longer_than_available["recommended_days"] == 10

    maximum = application.get_signal_stat_for_horizon("BTC-USD", None)
    assert maximum["stat_window"] == 10
    assert maximum["recommended_days"] == 10

    missing = application.get_signal_stat_for_horizon("ETH-USD", 7)
    assert missing == {
        "alpha": None,
        "alpha_prob": None,
        "strategy_avg_return": None,
        "strategy_profit_prob": None,
        "recommended_days": None,
        "stat_window": None,
    }


def test_signal_summary_outperformance_ratio_uses_terminal_values(monkeypatch, sample_market_frame):
    monkeypatch.setattr(
        application,
        "load_epoch_csv_for_ticker",
        lambda ticker: sample_market_frame.copy(),
    )
    monkeypatch.setattr(application, "get_transaction_cost_rate", lambda ticker: 0.0)

    application.cache.clear()
    result = application.compute_signals_for_ticker(
        "BTC-USD",
        None,
        csv_version=("test",),
    )

    assert result is not None
    bh_final = 1.0 + result["buy_hold_period_return"]
    ai_final = 1.0 + result["strategy_period_return"]
    assert result["outperformance_ratio"] == pytest.approx(ai_final / bh_final)
