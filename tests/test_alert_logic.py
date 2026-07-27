from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd

import app as application
from tools.send_signal_change_alerts import build_pending_event


def frame(signals):
    dates = pd.date_range("2026-07-20", periods=len(signals), freq="D")
    return pd.DataFrame(
        {
            "Close": [100.0 + index for index in range(len(signals))],
            "epoch_signal": signals,
        },
        index=dates,
    )


def item(signal=1, observed=date(2026, 7, 20), fingerprint=None):
    return SimpleNamespace(
        id=1,
        user_id=10,
        ticker="BTC-USD",
        last_observed_signal=signal,
        last_observed_signal_date=observed,
        last_observed_row_fingerprint=fingerprint,
    )


def test_first_observation_only_establishes_baseline():
    pending = build_pending_event(
        item(signal=None, observed=None),
        frame([1, 1]),
    )
    assert pending["baseline_only"] is True
    assert pending["event"] is None


def test_repeated_signals_do_not_create_email_event():
    pending = build_pending_event(item(signal=1), frame([1, 1, 1]))
    assert pending["event"] is None
    assert pending["final_signal"] == 1


def test_buy_hold_buy_creates_one_catch_up_with_two_transitions():
    pending = build_pending_event(item(signal=1), frame([1, 0, 1, 1]))
    event = pending["event"]
    assert event["event_type"] == "catch_up"
    assert event["final_signal"] == 1
    assert [(t["previous_signal"], t["current_signal"]) for t in event["transitions"]] == [
        (1, 0),
        (0, 1),
    ]


def test_buy_sell_hold_creates_one_catch_up_with_two_transitions():
    pending = build_pending_event(item(signal=1), frame([1, -1, 0, 0]))
    event = pending["event"]
    assert event["event_type"] == "catch_up"
    assert event["final_signal"] == 0
    assert len(event["transitions"]) == 2


def test_same_date_signal_change_is_revision():
    original_fingerprint = application.make_signal_row_fingerprint(
        "BTC-USD", date(2026, 7, 20), 1, 100.0
    )
    pending = build_pending_event(
        item(signal=1, fingerprint=original_fingerprint),
        frame([-1]),
    )
    event = pending["event"]
    assert event["event_type"] == "revision"
    assert event["final_signal"] == -1
