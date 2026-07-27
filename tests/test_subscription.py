from __future__ import annotations

from types import SimpleNamespace

import app as application


def subscription(interval="monthly", *, cancel=False):
    price_id = (
        application.STRIPE_PRO_MONTHLY_PRICE_ID
        if interval == "monthly"
        else application.STRIPE_PRO_ANNUAL_PRICE_ID
    )
    return {
        "id": "sub_test",
        "customer": "cus_test",
        "status": "active",
        "cancel_at_period_end": cancel,
        "current_period_end": 1_800_000_000,
        "items": {
            "data": [
                {
                    "id": "si_test",
                    "quantity": 1,
                    "price": {"id": price_id},
                }
            ]
        },
    }


def test_subscription_helpers_recognize_monthly_and_annual():
    assert application.stripe_subscription_uses_pro_price(subscription("monthly"))
    assert application.stripe_subscription_uses_pro_price(subscription("annual"))
    assert application.pro_subscription_item(subscription("annual"))[1] == "annual"


def test_management_payload_reports_pending_cancellation(create_user, app):
    user_id = create_user(
        subscription_type="pro",
        subscription_status="active",
    )
    with app.app_context():
        from extensions import db
        from models import User

        user = db.session.get(User, user_id)
        payload = application.subscription_management_payload(
            user,
            subscription("monthly", cancel=True),
        )
    assert payload["is_paid"] is True
    assert payload["current_interval"] == "monthly"
    assert payload["cancel_at_period_end"] is True


def test_billing_portal_switch_builds_annual_flow(
    client,
    create_user,
    login,
    monkeypatch,
):
    create_user(
        stripe_customer_id="cus_test",
        stripe_subscription_id="sub_test",
        subscription_type="pro",
        subscription_status="active",
    )
    assert login().status_code == 200

    captured = {}
    monkeypatch.setattr(
        application,
        "get_current_user_pro_subscription",
        lambda user: subscription("monthly", cancel=True),
    )

    def create_portal(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(url="https://billing.example/session")

    monkeypatch.setattr(
        application.stripe.billing_portal.Session,
        "create",
        create_portal,
    )

    response = client.post(
        "/billing-portal",
        json={"action": "switch", "billing_interval": "annual"},
    )
    assert response.status_code == 200
    flow = captured["flow_data"]["subscription_update_confirm"]
    assert flow["items"][0]["price"] == application.STRIPE_PRO_ANNUAL_PRICE_ID


def test_billing_portal_resume_removes_scheduled_cancellation(
    client,
    create_user,
    login,
    monkeypatch,
):
    create_user(
        stripe_customer_id="cus_test",
        stripe_subscription_id="sub_test",
        subscription_type="pro",
        subscription_status="active",
    )
    assert login().status_code == 200
    current = subscription("monthly", cancel=True)
    monkeypatch.setattr(
        application,
        "get_current_user_pro_subscription",
        lambda user: current,
    )

    calls = []

    def modify(subscription_id, **kwargs):
        calls.append((subscription_id, kwargs))
        updated = subscription("monthly", cancel=False)
        return updated

    monkeypatch.setattr(application.stripe.Subscription, "modify", modify)
    response = client.post("/billing-portal", json={"action": "resume"})
    assert response.status_code == 200
    assert calls == [("sub_test", {"cancel_at_period_end": False})]


def test_webhook_event_ledger_deduplicates_processed_event(app):
    event = {
        "id": "evt_test_1",
        "type": "customer.subscription.updated",
        "livemode": False,
        "created": 1_700_000_000,
        "data": {"object": {"id": "sub_test"}},
    }
    with app.app_context():
        record, state = application.begin_stripe_webhook_event(event)
        assert state == "claimed"
        application.finish_stripe_webhook_event(record, "processed")
        duplicate, duplicate_state = application.begin_stripe_webhook_event(event)
        assert duplicate.id == record.id
        assert duplicate_state == "duplicate"


def test_authoritative_subscription_prefers_active_status():
    selected = application.select_authoritative_pro_subscription(
        [
            {"id": "sub_old", "status": "canceled", "created": 200},
            {"id": "sub_live", "status": "active", "created": 100},
        ]
    )
    assert selected["id"] == "sub_live"
