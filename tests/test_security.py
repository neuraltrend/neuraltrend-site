from __future__ import annotations


def test_csrf_rejects_mutating_request_without_token(app, client):
    app.config["WTF_CSRF_ENABLED"] = True
    try:
        response = client.post(
            "/login",
            json={"email": "user@example.com", "password": "anything"},
        )
        assert response.status_code == 400
        payload = response.get_json()
        assert payload["code"] == "csrf_failed"
    finally:
        app.config["WTF_CSRF_ENABLED"] = False


def test_unsupported_ticker_is_rejected(client):
    response = client.post(
        "/equity",
        data={"ticker": "../../etc/passwd", "duration": "1y"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Unsupported asset ticker."


def test_pro_asset_is_locked_for_anonymous_user(client):
    response = client.post(
        "/equity",
        data={"ticker": "NVDA", "duration": "1y"},
    )
    assert response.status_code == 403
    payload = response.get_json()
    assert payload["upgrade_required"] is True


def test_oversized_request_is_rejected(client):
    response = client.post(
        "/login",
        data=b"x" * (65 * 1024),
        content_type="application/json",
    )
    assert response.status_code == 413


def test_stripe_webhook_can_exceed_ordinary_request_limit(client, monkeypatch):
    # The webhook route must reach its own configuration/signature checks rather
    # than Flask rejecting it at the ordinary 64 KiB browser/API threshold.
    response = client.post(
        "/stripe/webhook",
        data=b"x" * (65 * 1024),
        content_type="application/json",
        headers={"Stripe-Signature": "invalid"},
    )
    assert response.status_code != 413


def test_security_headers_and_public_methodology_cache_behavior(client):
    response = client.get("/methodology")
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy-Report-Only" in response.headers
    assert response.headers.get("X-Robots-Tag") is None
    assert "no-store" not in response.headers.get("Cache-Control", "")


def test_sensitive_unsubscribe_path_is_redacted(app):
    from operational_logging import safe_request_path
    with app.test_request_context("/signal-alerts/unsubscribe/super-secret-token"):
        assert safe_request_path() == "/signal-alerts/unsubscribe/[REDACTED]"
