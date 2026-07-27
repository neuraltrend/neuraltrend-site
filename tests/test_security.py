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
