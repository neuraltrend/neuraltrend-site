from __future__ import annotations

import re

from operational_logging import redact_text


def test_public_responses_include_request_id(client):
    response = client.get("/")
    assert response.status_code == 200
    request_id = response.headers.get("X-Request-ID")
    assert request_id is not None
    assert re.fullmatch(r"[0-9a-f]{16}", request_id)


def test_health_response_includes_request_id(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert re.fullmatch(r"[0-9a-f]{16}", response.headers["X-Request-ID"])


def test_log_redaction_removes_common_credentials_and_email():
    fake_stripe_secret = "sk_" + "live_" + "abcdefghijklmnopqrstuvwxyz"
    fake_webhook_secret = "wh" + "sec_" + "abcdefghijklmnopqrstuvwxyz"
    message = (
        "email=user@example.com password=hunter2 "
        f"stripe={fake_stripe_secret} "
        f"webhook={fake_webhook_secret} "
        "database=postgresql://neural:secret-password@db.example.com/site "
        "authorization=Bearer abc.def.ghi"
    )
    safe = redact_text(message)

    assert "user@example.com" not in safe
    assert "hunter2" not in safe
    assert "sk_live_" not in safe
    assert "whsec_" not in safe
    assert "secret-password" not in safe
    assert "abc.def.ghi" not in safe
    assert "REDACTED" in safe


def test_admin_operations_requires_login(client):
    assert client.get("/admin/operations").status_code == 401
    assert client.get("/admin/operations.json").status_code == 401


def test_non_admin_cannot_open_operations(authenticated_client):
    assert authenticated_client.get("/admin/operations").status_code == 404
    assert authenticated_client.get("/admin/operations.json").status_code == 404


def test_admin_operations_page_and_json_are_privacy_safe(admin_client):
    page = admin_client.get("/admin/operations")
    assert page.status_code == 200
    assert b"Operational status" in page.data

    response = admin_client.get("/admin/operations.json")
    assert response.status_code == 200
    assert response.headers["Cache-Control"].startswith("no-store")
    payload = response.get_json()

    assert payload["status"] in {"ok", "warning", "error"}
    assert payload["database"]["status"] == "ok"
    assert payload["storage"]["status"] == "ok"
    assert "users" in payload["database"]["counts"]

    serialized = response.get_data(as_text=True).lower()
    for forbidden in (
        "database_url",
        "stripe_secret_key",
        "email_pass",
        "password_hash",
        "@example.com",
        "postgresql://",
        "sk_test_",
        "whsec_",
    ):
        assert forbidden not in serialized
