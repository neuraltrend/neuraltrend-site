from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/dashboard",
        "/resources",
        "/subscription",
        "/contact",
        "/methodology",
        "/performance",
        "/privacy",
        "/terms",
        "/risk-disclaimer",
        "/refund-policy",
        "/ai-crypto-trading-signals",
        "/crypto-paper-trading-simulator",
        "/buy-and-hold-vs-ai-strategy",
    ],
)
def test_public_pages_render(client, path):
    response = client.get(path)
    assert response.status_code == 200


def test_health_check_reports_database_and_storage(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {
        "status": "ok",
        "checks": {
            "database": "ok",
            "forward_record_storage": "ok",
        },
    }
    assert response.headers["Cache-Control"].startswith("no-store")


def test_health_check_is_exempt_from_global_rate_limits(app):
    # The test environment intentionally imports the application with rate
    # limiting disabled, so Flask-Limiter does not create a storage backend.
    # Verify the exemption registration directly instead of trying to enable
    # and reset an extension that was never initialized in this process.
    import app as application
    from flask_limiter.constants import ExemptionScope

    exemption_scope = application.limiter.limit_manager.exemption_scope(
        app,
        "healthz",
        None,
    )

    assert exemption_scope & ExemptionScope.DEFAULT
    assert exemption_scope & ExemptionScope.APPLICATION
    assert exemption_scope & ExemptionScope.META


def test_admin_routes_require_login(client):
    assert client.get("/admin/signal-alerts").status_code == 401
    assert client.get("/admin/forward-record").status_code == 401


def test_non_admin_cannot_open_admin_dispatch(authenticated_client):
    response = authenticated_client.get("/admin/signal-alerts")
    assert response.status_code == 404


def test_admin_can_open_admin_dispatch(admin_client):
    response = admin_client.get("/admin/signal-alerts")
    assert response.status_code == 200
