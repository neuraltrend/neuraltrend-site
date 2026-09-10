from __future__ import annotations

import time

import app as application


def test_admin_routes_require_mfa(client, create_user):
    create_user(email="admin@example.com")
    login = client.post(
        "/login",
        json={"email": "admin@example.com", "password": "Correct-Horse-Battery-1!"},
    )
    assert login.status_code == 200
    response = client.get("/admin/operations")
    assert response.status_code == 302
    assert "/admin/mfa" in response.headers["Location"]


def test_admin_totp_helper_accepts_current_code():
    secret = "JBSWY3DPEHPK3PXP"
    now = int(time.time())
    code = application._totp_code(secret, now // 30)
    assert code is not None
    assert application.verify_admin_totp(secret, code, now=now) is True
    assert application.verify_admin_totp(secret, "000000", now=now) is False or code == "000000"
