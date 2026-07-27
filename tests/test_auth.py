from __future__ import annotations

from conftest import NEW_TEST_PASSWORD, TEST_PASSWORD

import app as application


def test_verified_user_can_login_and_logout(client, create_user):
    create_user()
    response = client.post(
        "/login",
        json={"email": " USER@example.com ", "password": TEST_PASSWORD},
    )
    assert response.status_code == 200
    assert response.get_json()["email"] == "user@example.com"

    me = client.get("/me").get_json()
    assert me["email"] == "user@example.com"
    assert me["is_paid"] is False

    logout = client.post("/logout")
    assert logout.status_code == 200
    assert client.get("/me").get_json()["email"] is None


def test_unverified_user_is_rejected(client, create_user):
    create_user(verified=False)
    response = client.post(
        "/login",
        json={"email": "user@example.com", "password": TEST_PASSWORD},
    )
    assert response.status_code == 403
    assert response.get_json()["verification_required"] is True


def test_invalid_password_does_not_reveal_account(client, create_user):
    create_user()
    existing = client.post(
        "/login",
        json={"email": "user@example.com", "password": "Wrong-password-123!"},
    )
    missing = client.post(
        "/login",
        json={"email": "missing@example.com", "password": "Wrong-password-123!"},
    )
    assert existing.status_code == missing.status_code == 401
    assert existing.get_json()["error"] == missing.get_json()["error"]


def test_change_password_keeps_current_session_and_revokes_other_session(
    app,
    create_user,
):
    create_user()
    first = app.test_client()
    second = app.test_client()

    for browser in (first, second):
        response = browser.post(
            "/login",
            json={"email": "user@example.com", "password": TEST_PASSWORD},
        )
        assert response.status_code == 200

    changed = first.post(
        "/change-password",
        json={
            "current_password": TEST_PASSWORD,
            "new_password": NEW_TEST_PASSWORD,
            "confirm_password": NEW_TEST_PASSWORD,
        },
    )
    assert changed.status_code == 200
    assert first.get("/me").get_json()["email"] == "user@example.com"
    assert second.get("/me").get_json()["email"] is None

    first.post("/logout")
    old_login = first.post(
        "/login",
        json={"email": "user@example.com", "password": TEST_PASSWORD},
    )
    new_login = first.post(
        "/login",
        json={"email": "user@example.com", "password": NEW_TEST_PASSWORD},
    )
    assert old_login.status_code == 401
    assert new_login.status_code == 200


def test_signup_verification_and_login(client, app):
    signup = client.post(
        "/signup",
        json={
            "email": "new-user@example.com",
            "password": TEST_PASSWORD,
        },
    )
    assert signup.status_code == 200

    with app.app_context():
        from models import User

        user = User.query.filter_by(email="new-user@example.com").one()
        assert user.is_verified is False
        token = application.generate_verification_token(user.email)

    verified = client.get(f"/verify/{token}")
    assert verified.status_code == 200
    assert b"verified successfully" in verified.data.lower()
    assert client.post(
        "/login",
        json={"email": "new-user@example.com", "password": TEST_PASSWORD},
    ).status_code == 200


def test_password_reset_token_is_one_time(client, app, create_user):
    user_id = create_user()
    with app.app_context():
        from extensions import db
        from models import User

        user = db.session.get(User, user_id)
        token = application.generate_reset_token(user)
        db.session.commit()

    begin = client.get(f"/reset-password/{token}", follow_redirects=True)
    assert begin.status_code == 200

    reset = client.post(
        "/reset-password",
        data={
            "password": NEW_TEST_PASSWORD,
            "confirm_password": NEW_TEST_PASSWORD,
        },
    )
    assert reset.status_code == 200
    assert b"success" in reset.data.lower()

    reused = client.get(f"/reset-password/{token}", follow_redirects=True)
    assert reused.status_code == 400
    assert b"invalid" in reused.data.lower()
