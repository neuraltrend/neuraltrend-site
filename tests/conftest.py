"""Isolated test configuration for NeuralTrend.

Environment variables are set before importing the application so SQLAlchemy,
Forward Record storage, Stripe IDs and email settings never point at production.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

SUITE_ROOT = Path(tempfile.mkdtemp(prefix="neuraltrend-tests-"))
DB_PATH = SUITE_ROOT / "neuraltrend.sqlite"
FORWARD_ROOT = SUITE_ROOT / "forward-record"

os.environ["NEURALTREND_TESTING"] = "1"
os.environ["SECRET_KEY"] = "neuraltrend-test-secret-key-not-for-production"
os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"
os.environ["BASE_URL"] = "http://localhost"
os.environ["EMAIL_USER"] = "test-sender@example.com"
os.environ["EMAIL_PASS"] = "test-email-password"
os.environ["ADMIN_EMAILS"] = "admin@example.com"
os.environ["STRIPE_SECRET_KEY"] = "sk_test_neuraltrend_unit_tests"
os.environ["STRIPE_PRO_PRICE_ID"] = "price_monthly_test"
os.environ["STRIPE_PRO_MONTHLY_PRICE_ID"] = "price_monthly_test"
os.environ["STRIPE_PRO_ANNUAL_PRICE_ID"] = "price_annual_test"
os.environ["STRIPE_WEBHOOK_SECRET"] = "whsec_neuraltrend_unit_tests"
os.environ["STRIPE_BILLING_PORTAL_CONFIGURATION_ID"] = "bpc_test"
os.environ["FORWARD_RECORD_STORAGE_DIR"] = str(FORWARD_ROOT)
os.environ["FORWARD_RECORD_PUBLIC_ENABLED"] = "false"
os.environ.pop("REDIS_URL", None)

import app as application  # noqa: E402
from extensions import db  # noqa: E402
from models import User  # noqa: E402

TEST_PASSWORD = "Correct-Horse-Battery-1!"
NEW_TEST_PASSWORD = "Correct-Horse-Battery-2!"


@pytest.fixture(scope="session")
def app():
    application.app.config.update(
        TESTING=True,
        SESSION_COOKIE_SECURE=False,
        WTF_CSRF_SSL_STRICT=False,
        WTF_CSRF_ENABLED=False,
        MAIL_SUPPRESS_SEND=True,
        PROPAGATE_EXCEPTIONS=True,
    )
    application.limiter.enabled = False
    return application.app


@pytest.fixture(autouse=True)
def isolated_state(app, monkeypatch):
    application.app.config["WTF_CSRF_ENABLED"] = False
    application.limiter.enabled = False
    monkeypatch.setattr(application.mail, "send", lambda message: None)

    with app.app_context():
        db.session.remove()
        db.drop_all()
        db.create_all()
        application.cache.clear()
        application.compute_signals_summary_cached.cache_clear()

    shutil.rmtree(FORWARD_ROOT, ignore_errors=True)
    FORWARD_ROOT.mkdir(parents=True, exist_ok=True)

    yield

    with app.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def runner(app):
    return app.test_cli_runner()


@pytest.fixture
def create_user(app):
    def _create(
        email="user@example.com",
        password=TEST_PASSWORD,
        *,
        verified=True,
        subscription_type="free",
        subscription_status="inactive",
        stripe_customer_id=None,
        stripe_subscription_id=None,
    ):
        with app.app_context():
            user = User(
                email=email.lower(),
                password_hash=application.bcrypt.generate_password_hash(
                    password
                ).decode("utf-8"),
                is_verified=verified,
                subscription_type=subscription_type,
                subscription_status=subscription_status,
                stripe_customer_id=stripe_customer_id,
                stripe_subscription_id=stripe_subscription_id,
            )
            db.session.add(user)
            db.session.commit()
            return user.id

    return _create


@pytest.fixture
def login(client):
    def _login(email="user@example.com", password=TEST_PASSWORD):
        return client.post(
            "/login",
            json={"email": email, "password": password},
        )

    return _login


@pytest.fixture
def authenticated_client(client, create_user, login):
    create_user()
    response = login()
    assert response.status_code == 200
    return client


@pytest.fixture
def admin_client(client, create_user, login):
    create_user(email="admin@example.com")
    response = login(email="admin@example.com")
    assert response.status_code == 200
    return client


@pytest.fixture
def pro_client(client, create_user, login):
    create_user(
        email="pro@example.com",
        subscription_type="pro",
        subscription_status="active",
    )
    response = login(email="pro@example.com")
    assert response.status_code == 200
    return client


@pytest.fixture
def sample_market_frame():
    import pandas as pd

    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    dates = pd.date_range(today - pd.Timedelta(days=5), periods=6, freq="D")
    frame = pd.DataFrame(
        {
            "Close": [100.0, 105.0, 103.0, 110.0, 108.0, 115.0],
            "epoch_signal": [0, 1, 1, 0, -1, 1],
        },
        index=dates,
    )
    frame.index.name = "Date"
    return frame


@pytest.fixture(scope="session", autouse=True)
def remove_suite_tempdir(request):
    def cleanup():
        shutil.rmtree(SUITE_ROOT, ignore_errors=True)

    request.addfinalizer(cleanup)
