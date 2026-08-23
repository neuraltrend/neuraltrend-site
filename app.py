from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    send_file,
    flash,
    Response,
    session,
    redirect,
    url_for,
    abort,
)
from flask_login import login_user, logout_user, login_required, current_user
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
from functools import lru_cache
from extensions import db, bcrypt, login_manager, csrf
from models import (
    User,
    LiveSimulation,
    LiveSimulationTrade,
    LiveSimulationEquity,
    StripeWebhookEvent,
    WatchlistItem,
    SignalAlertDelivery,
    ForwardRecordAsset,
    ForwardPublicationBatch,
)
from itsdangerous import URLSafeTimedSerializer
from flask_mail import Mail, Message
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.exceptions import RequestEntityTooLarge
import stripe
import traceback
import secrets
import hashlib
import hmac
import time
import math
import re
import unicodedata
import json
import shutil
from flask_wtf.csrf import CSRFError
from sqlalchemy import text
from operational_logging import (
    configure_operational_logging,
    register_request_observability,
)
from sqlalchemy.exc import IntegrityError
from pathlib import Path
from backup_manager import (
    create_database_backup,
    create_forward_backup,
    delete_backup,
    enforce_retention,
    list_backups,
    resolve_managed_file,
    sha256_file,
)

app = Flask(__name__)

TESTING_MODE = os.environ.get("NEURALTREND_TESTING", "").strip().lower() in {
    "1", "true", "yes", "on"
}

configure_operational_logging(app, testing_mode=TESTING_MODE)
register_request_observability(app)

# ✅ Fix proxy handling (Render-safe)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
app.config["RATELIMIT_ENABLED"] = not TESTING_MODE

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri=os.environ.get("REDIS_URL"),  # 🔑 important
    default_limits=["200 per day", "50 per hour"]
)

# ✅ SET CONFIG FIRST
secret = os.environ.get("SECRET_KEY")
if not secret:
    raise RuntimeError("SECRET_KEY not set!")

app.config["SECRET_KEY"] = secret

app.config.update(
    MAIL_SERVER="smtp.gmail.com",
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME=os.environ.get("EMAIL_USER"),
    MAIL_PASSWORD=os.environ.get("EMAIL_PASS"),
    MAIL_SUPPRESS_SEND=TESTING_MODE,
)

mail = Mail(app)

BASE_URL = os.environ.get("BASE_URL", "https://neuraltrend.org").rstrip("/")

# 🔐 REQUIRED FOR SESSIONS
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# 🔒 Cookie security (recommended)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = True  # important on Render HTTPS

# CSRF tokens are bound to the current Flask session. A longer time limit avoids
# interrupting users who keep the dashboard open for several hours.
app.config["WTF_CSRF_TIME_LIMIT"] = 12 * 60 * 60
app.config["WTF_CSRF_SSL_STRICT"] = True

# Bound browser request bodies. NeuralTrend forms and JSON payloads are tiny;
# rejecting oversized bodies reduces accidental/malicious memory and parser use.
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # 64 KiB
app.config["MAX_FORM_MEMORY_SIZE"] = 64 * 1024

# Test runs are isolated through environment variables before app import. Keep
# production defaults strict while making Flask's test client deterministic.
app.config["TESTING"] = TESTING_MODE
if TESTING_MODE:
    app.config["SESSION_COOKIE_SECURE"] = False
    app.config["WTF_CSRF_SSL_STRICT"] = False
    app.config["MAIL_SUPPRESS_SEND"] = True
    app.config["PROPAGATE_EXCEPTIONS"] = True

# Stripe
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")

# Backward compatibility: existing deployments can keep STRIPE_PRO_PRICE_ID as
# the monthly price. New deployments should prefer the explicit monthly name.
STRIPE_PRO_MONTHLY_PRICE_ID = (
    os.environ.get("STRIPE_PRO_MONTHLY_PRICE_ID")
    or os.environ.get("STRIPE_PRO_PRICE_ID")
)
STRIPE_PRO_ANNUAL_PRICE_ID = os.environ.get("STRIPE_PRO_ANNUAL_PRICE_ID")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
STRIPE_BILLING_PORTAL_CONFIGURATION_ID = os.environ.get(
    "STRIPE_BILLING_PORTAL_CONFIGURATION_ID"
)

STRIPE_PRO_PRICE_IDS_BY_INTERVAL = {
    "monthly": STRIPE_PRO_MONTHLY_PRICE_ID,
    "annual": STRIPE_PRO_ANNUAL_PRICE_ID,
}

db.init_app(app)
bcrypt.init_app(app)
login_manager.init_app(app)
csrf.init_app(app)
login_manager.login_view = "login"

@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({
        "error": "Login required"
    }), 401

# Data path
DATA_DIR = os.path.join(app.root_path, 'data')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "epoch_index-USD.csv")

# Ensure folder/file exist
os.makedirs(DATA_DIR, exist_ok=True)

# Approved Forward Record CSVs contain only Date, Close and epoch_signal. In
# production, point this at a persistent Render disk so admin-approved files
# survive deploys/restarts. The default keeps local development self-contained.
FORWARD_RECORD_STORAGE_EXPLICIT = bool(
    os.environ.get("FORWARD_RECORD_STORAGE_DIR", "").strip()
)
FORWARD_RECORD_STORAGE_DIR = os.path.realpath(
    os.environ.get(
        "FORWARD_RECORD_STORAGE_DIR",
        os.path.join(DATA_DIR, "forward_record_store"),
    )
)
os.makedirs(FORWARD_RECORD_STORAGE_DIR, exist_ok=True)

# Admin-managed backup storage. Configure this inside the Render persistent disk
# for retention across deploys; the safe default remains temporary.
BACKUP_STORAGE_EXPLICIT = bool(os.environ.get("NEURALTREND_BACKUP_DIR", "").strip())
BACKUP_STORAGE_DIR = os.path.realpath(
    os.environ.get("NEURALTREND_BACKUP_DIR", "/tmp/neuraltrend-admin-backups")
)
try:
    BACKUP_RETENTION_COUNT = max(1, min(
        int(os.environ.get("NEURALTREND_BACKUP_RETENTION", "10")), 100
    ))
except ValueError:
    BACKUP_RETENTION_COUNT = 10
    app.logger.warning("invalid_backup_retention_using_default")
os.makedirs(BACKUP_STORAGE_DIR, mode=0o700, exist_ok=True)

app.logger.info(
    "application_initialized testing=%s database_configured=%s redis_configured=%s "
    "email_configured=%s stripe_monthly_configured=%s "
    "stripe_annual_configured=%s stripe_webhook_configured=%s "
    "forward_record_storage_explicit=%s",
    TESTING_MODE,
    bool(os.environ.get("DATABASE_URL", "").strip()),
    bool(os.environ.get("REDIS_URL", "").strip()),
    bool(os.environ.get("EMAIL_USER", "").strip() and os.environ.get("EMAIL_PASS", "").strip()),
    bool(STRIPE_PRO_MONTHLY_PRICE_ID),
    bool(STRIPE_PRO_ANNUAL_PRICE_ID),
    bool(STRIPE_WEBHOOK_SECRET),
    FORWARD_RECORD_STORAGE_EXPLICIT,
)

@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None


@app.before_request
def enforce_authenticated_session_version():
    """
    Revoke existing authenticated sessions after a password reset.

    Existing sessions created before this migration are adopted at the user's
    current version. Future password resets increment the version, causing all
    older sessions to be logged out on their next request.
    """
    if not current_user.is_authenticated:
        return None

    current_version = int(getattr(current_user, "auth_version", 1) or 1)
    stored_version = session.get("auth_version")

    if stored_version is None:
        # Sessions created before auth-version tracking cannot be proven current.
        # Log them out once during deployment rather than allowing them to bypass
        # future password-reset revocation.
        logout_user()
        session.pop("auth_version", None)
        return None

    try:
        stored_version = int(stored_version)
    except (TypeError, ValueError):
        stored_version = -1

    if stored_version != current_version:
        logout_user()
        session.pop("auth_version", None)

    return None
    
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Too many requests. Please slow down and try again shortly."
    }), 429


@app.errorhandler(RequestEntityTooLarge)
def request_too_large_handler(error):
    return jsonify({
        "error": "Request body is too large."
    }), 413


@app.errorhandler(CSRFError)
def handle_csrf_error(error):
    message = (
        "Your security token is missing or expired. "
        "Refresh the page and try again."
    )

    wants_json = (
        request.is_json
        or request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or request.path.startswith((
            "/signup",
            "/login",
            "/logout",
            "/change-password",
            "/resend-verification",
            "/request-password-reset",
            "/request-delete-account",
            "/live-simulations",
            "/watchlist",
            "/backtest",
            "/equity",
            "/create-checkout-session",
            "/billing-portal",
        ))
    )

    app.logger.warning(
        "CSRF validation failed for path=%s method=%s reason=%s",
        request.path,
        request.method,
        error.description,
    )

    if wants_json:
        return jsonify({
            "error": message,
            "code": "csrf_failed",
        }), 400

    return render_template(
        "csrf_error.html",
        reason=message,
    ), 400

def get_serializer():
    return URLSafeTimedSerializer(app.config["SECRET_KEY"])

def generate_verification_token(email):
    return get_serializer().dumps(email, salt="email-confirm")

def confirm_verification_token(token, expiration=3600):
    try:
        email = get_serializer().loads(
            token,
            salt="email-confirm",
            max_age=expiration
        )
    except Exception:
        return None
    return email

def send_verification_email(user_email):
    token = generate_verification_token(user_email)

    verify_url = f"{BASE_URL}/verify/{token}"

    msg = Message(
        subject="Verify your NeuralTrend account",
        sender=app.config["MAIL_USERNAME"],
        recipients=[user_email]
    )

    msg.body = f"""
    Hi,
    
    Please verify your NeuralTrend account by clicking the link below:
    
    {verify_url}
    
    This verification link expires in 1 hour.
    
    If you did not create a NeuralTrend account, you can ignore this email.
    
    NeuralTrend
    """

    try:
        mail.send(msg)
        app.logger.info("Verification email sent")
    except Exception:
        app.logger.exception("Verification email delivery failed")

def normalize_email(email):
    return str(email or "").strip().lower()


EMAIL_MAX_LENGTH = 254
SIMULATION_NAME_MAX_LENGTH = 120
MIN_INITIAL_CASH = 1.0
MAX_INITIAL_CASH = 1_000_000_000.0


def validate_email_address(email):
    """Return a user-facing validation error, or None when the email is usable."""
    if not isinstance(email, str):
        return "Enter a valid email address."

    email = email.strip()
    if not email or len(email) > EMAIL_MAX_LENGTH:
        return "Enter a valid email address."

    if any(ch.isspace() or unicodedata.category(ch).startswith("C") for ch in email):
        return "Enter a valid email address."

    if email.count("@") != 1:
        return "Enter a valid email address."

    local_part, domain = email.rsplit("@", 1)
    if not local_part or len(local_part) > 64 or not domain:
        return "Enter a valid email address."

    try:
        ascii_domain = domain.encode("idna").decode("ascii")
    except UnicodeError:
        return "Enter a valid email address."

    if len(ascii_domain) > 253 or "." not in ascii_domain:
        return "Enter a valid email address."

    label_pattern = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
    if any(not label_pattern.fullmatch(label) for label in ascii_domain.split(".")):
        return "Enter a valid email address."

    # Keep the local-part validation deliberately conservative and compatible
    # with the addresses accepted by the current account UI.
    if not re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+", local_part):
        return "Enter a valid email address."

    return None


def get_json_object():
    """Return a JSON object or None. Arrays/scalars are not valid API bodies here."""
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else None


def parse_finite_number(value, field_name, minimum=None, maximum=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a valid number.")

    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite number.")

    if minimum is not None and number < minimum:
        raise ValueError(f"{field_name} must be at least {minimum:g}.")

    if maximum is not None and number > maximum:
        raise ValueError(f"{field_name} must be no more than {maximum:g}.")

    return number


def normalize_optional_currency_input(value, default_value=10000):
    """Accept user-friendly currency text such as 100$, $100, or $10,000."""
    if value is None:
        return default_value

    if isinstance(value, str):
        clean = value.strip()
        if not clean:
            return default_value
        clean = re.sub(r"[\s,$]", "", clean)
        return clean

    return value


def validate_simulation_name(value, allow_empty=False):
    if not isinstance(value, str):
        return None, "Simulation name must be text."

    clean = value.strip()

    if not clean:
        if allow_empty:
            return "", None
        return None, "Simulation name cannot be empty."

    if len(clean) > SIMULATION_NAME_MAX_LENGTH:
        return None, f"Simulation name must be {SIMULATION_NAME_MAX_LENGTH} characters or fewer."

    if any(unicodedata.category(ch).startswith("C") for ch in clean):
        return None, "Simulation name contains unsupported control characters."

    # Names are later displayed in several dashboard components. Reject markup
    # delimiters here as defense-in-depth; the frontend must still escape output.
    if any(ch in clean for ch in "<>{}"):
        return None, "Simulation name cannot contain <, >, {, or }."

    return clean, None

PASSWORD_MIN_LENGTH = 15
PASSWORD_MAX_LENGTH = 64
PASSWORD_MAX_UTF8_BYTES = 72
PASSWORD_RESET_TOKEN_SECONDS = 60 * 60
PASSWORD_RESET_GRANT_SECONDS = 15 * 60


def validate_password(password):
    """
    Validate passwords consistently for signup and password reset.

    The 72-byte encoded limit prevents silent bcrypt truncation. Passwords may
    otherwise contain spaces, Unicode, and punctuation; no composition rules
    are imposed.
    """
    if not isinstance(password, str):
        return "Password is required."

    if len(password) < PASSWORD_MIN_LENGTH:
        return f"Password must be at least {PASSWORD_MIN_LENGTH} characters."

    if len(password) > PASSWORD_MAX_LENGTH:
        return f"Password must be {PASSWORD_MAX_LENGTH} characters or fewer."

    if len(password.encode("utf-8")) > PASSWORD_MAX_UTF8_BYTES:
        return (
            "Password is too long when encoded. Use fewer multi-byte "
            "characters or a shorter passphrase."
        )

    return None


def hash_password_reset_nonce(nonce):
    return hashlib.sha256(str(nonce).encode("utf-8")).hexdigest()


def generate_reset_token(user):
    """
    Create a cryptographically random, one-time reset nonce.

    Only its SHA-256 digest is stored. Requesting a new link replaces the
    digest, immediately invalidating every older reset link for the account.
    """
    nonce = secrets.token_urlsafe(32)
    user.password_reset_token_hash = hash_password_reset_nonce(nonce)
    user.password_reset_requested_at = datetime.utcnow()

    return get_serializer().dumps(
        {
            "email": user.email,
            "nonce": nonce,
        },
        salt="password-reset",
    )


def confirm_reset_token(token, expiration=PASSWORD_RESET_TOKEN_SECONDS):
    try:
        payload = get_serializer().loads(
            token,
            salt="password-reset",
            max_age=expiration,
        )
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    email = normalize_email(payload.get("email"))
    nonce = payload.get("nonce")

    if not email or not isinstance(nonce, str) or not nonce:
        return None

    user = User.query.filter_by(email=email).first()

    if not user or not user.password_reset_token_hash:
        return None

    candidate_hash = hash_password_reset_nonce(nonce)

    if not hmac.compare_digest(
        user.password_reset_token_hash,
        candidate_hash,
    ):
        return None

    return user, candidate_hash


def clear_password_reset_session():
    for key in (
        "password_reset_user_id",
        "password_reset_token_hash",
        "password_reset_granted_at",
        "password_reset_invalid",
    ):
        session.pop(key, None)


def establish_password_reset_session(user, token_hash):
    clear_password_reset_session()
    session["password_reset_user_id"] = int(user.id)
    session["password_reset_token_hash"] = token_hash
    session["password_reset_granted_at"] = int(time.time())


def get_password_reset_user_from_session():
    user_id = session.get("password_reset_user_id")
    token_hash = session.get("password_reset_token_hash")
    granted_at = session.get("password_reset_granted_at")

    try:
        user_id = int(user_id)
        granted_at = int(granted_at)
    except (TypeError, ValueError):
        clear_password_reset_session()
        return None

    if time.time() - granted_at > PASSWORD_RESET_GRANT_SECONDS:
        clear_password_reset_session()
        return None

    if not isinstance(token_hash, str) or not token_hash:
        clear_password_reset_session()
        return None

    user = db.session.get(User, user_id)

    if not user or not user.password_reset_token_hash:
        clear_password_reset_session()
        return None

    if not hmac.compare_digest(user.password_reset_token_hash, token_hash):
        clear_password_reset_session()
        return None

    return user


def send_password_changed_email(user_email):
    msg = Message(
        subject="Your NeuralTrend password was changed",
        sender=app.config["MAIL_USERNAME"],
        recipients=[user_email],
    )

    msg.body = f"""Hi,

The password for your NeuralTrend account was changed successfully.

If you made this change, no further action is required.

If you did not change your password, contact NeuralTrend immediately and do not reuse the previous password.

NeuralTrend
"""

    try:
        mail.send(msg)
    except Exception:
        app.logger.exception(
            "Could not send password-change notification to %s",
            user_email,
        )

def generate_delete_token(email):
    return get_serializer().dumps(email, salt="delete-account")

def confirm_delete_token(token, expiration=3600):
    try:
        email = get_serializer().loads(token, salt="delete-account", max_age=expiration)
    except Exception:
        return None
    return email

ALLOWED_DURATION_DELTAS = {
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
    "1m": relativedelta(months=1),
    "1mo": relativedelta(months=1),
    "3m": relativedelta(months=3),
    "3mo": relativedelta(months=3),
    "6m": relativedelta(months=6),
    "6mo": relativedelta(months=6),
    "1y": relativedelta(years=1),
    "1yr": relativedelta(years=1),
    "2y": relativedelta(years=2),
    "2yr": relativedelta(years=2),
    "3y": relativedelta(years=3),
    "3yr": relativedelta(years=3),
    "5y": relativedelta(years=5),
    "5yr": relativedelta(years=5),
    "10y": relativedelta(years=10),
    "10yr": relativedelta(years=10),
}


def parse_duration(duration: str):
    """Parse only the bounded durations that NeuralTrend's UI supports."""
    if not isinstance(duration, str):
        raise ValueError("Unsupported duration.")

    clean = duration.strip().lower()
    delta = ALLOWED_DURATION_DELTAS.get(clean)

    if delta is None:
        raise ValueError("Unsupported duration. Choose a listed NeuralTrend horizon.")

    return delta


def duration_to_days(duration_str: str):
    """
    Convert a dashboard horizon to an approximate day count.

    ``max`` is intentionally represented by ``None`` so callers can use the
    complete validated CSV history without inventing an arbitrary year cap.
    Other dashboard horizon consumers still use ``parse_duration`` to convert
    the supported preset labels into day counts.
    """
    clean = str(duration_str or "").strip().lower()

    if clean == "max":
        return None

    delta = parse_duration(clean)

    if isinstance(delta, timedelta):
        return delta.days

    # The dashboard uses calendar slicing; these values are bounded approximations
    # used only to select a window from local market CSVs.
    return delta.years * 365 + delta.months * 30 + delta.days


def slice_epoch_data_for_period(df, period_days):
    """Return the exact dashboard horizon window anchored to the asset's latest row.

    Signal Board summaries and Equity Preview must use the same observations.
    Horizons are calendar-duration windows (for example, 1W spans exactly seven
    elapsed calendar days from the latest available market row). For stocks this
    naturally includes only the trading sessions that fall inside that calendar
    window; it must not be converted again into an approximate trading-day count.
    ``None`` means MAX and preserves the complete validated history.
    """
    if df is None or len(df) == 0:
        return df.copy() if hasattr(df, "copy") else df

    if period_days is None:
        return df.copy()

    if not isinstance(period_days, int) or period_days < 1:
        raise ValueError("Unsupported signal period.")

    end_date = df.index.max()
    start_date = end_date - pd.Timedelta(days=period_days)
    return df[df.index >= start_date].copy()


def parse_position_fraction(value):
    """Convert a finite percentage in (0, 100] to a decimal fraction."""
    if value is None:
        return 1.0

    text = str(value).strip().lower()
    text = text.replace("_pct", "").replace("pct", "").replace("%", "")

    pct = parse_finite_number(
        text,
        "Position size",
        minimum=1.0,
        maximum=100.0,
    )

    return pct / 100.0


def get_csv_version():
    """
    Returns a version number that changes whenever any CSV changes.
    """
    mtimes = []

    for fname in os.listdir(DATA_DIR):
        if fname.startswith("epoch_") and fname.endswith(".csv"):
            path = os.path.join(DATA_DIR, fname)
            mtimes.append(os.path.getmtime(path))

    # If no CSVs exist, still return something
    return max(mtimes) if mtimes else 0


def get_stat_csv_version():
    """Return a hashable signature for the optional ``stat_*.csv`` files.

    Signal-board statistics are generated independently from the market CSVs, so
    their cache invalidation must not depend on an ``epoch_*.csv`` being updated.
    A tuple of filename/mtime/size signatures is deterministic within a process
    and changes whenever a stat file is added, removed, or replaced.
    """
    signatures = []

    try:
        filenames = os.listdir(DATA_DIR)
    except OSError:
        return tuple()

    for fname in filenames:
        if not (fname.startswith("stat_") and fname.endswith(".csv")):
            continue

        path = os.path.join(DATA_DIR, fname)
        try:
            stat_result = os.stat(path)
        except OSError:
            continue

        signatures.append((fname, stat_result.st_mtime_ns, stat_result.st_size))

    return tuple(sorted(signatures))


FRESHNESS_STATUS_LABELS = {
    "current": "Current",
    "delayed": "Delayed",
    "stale": "Stale",
    "unknown": "Unknown",
}


def utc_timestamp_string(value):
    """Serialize a naive UTC datetime as an explicit ISO-8601 UTC timestamp."""
    if not isinstance(value, datetime):
        return None
    return value.replace(microsecond=0).isoformat() + "Z"


def count_market_weekdays_after(data_date, current_date):
    """Count Monday-Friday dates after data_date through current_date."""
    if not data_date or not current_date or data_date >= current_date:
        return 0

    count = 0
    cursor = data_date + timedelta(days=1)
    while cursor <= current_date:
        if cursor.weekday() < 5:
            count += 1
        cursor += timedelta(days=1)
    return count


def build_data_freshness_metadata(ticker, data_through=None):
    """
    Describe the freshness of the latest deployed market-data row.

    Crypto uses calendar-day lag because it trades continuously. Stocks use a
    weekday-aware lag so Friday data can remain current through a weekend.
    Exchange holidays are not modeled by this lightweight status indicator and
    are disclosed in the methodology page.
    """
    clean_ticker = normalize_ticker(ticker)
    checked_at = datetime.utcnow()

    try:
        if isinstance(data_through, pd.Timestamp):
            data_date = data_through.date()
        elif isinstance(data_through, datetime):
            data_date = data_through.date()
        elif hasattr(data_through, "isoformat") and not isinstance(data_through, str):
            data_date = data_through
        elif isinstance(data_through, str) and data_through.strip():
            data_date = datetime.strptime(data_through.strip()[:10], "%Y-%m-%d").date()
        else:
            market_data = load_epoch_csv_for_ticker(clean_ticker)
            data_date = market_data.index.max().date()

        csv_path = get_epoch_csv_path(clean_ticker)
        site_data_updated_at = datetime.utcfromtimestamp(os.path.getmtime(csv_path))
        today_utc = checked_at.date()

        if is_crypto_ticker(clean_ticker):
            lag_count = max((today_utc - data_date).days, 0)
            lag_unit = "calendar_days"
            expected_update_cadence = "Daily, including weekends"
        else:
            lag_count = count_market_weekdays_after(data_date, today_utc)
            lag_unit = "market_weekdays"
            expected_update_cadence = "Market weekdays"

        if lag_count <= 1:
            status = "current"
        elif lag_count == 2:
            status = "delayed"
        else:
            status = "stale"

        if status == "current":
            message = "Latest completed data is within the expected update cadence."
        elif status == "delayed":
            message = "Latest completed data is later than the usual update cadence."
        else:
            message = "Latest completed data is materially older than the usual update cadence."

        return {
            "data_through": data_date.isoformat(),
            "site_data_updated_at_utc": utc_timestamp_string(site_data_updated_at),
            "freshness_checked_at_utc": utc_timestamp_string(checked_at),
            "freshness_status": status,
            "freshness_label": FRESHNESS_STATUS_LABELS[status],
            "freshness_lag_count": lag_count,
            "freshness_lag_unit": lag_unit,
            "expected_update_cadence": expected_update_cadence,
            "freshness_message": message,
        }
    except Exception:
        app.logger.exception(
            "Could not determine data freshness for ticker=%s",
            clean_ticker,
        )
        return {
            "data_through": None,
            "site_data_updated_at_utc": None,
            "freshness_checked_at_utc": utc_timestamp_string(checked_at),
            "freshness_status": "unknown",
            "freshness_label": FRESHNESS_STATUS_LABELS["unknown"],
            "freshness_lag_count": None,
            "freshness_lag_unit": None,
            "expected_update_cadence": None,
            "freshness_message": "Freshness could not be determined for this asset.",
        }

cache = {}  # simple in-memory cache per ticker

SUPPORTED_TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'NVDA', 'AAPL', 'GOOGL', 'MSFT', "1INCH-USD", "3ULL-USD", "AAVE-USD","ABBV", "ACE-USD",
               "ACH-USD", "ADA-USD", "AERO-USD", "AEVO-USD", "AGI-USD", "AIOZ-USD", "AIT-USD", "AIXBT-USD", "AKT-USD", "ALEPH-USD",
               "ALGO-USD", "ALI-USD", "ALPH-USD", "ALT-USD", "ALU-USD", "ALVA-USD", "AMP-USD", 'AMZN', "ANKR-USD", "ANON-USD", "ANYONE-USD", "APT-USD",
               "APU-USD", "AR-USD", "ARB-USD", "ARC-USD", "ASML", "ASTR-USD", "ATLAS-USD", "ATOM-USD", "AURY-USD", "AUTOS-USD", "AVAX-USD", 'AVGO', 
               "AXL-USD", "AXS-USD", "BAI-USD", "BAL-USD", "BANANA-USD", "BAND-USD", "BASEDAI-USD", "BAZED-USD", "BCH-USD",
               "BCUT-USD", "BEAM-USD", "BGB-USD", "BIGTIME-USD", "BLUR-USD", "BNB-USD", "BNT-USD", "BONK-USD", "BRETT-USD", 'BRKB', 
               "BYTES-USD", "CAKE-USD", "CELO-USD", "CERE-USD", "CETUS-USD", "CFG-USD", "CGPT-USD", "CHAPZ-USD", "CHAT-USD", "CHEX-USD", "CHZ-USD", 
               "COMP-USD", "COST", "COTI-USD", "CPOOL-USD", "CREDI-USD", "CREO-USD", "CRO-USD", "CROWN-USD", "CRU-USD", "CRV-USD", "CTC-USD", "CVC-USD",
               "DARK-USD", "DCK-USD", "DEVVE-USD", "DEXE-USD", "DIMO-USD", "DIO-USD", "DOGE-USD", "DOME-USD", "DOMI-USD", "DOT-USD", "DRIFT-USD", 
               "DSYNC-USD", "DYDX-USD", "DYM-USD", "EDU-USD", "ENA-USD", "ENJ-USD", "ENQAI-USD", "FAR-USD", "FET-USD", "FIDA-USD", 
               "FIL-USD", "FLIP-USD", "FLOW-USD", "FLR-USD", "FLUX-USD", "FOXY-USD", "FUELX-USD", "FYN-USD", "GEAR-USD", "GFAL-USD",
               "GHX-USD", "GLQ-USD", "GMEE-USD", "GMRX-USD", "GMT-USD", "GMX-USD", "GODS-USD", "GPU-USD", "GRIFFAIN-USD", "GRT-USD",
               "GSWIFT-USD", "GTAI-USD", "GTC-USD", "HASHAI-USD", "HBAR-USD", "HEART-USD", "HELLO-USD", "HNT-USD", "HONEY-USD",
               "HXD-USD", "HYPC-USD", "HYPE-USD", "IAG-USD", "ICP-USD", "ILV-USD", "IMX-USD", "INJ-USD", "INSP-USD", "IOTX-USD", "IVPAY-USD",
               "JASMY-USD", "JNJ", "JOE-USD", "JPM", "JST-USD", "JTO-USD", "JUP-USD", "KARATE-USD", "KARRAT-USD", "KAS-USD", "KATA-USD", "KOMPETE-USD",
               "KRL-USD", "LAI-USD", "LEO-USD", "LFNTY-USD", "LIKE-USD", "LINK-USD", "LLY", "LMWR-USD", "LPT-USD", "LRC-USD", "LTC-USD", "MA",
               "MAGIC-USD", "MASK-USD", "MAVIA-USD", "MBS-USD", 'META', "METIS-USD", "MEW-USD", "MINA-USD", "ML-USD", "MLN-USD", "MNDE-USD",
               "MNT-USD", "MOODENG-USD", "MPLX-USD", "MU", "MUBI-USD", "MXM-USD", "MYRIA-USD", "MYRO-USD", "NAKA-USD", 
               "NEAR-USD", "NEON-USD", "NEURAL-USD", "NEXO-USD", "NMT-USD", "NOS-USD", "NTRN-USD", "NU-USD", "NXRA-USD", "OGN-USD", "OKB-USD",
               "OLAS-USD", "OMG-USD", "ONDO-USD", "OP-USD", "ORAI-USD", "ORCA-USD", "ORCL", "ORDI-USD", "OTK-USD", "OXT-USD", "PAAL-USD", 
               "PANDORA-USD", "PDA-USD", "PENDLE-USD", "PENG-USD", "PENGU-USD", "PEPE-USD", "PERP-USD", "PHA-USD", 
               "PIN-USD", "PIXEL-USD", "POL-USD", "POLS-USD", "POLYX-USD", "PORTAL-USD", "PRIME-USD", "PROPC-USD", "PYR-USD", 
               "PYTH-USD", "QANX-USD", "QI-USD", "QNT-USD", "RAY-USD", "RARE-USD", "RARI-USD", "REN-USD", "RENDER-USD", 
               "REQ-USD", "RIO-USD", "RLB-USD", "RMRK-USD", "RON-USD", "ROOT-USD", "RSC-USD", "RSR-USD", "RSS3-USD",
               "RUNE-USD", "SAFE-USD", "SC-USD", "SEI-USD", "SENATE-USD", "SERSH-USD", "SHDW-USD", "SHIB-USD", 
               "SHIDO-USD", "SHRAP-USD", "SIDUS-USD", "SIPHER-USD", "SKL-USD", "SPEC-USD", "SPELL-USD", "SRM-USD", "SSV-USD", 
               "STEP-USD", "STG-USD", "STORJ-USD", "STRK-USD", "SUI-USD", "SUNDOG-USD", "SUPER-USD", "TAI-USD", "TAO-USD", 'TCEHY', 
               "TET-USD", "TFUEL-USD", "THETA-USD", "TLOS-USD", "TON-USD", "TRAC-USD", "TRIAS-USD", "TRU-USD", 'TSLA', 'TSM', "TURBO-USD",
               "UNI-USD", "UNIBOT-USD", "UOS-USD", 'V', "VAI-USD", "VET-USD", "VIA-USD", "VIRTUAL-USD", "VOO", "VR-USD", "VRA-USD",
               "WAXP-USD", "WHALES-USD", "WIF-USD", "WIFI-USD", "WILD-USD", "WINR-USD", "WLD-USD", "WMT", "WMTX-USD", "XAI-USD", 
               "XCAD-USD", "XLM-USD", "XMR-USD", "XOM", "XTZ-USD", "XYO-USD", "YGG-USD", "ZBCN-USD", "ZEN-USD", "ZEREBRO-USD", "ZETA-USD", 
               "ZIG-USD", "ZKJ-USD", "ZRX-USD"]

TOP_FREE_TICKERS = {"BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"}
ADMIN_ONLY_TICKERS = {"JNJ", "LLY"}
ALL_SUPPORTED_TICKERS = frozenset(
    str(ticker).strip().upper()
    for ticker in [*SUPPORTED_TICKERS, *ADMIN_ONLY_TICKERS]
)

FREE_LIVE_SIMULATION_LIMIT = 5
PAID_LIVE_SIMULATION_LIMIT = 100

FREE_WATCHLIST_LIMIT = 4
PAID_WATCHLIST_LIMIT = 100
SIGNAL_ALERT_UNSUBSCRIBE_SALT = "signal-alert-unsubscribe"

PAID_SUBSCRIPTION_STATUSES = {"active", "trialing"}

def is_admin_user(user):
    if (
        user is None
        or not getattr(user, "is_authenticated", False)
        or not getattr(user, "email", None)
    ):
        return False

    admin_emails = {
        email.strip().lower()
        for email in os.environ.get("ADMIN_EMAILS", "").split(",")
        if email.strip()
    }

    return user.email.lower() in admin_emails

def is_paid_user(user):
    if is_admin_user(user):
        return True

    return (
        user is not None
        and getattr(user, "is_authenticated", False)
        and user.subscription_type == "pro"
        and user.subscription_status in PAID_SUBSCRIPTION_STATUSES
    )

def get_live_simulation_limit_for_user(user):
    if is_admin_user(user):
        return None  # Admin has unlimited live simulations

    if is_paid_user(user):
        return PAID_LIVE_SIMULATION_LIMIT

    return FREE_LIVE_SIMULATION_LIMIT

def can_view_full_signals_for_ticker(user, ticker):
    ticker = normalize_ticker(ticker)

    # Admin-only assets are invisible to everyone except admin,
    # even if the user is subscribed/Pro.
    if not can_user_see_ticker(user, ticker):
        return False

    if ticker in TOP_FREE_TICKERS:
        return True

    return is_paid_user(user)

def require_signal_access_or_403(ticker):
    ticker = normalize_ticker(ticker)

    support_error = require_supported_ticker_or_400(ticker)
    if support_error:
        return support_error

    visibility_error = require_ticker_visible_or_404(ticker)
    if visibility_error:
        return visibility_error

    if can_view_full_signals_for_ticker(current_user, ticker):
        return None

    return jsonify({
        "error": "This asset is available in Pro. Upgrade to unlock full signals, equity preview, and backtesting.",
        "upgrade_required": True,
        "ticker": ticker
    }), 403

def normalize_ticker(ticker):
    return str(ticker or "").strip().upper()


def is_supported_ticker(ticker):
    return normalize_ticker(ticker) in ALL_SUPPORTED_TICKERS


def require_supported_ticker_or_400(ticker):
    clean = normalize_ticker(ticker)
    if clean not in ALL_SUPPORTED_TICKERS:
        return jsonify({"error": "Unsupported asset ticker."}), 400
    return None


def get_epoch_csv_path(ticker):
    """Return a canonical local CSV path only for an allowlisted ticker."""
    clean = normalize_ticker(ticker)
    if clean not in ALL_SUPPORTED_TICKERS:
        raise ValueError("Unsupported asset ticker.")

    base_symbol = clean.split("-", 1)[0]
    if not re.fullmatch(r"[A-Z0-9.]{1,20}", base_symbol):
        raise ValueError("Unsupported asset ticker.")

    data_root = os.path.realpath(DATA_DIR)
    csv_path = os.path.realpath(os.path.join(data_root, f"epoch_{base_symbol}.csv"))

    try:
        inside_data_dir = os.path.commonpath([data_root, csv_path]) == data_root
    except ValueError:
        inside_data_dir = False

    if not inside_data_dir:
        raise ValueError("Unsupported asset ticker.")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError("Market data is unavailable for this asset.")

    return csv_path


def get_stat_csv_path(ticker, *, must_exist=False):
    """Return the optional model-statistics CSV path for an allowlisted ticker.

    ``BTC-USD`` maps to ``data/stat_BTC.csv``; stocks such as ``AAPL`` map to
    ``data/stat_AAPL.csv``. Missing stat files are expected while the long-running
    statistics generation process is still being completed.
    """
    clean = normalize_ticker(ticker)
    if clean not in ALL_SUPPORTED_TICKERS:
        raise ValueError("Unsupported asset ticker.")

    base_symbol = clean.split("-", 1)[0]
    if not re.fullmatch(r"[A-Z0-9.]{1,20}", base_symbol):
        raise ValueError("Unsupported asset ticker.")

    data_root = os.path.realpath(DATA_DIR)
    csv_path = os.path.realpath(os.path.join(data_root, f"stat_{base_symbol}.csv"))

    try:
        inside_data_dir = os.path.commonpath([data_root, csv_path]) == data_root
    except ValueError:
        inside_data_dir = False

    if not inside_data_dir:
        raise ValueError("Unsupported asset ticker.")

    if not os.path.isfile(csv_path):
        if must_exist:
            raise FileNotFoundError("Model statistics are unavailable for this asset.")
        return None

    return csv_path


def get_signal_stat_for_horizon(ticker, period_days):
    """Return optional model statistics for the requested Signal Board horizon.

    ``window`` in ``stat_<asset>.csv`` is measured in calendar-day units,
    matching the dashboard horizon mapping (1W=7, 1M=30, 3M=90, etc.).

    The selected-window statistics are:
    - ``alpha``: average AI/B&H terminal-value ratio
    - ``alpha_prob``: probability AI outperforms B&H
    - ``strategy_avg_return``: average AI terminal/initial value ratio
    - ``strategy_profit_prob``: probability the AI strategy finishes in profit

    ``recommended_days`` is independent of the selected dashboard horizon: it is
    the smallest available ``window`` whose ``strategy_prob`` is at least 0.50.

    For MAX, or when the requested horizon is longer than the generated
    statistics history, the longest available window is used. Missing/malformed
    stat files return ``None`` values and never suppress the asset from the
    Signal Board.
    """
    empty_result = {
        "alpha": None,
        "alpha_prob": None,
        "strategy_avg_return": None,
        "strategy_profit_prob": None,
        "recommended_days": None,
        "stat_window": None,
    }

    try:
        csv_path = get_stat_csv_path(ticker)
    except (ValueError, OSError):
        return empty_result.copy()

    if not csv_path:
        return empty_result.copy()

    wanted_columns = {
        "window",
        "alpha",
        "alpha_prob",
        "strategy_return",
        "strategy_prob",
    }

    try:
        stats_df = pd.read_csv(
            csv_path,
            usecols=lambda column: column in wanted_columns,
        )
    except Exception:
        app.logger.warning(
            "Could not read signal statistics ticker=%s path=%s",
            ticker,
            csv_path,
            exc_info=True,
        )
        return empty_result.copy()

    if "window" not in stats_df.columns:
        app.logger.warning(
            "Signal statistics missing window column ticker=%s columns=%s",
            ticker,
            sorted(stats_df.columns),
        )
        return empty_result.copy()

    # Older/partially generated stat files should remain safe to display.
    for column in ("alpha", "alpha_prob", "strategy_return", "strategy_prob"):
        if column not in stats_df.columns:
            stats_df[column] = math.nan

    cleaned = stats_df.loc[
        :,
        ["window", "alpha", "alpha_prob", "strategy_return", "strategy_prob"],
    ].copy()

    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=["window"])
    cleaned = cleaned[cleaned["window"] >= 1]
    cleaned = cleaned.sort_values("window").drop_duplicates("window", keep="last")

    if cleaned.empty:
        return empty_result.copy()

    # Smallest horizon at which the historical probability of the AI strategy
    # finishing in profit reaches at least 50%.
    recommended_candidates = cleaned[
        cleaned["strategy_prob"].notna() & (cleaned["strategy_prob"] >= 0.50)
    ]
    recommended_days = (
        int(recommended_candidates.iloc[0]["window"])
        if not recommended_candidates.empty
        else None
    )

    if period_days is None:
        selected = cleaned.iloc[-1]
    else:
        target_window = max(1, int(period_days))
        eligible = cleaned[cleaned["window"] <= target_window]
        selected = eligible.iloc[-1] if not eligible.empty else cleaned.iloc[0]

    def finite_or_none(value):
        if pd.isna(value):
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    return {
        "alpha": finite_or_none(selected["alpha"]),
        "alpha_prob": finite_or_none(selected["alpha_prob"]),
        "strategy_avg_return": finite_or_none(selected["strategy_return"]),
        "strategy_profit_prob": finite_or_none(selected["strategy_prob"]),
        "recommended_days": recommended_days,
        "stat_window": int(selected["window"]),
    }


def get_forward_record_csv_path(ticker, record_mode="public", *, must_exist=False):
    """Return the canonical approved Forward Record CSV path.

    These files are deliberately separate from working model CSVs. Updating a
    working CSV never changes customer-facing performance until an admin
    approves the corresponding compact file update.
    """
    clean = normalize_ticker(ticker)
    mode = str(record_mode or "public").strip().lower()
    if clean not in ALL_SUPPORTED_TICKERS:
        raise ValueError("Unsupported asset ticker.")
    if mode not in {"sandbox", "public"}:
        raise ValueError("Forward Record mode must be sandbox or public.")
    if not re.fullmatch(r"[A-Z0-9.-]{1,30}", clean):
        raise ValueError("Unsupported asset ticker.")

    root = os.path.realpath(FORWARD_RECORD_STORAGE_DIR)
    mode_root = os.path.realpath(os.path.join(root, mode))
    os.makedirs(mode_root, exist_ok=True)
    csv_path = os.path.realpath(os.path.join(mode_root, f"{clean}.csv"))
    try:
        inside_root = os.path.commonpath([mode_root, csv_path]) == mode_root
    except ValueError:
        inside_root = False
    if not inside_root:
        raise ValueError("Unsupported Forward Record path.")
    if must_exist and not os.path.isfile(csv_path):
        raise FileNotFoundError("Approved Forward Record data is unavailable for this asset.")
    return csv_path


def load_forward_record_csv(ticker, record_mode="public"):
    path = get_forward_record_csv_path(ticker, record_mode, must_exist=True)
    data = pd.read_csv(
        path,
        usecols=["Date", "Close", "epoch_signal"],
        parse_dates=["Date"],
    )
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data["epoch_signal"] = pd.to_numeric(data["epoch_signal"], errors="coerce")
    data = data.dropna(subset=["Date", "Close", "epoch_signal"]).copy()
    data = data[
        data["Close"].map(math.isfinite)
        & data["epoch_signal"].map(math.isfinite)
        & (data["Close"] > 0)
        & data["epoch_signal"].isin([-1, 0, 1])
    ].copy()
    data["epoch_signal"] = data["epoch_signal"].astype(int)
    data = data.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    data.set_index("Date", inplace=True)
    return data


def is_admin_only_ticker(ticker):
    return normalize_ticker(ticker) in ADMIN_ONLY_TICKERS

def can_user_see_ticker(user, ticker):
    ticker = normalize_ticker(ticker)

    if ticker not in ALL_SUPPORTED_TICKERS:
        return False

    if is_admin_only_ticker(ticker) and not is_admin_user(user):
        return False

    # Retired/removed Forward Record assets remain inspectable by admins while
    # their working CSV is retained, but ordinary customers no longer see them
    # in Signal Overview, searches, backtests, watchlists, or simulations.
    if public_forward_asset_is_retired(ticker) and not is_admin_user(user):
        return False

    return True

def get_supported_tickers_for_user(user):
    """
    User-visible ticker list for dropdowns.

    Admin sees normal supported tickers plus admin-only tickers.
    Non-admin users see only supported tickers, excluding admin-only tickers.
    """
    if is_admin_user(user):
        return get_all_signal_board_tickers()

    inactive_tickers = get_retired_public_ticker_set()
    return [
        ticker for ticker in SUPPORTED_TICKERS
        if not is_admin_only_ticker(ticker)
        and ticker not in inactive_tickers
    ]

def require_ticker_visible_or_404(ticker):
    if not can_user_see_ticker(current_user, ticker):
        return jsonify({
            "error": "Asset not available."
        }), 404

    return None

def unique_ticker_list(*ticker_groups):
    """
    Combines ticker lists/sets while preserving order and removing duplicates.
    """
    seen = set()
    output = []

    for group in ticker_groups:
        for ticker in group:
            clean_ticker = normalize_ticker(ticker)

            if not clean_ticker or clean_ticker in seen:
                continue

            seen.add(clean_ticker)
            output.append(clean_ticker)

    return output

def get_all_signal_board_tickers():
    """
    Internal processing list.

    Includes normal supported tickers plus admin-only tickers.
    This allows admin-only tickers to appear for admin even if they are
    not listed in SUPPORTED_TICKERS.
    """
    return unique_ticker_list(
        SUPPORTED_TICKERS,
        sorted(ADMIN_ONLY_TICKERS)
    )

def compute_signals_for_ticker(ticker, period_days=365*10, csv_version=None):
    ticker = normalize_ticker(ticker)
    if ticker not in ALL_SUPPORTED_TICKERS:
        raise ValueError("Unsupported asset ticker.")

    # ``None`` means MAX: use every validated row available for this asset.
    if period_days is not None and (
        not isinstance(period_days, int)
        or period_days < 1
        or period_days > 3650
    ):
        raise ValueError("Unsupported signal period.")

    effective_csv_version = csv_version if csv_version is not None else get_csv_version()
    cache_key = (ticker, period_days, effective_csv_version)
    if cache_key in cache:
        return cache[cache_key]

    df = load_epoch_csv_for_ticker(ticker)

    if len(df) < 2:
        app.logger.info("Not enough data for signal summary ticker=%s", ticker)
        return None

    # Signal Board returns and Equity Preview use one canonical horizon slice:
    # a calendar-duration window anchored to this asset's latest available row.
    # This is especially important for stocks, where converting 7 calendar days
    # into ~4 trading rows caused the Signal Board and 1W equity chart to disagree.
    df = slice_epoch_data_for_period(df, period_days)
    transaction_cost = get_transaction_cost_rate(ticker)

    if len(df) < 2:
        return None

    prices = df["Close"].astype(float)

    # Use the exact same normalized benchmark builder as Equity Preview so the
    # Signal Board B&H return is mathematically identical to the plotted curve.
    _benchmark_quantity, _benchmark_cash_balance, buy_hold_equity_curve = (
        build_buy_and_hold_benchmark(
            initial_cash=1.0,
            prices=prices.to_numpy(dtype=float),
            ticker=ticker,
            transaction_cost_rate=transaction_cost,
            enforce_whole_stock_shares=False,
        )
    )

    cash = 1.0
    shares = 0.0
    strategy_equity_curve = []
    exposure_flags = []
    executed_trade_count = 0

    for date_index, row in df.iterrows():
        signal = int(row["epoch_signal"])
        price = float(row["Close"])

        if signal == 1 and shares == 0:
            gross_buy_budget = cash / (1 + transaction_cost)
            shares = gross_buy_budget / price
            cash = 0.0
            executed_trade_count += 1

        elif signal == -1 and shares > 0:
            cash = (shares * price) * (1 - transaction_cost)
            shares = 0.0
            executed_trade_count += 1

        strategy_equity_curve.append(cash + shares * price)
        exposure_flags.append(shares > 0)

    buy_hold_value = float(buy_hold_equity_curve[-1])
    strategy_value = float(strategy_equity_curve[-1])
    buy_hold_period_return = buy_hold_value - 1
    strategy_period_return = strategy_value - 1
    return_spread = strategy_period_return - buy_hold_period_return
    outperformance_ratio = (
        strategy_value / buy_hold_value
        if math.isfinite(strategy_value)
        and math.isfinite(buy_hold_value)
        and buy_hold_value > 0
        else None
    )

    output = {
        "today": int(df["epoch_signal"].iloc[-1]),
        "yesterday": int(df["epoch_signal"].iloc[-2]),
        "last_week": int(df["epoch_signal"].iloc[-8]) if len(df) >= 8 else int(df["epoch_signal"].iloc[-1]),
        "last_month": int(df["epoch_signal"].iloc[-31]) if len(df) >= 31 else int(df["epoch_signal"].iloc[-1]),
        "buy_hold_period_return": buy_hold_period_return,
        "strategy_period_return": strategy_period_return,
        # Decimal percentage-point spread retained for the Equity Preview.
        "return_spread": return_spread,
        # Terminal AI equity divided by terminal Buy & Hold equity for this
        # selected (most recent) horizon. 1.00 means equal ending value.
        "outperformance_ratio": outperformance_ratio,
        "strategy_max_drawdown": calculate_max_drawdown(
            [1.0, *strategy_equity_curve]
        ),
        "buy_hold_max_drawdown": calculate_max_drawdown(
            [1.0, *buy_hold_equity_curve]
        ),
        "strategy_annualized_volatility": calculate_annualized_volatility(
            strategy_equity_curve,
            ticker,
        ),
        "buy_hold_annualized_volatility": calculate_annualized_volatility(
            buy_hold_equity_curve,
            ticker,
        ),
        "sharpe_ratio": calculate_sharpe_from_equity_curve(
            strategy_equity_curve,
            ticker,
        ),
        "strategy_market_exposure": calculate_market_exposure(exposure_flags),
        "executed_trade_count": executed_trade_count,
        "observation_count": len(df),
        "coverage_start": df.index.min().date().isoformat(),
        "coverage_end": df.index.max().date().isoformat(),
        # Freshness itself is calculated per request so it can become delayed or
        # stale even when the process-level performance cache remains populated.
        "data_through": df.index.max().date().isoformat(),
    }

    cache[cache_key] = output
    return output


PUBLIC_MODEL_CHANGE_LOG_PATH = os.path.join(
    DATA_DIR,
    "public_model_change_log.json",
)


def public_performance_value_class(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nt-methodology-value-neutral"

    if not math.isfinite(number) or number == 0:
        return "nt-methodology-value-neutral"

    return (
        "nt-methodology-value-positive"
        if number > 0
        else "nt-methodology-value-negative"
    )


def format_public_percent(value, *, points=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"

    if not math.isfinite(number):
        return "—"

    suffix = " pts" if points else "%"
    return f"{number * 100:+.1f}{suffix}"


def format_public_unsigned_percent(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"

    if not math.isfinite(number):
        return "—"

    return f"{number * 100:.1f}%"


def public_signal_label(value):
    try:
        signal = int(value)
    except (TypeError, ValueError):
        signal = 0

    if signal == 1:
        return "BUY", "buy"
    if signal == -1:
        return "SELL", "sell"
    return "HOLD", "hold"


def load_public_model_change_log():
    """Load the version-controlled public model/history change log safely."""
    try:
        with open(PUBLIC_MODEL_CHANGE_LOG_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError):
        app.logger.exception("Could not load public model change log")
        return []

    if not isinstance(payload, list):
        app.logger.error("Public model change log must contain a JSON list")
        return []

    required_fields = (
        "date",
        "scope",
        "change_type",
        "title",
        "summary",
        "historical_signals_changed",
        "affected_period",
    )
    entries = []

    for raw_entry in payload[:100]:
        if not isinstance(raw_entry, dict):
            continue

        entry = {
            field: str(raw_entry.get(field, "")).strip()[:2000]
            for field in required_fields
        }

        if not entry["date"] or not entry["title"] or not entry["summary"]:
            continue

        entries.append(entry)

    entries.sort(key=lambda item: item["date"], reverse=True)
    return entries


def build_methodology_featured_performance():
    """Build a small public one-year snapshot for the four Free assets."""
    rows = []
    csv_version = get_csv_version()

    inactive_tickers = get_retired_public_ticker_set()
    for ticker in ("BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"):
        if ticker in inactive_tickers:
            continue
        try:
            summary = compute_signals_for_ticker(
                ticker,
                period_days=365,
                csv_version=csv_version,
            )
            market_data = load_epoch_csv_for_ticker(ticker)
            snapshot_start = datetime.today().date() - timedelta(days=365)
            market_data = market_data[
                market_data.index >= pd.to_datetime(snapshot_start)
            ].copy()

            if not summary or market_data.empty:
                continue

            strategy_return = summary.get("strategy_period_return")
            buy_hold_return = summary.get("buy_hold_period_return")
            return_spread = summary.get("return_spread")
            strategy_max_drawdown = summary.get("strategy_max_drawdown")
            buy_hold_max_drawdown = summary.get("buy_hold_max_drawdown")
            strategy_market_exposure = summary.get("strategy_market_exposure")
            signal, signal_class = public_signal_label(summary.get("today"))

            rows.append({
                "ticker": ticker,
                "coverage_start": market_data.index.min().date().isoformat(),
                "coverage_end": market_data.index.max().date().isoformat(),
                "observations": f"{len(market_data):,}",
                "strategy_return": format_public_percent(strategy_return),
                "buy_hold_return": format_public_percent(buy_hold_return),
                "return_spread": format_public_percent(
                    return_spread,
                    points=True,
                ),
                "strategy_max_drawdown": format_public_percent(strategy_max_drawdown),
                "buy_hold_max_drawdown": format_public_percent(buy_hold_max_drawdown),
                "sharpe_ratio": f"{float(summary.get('sharpe_ratio') or 0):.2f}",
                "executed_trade_count": f"{int(summary.get('executed_trade_count') or 0):,}",
                "market_exposure": format_public_unsigned_percent(strategy_market_exposure),
                "strategy_class": public_performance_value_class(strategy_return),
                "benchmark_class": public_performance_value_class(buy_hold_return),
                "spread_class": public_performance_value_class(return_spread),
                "strategy_drawdown_class": public_performance_value_class(strategy_max_drawdown),
                "benchmark_drawdown_class": public_performance_value_class(buy_hold_max_drawdown),
                "signal": signal,
                "signal_class": signal_class,
                "freshness": build_data_freshness_metadata(
                    ticker,
                    summary.get("data_through") or market_data.index.max().date(),
                ),
            })
        except Exception:
            app.logger.exception(
                "Could not build methodology snapshot for ticker=%s",
                ticker,
            )

    return rows


def build_methodology_coverage():
    inactive_tickers = get_retired_public_ticker_set()
    public_tickers = [
        ticker
        for ticker in SUPPORTED_TICKERS
        if ticker not in ADMIN_ONLY_TICKERS
        and ticker not in inactive_tickers
    ]

    return {
        "asset_count": len(public_tickers),
        "crypto_count": sum(
            1 for ticker in public_tickers if ticker.endswith("-USD")
        ),
        "stock_count": sum(
            1 for ticker in public_tickers if not ticker.endswith("-USD")
        ),
    }


# --------------------
# Approved CSV-backed Forward Record
# --------------------

FORWARD_RECORD_INITIAL_CASH = 10_000.0
FORWARD_RECORD_MODES = {"sandbox", "public"}


def env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def public_forward_record_started():
    return (
        ForwardRecordAsset.query
        .filter(
            ForwardRecordAsset.record_mode == "public",
            ForwardRecordAsset.status.in_(["active", "retired"]),
        )
        .first()
        is not None
    )


def public_forward_record_enabled():
    """Allow launch by flag, but never hide an already-started record."""
    return (
        env_flag("FORWARD_RECORD_PUBLIC_ENABLED", default=False)
        or public_forward_record_started()
    )


def normalize_forward_record_mode(value):
    mode = str(value or "public").strip().lower()
    if mode not in FORWARD_RECORD_MODES:
        raise ValueError("Forward Record mode must be sandbox or public.")
    return mode


def format_forward_percent(value, *, points=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    suffix = " pp" if points else "%"
    return f"{number * 100:+.1f}{suffix}"


def format_forward_ratio(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.2f}"


def get_forward_record_assets(
    record_mode="public",
    *,
    active_only=False,
    include_removed=False,
):
    mode = normalize_forward_record_mode(record_mode)
    query = ForwardRecordAsset.query.filter_by(record_mode=mode)
    if active_only:
        query = query.filter_by(status="active")
    elif not include_removed:
        query = query.filter(ForwardRecordAsset.status != "removed")
    return query.order_by(
        ForwardRecordAsset.status.asc(),
        ForwardRecordAsset.ticker.asc(),
    ).all()


def get_forward_record_tickers(
    record_mode="public",
    *,
    active_only=False,
    include_removed=False,
):
    return [
        asset.ticker
        for asset in get_forward_record_assets(
            record_mode,
            active_only=active_only,
            include_removed=include_removed,
        )
    ]


def get_forward_record_asset(ticker, record_mode="public"):
    clean = normalize_ticker(ticker)
    if not clean:
        return None
    return ForwardRecordAsset.query.filter_by(
        record_mode=normalize_forward_record_mode(record_mode),
        ticker=clean,
    ).first()


def public_forward_asset_is_retired(ticker):
    """True for assets hidden from ordinary product tools after retirement."""
    asset = get_forward_record_asset(ticker, "public")
    return bool(asset and asset.status in {"retired", "removed"})


def get_retired_public_ticker_set():
    return {
        row[0]
        for row in (
            db.session.query(ForwardRecordAsset.ticker)
            .filter(
                ForwardRecordAsset.record_mode == "public",
                ForwardRecordAsset.status.in_(["retired", "removed"]),
            )
            .all()
        )
    }


def _empty_forward_record(asset, mode, *, data_available=True):
    is_retired = asset.status == "retired"
    end_date = (
        asset.retirement_date
        if is_retired and asset.retirement_date
        else asset.approved_through_date or asset.start_date
    )
    return {
        "record_mode": mode,
        "ticker": asset.ticker,
        "asset_status": asset.status,
        "is_retired": is_retired,
        "data_available": data_available,
        "retired_at": asset.retired_at,
        "retired_date": (
            asset.retirement_date.isoformat()
            if asset.retirement_date else None
        ),
        "removal_after_date": (
            asset.removal_after_date.isoformat()
            if asset.removal_after_date else None
        ),
        "retirement_reason": asset.retirement_reason,
        "record_started": asset.enrolled_at,
        "record_started_date": asset.start_date.isoformat(),
        "record_age_days": max(0, (end_date - asset.start_date).days),
        "approved_through_date": (
            asset.approved_through_date.isoformat()
            if asset.approved_through_date else None
        ),
        "last_approved_at": asset.last_approved_at,
        "publication_count": 0,
        "observation_count": 0,
        "latest_signal": "—",
        "latest_signal_class": "hold",
        "pending_execution": False,
        "strategy_return": None,
        "benchmark_return": None,
        "return_spread": None,
        "strategy_max_drawdown": None,
        "benchmark_max_drawdown": None,
        "strategy_volatility": None,
        "sharpe_ratio": None,
        "market_exposure": None,
        "trade_count": 0,
        "strategy_return_display": "—",
        "benchmark_return_display": "—",
        "return_spread_display": "—",
        "strategy_max_drawdown_display": "—",
        "benchmark_max_drawdown_display": "—",
        "strategy_volatility_display": "—",
        "sharpe_ratio_display": "Not enough history",
        "market_exposure_display": "—",
        "chart": {"dates": [], "strategy": [], "benchmark": []},
        "recent_publications": [],
    }


def build_forward_record_performance(ticker, record_mode="public"):
    """Calculate one asset's approved record directly from its slim CSV.

    This intentionally mirrors the historical backtest treatment: the fixed
    start boundary is the asset's approved tracking start date, the end is the
    latest approved row (or retirement date), BUY/SELL actions use each row's
    closing price, transaction costs are included, and slippage is excluded.
    """
    mode = normalize_forward_record_mode(record_mode)
    clean = normalize_ticker(ticker)
    asset = get_forward_record_asset(clean, mode)
    if asset is None or asset.status == "removed":
        return None

    try:
        data = load_forward_record_csv(clean, mode)
        data_available = True
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError):
        data = pd.DataFrame(columns=["Close", "epoch_signal"])
        data.index = pd.DatetimeIndex([], name="Date")
        data_available = False

    if not data.empty:
        data = data[data.index.date >= asset.start_date].copy()
        if asset.status == "retired" and asset.retirement_date:
            data = data[data.index.date <= asset.retirement_date].copy()
        if asset.approved_through_date:
            data = data[data.index.date <= asset.approved_through_date].copy()

    if data.empty:
        return _empty_forward_record(asset, mode, data_available=data_available)

    initial_cash = FORWARD_RECORD_INITIAL_CASH
    transaction_cost_rate = get_transaction_cost_rate(clean)
    strategy_cash = initial_cash
    strategy_quantity = 0.0
    benchmark_cash = initial_cash
    benchmark_quantity = 0.0
    benchmark_started = False
    trade_count = 0
    exposure_flags = []
    chart_dates = []
    strategy_values = []
    benchmark_values = []

    for index, row in data.iterrows():
        close_price = float(row["Close"])
        signal = int(row["epoch_signal"])
        if not math.isfinite(close_price) or close_price <= 0:
            continue

        if not benchmark_started:
            benchmark_quantity = benchmark_cash / (
                close_price * (1 + transaction_cost_rate)
            )
            benchmark_cash -= benchmark_quantity * close_price * (
                1 + transaction_cost_rate
            )
            benchmark_started = True

        if signal == 1 and strategy_quantity <= 0 and strategy_cash > 0:
            strategy_quantity = strategy_cash / (
                close_price * (1 + transaction_cost_rate)
            )
            strategy_cash -= strategy_quantity * close_price * (
                1 + transaction_cost_rate
            )
            strategy_cash = max(0.0, strategy_cash)
            trade_count += 1
        elif signal == -1 and strategy_quantity > 0:
            strategy_cash += strategy_quantity * close_price * (
                1 - transaction_cost_rate
            )
            strategy_quantity = 0.0
            trade_count += 1

        strategy_value = strategy_cash + strategy_quantity * close_price
        benchmark_value = benchmark_cash + benchmark_quantity * close_price
        chart_dates.append(index.date().isoformat())
        strategy_values.append(float(strategy_value))
        benchmark_values.append(float(benchmark_value))
        exposure_flags.append(strategy_quantity > 0)

    if not strategy_values:
        return _empty_forward_record(asset, mode, data_available=data_available)

    metric_strategy_values = [initial_cash, *strategy_values]
    metric_benchmark_values = [initial_cash, *benchmark_values]
    strategy_return = strategy_values[-1] / initial_cash - 1
    benchmark_return = benchmark_values[-1] / initial_cash - 1
    return_spread = strategy_return - benchmark_return
    strategy_drawdown = calculate_max_drawdown(metric_strategy_values)
    benchmark_drawdown = calculate_max_drawdown(metric_benchmark_values)
    strategy_volatility = calculate_annualized_volatility(
        metric_strategy_values,
        clean,
    )
    sharpe_ratio = calculate_sharpe_from_equity_curve(
        metric_strategy_values,
        clean,
    )
    market_exposure = calculate_market_exposure(exposure_flags)

    last_row = data.iloc[-1]
    latest_signal = signal_label(int(last_row["epoch_signal"]))
    end_date = (
        asset.retirement_date
        if asset.status == "retired" and asset.retirement_date
        else data.index.max().date()
    )
    recent_rows = list(data.tail(12).iloc[::-1].iterrows())

    return {
        "record_mode": mode,
        "ticker": clean,
        "asset_status": asset.status,
        "is_retired": asset.status == "retired",
        "data_available": data_available,
        "retired_at": asset.retired_at,
        "retired_date": (
            asset.retirement_date.isoformat()
            if asset.retirement_date else None
        ),
        "removal_after_date": (
            asset.removal_after_date.isoformat()
            if asset.removal_after_date else None
        ),
        "retirement_reason": asset.retirement_reason,
        "record_started": asset.enrolled_at,
        "record_started_date": asset.start_date.isoformat(),
        "record_age_days": max(0, (end_date - asset.start_date).days),
        "approved_through_date": data.index.max().date().isoformat(),
        "last_approved_at": asset.last_approved_at,
        "publication_count": len(data),
        "observation_count": len(strategy_values),
        "latest_signal": latest_signal,
        "latest_signal_class": latest_signal.lower(),
        "pending_execution": False,
        "strategy_return": strategy_return,
        "benchmark_return": benchmark_return,
        "return_spread": return_spread,
        "strategy_max_drawdown": strategy_drawdown,
        "benchmark_max_drawdown": benchmark_drawdown,
        "strategy_volatility": strategy_volatility,
        "sharpe_ratio": sharpe_ratio,
        "market_exposure": market_exposure,
        "trade_count": trade_count,
        "strategy_return_display": format_forward_percent(strategy_return),
        "benchmark_return_display": format_forward_percent(benchmark_return),
        "return_spread_display": format_forward_percent(return_spread, points=True),
        "strategy_max_drawdown_display": format_forward_percent(strategy_drawdown),
        "benchmark_max_drawdown_display": format_forward_percent(benchmark_drawdown),
        "strategy_volatility_display": format_forward_percent(strategy_volatility),
        "sharpe_ratio_display": (
            format_forward_ratio(sharpe_ratio)
            if len(strategy_values) >= 3 else "Not enough history"
        ),
        "market_exposure_display": (
            format_public_unsigned_percent(market_exposure)
            if market_exposure is not None else "—"
        ),
        "chart": {
            "dates": chart_dates,
            "strategy": [round(value, 4) for value in strategy_values],
            "benchmark": [round(value, 4) for value in benchmark_values],
        },
        "recent_publications": [
            {
                "published_at": None,
                "source_data_date": index.date(),
                "signal": signal_label(int(row["epoch_signal"])),
                "signal_class": signal_label(int(row["epoch_signal"])).lower(),
                "status": "Approved",
                "status_class": "regular",
            }
            for index, row in recent_rows
        ],
    }


def build_forward_record_summary(record_mode="public"):
    mode = normalize_forward_record_mode(record_mode)
    all_assets = get_forward_record_assets(mode, include_removed=True)
    visible_assets = [asset for asset in all_assets if asset.status != "removed"]
    if not visible_assets:
        return {
            "record_mode": mode,
            "started": False,
            "asset_count": 0,
            "active_asset_count": 0,
            "retired_asset_count": 0,
            "removed_asset_count": len(all_assets),
            "record_started_date": None,
        }

    active_count = sum(1 for asset in visible_assets if asset.status == "active")
    retired_count = sum(1 for asset in visible_assets if asset.status == "retired")
    first_start = min(asset.start_date for asset in visible_assets)
    return {
        "record_mode": mode,
        "started": True,
        "asset_count": len(visible_assets),
        "active_asset_count": active_count,
        "retired_asset_count": retired_count,
        "removed_asset_count": sum(1 for asset in all_assets if asset.status == "removed"),
        "record_started_date": first_start.isoformat(),
    }


def build_forward_record_home_summary():
    if not public_forward_record_enabled():
        return {
            "record_mode": "public",
            "started": False,
            "asset_count": 0,
            "active_asset_count": 0,
            "retired_asset_count": 0,
            "removed_asset_count": 0,
            "record_started_date": None,
        }
    return build_forward_record_summary("public")


# --------------------
# Live simulation helpers
# --------------------

def is_crypto_ticker(ticker: str) -> bool:
    return str(ticker).upper().endswith("-USD")


def get_asset_type(ticker: str) -> str:
    return "crypto" if is_crypto_ticker(ticker) else "stock"


def get_transaction_cost_rate(ticker: str) -> float:
    """
    Same transaction-cost logic used by EpochSignaler:
    crypto = 1% per side
    stock = 0.1% per side
    """
    return 0.01 if is_crypto_ticker(ticker) else 0.001


def get_return_periods_per_year(ticker: str) -> int:
    """Return the annualization basis used for daily observations."""
    return 365 if is_crypto_ticker(ticker) else 252


def calculate_sharpe_from_equity_curve(
    equity_values,
    ticker: str,
    risk_free_rate_annual: float = 0.01,
) -> float:
    """
    Calculate annualized Sharpe from the displayed strategy equity curve.

    This intentionally uses strategy-equity returns rather than underlying
    asset-price returns, so the reported risk-adjusted result matches the
    strategy shown to the user.
    """
    series = pd.Series(equity_values, dtype="float64")
    series = series.replace([float("inf"), float("-inf")], pd.NA).dropna()

    if len(series) < 3 or (series <= 0).any():
        return 0.0

    returns = (
        series.pct_change()
        .replace([float("inf"), float("-inf")], pd.NA)
        .dropna()
    )

    if len(returns) < 2:
        return 0.0

    periods_per_year = get_return_periods_per_year(ticker)
    risk_free_per_period = (1 + risk_free_rate_annual) ** (1 / periods_per_year) - 1
    excess_returns = returns - risk_free_per_period
    volatility = excess_returns.std(ddof=1)

    if not math.isfinite(float(volatility)) or volatility <= 0:
        return 0.0

    sharpe = (excess_returns.mean() / volatility) * math.sqrt(periods_per_year)

    if not math.isfinite(float(sharpe)):
        return 0.0

    return float(sharpe)


def calculate_max_drawdown(equity_values) -> float:
    """Return the largest peak-to-trough decline as a negative decimal."""
    series = pd.Series(equity_values, dtype="float64")
    series = series.replace([float("inf"), float("-inf")], pd.NA).dropna()

    if series.empty or (series <= 0).any():
        return 0.0

    running_peak = series.cummax()
    drawdowns = (series / running_peak) - 1
    maximum_drawdown = float(drawdowns.min())

    if not math.isfinite(maximum_drawdown):
        return 0.0

    return min(0.0, maximum_drawdown)


def calculate_annualized_volatility(equity_values, ticker: str) -> float:
    """Annualize daily equity-return volatility using the asset calendar."""
    series = pd.Series(equity_values, dtype="float64")
    series = series.replace([float("inf"), float("-inf")], pd.NA).dropna()

    if len(series) < 3 or (series <= 0).any():
        return 0.0

    returns = (
        series.pct_change()
        .replace([float("inf"), float("-inf")], pd.NA)
        .dropna()
    )

    if len(returns) < 2:
        return 0.0

    volatility = returns.std(ddof=1) * math.sqrt(
        get_return_periods_per_year(ticker)
    )

    if not math.isfinite(float(volatility)):
        return 0.0

    return max(0.0, float(volatility))


def calculate_market_exposure(exposure_flags) -> float:
    """Return the share of displayed observations with an open strategy position."""
    flags = [bool(value) for value in exposure_flags]
    if not flags:
        return 0.0
    return float(sum(flags) / len(flags))


def build_buy_and_hold_benchmark(
    initial_cash: float,
    prices,
    ticker: str,
    transaction_cost_rate: float,
    *,
    enforce_whole_stock_shares: bool,
):
    """
    Build a buy-and-hold equity curve with entry cost and residual cash.

    For real-cash stock simulations/backtests, whole shares are enforced and
    any uninvested cash remains part of benchmark equity. For normalized
    previews, fractional notional is allowed so a $1 base remains meaningful.
    """
    price_series = pd.Series(prices, dtype="float64")

    if price_series.empty:
        raise ValueError("Benchmark requires at least one price.")

    if (
        price_series.isna().any()
        or (~price_series.map(math.isfinite)).any()
        or (price_series <= 0).any()
    ):
        raise ValueError("Benchmark contains an invalid market price.")

    if not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("Benchmark initial cash must be positive and finite.")

    first_price = float(price_series.iloc[0])
    gross_budget = initial_cash / (1 + transaction_cost_rate)
    raw_quantity = gross_budget / first_price

    if enforce_whole_stock_shares and not is_crypto_ticker(ticker):
        quantity = float(math.floor(raw_quantity))
    else:
        quantity = float(raw_quantity)

    entry_total = quantity * first_price * (1 + transaction_cost_rate)
    residual_cash = max(0.0, float(initial_cash - entry_total))
    equity_curve = (price_series * quantity + residual_cash).astype(float).tolist()

    return quantity, residual_cash, equity_curve


def load_epoch_csv_for_ticker(ticker: str) -> pd.DataFrame:
    """
    Loads the local epoch CSV for a ticker.
    Example:
    BTC-USD -> data/epoch_BTC.csv
    ETH-USD -> data/epoch_ETH.csv
    AAPL -> data/epoch_AAPL.csv
    """
    ticker = normalize_ticker(ticker)
    csv_path = get_epoch_csv_path(ticker)

    df = pd.read_csv(
        csv_path,
        usecols=["Date", "Close", "epoch_signal"],
        parse_dates=["Date"]
    )

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["epoch_signal"] = pd.to_numeric(df["epoch_signal"], errors="coerce")

    df = df.dropna(subset=["Date", "Close", "epoch_signal"]).copy()

    finite_close = df["Close"].map(math.isfinite)
    finite_signal = df["epoch_signal"].map(math.isfinite)
    valid_signal = df["epoch_signal"].isin([-1, 0, 1])

    df = df[finite_close & finite_signal & valid_signal & (df["Close"] > 0)].copy()
    df["epoch_signal"] = df["epoch_signal"].astype(int)
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df.set_index("Date", inplace=True)

    if len(df) < 1:
        raise ValueError("Market data has no valid rows for this asset.")

    return df


def normalize_live_quantity_for_buy(ticker, raw_quantity, price, cash_allocation, transaction_cost_rate):
    """
    Crypto: fractional quantity allowed.
    Stock: whole shares only. Round to nearest integer, but never exceed available allocation.
    """
    if is_crypto_ticker(ticker):
        return float(raw_quantity)

    quantity = int(round(raw_quantity))

    while quantity > 0:
        gross_amount = quantity * price
        transaction_cost = gross_amount * transaction_cost_rate
        total_needed = gross_amount + transaction_cost

        if total_needed <= cash_allocation + 1e-9:
            return float(quantity)

        quantity -= 1

    return 0.0


def normalize_live_quantity_for_sell(ticker, raw_quantity, current_position):
    """
    Crypto: fractional sell allowed.
    Stock: whole-share sell only.
    """
    if current_position <= 0:
        return 0.0

    if is_crypto_ticker(ticker):
        return float(min(raw_quantity, current_position))

    quantity = int(round(raw_quantity))
    quantity = min(quantity, int(current_position))

    if quantity < 1:
        return 0.0

    return float(quantity)


def get_latest_equity_point(simulation_id):
    return (
        LiveSimulationEquity.query
        .filter_by(simulation_id=simulation_id)
        .order_by(LiveSimulationEquity.equity_date.desc())
        .first()
    )

def enforce_free_live_simulation_limit_for_user(user):
    """
    For Free users, keep only the newest allowed active live simulations up to the Free limit.
    Extra active simulations are paused.
    """
    if not user or is_paid_user(user):
        return 0

    limit = get_live_simulation_limit_for_user(user)

    active_sims = (
        LiveSimulation.query
        .filter_by(user_id=user.id, status="active")
        .order_by(LiveSimulation.updated_at.desc(), LiveSimulation.id.desc())
        .all()
    )

    kept_count = 0
    paused_count = 0

    for sim in active_sims:
        ticker = str(sim.ticker or "").strip().upper()

        # Pro-only tickers should never stay active for Free users.
        if ticker not in TOP_FREE_TICKERS:
            sim.status = "paused"
            paused_count += 1
            continue

        # Keep only the newest 5 active Free-allowed simulations.
        if kept_count < limit:
            kept_count += 1
        else:
            sim.status = "paused"
            paused_count += 1

    if paused_count:
        db.session.commit()

    return paused_count

_LIVE_SIM_LATEST_CSV_UNSET = object()

def live_simulation_summary(
    sim,
    *,
    latest_csv_date=_LIVE_SIM_LATEST_CSV_UNSET,
    equity_points=None,
):
    # Reuse a single ordered equity query for the latest point and all horizon
    # calculations. The old implementation queried the same history twice per
    # simulation, which became expensive for accounts with many simulations.
    if equity_points is None:
        equity_points = (
            LiveSimulationEquity.query
            .filter_by(simulation_id=sim.id)
            .order_by(LiveSimulationEquity.equity_date.asc())
            .all()
        )

    latest = equity_points[-1] if equity_points else None

    if latest_csv_date is _LIVE_SIM_LATEST_CSV_UNSET:
        latest_csv_date = get_latest_csv_date_for_ticker(sim.ticker)
    
    strategy_value = latest.strategy_value if latest else sim.initial_cash
    benchmark_value = latest.benchmark_value if latest else sim.initial_cash

    strategy_return = (
        (strategy_value / sim.initial_cash) - 1
        if sim.initial_cash else 0.0
    )

    benchmark_return = (
        (benchmark_value / sim.initial_cash) - 1
        if sim.initial_cash else 0.0
    )

    value_difference = strategy_value - benchmark_value
    return_spread = strategy_return - benchmark_return

    trade_count = LiveSimulationTrade.query.filter_by(
        simulation_id=sim.id
    ).count()

    source_freshness = build_data_freshness_metadata(
        sim.ticker,
        latest_csv_date,
    )

    data = sim.to_dict()
    data.update({
        "latest_strategy_value": strategy_value,
        "latest_benchmark_value": benchmark_value,
        "strategy_return": strategy_return,
        "benchmark_return": benchmark_return,
        "value_difference": value_difference,
        # Decimal percentage-point spread between strategy and benchmark returns.
        "return_spread": return_spread,
        "trade_count": trade_count,
        "latest_equity_date": latest.equity_date.isoformat() if latest else None,
        "latest_signal": int(latest.signal) if latest else None,
        "latest_close_price": float(latest.close_price) if latest else None,
        "latest_csv_date": latest_csv_date.isoformat() if latest_csv_date else None,
        "simulation_through": latest.equity_date.isoformat() if latest else None,
        **source_freshness,
        "is_current_with_csv": (
            sim.last_processed_date == latest_csv_date
            if sim.last_processed_date and latest_csv_date else False
        ),
        "horizon_returns": build_live_sim_horizon_returns(
            sim,
            points=equity_points,
        ),
    })

    return data

def live_simulation_curve_payload(sim):
    """Return the chart/trade payload without recomputing summary metrics.

    The Signal Board-style Live Simulation table already has the simulation
    summary. Loading a selected chart should therefore only fetch the curve and
    trades instead of re-reading the full history again for summary/horizon
    calculations.
    """
    equity_points = (
        LiveSimulationEquity.query
        .filter_by(simulation_id=sim.id)
        .order_by(LiveSimulationEquity.equity_date.asc())
        .all()
    )

    trades = (
        LiveSimulationTrade.query
        .filter_by(simulation_id=sim.id)
        .order_by(LiveSimulationTrade.trade_date.asc(), LiveSimulationTrade.id.asc())
        .all()
    )

    return {
        "dates": [p.equity_date.isoformat() for p in equity_points],
        "strategy_curve": [p.strategy_value for p in equity_points],
        "benchmark_curve": [p.benchmark_value for p in equity_points],
        "signals": [p.signal for p in equity_points],
        "close_prices": [p.close_price for p in equity_points],
        "trades": [t.to_dict() for t in trades],
    }


def live_simulation_detail(sim):
    equity_points = (
        LiveSimulationEquity.query
        .filter_by(simulation_id=sim.id)
        .order_by(LiveSimulationEquity.equity_date.asc())
        .all()
    )

    trades = (
        LiveSimulationTrade.query
        .filter_by(simulation_id=sim.id)
        .order_by(LiveSimulationTrade.trade_date.asc(), LiveSimulationTrade.id.asc())
        .all()
    )

    summary = live_simulation_summary(
        sim,
        equity_points=equity_points,
    )

    summary.update({
        "dates": [p.equity_date.isoformat() for p in equity_points],
        "strategy_curve": [p.strategy_value for p in equity_points],
        "benchmark_curve": [p.benchmark_value for p in equity_points],
        "signals": [p.signal for p in equity_points],
        "close_prices": [p.close_price for p in equity_points],
        "trades": [t.to_dict() for t in trades],
    })

    return summary


def build_live_simulation_portfolio_total(
    simulations,
    *,
    summaries=None,
    include_curve=False,
    view="open",
):
    """Build a combined portfolio view across the supplied live simulations.

    The combined equity curve sums every simulation's strategy and benchmark
    equity. Simulations that started later are carried at their initial cash
    before their own start date, which keeps the combined capital base stable
    and avoids artificial jumps simply because a new simulation was created.
    """
    simulations = list(simulations or [])
    if not simulations:
        return None

    if summaries is None:
        summaries = [live_simulation_summary(sim) for sim in simulations]
    else:
        summaries = list(summaries)

    summary_by_id = {
        int(item.get("id")): item
        for item in summaries
        if item.get("id") is not None
    }

    total_initial_cash = sum(
        safe_live_sim_float(sim.initial_cash) or 0.0
        for sim in simulations
    )
    total_cash_balance = sum(
        safe_live_sim_float(sim.cash_balance) or 0.0
        for sim in simulations
    )

    latest_strategy_value = 0.0
    latest_benchmark_value = 0.0
    total_trade_count = 0

    for sim in simulations:
        item = summary_by_id.get(int(sim.id), {})
        initial_cash = safe_live_sim_float(sim.initial_cash) or 0.0

        strategy_value = (
            safe_live_sim_float(item.get("latest_strategy_value"))
            if item
            else None
        )
        benchmark_value = (
            safe_live_sim_float(item.get("latest_benchmark_value"))
            if item
            else None
        )

        latest_strategy_value += (
            strategy_value if strategy_value is not None else initial_cash
        )
        latest_benchmark_value += (
            benchmark_value if benchmark_value is not None else initial_cash
        )
        total_trade_count += int(item.get("trade_count") or 0) if item else 0

    strategy_return = safe_live_sim_return(
        latest_strategy_value,
        total_initial_cash,
    )
    benchmark_return = safe_live_sim_return(
        latest_benchmark_value,
        total_initial_cash,
    )
    return_spread = None
    if strategy_return is not None and benchmark_return is not None:
        return_spread = strategy_return - benchmark_return

    value_difference = latest_strategy_value - latest_benchmark_value

    start_dates = [sim.start_date for sim in simulations if sim.start_date]
    earliest_start_date = min(start_dates) if start_dates else None

    latest_equity_dates = []
    data_through_dates = []
    site_update_timestamps = []
    freshness_statuses = []
    all_current_with_csv = True

    for sim in simulations:
        item = summary_by_id.get(int(sim.id), {})
        latest_equity_date = item.get("latest_equity_date") if item else None
        data_through = item.get("data_through") if item else None
        site_update = item.get("site_data_updated_at_utc") if item else None
        freshness_status = str(item.get("freshness_status") or "unknown") if item else "unknown"

        if latest_equity_date:
            try:
                latest_equity_dates.append(
                    datetime.strptime(str(latest_equity_date)[:10], "%Y-%m-%d").date()
                )
            except ValueError:
                pass

        if data_through:
            try:
                data_through_dates.append(
                    datetime.strptime(str(data_through)[:10], "%Y-%m-%d").date()
                )
            except ValueError:
                pass

        if site_update:
            site_update_timestamps.append(str(site_update))

        freshness_statuses.append(freshness_status)
        if not bool(item.get("is_current_with_csv")):
            all_current_with_csv = False

    # A combined portfolio is only as current as its least-current component.
    freshness_rank = {
        "current": 0,
        "delayed": 1,
        "stale": 2,
        "unknown": 3,
    }
    portfolio_freshness_status = max(
        freshness_statuses or ["unknown"],
        key=lambda value: freshness_rank.get(value, 3),
    )
    portfolio_freshness_label = FRESHNESS_STATUS_LABELS.get(
        portfolio_freshness_status,
        "Unknown",
    )

    freshness_counts = {
        "current": 0,
        "delayed": 0,
        "stale": 0,
        "unknown": 0,
    }
    for freshness_status in freshness_statuses:
        normalized_status = (
            freshness_status
            if freshness_status in freshness_counts
            else "unknown"
        )
        freshness_counts[normalized_status] += 1

    common_simulation_through = (
        min(latest_equity_dates).isoformat()
        if latest_equity_dates and len(latest_equity_dates) == len(simulations)
        else None
    )
    common_data_through = (
        min(data_through_dates).isoformat()
        if data_through_dates and len(data_through_dates) == len(simulations)
        else None
    )
    latest_equity_date = (
        max(latest_equity_dates).isoformat()
        if latest_equity_dates
        else earliest_start_date.isoformat() if earliest_start_date else None
    )

    # Build portfolio-level horizon metrics by weighting every simulation by
    # its actual capital base instead of averaging individual percentages.
    horizon_returns = {}
    for horizon_key in LIVE_SIM_HORIZON_DAYS.keys():
        base_strategy_value = 0.0
        base_benchmark_value = 0.0
        current_strategy_value = 0.0
        current_benchmark_value = 0.0
        base_dates = []
        latest_dates = []

        for sim in simulations:
            item = summary_by_id.get(int(sim.id), {})
            metrics = (item.get("horizon_returns") or {}).get(horizon_key)
            if metrics is None:
                metrics = (item.get("horizon_returns") or {}).get("since_start") or {}

            initial_cash = safe_live_sim_float(sim.initial_cash) or 0.0
            current_strategy = (
                safe_live_sim_float(item.get("latest_strategy_value"))
                if item
                else None
            )
            current_benchmark = (
                safe_live_sim_float(item.get("latest_benchmark_value"))
                if item
                else None
            )
            current_strategy = current_strategy if current_strategy is not None else initial_cash
            current_benchmark = current_benchmark if current_benchmark is not None else initial_cash

            base_strategy = safe_live_sim_float(metrics.get("base_strategy_value"))
            base_benchmark = safe_live_sim_float(metrics.get("base_benchmark_value"))
            if base_strategy is None:
                base_strategy = initial_cash
            if base_benchmark is None:
                base_benchmark = initial_cash

            base_strategy_value += base_strategy
            base_benchmark_value += base_benchmark
            current_strategy_value += current_strategy
            current_benchmark_value += current_benchmark

            if metrics.get("base_date"):
                base_dates.append(str(metrics.get("base_date")))
            if metrics.get("latest_date"):
                latest_dates.append(str(metrics.get("latest_date")))

        portfolio_strategy_return = safe_live_sim_return(
            current_strategy_value,
            base_strategy_value,
        )
        portfolio_benchmark_return = safe_live_sim_return(
            current_benchmark_value,
            base_benchmark_value,
        )
        portfolio_return_spread = None
        if portfolio_strategy_return is not None and portfolio_benchmark_return is not None:
            portfolio_return_spread = portfolio_strategy_return - portfolio_benchmark_return

        strategy_change = current_strategy_value - base_strategy_value
        benchmark_change = current_benchmark_value - base_benchmark_value

        horizon_returns[horizon_key] = {
            "base_date": min(base_dates) if base_dates else None,
            "latest_date": max(latest_dates) if latest_dates else latest_equity_date,
            "strategy_return": portfolio_strategy_return,
            "benchmark_return": portfolio_benchmark_return,
            "return_spread": portfolio_return_spread,
            "strategy_change": strategy_change,
            "benchmark_change": benchmark_change,
            "value_difference": strategy_change - benchmark_change,
            "strategy_value": current_strategy_value,
            "benchmark_value": current_benchmark_value,
            "base_strategy_value": base_strategy_value,
            "base_benchmark_value": base_benchmark_value,
        }

    output = {
        "id": "portfolio-total",
        "is_portfolio_total": True,
        "name": "Portfolio Total",
        "ticker": "All assets",
        "asset_type": "mixed",
        "status": str(view or "open"),
        "view": str(view or "open"),
        "simulation_count": len(simulations),
        "initial_cash": total_initial_cash,
        "cash_balance": total_cash_balance,
        "position_quantity": None,
        "position_size_pct": None,
        "start_date": earliest_start_date.isoformat() if earliest_start_date else None,
        "latest_strategy_value": latest_strategy_value,
        "latest_benchmark_value": latest_benchmark_value,
        "strategy_return": strategy_return,
        "benchmark_return": benchmark_return,
        "value_difference": value_difference,
        "return_spread": return_spread,
        "trade_count": total_trade_count,
        "latest_equity_date": latest_equity_date,
        "latest_signal": None,
        "latest_close_price": None,
        "latest_csv_date": common_data_through,
        "data_through": common_data_through,
        "simulation_through": common_simulation_through,
        "site_data_updated_at_utc": max(site_update_timestamps) if site_update_timestamps else None,
        "freshness_status": portfolio_freshness_status,
        "freshness_label": portfolio_freshness_label,
        "freshness_message": (
            "Combined portfolio freshness is summarized across all simulations in this view."
        ),
        "freshness_counts": freshness_counts,
        "is_current_with_csv": all_current_with_csv,
        "horizon_returns": horizon_returns,
    }

    if include_curve:
        sim_ids = [sim.id for sim in simulations]
        equity_points = (
            LiveSimulationEquity.query
            .filter(LiveSimulationEquity.simulation_id.in_(sim_ids))
            .order_by(
                LiveSimulationEquity.equity_date.asc(),
                LiveSimulationEquity.simulation_id.asc(),
            )
            .all()
        )

        points_by_sim = {int(sim.id): [] for sim in simulations}
        all_dates = set(start_dates)
        for point in equity_points:
            points_by_sim.setdefault(int(point.simulation_id), []).append(point)
            if point.equity_date:
                all_dates.add(point.equity_date)

        ordered_dates = sorted(date for date in all_dates if date is not None)
        states = {}
        for sim in simulations:
            states[int(sim.id)] = {
                "points": points_by_sim.get(int(sim.id), []),
                "index": -1,
                "strategy": safe_live_sim_float(sim.initial_cash) or 0.0,
                "benchmark": safe_live_sim_float(sim.initial_cash) or 0.0,
            }

        strategy_curve = []
        benchmark_curve = []

        for equity_date in ordered_dates:
            portfolio_strategy_value = 0.0
            portfolio_benchmark_value = 0.0

            for sim in simulations:
                state = states[int(sim.id)]
                points = state["points"]

                while (
                    state["index"] + 1 < len(points)
                    and points[state["index"] + 1].equity_date <= equity_date
                ):
                    state["index"] += 1
                    point = points[state["index"]]
                    strategy_value = safe_live_sim_float(point.strategy_value)
                    benchmark_value = safe_live_sim_float(point.benchmark_value)
                    if strategy_value is not None:
                        state["strategy"] = strategy_value
                    if benchmark_value is not None:
                        state["benchmark"] = benchmark_value

                portfolio_strategy_value += state["strategy"]
                portfolio_benchmark_value += state["benchmark"]

            strategy_curve.append(portfolio_strategy_value)
            benchmark_curve.append(portfolio_benchmark_value)

        output.update({
            "dates": [date.isoformat() for date in ordered_dates],
            "strategy_curve": strategy_curve,
            "benchmark_curve": benchmark_curve,
            "signals": [],
            "close_prices": [],
            "trades": [],
            "curve_note": (
                "Each simulation contributes its initial cash before its own start date, "
                "then its recorded equity thereafter."
            ),
        })

        if strategy_curve:
            output["latest_strategy_value"] = strategy_curve[-1]
        if benchmark_curve:
            output["latest_benchmark_value"] = benchmark_curve[-1]

    return output


def can_access_live_sim_ticker_for_user(user, ticker):
    return can_use_live_simulation_ticker(user, ticker)

def freeze_locked_live_simulations_for_user(user):
    """
    If a user is no longer Pro, pause active simulations for Pro-only tickers.
    They remain visible as old/frozen paper simulations, but they stop processing new CSV rows.
    """
    if not user or is_paid_user(user):
        return 0

    locked_active_sims = (
        LiveSimulation.query
        .filter(
            LiveSimulation.user_id == user.id,
            ~LiveSimulation.ticker.in_(list(TOP_FREE_TICKERS)),
            LiveSimulation.status == "active"
        )
        .all()
    )

    for sim in locked_active_sims:
        sim.status = "paused"

    if locked_active_sims:
        db.session.commit()

    return len(locked_active_sims)


def live_sim_can_update_for_user(user, sim):
    if not sim:
        return False

    if sim.status != "active":
        return False

    return can_access_live_sim_ticker_for_user(user, sim.ticker)

def validate_live_simulation_accounting_state(sim):
    """
    Reject non-finite or materially negative persisted accounting state.

    Tiny floating-point residues are normalized to zero. Larger negative values
    indicate corrupted state and must not be carried into another refresh.
    """
    numeric_fields = {
        "initial_cash": sim.initial_cash,
        "cash_balance": sim.cash_balance,
        "position_quantity": sim.position_quantity,
        "position_size_pct": sim.position_size_pct,
        "transaction_cost_rate": sim.transaction_cost_rate,
        "benchmark_quantity": sim.benchmark_quantity,
        "benchmark_cash_balance": sim.benchmark_cash_balance,
    }

    normalized = {}

    for field_name, raw_value in numeric_fields.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Live simulation {sim.id} has invalid {field_name}."
            )

        if not math.isfinite(value):
            raise ValueError(
                f"Live simulation {sim.id} has non-finite {field_name}."
            )

        normalized[field_name] = value

    if normalized["initial_cash"] <= 0:
        raise ValueError(
            f"Live simulation {sim.id} has invalid initial cash."
        )

    if normalized["cash_balance"] < -1e-7:
        raise ValueError(
            f"Live simulation {sim.id} has a negative cash balance."
        )

    if normalized["position_quantity"] < -1e-12:
        raise ValueError(
            f"Live simulation {sim.id} has a negative position."
        )

    if not 0 < normalized["position_size_pct"] <= 100:
        raise ValueError(
            f"Live simulation {sim.id} has invalid position size."
        )

    if not 0 <= normalized["transaction_cost_rate"] < 1:
        raise ValueError(
            f"Live simulation {sim.id} has invalid transaction cost."
        )

    if normalized["benchmark_quantity"] < -1e-12:
        raise ValueError(
            f"Live simulation {sim.id} has a negative benchmark quantity."
        )

    if normalized["benchmark_cash_balance"] < -1e-7:
        raise ValueError(
            f"Live simulation {sim.id} has a negative benchmark cash balance."
        )

    if abs(normalized["cash_balance"]) < 1e-10:
        sim.cash_balance = 0.0

    if abs(normalized["position_quantity"]) < 1e-12:
        sim.position_quantity = 0.0

    if abs(normalized["benchmark_quantity"]) < 1e-12:
        sim.benchmark_quantity = 0.0

    if abs(normalized["benchmark_cash_balance"]) < 1e-10:
        sim.benchmark_cash_balance = 0.0


def update_live_simulation_from_csv(sim, user=None):
    """
    Atomically process fresh CSV rows for one live simulation.

    Reliability guarantees:
    - SELECT ... FOR UPDATE serializes concurrent refreshes for this simulation.
    - last_processed_date is re-read only after the row lock is acquired.
    - one transaction contains every balance, trade, equity, and checkpoint update.
    - any exception rolls the entire refresh back.
    - existing legacy equity/trade rows are treated idempotently rather than
      generating a second trade for the same simulation date.
    """
    simulation_id = getattr(sim, "id", None)

    try:
        simulation_id = int(simulation_id)
    except (TypeError, ValueError):
        raise ValueError("A persisted live simulation is required.")

    try:
        # populate_existing forces values already present in the identity map to
        # be refreshed from PostgreSQL after the row lock is acquired.
        locked_sim = (
            db.session.query(LiveSimulation)
            .filter(LiveSimulation.id == simulation_id)
            .populate_existing()
            .with_for_update()
            .one_or_none()
        )

        if locked_sim is None:
            db.session.rollback()
            return None

        if locked_sim.status != "active":
            db.session.commit()
            return locked_sim

        owner = user

        if (
            owner is None
            or getattr(owner, "id", None) != locked_sim.user_id
        ):
            owner = db.session.get(User, locked_sim.user_id)

        if (
            owner is not None
            and not can_access_live_sim_ticker_for_user(
                owner,
                locked_sim.ticker,
            )
        ):
            db.session.commit()
            return locked_sim

        validate_live_simulation_accounting_state(locked_sim)

        df = load_epoch_csv_for_ticker(locked_sim.ticker)

        if locked_sim.last_processed_date:
            new_rows = df[df.index.date > locked_sim.last_processed_date]
        else:
            new_rows = df[df.index.date >= locked_sim.start_date]

        if new_rows.empty:
            # Release the row lock immediately instead of holding it until the
            # request teardown.
            db.session.commit()
            return locked_sim

        row_dates = [date_index.date() for date_index in new_rows.index]

        existing_points = (
            LiveSimulationEquity.query
            .filter(
                LiveSimulationEquity.simulation_id == locked_sim.id,
                LiveSimulationEquity.equity_date.in_(row_dates),
            )
            .all()
        )

        points_by_date = {
            point.equity_date: point
            for point in existing_points
        }

        existing_trades = (
            LiveSimulationTrade.query
            .filter(
                LiveSimulationTrade.simulation_id == locked_sim.id,
                LiveSimulationTrade.trade_date.in_(row_dates),
            )
            .order_by(
                LiveSimulationTrade.trade_date.asc(),
                LiveSimulationTrade.id.asc(),
            )
            .all()
        )

        trades_by_date = {}

        for trade in existing_trades:
            if trade.trade_date in trades_by_date:
                raise RuntimeError(
                    "Duplicate legacy trades exist for live simulation "
                    f"{locked_sim.id} on {trade.trade_date}. "
                    "Run the Step 10 migration preflight before deployment."
                )

            trades_by_date[trade.trade_date] = trade

        position_fraction = float(locked_sim.position_size_pct) / 100.0
        transaction_cost_rate = float(
            locked_sim.transaction_cost_rate
        )

        for date_index, row in new_rows.iterrows():
            equity_date = date_index.date()
            price = float(row["Close"])
            signal = int(row["epoch_signal"])

            if (
                not math.isfinite(price)
                or price <= 0
                or signal not in {-1, 0, 1}
            ):
                raise ValueError(
                    f"Invalid market row for {locked_sim.ticker} "
                    f"on {equity_date}."
                )

            existing_point = points_by_date.get(equity_date)

            if existing_point is not None:
                # A persisted daily point is the canonical post-day state. This
                # recovers safely from a legacy stale checkpoint without
                # executing the day's trade a second time.
                locked_sim.cash_balance = float(
                    existing_point.cash_balance
                )
                locked_sim.position_quantity = float(
                    existing_point.position_quantity
                )
                locked_sim.last_processed_date = equity_date
                validate_live_simulation_accounting_state(locked_sim)
                continue

            existing_trade = trades_by_date.get(equity_date)

            if existing_trade is not None:
                # A legacy trade without its equity point is recoverable because
                # each trade records the post-trade cash and position state.
                locked_sim.cash_balance = float(
                    existing_trade.cash_after
                )
                locked_sim.position_quantity = float(
                    existing_trade.position_after
                )
                validate_live_simulation_accounting_state(locked_sim)

            else:
                # --------------------
                # BUY
                # --------------------
                if signal == 1 and locked_sim.cash_balance > 0:
                    cash_allocation = (
                        locked_sim.cash_balance
                        * position_fraction
                    )

                    gross_buy_budget = (
                        cash_allocation
                        / (1 + transaction_cost_rate)
                    )
                    raw_quantity = gross_buy_budget / price

                    quantity = normalize_live_quantity_for_buy(
                        ticker=locked_sim.ticker,
                        raw_quantity=raw_quantity,
                        price=price,
                        cash_allocation=cash_allocation,
                        transaction_cost_rate=transaction_cost_rate,
                    )

                    if quantity > 0:
                        gross_amount = quantity * price
                        transaction_cost = (
                            gross_amount
                            * transaction_cost_rate
                        )
                        total_cash_used = (
                            gross_amount
                            + transaction_cost
                        )

                        if (
                            total_cash_used
                            <= locked_sim.cash_balance + 1e-9
                        ):
                            locked_sim.position_quantity += quantity
                            locked_sim.cash_balance -= total_cash_used

                            trade = LiveSimulationTrade(
                                simulation_id=locked_sim.id,
                                trade_date=equity_date,
                                ticker=locked_sim.ticker,
                                signal=1,
                                price=price,
                                quantity=quantity,
                                gross_amount=gross_amount,
                                transaction_cost=transaction_cost,
                                cash_after=locked_sim.cash_balance,
                                position_after=(
                                    locked_sim.position_quantity
                                ),
                            )
                            db.session.add(trade)
                            trades_by_date[equity_date] = trade

                # --------------------
                # SELL
                # --------------------
                elif (
                    signal == -1
                    and locked_sim.position_quantity > 0
                ):
                    raw_quantity = (
                        locked_sim.position_quantity
                        * position_fraction
                    )

                    quantity = normalize_live_quantity_for_sell(
                        ticker=locked_sim.ticker,
                        raw_quantity=raw_quantity,
                        current_position=(
                            locked_sim.position_quantity
                        ),
                    )

                    if quantity > 0:
                        gross_amount = quantity * price
                        transaction_cost = (
                            gross_amount
                            * transaction_cost_rate
                        )
                        net_cash_received = (
                            gross_amount
                            - transaction_cost
                        )

                        locked_sim.position_quantity -= quantity

                        if (
                            locked_sim.position_quantity
                            < 1e-12
                        ):
                            locked_sim.position_quantity = 0.0

                        locked_sim.cash_balance += net_cash_received

                        trade = LiveSimulationTrade(
                            simulation_id=locked_sim.id,
                            trade_date=equity_date,
                            ticker=locked_sim.ticker,
                            signal=-1,
                            price=price,
                            quantity=quantity,
                            gross_amount=gross_amount,
                            transaction_cost=transaction_cost,
                            cash_after=locked_sim.cash_balance,
                            position_after=(
                                locked_sim.position_quantity
                            ),
                        )
                        db.session.add(trade)
                        trades_by_date[equity_date] = trade

                validate_live_simulation_accounting_state(locked_sim)

            strategy_value = (
                locked_sim.cash_balance
                + (locked_sim.position_quantity * price)
            )
            benchmark_value = (
                locked_sim.benchmark_cash_balance
                + (locked_sim.benchmark_quantity * price)
            )

            if (
                not math.isfinite(strategy_value)
                or not math.isfinite(benchmark_value)
                or strategy_value < -1e-7
                or benchmark_value < -1e-7
            ):
                raise ValueError(
                    "Live simulation produced invalid equity values "
                    f"for {locked_sim.id} on {equity_date}."
                )

            point = LiveSimulationEquity(
                simulation_id=locked_sim.id,
                equity_date=equity_date,
                ticker=locked_sim.ticker,
                signal=signal,
                close_price=price,
                cash_balance=locked_sim.cash_balance,
                position_quantity=locked_sim.position_quantity,
                strategy_value=strategy_value,
                benchmark_value=benchmark_value,
            )
            db.session.add(point)
            points_by_date[equity_date] = point
            locked_sim.last_processed_date = equity_date

        # Flush first so uniqueness or database validation failures are caught
        # inside this helper and trigger a complete rollback.
        db.session.flush()
        db.session.commit()
        return locked_sim

    except IntegrityError:
        db.session.rollback()
        app.logger.exception(
            "Live simulation uniqueness conflict: simulation_id=%s",
            simulation_id,
        )
        raise
    except Exception:
        db.session.rollback()
        app.logger.exception(
            "Live simulation refresh rolled back: simulation_id=%s",
            simulation_id,
        )
        raise

LIVE_SIM_HORIZON_DAYS = {
    "1d": 1,
    "1w": 7,
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "since_start": None,
}

# Combined portfolio curves can contain many thousands of equity observations.
# Cache only completed user/status/fingerprint results. The fingerprint changes
# whenever an included simulation is updated, so current data never reuses an
# obsolete curve. A small bounded cache prevents unbounded process memory use.
LIVE_SIM_PORTFOLIO_CURVE_CACHE = {}
LIVE_SIM_PORTFOLIO_CURVE_CACHE_MAX = 48


def live_sim_portfolio_curve_cache_key(user_id, view, simulations):
    fingerprint = tuple(
        sorted(
            (
                int(sim.id),
                str(sim.status or ""),
                sim.updated_at.isoformat() if sim.updated_at else "",
                sim.last_processed_date.isoformat() if sim.last_processed_date else "",
                float(sim.initial_cash or 0.0),
            )
            for sim in simulations
        )
    )
    return (int(user_id), str(view or "open"), fingerprint)


def get_cached_live_sim_portfolio_curve(cache_key):
    cached = LIVE_SIM_PORTFOLIO_CURVE_CACHE.get(cache_key)
    if cached is None:
        return None
    return cached


def store_cached_live_sim_portfolio_curve(cache_key, portfolio):
    if len(LIVE_SIM_PORTFOLIO_CURVE_CACHE) >= LIVE_SIM_PORTFOLIO_CURVE_CACHE_MAX:
        # Dicts preserve insertion order on supported Python versions.
        oldest_key = next(iter(LIVE_SIM_PORTFOLIO_CURVE_CACHE), None)
        if oldest_key is not None:
            LIVE_SIM_PORTFOLIO_CURVE_CACHE.pop(oldest_key, None)
    LIVE_SIM_PORTFOLIO_CURVE_CACHE[cache_key] = portfolio


def safe_live_sim_return(current_value, base_value):
    try:
        current_value = float(current_value)
        base_value = float(base_value)

        if base_value <= 0:
            return None

        return (current_value / base_value) - 1

    except Exception:
        return None


def safe_live_sim_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def get_live_sim_horizon_base_point(points, latest_point, horizon_key):
    if not points or latest_point is None:
        return None

    days = LIVE_SIM_HORIZON_DAYS.get(horizon_key)

    if days is None:
        return None

    target_date = latest_point.equity_date - timedelta(days=days)

    older_points = [
        point for point in points
        if point.equity_date <= target_date
    ]

    if older_points:
        return older_points[-1]

    # If the simulation is younger than the selected horizon,
    # use the earliest available point.
    return points[0]


def build_live_sim_horizon_returns(sim, *, points=None):
    if points is None:
        points = (
            LiveSimulationEquity.query
            .filter_by(simulation_id=sim.id)
            .order_by(LiveSimulationEquity.equity_date.asc())
            .all()
        )

    if not points:
        return {}

    latest_point = points[-1]

    latest_strategy_value = safe_live_sim_float(latest_point.strategy_value)
    latest_benchmark_value = safe_live_sim_float(latest_point.benchmark_value)

    if latest_strategy_value is None or latest_benchmark_value is None:
        return {}

    output = {}

    for horizon_key in LIVE_SIM_HORIZON_DAYS.keys():

        if horizon_key == "since_start":
            base_strategy_value = safe_live_sim_float(sim.initial_cash)
            base_benchmark_value = safe_live_sim_float(sim.initial_cash)
            base_date = sim.start_date.isoformat() if sim.start_date else None
        else:
            base_point = get_live_sim_horizon_base_point(
                points,
                latest_point,
                horizon_key
            )

            if base_point is None:
                continue

            base_strategy_value = safe_live_sim_float(base_point.strategy_value)
            base_benchmark_value = safe_live_sim_float(base_point.benchmark_value)
            base_date = base_point.equity_date.isoformat() if base_point.equity_date else None

        if base_strategy_value is None or base_benchmark_value is None:
            continue

        strategy_return = safe_live_sim_return(
            latest_strategy_value,
            base_strategy_value
        )

        benchmark_return = safe_live_sim_return(
            latest_benchmark_value,
            base_benchmark_value
        )

        strategy_change = latest_strategy_value - base_strategy_value
        benchmark_change = latest_benchmark_value - base_benchmark_value

        return_spread = None
        if strategy_return is not None and benchmark_return is not None:
            return_spread = strategy_return - benchmark_return

        value_difference = strategy_change - benchmark_change

        output[horizon_key] = {
            "base_date": base_date,
            "latest_date": latest_point.equity_date.isoformat() if latest_point.equity_date else None,

            "strategy_return": strategy_return,
            "benchmark_return": benchmark_return,
            "return_spread": return_spread,

            "strategy_change": strategy_change,
            "benchmark_change": benchmark_change,
            "value_difference": value_difference,

            "strategy_value": latest_strategy_value,
            "benchmark_value": latest_benchmark_value,
            "base_strategy_value": base_strategy_value,
            "base_benchmark_value": base_benchmark_value,
        }

    return output

def can_use_live_simulation_ticker(user, ticker):
    ticker = normalize_ticker(ticker)

    if ticker not in ALL_SUPPORTED_TICKERS:
        return False

    # Admin-only assets are blocked for non-admin users,
    # including subscribed/Pro users.
    if not can_user_see_ticker(user, ticker):
        return False

    if is_paid_user(user):
        return True

    return ticker in TOP_FREE_TICKERS

def require_live_simulation_ticker_access(ticker):
    ticker = normalize_ticker(ticker)

    support_error = require_supported_ticker_or_400(ticker)
    if support_error:
        return support_error

    if can_use_live_simulation_ticker(current_user, ticker):
        return None

    return jsonify({
        "error": (
            "Live simulations for this asset are available on NeuralTrend Pro. "
            "Free accounts can create live simulations only for BTC-USD, ETH-USD, SOL-USD, and XRP-USD."
        ),
        "upgrade_required": True,
        "allowed_tickers": sorted(TOP_FREE_TICKERS)
    }), 403

def visible_live_simulation_query():
    query = LiveSimulation.query.filter_by(user_id=current_user.id)

    # Non-admin users should never see admin-only simulations,
    # even if they are Pro.
    if not is_admin_user(current_user):
        query = query.filter(
            ~LiveSimulation.ticker.in_(list(ADMIN_ONLY_TICKERS))
        )

    # Free users can only see Free live-simulation tickers.
    if not is_paid_user(current_user):
        query = query.filter(
            LiveSimulation.ticker.in_(list(TOP_FREE_TICKERS))
        )

    return query

def get_latest_csv_date_for_ticker(ticker):
    try:
        df = load_epoch_csv_for_ticker(ticker)
        return df.index.max().date()
    except Exception:
        app.logger.exception(
            "Could not read latest CSV date ticker=%s",
            normalize_ticker(ticker),
        )
        return None

# --------------------
# Routes
# --------------------

@app.route("/healthz")
@limiter.exempt
def healthz():
    """Lightweight readiness check for Render and deployment smoke tests.

    The response intentionally exposes only component status, never database
    credentials, filesystem paths, Stripe configuration, or user information.
    """
    checks = {
        "database": "ok",
        "forward_record_storage": "ok",
    }
    healthy = True

    try:
        db.session.execute(text("SELECT 1")).scalar()
    except Exception:
        db.session.rollback()
        app.logger.exception("Health check database probe failed")
        checks["database"] = "error"
        healthy = False

    try:
        storage_ready = (
            os.path.isdir(FORWARD_RECORD_STORAGE_DIR)
            and os.access(FORWARD_RECORD_STORAGE_DIR, os.R_OK | os.W_OK)
        )
    except OSError:
        storage_ready = False

    if not storage_ready:
        checks["forward_record_storage"] = "error"
        healthy = False

    response = jsonify({
        "status": "ok" if healthy else "degraded",
        "checks": checks,
    })
    response.status_code = 200 if healthy else 503
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.route("/resources")
def resources():
    return render_template("resources.html")


@app.route("/ai-crypto-trading-signals")
def ai_crypto_trading_signals():
    return render_template("ai_crypto_trading_signals.html")

@app.route("/crypto-paper-trading-simulator")
def crypto_paper_trading_simulator():
    return render_template("crypto_paper_trading_simulator.html")

@app.route("/buy-and-hold-vs-ai-strategy")
def buy_and_hold_vs_ai_strategy():
    return render_template("buy_and_hold_vs_ai_strategy.html")

@app.route("/signup", methods=["POST"])
@limiter.limit("3 per minute")
def signup():    
    data = get_json_object()

    if data is None:
        return jsonify({"error": "A JSON object is required."}), 400

    email = normalize_email(data.get("email"))
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    email_error = validate_email_address(email)
    if email_error:
        return jsonify({"error": email_error}), 400

    password_error = validate_password(password)
    if password_error:
        return jsonify({"error": password_error}), 400

    existing_user = User.query.filter_by(email=email).first()

    if existing_user:
        if not existing_user.is_verified:
            send_verification_email(email)

            return jsonify({
                "message": "This account already exists but is not verified. We sent a new verification email. Please check your inbox and spam folder."
            })

        return jsonify({"error": "User already exists. Please log in."}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

    new_user = User(
        email=email,
        password_hash=hashed_password,
        is_verified=False
    )
    
    db.session.add(new_user)
    db.session.commit()
    
    send_verification_email(email)
    
    return jsonify({
        "message": "Account created. Please check your email to verify."
    })

@app.route("/resend-verification", methods=["POST"])
@limiter.limit("3 per minute")
def resend_verification():
    data = get_json_object() or {}

    email = normalize_email(data.get("email"))

    if not email:
        return jsonify({
            "error": "Email is required."
        }), 400

    if validate_email_address(email):
        return jsonify({
            "message": "If this email belongs to an unverified NeuralTrend account, a new verification email has been sent. Please check your inbox and spam folder."
        })

    user = User.query.filter_by(email=email).first()

    # Avoid exposing whether an account exists.
    if user and not user.is_verified:
        send_verification_email(email)

    return jsonify({
        "message": "If this email belongs to an unverified NeuralTrend account, a new verification email has been sent. Please check your inbox and spam folder."
    })

@app.route("/verify/<token>")
def verify_email(token):
    email = confirm_verification_token(token)

    if not email:
        return "Verification link expired or invalid."

    user = User.query.filter_by(email=email).first()

    if not user:
        return "User not found."

    user.is_verified = True
    db.session.commit()

    return "Email verified successfully! You can now log in."

@app.route("/login", methods=["POST"])
@limiter.limit("5 per minute")
def login():
    data = get_json_object() or {}

    email = normalize_email(data.get("email"))
    password = data.get("password")

    if not email or not isinstance(password, str) or not password:
        return jsonify({"error": "Email and password required"}), 400

    if validate_email_address(email):
        return jsonify({"error": "Invalid email or password"}), 401

    # Bound work before invoking bcrypt. Existing NeuralTrend passwords were
    # created with bcrypt, whose input must not exceed 72 encoded bytes.
    if len(password.encode("utf-8")) > PASSWORD_MAX_UTF8_BYTES:
        return jsonify({"error": "Invalid email or password"}), 401

    user = User.query.filter_by(email=email).first()

    # 🔒 Prevent user enumeration
    if not user:
        return jsonify({"error": "Invalid email or password"}), 401

    # 🔒 Check lockout
    if user.locked_until and user.locked_until > datetime.utcnow():
        return jsonify({
            "error": "Account locked. Try again later."
        }), 403

    # 🔒 Check password
    if not bcrypt.check_password_hash(user.password_hash, password):
        user.failed_attempts += 1

        if user.failed_attempts >= 5:
            user.locked_until = datetime.utcnow() + timedelta(minutes=15)
            user.failed_attempts = 0

        db.session.commit()

        return jsonify({"error": "Invalid email or password"}), 401

    # ✅ Successful password check → reset counters
    user.failed_attempts = 0
    user.locked_until = None
    
    # 🔒 Require verification
    if not user.is_verified:
        db.session.commit()

        return jsonify({
            "error": "Please verify your email first. Check your inbox and spam folder, or resend the verification email.",
            "verification_required": True,
            "email": user.email
        }), 403
    
    # ✅ User is verified and will actually log in
    user.last_login = datetime.utcnow()
    
    db.session.commit()
    
    login_user(user)
    session["auth_version"] = int(getattr(user, "auth_version", 1) or 1)

    return jsonify({
        "message": "Logged in successfully",
        "user_id": user.id,
        "email": user.email
    })

@app.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    session.pop("auth_version", None)
    clear_password_reset_session()
    return jsonify({"message": "Logged out"})


@app.route("/change-password", methods=["POST"])
@login_required
@limiter.limit("5 per 15 minutes")
def change_password():
    """Change the authenticated user's password after re-authentication.

    The current browser remains authenticated by adopting the incremented
    auth_version. Every other existing session is revoked on its next request.
    """
    data = get_json_object()
    if data is None:
        return jsonify({"error": "A JSON object is required."}), 400

    current_password = data.get("current_password")
    new_password = data.get("new_password")
    confirm_password = data.get("confirm_password")

    if not isinstance(current_password, str) or not current_password:
        return jsonify({"error": "Current password is required."}), 400

    # Bound bcrypt work and prevent silent truncation behavior.
    if len(current_password.encode("utf-8")) > PASSWORD_MAX_UTF8_BYTES:
        return jsonify({"error": "Current password is incorrect."}), 400

    if not bcrypt.check_password_hash(current_user.password_hash, current_password):
        app.logger.warning(
            "Authenticated password-change reauthentication failed for user_id=%s",
            current_user.id,
        )
        return jsonify({"error": "Current password is incorrect."}), 400

    password_error = validate_password(new_password)
    if password_error:
        return jsonify({"error": password_error}), 400

    if new_password != confirm_password:
        return jsonify({"error": "The two new password entries do not match."}), 400

    if bcrypt.check_password_hash(current_user.password_hash, new_password):
        return jsonify({
            "error": "Choose a password different from your current password."
        }), 400

    user_id = current_user.id
    user_email = current_user.email

    try:
        user = (
            User.query
            .filter_by(id=user_id)
            .with_for_update()
            .first()
        )

        if not user:
            db.session.rollback()
            return jsonify({"error": "Account not found."}), 404

        # Re-check against the locked row in case another password change
        # completed between the first verification and the row lock.
        if not bcrypt.check_password_hash(user.password_hash, current_password):
            db.session.rollback()
            return jsonify({
                "error": "Your password changed in another session. Please log in again."
            }), 409

        user.password_hash = bcrypt.generate_password_hash(
            new_password
        ).decode("utf-8")
        user.password_changed_at = datetime.utcnow()
        user.password_reset_token_hash = None
        user.password_reset_requested_at = None
        user.auth_version = int(user.auth_version or 1) + 1
        user.failed_attempts = 0
        user.locked_until = None
        db.session.commit()

        # Keep this browser logged in while revoking every older session.
        session["auth_version"] = int(user.auth_version)
        clear_password_reset_session()

    except Exception:
        db.session.rollback()
        app.logger.exception(
            "Authenticated password change failed for user_id=%s",
            user_id,
        )
        return jsonify({
            "error": "Could not change your password. Please try again."
        }), 500

    send_password_changed_email(user_email)
    app.logger.info("Password changed from account menu for user_id=%s", user_id)

    return jsonify({
        "message": (
            "Password changed successfully. Other signed-in sessions have "
            "been logged out."
        )
    })


@app.route("/me")
def me():
    if current_user.is_authenticated:
        payload = {
            "email": current_user.email,
            "subscription_type": current_user.subscription_type,
            "subscription_status": current_user.subscription_status,
            "is_paid": is_paid_user(current_user),
            "is_admin": is_admin_user(current_user),
            "live_simulation_limit": get_live_simulation_limit_for_user(current_user),
            "watchlist_limit": get_watchlist_limit_for_user(current_user),
            "signal_email_alerts_available": is_paid_user(current_user),
        }
    else:
        payload = {
            "email": None,
            "subscription_type": "anonymous",
            "subscription_status": "none",
            "is_paid": False,
            "live_simulation_limit": 0,
            "watchlist_limit": 0,
            "signal_email_alerts_available": False,
        }

    response = jsonify(payload)
    response.headers["Cache-Control"] = "private, no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Vary"] = "Cookie"
    return response



# --------------------
# Admin operational monitoring
# --------------------

def _utc_iso(value):
    if value is None:
        return None
    if getattr(value, "tzinfo", None) is None:
        return value.isoformat() + "Z"
    return value.isoformat()


def _forward_storage_inventory():
    file_count = 0
    total_bytes = 0
    for root, dirs, files in os.walk(FORWARD_RECORD_STORAGE_DIR):
        dirs[:] = [
            name for name in dirs
            if not os.path.islink(os.path.join(root, name))
        ]
        for name in files:
            path = os.path.join(root, name)
            try:
                if os.path.islink(path) or not os.path.isfile(path):
                    continue
                file_count += 1
                total_bytes += os.path.getsize(path)
            except OSError:
                continue
    return file_count, total_bytes


def build_operational_status():
    """Build a read-only, admin-safe production status summary.

    The result contains counts and component states only. It intentionally
    excludes user email addresses, database URLs, filesystem paths, Stripe
    identifiers, request bodies, tokens, and other secrets.
    """
    now = datetime.utcnow()
    warnings = []
    errors = []

    result = {
        "generated_at": _utc_iso(now),
        "status": "ok",
        "release": {
            "service": os.environ.get("RENDER_SERVICE_NAME") or "unknown",
            "commit": (os.environ.get("RENDER_GIT_COMMIT") or "unknown")[:12],
            "environment": os.environ.get("RENDER") and "render" or "local",
        },
        "configuration": {
            "database_configured": bool(os.environ.get("DATABASE_URL", "").strip()),
            "redis_configured": bool(os.environ.get("REDIS_URL", "").strip()),
            "email_configured": bool(
                os.environ.get("EMAIL_USER", "").strip()
                and os.environ.get("EMAIL_PASS", "").strip()
            ),
            "stripe_monthly_configured": bool(STRIPE_PRO_MONTHLY_PRICE_ID),
            "stripe_annual_configured": bool(STRIPE_PRO_ANNUAL_PRICE_ID),
            "stripe_webhook_configured": bool(STRIPE_WEBHOOK_SECRET),
            "stripe_portal_configured": bool(
                STRIPE_BILLING_PORTAL_CONFIGURATION_ID
            ),
            "forward_record_storage_explicit": FORWARD_RECORD_STORAGE_EXPLICIT,
        },
        "database": {"status": "ok", "counts": {}},
        "processing": {
            "failed_webhooks_24h": None,
            "stale_webhooks": None,
            "latest_webhook_failure_at": None,
            "latest_webhook_failure_type": None,
            "failed_alerts_24h": None,
            "stale_alerts": None,
            "latest_alert_failure_at": None,
        },
        "forward_record": {
            "assets": {
                "sandbox": {"active": 0, "retired": 0, "removed": 0},
                "public": {"active": 0, "retired": 0, "removed": 0},
            },
            "latest_publication_at": None,
            "latest_publication_mode": None,
            "latest_publication_count": 0,
        },
        "storage": {"status": "ok"},
        "market_data": {"status": "ok"},
        "warnings": warnings,
        "errors": errors,
    }

    if not result["configuration"]["redis_configured"]:
        warnings.append(
            "REDIS_URL is not configured; rate limits may not be shared "
            "across multiple web workers."
        )
    if not FORWARD_RECORD_STORAGE_EXPLICIT:
        warnings.append(
            "Forward Record storage is using the application directory "
            "instead of an explicitly mounted persistent path."
        )

    try:
        db.session.execute(text("SELECT 1")).scalar()

        result["database"]["counts"] = {
            "users": User.query.count(),
            "active_pro_users": User.query.filter(
                User.subscription_type == "pro",
                User.subscription_status.in_(tuple(PAID_SUBSCRIPTION_STATUSES)),
            ).count(),
            "active_live_simulations": LiveSimulation.query.filter_by(
                status="active"
            ).count(),
            "watchlist_items": WatchlistItem.query.count(),
            "enabled_email_alerts": WatchlistItem.query.filter_by(
                email_alert_enabled=True
            ).count(),
        }

        webhook_cutoff = now - timedelta(hours=24)
        processing_cutoff = now - timedelta(minutes=15)
        failed_webhooks = StripeWebhookEvent.query.filter(
            StripeWebhookEvent.processing_status == "failed",
            StripeWebhookEvent.received_at >= webhook_cutoff,
        ).count()
        stale_webhooks = StripeWebhookEvent.query.filter(
            StripeWebhookEvent.processing_status == "processing",
            StripeWebhookEvent.processing_started_at < processing_cutoff,
        ).count()
        latest_webhook_failure = StripeWebhookEvent.query.filter_by(
            processing_status="failed"
        ).order_by(StripeWebhookEvent.received_at.desc()).first()

        failed_alerts = SignalAlertDelivery.query.filter(
            SignalAlertDelivery.processing_status == "failed",
            SignalAlertDelivery.created_at >= webhook_cutoff,
        ).count()
        stale_alerts = SignalAlertDelivery.query.filter(
            SignalAlertDelivery.processing_status == "processing",
            SignalAlertDelivery.processing_started_at < processing_cutoff,
        ).count()
        latest_alert_failure = SignalAlertDelivery.query.filter_by(
            processing_status="failed"
        ).order_by(SignalAlertDelivery.created_at.desc()).first()

        result["processing"] = {
            "failed_webhooks_24h": failed_webhooks,
            "stale_webhooks": stale_webhooks,
            "latest_webhook_failure_at": _utc_iso(
                latest_webhook_failure.received_at
                if latest_webhook_failure else None
            ),
            "latest_webhook_failure_type": (
                latest_webhook_failure.event_type
                if latest_webhook_failure else None
            ),
            "failed_alerts_24h": failed_alerts,
            "stale_alerts": stale_alerts,
            "latest_alert_failure_at": _utc_iso(
                latest_alert_failure.created_at
                if latest_alert_failure else None
            ),
        }

        forward_counts = {}
        for mode in ("sandbox", "public"):
            forward_counts[mode] = {
                status: ForwardRecordAsset.query.filter_by(
                    record_mode=mode,
                    status=status,
                ).count()
                for status in ("active", "retired", "removed")
            }

        latest_batch = ForwardPublicationBatch.query.order_by(
            ForwardPublicationBatch.published_at.desc()
        ).first()
        result["forward_record"] = {
            "assets": forward_counts,
            "latest_publication_at": _utc_iso(
                latest_batch.published_at if latest_batch else None
            ),
            "latest_publication_mode": (
                latest_batch.record_mode if latest_batch else None
            ),
            "latest_publication_count": (
                latest_batch.publication_count if latest_batch else 0
            ),
        }

        if failed_webhooks:
            warnings.append(
                f"{failed_webhooks} Stripe webhook event(s) failed in the last 24 hours."
            )
        if stale_webhooks:
            warnings.append(
                f"{stale_webhooks} Stripe webhook event(s) have remained in processing for more than 15 minutes."
            )
        if failed_alerts:
            warnings.append(
                f"{failed_alerts} signal-alert delivery attempt(s) failed in the last 24 hours."
            )
        if stale_alerts:
            warnings.append(
                f"{stale_alerts} signal-alert delivery attempt(s) have remained in processing for more than 15 minutes."
            )
    except Exception:
        db.session.rollback()
        app.logger.exception("Operational status database checks failed")
        result["database"]["status"] = "error"
        errors.append("Database operational checks failed.")

    try:
        storage_ready = (
            os.path.isdir(FORWARD_RECORD_STORAGE_DIR)
            and os.access(FORWARD_RECORD_STORAGE_DIR, os.R_OK | os.W_OK)
        )
        if not storage_ready:
            raise OSError("storage directory is not readable and writable")

        usage = shutil.disk_usage(FORWARD_RECORD_STORAGE_DIR)
        file_count, total_bytes = _forward_storage_inventory()
        free_percent = (usage.free / usage.total * 100.0) if usage.total else 0.0
        result["storage"].update({
            "file_count": file_count,
            "used_by_forward_record_bytes": total_bytes,
            "disk_free_bytes": usage.free,
            "disk_total_bytes": usage.total,
            "disk_free_percent": round(free_percent, 1),
        })
        if usage.free < 512 * 1024 * 1024 or free_percent < 10.0:
            warnings.append("Persistent storage is running low on free space.")
    except Exception:
        app.logger.exception("Operational status storage checks failed")
        result["storage"]["status"] = "error"
        errors.append("Forward Record storage checks failed.")

    try:
        latest_date = get_latest_csv_date_for_ticker("BTC-USD")
        if latest_date is None:
            raise ValueError("BTC-USD latest date is unavailable")
        age_days = max(0, (now.date() - latest_date).days)
        result["market_data"].update({
            "btc_latest_date": latest_date.isoformat(),
            "age_days": age_days,
        })
        if age_days > 3:
            warnings.append(
                f"BTC-USD working market data is {age_days} days old."
            )
    except Exception:
        app.logger.exception("Operational status market-data check failed")
        result["market_data"]["status"] = "error"
        errors.append("BTC-USD market-data check failed.")

    if errors:
        result["status"] = "error"
    elif warnings:
        result["status"] = "warning"

    return result


@app.route("/admin/operations")
@login_required
def admin_operations():
    if not is_admin_user(current_user):
        abort(404)
    return render_template(
        "admin_operations.html",
        operations=build_operational_status(),
        backup_recovery=_backup_recovery_summary(),
        format_bytes=_format_backup_bytes,
        active_admin_page="operations",
    )


@app.route("/admin/operations.json")
@login_required
def admin_operations_json():
    if not is_admin_user(current_user):
        abort(404)
    response = jsonify(build_operational_status())
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


def _format_backup_bytes(value):
    value = float(value or 0)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024


def _backup_checksum_status(filename):
    try:
        backup = resolve_managed_file(Path(BACKUP_STORAGE_DIR), filename, allow_checksum=False)
        sidecar = Path(str(backup) + ".sha256")
        if not backup.is_file() or not sidecar.is_file():
            return "missing"
        expected = sidecar.read_text(encoding="utf-8").strip().split()[0].lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            return "invalid"
        return "verified" if hmac.compare_digest(expected, sha256_file(backup)) else "mismatch"
    except Exception:
        app.logger.exception("backup_checksum_status_failed filename=%s", filename)
        return "error"


def _backup_recovery_summary():
    backups = list_backups(Path(BACKUP_STORAGE_DIR))
    now = datetime.utcnow().replace(tzinfo=None)
    summary = {}
    for kind in ("database", "forward_record"):
        item = next((entry for entry in backups if entry.kind == kind), None)
        if item is None:
            summary[kind] = {"available": False, "status": "missing"}
            continue
        modified = item.modified_at.replace(tzinfo=None)
        age_hours = max(0, int((now - modified).total_seconds() // 3600))
        summary[kind] = {
            "available": True,
            "item": item,
            "age_hours": age_hours,
            "checksum_status": _backup_checksum_status(item.name),
            "status": "ready" if item.checksum_available and _backup_checksum_status(item.name) == "verified" else "attention",
        }
    summary["ready"] = all(summary[k].get("status") == "ready" for k in ("database", "forward_record"))
    return summary


@app.route("/admin/backups")
@login_required
def admin_backups():
    if not is_admin_user(current_user):
        abort(404)
    response = app.make_response(render_template(
        "admin_backups.html",
        backups=list_backups(Path(BACKUP_STORAGE_DIR)),
        persistent=BACKUP_STORAGE_EXPLICIT,
        retention=BACKUP_RETENTION_COUNT,
        format_bytes=_format_backup_bytes,
        active_admin_page="backups",
    ))
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response




@app.route("/admin/recovery")
@login_required
def admin_recovery():
    if not is_admin_user(current_user):
        abort(404)
    response = app.make_response(render_template(
        "admin_recovery.html",
        recovery=_backup_recovery_summary(),
        persistent=BACKUP_STORAGE_EXPLICIT,
        retention=BACKUP_RETENTION_COUNT,
        format_bytes=_format_backup_bytes,
        active_admin_page="recovery",
    ))
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.route("/admin/backups/create", methods=["POST"])
@login_required
@limiter.limit("4 per hour")
def admin_backup_create():
    if not is_admin_user(current_user):
        abort(404)
    kind = request.form.get("kind", "").strip()
    try:
        if kind == "database":
            database_url = os.environ.get("DATABASE_URL", "").strip()
            if not database_url:
                raise RuntimeError("DATABASE_URL is not configured.")
            created = create_database_backup(Path(BACKUP_STORAGE_DIR), database_url)
        elif kind == "forward_record":
            created = create_forward_backup(
                Path(BACKUP_STORAGE_DIR),
                Path(FORWARD_RECORD_STORAGE_DIR),
                os.environ.get("RENDER_GIT_COMMIT", "unknown"),
            )
        else:
            abort(400)
        removed = enforce_retention(Path(BACKUP_STORAGE_DIR), BACKUP_RETENTION_COUNT)
        app.logger.warning(
            "admin_backup_created kind=%s filename=%s retention_removed=%s user_id=%s",
            kind, created.name, len(removed), current_user.id,
        )
        flash(f"Created and verified {created.name}.", "success")
    except Exception as exc:
        app.logger.exception("admin_backup_creation_failed kind=%s user_id=%s", kind, current_user.id)
        flash(f"Backup creation failed: {exc}", "error")
    return redirect(url_for("admin_backups"))


@app.route("/admin/backups/download/<path:filename>")
@login_required
@limiter.limit("30 per hour")
def admin_backup_download(filename):
    if not is_admin_user(current_user):
        abort(404)
    try:
        path = resolve_managed_file(Path(BACKUP_STORAGE_DIR), filename)
    except ValueError:
        abort(404)
    if not path.is_file():
        abort(404)
    app.logger.warning("admin_backup_downloaded filename=%s user_id=%s", path.name, current_user.id)
    response = send_file(path, as_attachment=True, download_name=path.name, conditional=True)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.route("/admin/backups/delete/<path:filename>", methods=["POST"])
@login_required
@limiter.limit("20 per hour")
def admin_backup_delete(filename):
    if not is_admin_user(current_user):
        abort(404)
    try:
        delete_backup(Path(BACKUP_STORAGE_DIR), filename)
    except (ValueError, FileNotFoundError):
        abort(404)
    app.logger.warning("admin_backup_deleted filename=%s user_id=%s", filename, current_user.id)
    flash(f"Deleted {filename} and its checksum.", "success")
    return redirect(url_for("admin_backups"))


# --------------------
# Manual admin signal-alert dispatch
# --------------------

@app.route("/admin/signal-alerts", methods=["GET", "POST"])
@login_required
@limiter.limit("12 per minute")
def admin_signal_alerts():
    """Admin-only manual controls for alerts and Forward Record modes.

    Alert dispatch, private sandbox publication, and public publication are
    intentionally independent actions. Public assets are enrolled prospectively
    one by one, and nothing runs automatically.
    """
    if not is_admin_user(current_user):
        abort(404)

    dispatch_summary = None
    publication_summary = None
    retirement_summary = None
    removal_summary = None
    reset_summary = None
    dispatch_output = []
    page_error = None
    selected_ticker = ""
    selected_action = None
    public_enabled = public_forward_record_enabled()
    public_record_summary = build_forward_record_summary("public")

    def approval_is_valid(key, *, ticker, signature_key="approval_signature"):
        approval = session.get(key)
        preview_age = None
        try:
            preview_age = int(time.time()) - int(approval.get("created_at"))
        except (AttributeError, TypeError, ValueError):
            preview_age = None
        return (
            isinstance(approval, dict)
            and preview_age is not None
            and 0 <= preview_age <= 15 * 60
            and approval.get("ticker", "") == ticker
            and bool(approval.get(signature_key))
        )

    if request.method == "POST":
        selected_action = (request.form.get("action") or "").strip().lower()
        selected_ticker = normalize_ticker(request.form.get("ticker"))
        allowed_actions = {
            "alert_preview",
            "alert_send",
            "sandbox_preview",
            "sandbox_publish",
            "sandbox_reset",
            "public_preview",
            "public_publish",
            "public_retire_preview",
            "public_retire",
            "public_remove_preview",
            "public_remove",
        }

        if selected_action not in allowed_actions:
            page_error = "Choose one of the available admin actions."

        if not page_error and selected_ticker:
            if selected_action in {"public_retire_preview", "public_retire"}:
                tracked_asset = get_forward_record_asset(selected_ticker, "public")
                if tracked_asset is None:
                    page_error = "That asset has not entered the public Forward Record."
                elif tracked_asset.status != "active":
                    page_error = f"That asset is already {tracked_asset.status}."
            elif selected_action in {"public_remove_preview", "public_remove"}:
                tracked_asset = get_forward_record_asset(selected_ticker, "public")
                if tracked_asset is None:
                    page_error = "That asset has not entered the public Forward Record."
                elif tracked_asset.status != "retired":
                    page_error = "Only retired assets can have their public CSV removed."
            else:
                try:
                    get_epoch_csv_path(selected_ticker)
                    if selected_ticker in ADMIN_ONLY_TICKERS:
                        raise ValueError("Admin-only assets are not included.")
                except (ValueError, FileNotFoundError):
                    page_error = "That ticker does not have a supported public signal file."

        if not page_error and selected_action.startswith("public_") and not public_enabled:
            page_error = (
                "Public Forward Record publication is locked. Keep it locked during "
                "pre-launch testing; set FORWARD_RECORD_PUBLIC_ENABLED=true only when "
                "you are ready to start the customer-facing record."
            )

        if (
            not page_error
            and selected_action in {"public_preview", "public_publish"}
            and not public_record_summary.get("started")
            and not selected_ticker
        ):
            page_error = (
                "Choose one supported ticker for the first public Forward Record "
                "publication. Assets enter the public record individually."
            )

        if (
            not page_error
            and selected_action in {
                "public_retire_preview",
                "public_retire",
                "public_remove_preview",
                "public_remove",
            }
            and not selected_ticker
        ):
            page_error = "Choose one public asset for this lifecycle action."

        confirmation = (request.form.get("confirmation") or "").strip()
        required_confirmation = None
        if selected_action == "alert_send":
            required_confirmation = "SEND ALERTS"
        elif selected_action == "sandbox_publish":
            required_confirmation = "PUBLISH SANDBOX"
        elif selected_action == "sandbox_reset":
            required_confirmation = "RESET SANDBOX"
        elif selected_action == "public_publish":
            required_confirmation = (
                "START PUBLIC RECORD"
                if not public_record_summary.get("started")
                else "PUBLISH PUBLIC"
            )
        elif selected_action == "public_retire":
            required_confirmation = f"RETIRE {selected_ticker}"
        elif selected_action == "public_remove":
            required_confirmation = f"REMOVE {selected_ticker}"

        if (
            not page_error
            and required_confirmation
            and confirmation != required_confirmation
        ):
            page_error = f'Type "{required_confirmation}" exactly before continuing.'

        if not page_error and selected_action == "alert_send":
            if not approval_is_valid(
                "signal_alert_dispatch_preview",
                ticker=selected_ticker,
            ):
                page_error = (
                    "Preview this exact alert selection first, then send it within "
                    "15 minutes."
                )

        if not page_error and selected_action in {"sandbox_publish", "public_publish"}:
            mode = "sandbox" if selected_action.startswith("sandbox") else "public"
            if not approval_is_valid(
                f"forward_{mode}_preview",
                ticker=selected_ticker,
            ):
                page_error = (
                    f"Preview this exact {mode} publication selection first, then "
                    "approve it within 15 minutes."
                )

        if not page_error and selected_action == "public_retire":
            if not approval_is_valid(
                "forward_public_retirement_preview",
                ticker=selected_ticker,
            ):
                page_error = (
                    "Preview this exact retirement first, then approve it within "
                    "15 minutes."
                )

        if not page_error and selected_action == "public_remove":
            if not approval_is_valid(
                "forward_public_removal_preview",
                ticker=selected_ticker,
            ):
                page_error = (
                    "Preview this exact data removal first, then approve it within "
                    "15 minutes."
                )

        if not page_error:
            import io
            from contextlib import redirect_stderr, redirect_stdout
            from tools.publish_forward_record import (
                build_publication_preview,
                build_removal_preview,
                build_retirement_preview,
                print_preview,
                print_removal_preview,
                print_retirement_preview,
                publish_forward_batch,
                remove_retired_public_data,
                reset_sandbox_record,
                retire_public_asset,
            )
            from tools.send_signal_change_alerts import run_dispatch

            output_buffer = io.StringIO()
            try:
                with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                    if selected_action == "alert_preview":
                        dispatch_summary = run_dispatch(
                            dry_run=True,
                            limit=1000,
                            stability_minutes=0,
                            ticker=(selected_ticker or None),
                        )
                    elif selected_action == "alert_send":
                        approval = session.get("signal_alert_dispatch_preview", {})
                        dispatch_summary = run_dispatch(
                            dry_run=False,
                            limit=25,
                            stability_minutes=0,
                            ticker=(selected_ticker or None),
                            expected_approval_signature=approval.get(
                                "approval_signature"
                            ),
                        )
                    elif selected_action in {"sandbox_preview", "public_preview"}:
                        mode = (
                            "sandbox"
                            if selected_action == "sandbox_preview"
                            else "public"
                        )
                        publication_summary = build_publication_preview(
                            ticker=(selected_ticker or None),
                            record_mode=mode,
                        )
                        print_preview(publication_summary)
                    elif selected_action in {"sandbox_publish", "public_publish"}:
                        mode = (
                            "sandbox"
                            if selected_action == "sandbox_publish"
                            else "public"
                        )
                        approval = session.get(f"forward_{mode}_preview", {})
                        publication_summary = publish_forward_batch(
                            admin_user_id=current_user.id,
                            expected_approval_signature=approval.get(
                                "approval_signature"
                            ),
                            ticker=(selected_ticker or None),
                            record_mode=mode,
                        )
                    elif selected_action == "public_retire_preview":
                        retirement_summary = build_retirement_preview(
                            selected_ticker,
                            request.form.get("retirement_reason"),
                            request.form.get("removal_after_date"),
                        )
                        print_retirement_preview(retirement_summary)
                    elif selected_action == "public_retire":
                        approval = session.get(
                            "forward_public_retirement_preview",
                            {},
                        )
                        retirement_summary = retire_public_asset(
                            admin_user_id=current_user.id,
                            ticker=selected_ticker,
                            reason=request.form.get("retirement_reason"),
                            removal_after_date=request.form.get("removal_after_date"),
                            expected_approval_signature=approval.get(
                                "approval_signature"
                            ),
                        )
                    elif selected_action == "public_remove_preview":
                        removal_summary = build_removal_preview(selected_ticker)
                        print_removal_preview(removal_summary)
                    elif selected_action == "public_remove":
                        approval = session.get(
                            "forward_public_removal_preview",
                            {},
                        )
                        removal_summary = remove_retired_public_data(
                            admin_user_id=current_user.id,
                            ticker=selected_ticker,
                            expected_approval_signature=approval.get(
                                "approval_signature"
                            ),
                        )
                    elif selected_action == "sandbox_reset":
                        reset_summary = reset_sandbox_record(
                            admin_user_id=current_user.id
                        )
            except Exception as error:
                db.session.rollback()
                app.logger.exception(
                    "Manual admin action failed admin_user_id=%s action=%s ticker=%s",
                    current_user.id,
                    selected_action,
                    selected_ticker or "ALL",
                )
                page_error = f"Admin action failed: {error}"

            dispatch_output = [
                line
                for line in output_buffer.getvalue().splitlines()
                if line.strip()
            ][-500:]
            db.session.expire_all()

            if dispatch_summary and selected_action == "alert_preview":
                if (
                    dispatch_summary.get("failures", 0) == 0
                    and dispatch_summary.get("waiting", 0) == 0
                ):
                    session["signal_alert_dispatch_preview"] = {
                        "approval_signature": dispatch_summary.get(
                            "approval_signature"
                        ),
                        "ticker": selected_ticker,
                        "created_at": int(time.time()),
                    }
                else:
                    session.pop("signal_alert_dispatch_preview", None)
                    page_error = (
                        "Alert preview found a source/read failure. Resolve it and "
                        "preview again."
                    )
            elif dispatch_summary and selected_action == "alert_send":
                session.pop("signal_alert_dispatch_preview", None)
                if dispatch_summary.get("approval_mismatch"):
                    page_error = (
                        "Eligible recipients, preferences, checkpoints, or source data "
                        "changed after preview. No email was sent; preview again."
                    )

            if publication_summary and selected_action in {
                "sandbox_preview",
                "public_preview",
            }:
                mode = publication_summary.get("record_mode", "sandbox")
                if publication_summary.get("error_count", 0) == 0:
                    session[f"forward_{mode}_preview"] = {
                        "approval_signature": publication_summary.get(
                            "approval_signature"
                        ),
                        "ticker": selected_ticker,
                        "created_at": int(time.time()),
                    }
                else:
                    session.pop(f"forward_{mode}_preview", None)
                    page_error = (
                        f"{mode.title()} preview found a source-data error. "
                        "Resolve it and preview again."
                    )
            elif publication_summary and selected_action in {
                "sandbox_publish",
                "public_publish",
            }:
                mode = publication_summary.get("record_mode", "sandbox")
                session.pop(f"forward_{mode}_preview", None)
                if publication_summary.get("approval_mismatch"):
                    page_error = (
                        "The source files changed after preview. Nothing was "
                        "published; preview again."
                    )

            if retirement_summary and selected_action == "public_retire_preview":
                session["forward_public_retirement_preview"] = {
                    "approval_signature": retirement_summary.get(
                        "approval_signature"
                    ),
                    "ticker": selected_ticker,
                    "created_at": int(time.time()),
                }
            elif retirement_summary and selected_action == "public_retire":
                session.pop("forward_public_retirement_preview", None)
                if retirement_summary.get("approval_mismatch"):
                    page_error = (
                        "The public record or alert state changed after preview. "
                        "Nothing was retired; preview again."
                    )

            if removal_summary and selected_action == "public_remove_preview":
                session["forward_public_removal_preview"] = {
                    "approval_signature": removal_summary.get("approval_signature"),
                    "ticker": selected_ticker,
                    "created_at": int(time.time()),
                }
            elif removal_summary and selected_action == "public_remove":
                session.pop("forward_public_removal_preview", None)
                if removal_summary.get("approval_mismatch"):
                    page_error = (
                        "The retired record or approved file changed after preview. "
                        "Nothing was removed; preview again."
                    )

            if reset_summary is not None:
                session.pop("forward_sandbox_preview", None)

            app.logger.info(
                "Manual admin action admin_user_id=%s action=%s ticker=%s "
                "publication=%s alerts=%s retirement=%s removal=%s reset=%s",
                current_user.id,
                selected_action,
                selected_ticker or "ALL",
                {
                    "mode": publication_summary.get("record_mode"),
                    "pending": publication_summary.get("candidate_count", 0),
                    "published": publication_summary.get("published", 0),
                    "errors": publication_summary.get("error_count", 0),
                    "batch_id": publication_summary.get("batch_id"),
                } if publication_summary else None,
                {
                    "mode": dispatch_summary.get("mode"),
                    "pending_or_sent": dispatch_summary.get("pending_or_sent", 0),
                    "failures": dispatch_summary.get("failures", 0),
                    "approval_mismatch": dispatch_summary.get(
                        "approval_mismatch", False
                    ),
                } if dispatch_summary else None,
                {
                    "ticker": retirement_summary.get("ticker"),
                    "retired": retirement_summary.get("retired", False),
                    "approval_mismatch": retirement_summary.get(
                        "approval_mismatch", False
                    ),
                    "disabled_alert_count": retirement_summary.get(
                        "disabled_alert_count", 0
                    ),
                } if retirement_summary else None,
                {
                    "ticker": removal_summary.get("ticker"),
                    "removed": removal_summary.get("removed", False),
                    "approval_mismatch": removal_summary.get(
                        "approval_mismatch", False
                    ),
                } if removal_summary else None,
                reset_summary,
            )

    enabled_alert_count = WatchlistItem.query.filter_by(
        email_alert_enabled=True
    ).count()
    enabled_asset_count = (
        db.session.query(WatchlistItem.ticker)
        .filter(WatchlistItem.email_alert_enabled.is_(True))
        .distinct()
        .count()
    )
    sandbox_summary = build_forward_record_summary("sandbox")
    public_record_summary = build_forward_record_summary("public")
    recent_deliveries = (
        db.session.query(SignalAlertDelivery, User.email)
        .join(User, User.id == SignalAlertDelivery.user_id)
        .order_by(SignalAlertDelivery.id.desc())
        .limit(25)
        .all()
    )
    recent_publication_batches = (
        ForwardPublicationBatch.query
        .order_by(ForwardPublicationBatch.id.desc())
        .limit(15)
        .all()
    )
    public_confirmation_phrase = (
        "START PUBLIC RECORD"
        if not public_record_summary.get("started")
        else "PUBLISH PUBLIC"
    )
    active_public_assets = get_forward_record_assets(
        "public",
        active_only=True,
    )
    retired_public_assets = [
        asset
        for asset in get_forward_record_assets("public")
        if asset.status == "retired"
    ]
    removed_public_assets = [
        asset
        for asset in get_forward_record_assets("public", include_removed=True)
        if asset.status == "removed"
    ]
    default_removal_after_date = (datetime.utcnow().date() + timedelta(days=365)).isoformat()

    response = app.make_response(render_template(
        "admin_signal_alerts.html",
        dispatch_summary=dispatch_summary,
        publication_summary=publication_summary,
        retirement_summary=retirement_summary,
        removal_summary=removal_summary,
        reset_summary=reset_summary,
        dispatch_output=dispatch_output,
        page_error=page_error,
        selected_ticker=selected_ticker,
        selected_action=selected_action,
        enabled_alert_count=enabled_alert_count,
        enabled_asset_count=enabled_asset_count,
        sandbox_summary=sandbox_summary,
        public_record_summary=public_record_summary,
        public_record_enabled=public_enabled,
        public_confirmation_phrase=public_confirmation_phrase,
        active_public_assets=active_public_assets,
        retired_public_assets=retired_public_assets,
        removed_public_assets=removed_public_assets,
        default_removal_after_date=default_removal_after_date,
        forward_record_storage_explicit=FORWARD_RECORD_STORAGE_EXPLICIT,
        recent_deliveries=recent_deliveries,
        recent_publication_batches=recent_publication_batches,
    ))
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    return response


# --------------------
# Watchlists and signal-change alerts
# --------------------

def get_watchlist_limit_for_user(user):
    if is_admin_user(user):
        return None
    return PAID_WATCHLIST_LIMIT if is_paid_user(user) else FREE_WATCHLIST_LIMIT


def signal_label(value):
    try:
        signal = int(value)
    except (TypeError, ValueError):
        signal = 0

    if signal == 1:
        return "BUY"
    if signal == -1:
        return "SELL"
    return "HOLD"


def make_signal_row_fingerprint(ticker, signal_date, signal, close_price):
    """Return a stable digest for the published parts of one signal row.

    The alert worker uses this to distinguish a later revision to an already
    observed signal date from a genuinely new dated row. Only the public signal
    and its reference closing price are included; proprietary feature columns
    are intentionally excluded.
    """
    clean_ticker = normalize_ticker(ticker)
    if isinstance(signal_date, datetime):
        signal_date = signal_date.date()

    normalized_close = format(float(close_price), ".12g")
    raw = (
        f"{clean_ticker}:{signal_date.isoformat()}:"
        f"{int(signal)}:{normalized_close}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_latest_signal_snapshot(ticker):
    clean_ticker = normalize_ticker(ticker)
    market_data = load_epoch_csv_for_ticker(clean_ticker)

    if market_data.empty:
        raise ValueError("Market data has no valid rows for this asset.")

    latest_index = market_data.index.max()
    latest_row = market_data.loc[latest_index]

    # Defensive handling in case a duplicated pandas index ever returns a frame.
    if isinstance(latest_row, pd.DataFrame):
        latest_row = latest_row.iloc[-1]

    signal = int(latest_row["epoch_signal"])
    close_price = float(latest_row["Close"])

    if signal not in {-1, 0, 1}:
        raise ValueError("The latest signal is invalid.")

    if not math.isfinite(close_price) or close_price <= 0:
        raise ValueError("The latest market price is invalid.")

    signal_date = latest_index.date()

    return {
        "ticker": clean_ticker,
        "signal": signal,
        "signal_label": signal_label(signal),
        "signal_date": signal_date,
        "close_price": close_price,
        "row_fingerprint": make_signal_row_fingerprint(
            clean_ticker,
            signal_date,
            signal,
            close_price,
        ),
        **build_data_freshness_metadata(clean_ticker, signal_date),
    }


def watchlist_access_count_for_user(user):
    query = WatchlistItem.query.filter_by(user_id=user.id)

    if is_admin_user(user) or is_paid_user(user):
        return query.count()

    # Preserve locked Pro items after a downgrade, but do not let them consume
    # the four Free slots available for the Free-access assets.
    return query.filter(WatchlistItem.ticker.in_(list(TOP_FREE_TICKERS))).count()


def serialize_watchlist_item(item, user, *, retired_tickers=None):
    ticker = normalize_ticker(item.ticker)
    full_signal_access = can_view_full_signals_for_ticker(user, ticker)
    visible_to_user = can_user_see_ticker(user, ticker)
    locked = not (visible_to_user and full_signal_access)
    retired_from_public_tracking = (
        ticker in retired_tickers
        if retired_tickers is not None
        else public_forward_asset_is_retired(ticker)
    )

    payload = {
        **item.to_dict(),
        "ticker": ticker,
        "locked": locked,
        "asset_available": visible_to_user,
        "retired_from_public_tracking": retired_from_public_tracking,
        "can_enable_email_alert": bool(
            is_paid_user(user)
            and visible_to_user
            and full_signal_access
            and not retired_from_public_tracking
        ),
        "signal": None,
        "signal_label": None,
        "signal_date": None,
        "close_price": None,
        "data_through": None,
        "site_data_updated_at_utc": None,
        "freshness_status": "unknown",
        "freshness_label": "Unknown",
    }

    # Signal-board data is already loaded by the dashboard. The watchlist API
    # returns membership and alert state only, avoiding up to 100 redundant CSV
    # reads on every page load.
    return payload


def generate_signal_alert_unsubscribe_token(user):
    return get_serializer().dumps(
        {
            "user_id": int(user.id),
            "email": normalize_email(user.email),
        },
        salt=SIGNAL_ALERT_UNSUBSCRIBE_SALT,
    )


def confirm_signal_alert_unsubscribe_token(token):
    try:
        payload = get_serializer().loads(
            token,
            salt=SIGNAL_ALERT_UNSUBSCRIBE_SALT,
        )
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    try:
        user_id = int(payload.get("user_id"))
    except (TypeError, ValueError):
        return None

    email = normalize_email(payload.get("email"))
    user = db.session.get(User, user_id)

    if not user or not email or normalize_email(user.email) != email:
        return None

    return user


def render_signal_alert_unsubscribe_page(
    *,
    token,
    user=None,
    success=False,
    error=None,
    disabled_count=0,
    status_code=200,
):
    response = app.make_response(render_template(
        "unsubscribe_signal_alerts.html",
        token=token,
        masked_email=(
            mask_email_for_display(user.email)
            if user is not None else "your account"
        ),
        valid=(user is not None),
        success=success,
        error=error,
        disabled_count=disabled_count,
    ))
    response.status_code = status_code
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    return response


@app.route("/watchlist", methods=["GET"])
@login_required
@limiter.limit("60 per minute")
def get_watchlist():
    items = (
        WatchlistItem.query
        .filter_by(user_id=current_user.id)
        .order_by(WatchlistItem.created_at.asc(), WatchlistItem.id.asc())
        .all()
    )

    retired_tickers = get_retired_public_ticker_set()

    # Downgrades, access-policy changes, or public retirement immediately stop
    # email delivery while preserving the saved watchlist item for removal.
    preferences_changed = False
    for item in items:
        if (
            item.email_alert_enabled
            and not (
                is_paid_user(current_user)
                and can_view_full_signals_for_ticker(current_user, item.ticker)
                and normalize_ticker(item.ticker) not in retired_tickers
            )
        ):
            item.email_alert_enabled = False
            preferences_changed = True

    if preferences_changed:
        db.session.commit()

    limit = get_watchlist_limit_for_user(current_user)
    used_count = watchlist_access_count_for_user(current_user)

    return jsonify({
        "items": [
            serialize_watchlist_item(
                item,
                current_user,
                retired_tickers=retired_tickers,
            )
            for item in items
        ],
        "limit": limit,
        "used_count": used_count,
        "is_paid": is_paid_user(current_user),
        "email_alerts_available": is_paid_user(current_user),
    })


@app.route("/watchlist", methods=["POST"])
@login_required
@limiter.limit("20 per minute")
def add_watchlist_item():
    data = get_json_object()
    if data is None:
        return jsonify({"error": "A JSON object is required."}), 400

    ticker = normalize_ticker(data.get("ticker"))

    support_error = require_supported_ticker_or_400(ticker)
    if support_error:
        return support_error

    visibility_error = require_ticker_visible_or_404(ticker)
    if visibility_error:
        return visibility_error

    if not can_view_full_signals_for_ticker(current_user, ticker):
        return jsonify({
            "error": "This asset requires NeuralTrend Pro before it can be added to your watchlist.",
            "upgrade_required": True,
            "ticker": ticker,
        }), 403

    # Serialize limit-sensitive additions for this account.
    db.session.query(User.id).filter(
        User.id == current_user.id
    ).with_for_update().one()

    existing = WatchlistItem.query.filter_by(
        user_id=current_user.id,
        ticker=ticker,
    ).first()

    if existing:
        return jsonify({
            "message": "Asset is already on your watchlist.",
            "item": serialize_watchlist_item(existing, current_user),
        })

    limit = get_watchlist_limit_for_user(current_user)
    used_count = watchlist_access_count_for_user(current_user)

    if limit is not None and used_count >= limit:
        return jsonify({
            "error": f"Watchlist limit reached. Your current limit is {limit}.",
            "upgrade_required": not is_paid_user(current_user),
            "limit": limit,
        }), 403

    try:
        snapshot = get_latest_signal_snapshot(ticker)
        item = WatchlistItem(
            user_id=current_user.id,
            ticker=ticker,
            email_alert_enabled=False,
            last_observed_signal=snapshot["signal"],
            last_observed_signal_date=snapshot["signal_date"],
            last_observed_row_fingerprint=snapshot["row_fingerprint"],
        )
        db.session.add(item)
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        existing = WatchlistItem.query.filter_by(
            user_id=current_user.id,
            ticker=ticker,
        ).first()
        if existing:
            return jsonify({
                "message": "Asset is already on your watchlist.",
                "item": serialize_watchlist_item(existing, current_user),
            })
        raise
    except (ValueError, FileNotFoundError) as error:
        db.session.rollback()
        return jsonify({"error": str(error)}), 400
    except Exception:
        db.session.rollback()
        app.logger.exception(
            "Could not add watchlist item user_id=%s ticker=%s",
            current_user.id,
            ticker,
        )
        return jsonify({"error": "Could not update your watchlist."}), 500

    return jsonify({
        "message": f"{ticker} added to your watchlist.",
        "item": serialize_watchlist_item(item, current_user),
    }), 201


@app.route("/watchlist/<ticker>", methods=["DELETE"])
@login_required
@limiter.limit("20 per minute")
def remove_watchlist_item(ticker):
    clean_ticker = normalize_ticker(ticker)
    item = WatchlistItem.query.filter_by(
        user_id=current_user.id,
        ticker=clean_ticker,
    ).first()

    if not item:
        return jsonify({"error": "Watchlist item not found."}), 404

    db.session.delete(item)
    db.session.commit()

    return jsonify({
        "message": f"{clean_ticker} removed from your watchlist."
    })


@app.route("/watchlist/<ticker>/alerts", methods=["PATCH"])
@login_required
@limiter.limit("20 per minute")
def update_watchlist_alert(ticker):
    data = get_json_object()
    if data is None:
        return jsonify({"error": "A JSON object is required."}), 400

    enabled = data.get("enabled")
    if not isinstance(enabled, bool):
        return jsonify({"error": "enabled must be true or false."}), 400

    clean_ticker = normalize_ticker(ticker)
    item = (
        WatchlistItem.query
        .filter_by(user_id=current_user.id, ticker=clean_ticker)
        .with_for_update()
        .first()
    )

    if not item:
        return jsonify({"error": "Add this asset to your watchlist first."}), 404

    if enabled:
        if public_forward_asset_is_retired(clean_ticker):
            return jsonify({
                "error": "Signal alerts are no longer available because this asset has been retired from active public tracking.",
            }), 409

        if not is_paid_user(current_user):
            return jsonify({
                "error": "Email signal-change alerts are available with NeuralTrend Pro.",
                "upgrade_required": True,
            }), 403

        if not can_view_full_signals_for_ticker(current_user, clean_ticker):
            return jsonify({
                "error": "This asset is not available for signal alerts.",
            }), 403

        try:
            snapshot = get_latest_signal_snapshot(clean_ticker)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 400

        # Establish the current published signal as the baseline. This prevents
        # enabling alerts from sending a retroactive or misleading change email.
        item.last_observed_signal = snapshot["signal"]
        item.last_observed_signal_date = snapshot["signal_date"]
        item.last_observed_row_fingerprint = snapshot["row_fingerprint"]

    item.email_alert_enabled = enabled
    db.session.commit()

    return jsonify({
        "message": (
            f"Email signal-change alerts enabled for {clean_ticker}."
            if enabled else
            f"Email signal-change alerts disabled for {clean_ticker}."
        ),
        "item": serialize_watchlist_item(item, current_user),
    })


@app.route("/signal-alerts/unsubscribe/<token>", methods=["GET", "POST"])
@limiter.limit("10 per hour")
def unsubscribe_signal_alerts(token):
    user = confirm_signal_alert_unsubscribe_token(token)

    if not user:
        return render_signal_alert_unsubscribe_page(
            token=token,
            error="This unsubscribe link is invalid.",
            status_code=400,
        )

    if request.method == "GET":
        return render_signal_alert_unsubscribe_page(
            token=token,
            user=user,
        )

    items = WatchlistItem.query.filter_by(
        user_id=user.id,
        email_alert_enabled=True,
    ).all()

    for item in items:
        item.email_alert_enabled = False

    db.session.commit()

    return render_signal_alert_unsubscribe_page(
        token=token,
        user=user,
        success=True,
        disabled_count=len(items),
    )


def get_live_sim_latest_csv_dates(simulations):
    """Read each unique ticker CSV once for a Live Simulation request."""
    latest_dates = {}
    for sim in simulations:
        ticker = normalize_ticker(sim.ticker)
        if ticker in latest_dates:
            continue
        latest_dates[ticker] = get_latest_csv_date_for_ticker(ticker)
    return latest_dates


def refresh_live_simulations_if_needed(simulations, user, latest_csv_dates):
    """Refresh only simulations that are actually behind their source CSV.

    Previously opening the Live Simulation board acquired a row lock and read a
    full market CSV for every active simulation, even when almost all of them
    were already current. On large accounts this could keep a Gunicorn worker
    busy long enough for the proxy to return a transient 502.
    """
    for sim in simulations:
        if not live_sim_can_update_for_user(user, sim):
            continue

        ticker = normalize_ticker(sim.ticker)
        latest_csv_date = latest_csv_dates.get(ticker)

        if (
            latest_csv_date is not None
            and sim.last_processed_date is not None
            and sim.last_processed_date >= latest_csv_date
        ):
            continue

        try:
            update_live_simulation_from_csv(sim, user)
        except Exception:
            app.logger.exception(
                "Live simulation list refresh failed: simulation_id=%s",
                sim.id,
            )


# --------------------
# Live simulation API
# --------------------

@app.route("/live-simulations", methods=["GET"])
@login_required
def list_live_simulations():
    freeze_locked_live_simulations_for_user(current_user)
    enforce_free_live_simulation_limit_for_user(current_user)
    
    requested_status = request.args.get("status", "open").strip().lower()

    query = visible_live_simulation_query()

    if requested_status == "active":
        query = query.filter_by(status="active")
    elif requested_status == "paused":
        query = query.filter_by(status="paused")
    elif requested_status == "archived":
        query = query.filter_by(status="archived")
    elif requested_status == "all":
        pass
    else:
        # Default view: active + paused
        requested_status = "open"
        query = query.filter(LiveSimulation.status.in_(["active", "paused"]))

    sims = query.order_by(LiveSimulation.created_at.desc()).all()
    latest_csv_dates = get_live_sim_latest_csv_dates(sims)

    should_refresh = str(request.args.get("refresh", "1")).strip().lower() not in {
        "0", "false", "no"
    }

    if should_refresh:
        refresh_live_simulations_if_needed(
            sims,
            current_user,
            latest_csv_dates,
        )
        sims = query.order_by(LiveSimulation.created_at.desc()).all()

    active_count = visible_live_simulation_query().filter_by(status="active").count()
    paused_count = visible_live_simulation_query().filter_by(status="paused").count()
    archived_count = visible_live_simulation_query().filter_by(status="archived").count()

    open_count = active_count + paused_count
    all_count = open_count + archived_count

    user_limit = get_live_simulation_limit_for_user(current_user)
    simulation_summaries = [
        live_simulation_summary(
            sim,
            latest_csv_date=latest_csv_dates.get(normalize_ticker(sim.ticker)),
        )
        for sim in sims
    ]
    portfolio_total = build_live_simulation_portfolio_total(
        sims,
        summaries=simulation_summaries,
        include_curve=False,
        view=requested_status,
    )

    return jsonify({
        "limit": user_limit,
        "is_paid": is_paid_user(current_user),
        "is_admin": is_admin_user(current_user),
        "count": len(sims),
        "used_count": open_count,
        "active_count": active_count,
        "paused_count": paused_count,
        "archived_count": archived_count,
        "open_count": open_count,
        "all_count": all_count,
        "view": requested_status,
        "simulations": simulation_summaries,
        "portfolio_total": portfolio_total,
    })

@app.route("/live-simulations", methods=["POST"])
@login_required
def create_live_simulation():
    data = get_json_object()
    if data is None:
        return jsonify({"error": "A JSON object is required."}), 400

    ticker = normalize_ticker(data.get("ticker", "BTC-USD"))
    name, name_error = validate_simulation_name(
        data.get("name", ""),
        allow_empty=True,
    )
    if name_error:
        return jsonify({"error": name_error}), 400

    access_error = require_live_simulation_ticker_access(ticker)
    if access_error:
        return access_error

    try:
        initial_cash = parse_finite_number(
            normalize_optional_currency_input(data.get("initial_cash"), 10000),
            "Initial cash",
            minimum=MIN_INITIAL_CASH,
            maximum=MAX_INITIAL_CASH,
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    try:
        position_fraction = parse_position_fraction(
            data.get("position_size_pct", 100)
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    position_size_pct = position_fraction * 100

    # Serialize limit-sensitive operations for this account. This prevents two
    # concurrent create requests from both observing the same available slot.
    db.session.query(User.id).filter(
        User.id == current_user.id
    ).with_for_update().one()

    active_count = (
        LiveSimulation.query
        .filter(
            LiveSimulation.user_id == current_user.id,
            LiveSimulation.status.in_(["active", "paused"])
        )
        .count()
    )

    user_limit = get_live_simulation_limit_for_user(current_user)

    if user_limit is not None and active_count >= user_limit:
        return jsonify({
            "error": (
                f"Simulation limit reached. Your current limit is {user_limit}. "
                "Upgrade to Pro to unlock up to 100 live simulations."
            ),
            "upgrade_required": not is_paid_user(current_user),
            "limit": user_limit
        }), 403

    try:
        df = load_epoch_csv_for_ticker(ticker)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Could not load market data for live simulation ticker=%s", ticker)
        return jsonify({"error": "Market data is temporarily unavailable."}), 503

    latest_date = df.index.max().date()
    latest_price = float(df.loc[df.index.max(), "Close"])
    if not math.isfinite(latest_price) or latest_price <= 0:
        return jsonify({"error": "The latest market price is unavailable."}), 503

    transaction_cost_rate = get_transaction_cost_rate(ticker)
    asset_type = get_asset_type(ticker)

    # Buy & Hold benchmark:
    # include entry cost and preserve any uninvested stock cash.
    benchmark_quantity, benchmark_cash_balance, _ = build_buy_and_hold_benchmark(
        initial_cash=initial_cash,
        prices=[latest_price],
        ticker=ticker,
        transaction_cost_rate=transaction_cost_rate,
        enforce_whole_stock_shares=True,
    )

    if not is_crypto_ticker(ticker) and benchmark_quantity < 1:
        return jsonify({
            "error": "Initial cash is too small to buy at least one whole share for the buy-and-hold benchmark."
        }), 400

    if not name:
        if float(initial_cash).is_integer():
            initial_cash_name = str(int(initial_cash))
        else:
            initial_cash_name = f"{initial_cash:.8f}".rstrip("0").rstrip(".")
        name = f"{ticker}_{initial_cash_name}_{position_size_pct:.0f}%"

    sim = LiveSimulation(
        user_id=current_user.id,
        name=name,
        ticker=ticker,
        asset_type=asset_type,
        initial_cash=initial_cash,
        cash_balance=initial_cash,
        position_quantity=0.0,
        position_size_pct=position_size_pct,
        transaction_cost_rate=transaction_cost_rate,
        benchmark_quantity=benchmark_quantity,
        benchmark_cash_balance=benchmark_cash_balance,
        start_date=latest_date,
        last_processed_date=None,
        status="active"
    )

    db.session.add(sim)

    try:
        # Flush assigns the simulation ID without committing. The initial
        # market-data processing then commits the simulation, balances, trade,
        # equity point, and checkpoint as one atomic transaction.
        db.session.flush()
        sim = update_live_simulation_from_csv(sim, current_user)

        if sim is None:
            raise RuntimeError(
                "The new live simulation disappeared before initialization."
            )

    except Exception:
        db.session.rollback()
        app.logger.exception(
            "Live simulation creation rolled back for user_id=%s ticker=%s",
            current_user.id,
            ticker,
        )
        return jsonify({
            "error": (
                "Simulation could not be created because its initial "
                "market-data processing failed."
            )
        }), 500

    return jsonify({
        "message": "Live simulation created.",
        "simulation": live_simulation_detail(sim)
    }), 201

@app.route("/live-simulations/portfolio", methods=["GET"])
@login_required
def get_live_simulation_portfolio():
    freeze_locked_live_simulations_for_user(current_user)
    enforce_free_live_simulation_limit_for_user(current_user)

    requested_status = request.args.get("status", "open").strip().lower()
    query = visible_live_simulation_query()

    if requested_status == "active":
        query = query.filter_by(status="active")
    elif requested_status == "paused":
        query = query.filter_by(status="paused")
    elif requested_status == "archived":
        query = query.filter_by(status="archived")
    elif requested_status == "all":
        pass
    else:
        requested_status = "open"
        query = query.filter(LiveSimulation.status.in_(["active", "paused"]))

    requested_ids_raw = str(request.args.get("ids", "") or "").strip()
    requested_ids = None
    if requested_ids_raw:
        try:
            requested_ids = sorted({
                int(value)
                for value in requested_ids_raw.split(",")
                if str(value).strip()
            })
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid simulation filter."}), 400

        requested_ids = [value for value in requested_ids if value > 0]
        if not requested_ids:
            return jsonify({"error": "No simulations match the current filters."}), 404
        if len(requested_ids) > 100:
            return jsonify({"error": "Too many simulations were requested."}), 400

        query = query.filter(LiveSimulation.id.in_(requested_ids))

    simulations = query.order_by(LiveSimulation.created_at.desc()).all()
    if not simulations:
        return jsonify({"error": "No simulations match the current filters."}), 404

    skip_refresh = str(request.args.get("skip_refresh", "")).strip().lower() in {"1", "true", "yes"}
    if not skip_refresh:
        for sim in simulations:
            if not live_sim_can_update_for_user(current_user, sim):
                continue
            try:
                update_live_simulation_from_csv(sim, current_user)
            except Exception:
                app.logger.exception(
                    "Live simulation portfolio refresh failed: simulation_id=%s",
                    sim.id,
                )

        simulations = query.order_by(LiveSimulation.created_at.desc()).all()
    cache_key = live_sim_portfolio_curve_cache_key(
        current_user.id,
        requested_status,
        simulations,
    )
    cached_portfolio = get_cached_live_sim_portfolio_curve(cache_key)
    if cached_portfolio is not None:
        response = jsonify({"portfolio": cached_portfolio})
        response.headers["X-NeuralTrend-Portfolio-Cache"] = "HIT"
        response.headers["Cache-Control"] = "private, no-store, max-age=0"
        return response

    summaries = [live_simulation_summary(sim) for sim in simulations]
    portfolio = build_live_simulation_portfolio_total(
        simulations,
        summaries=summaries,
        include_curve=True,
        view=requested_status,
    )

    if portfolio is None:
        return jsonify({"error": "No simulations are available in this view."}), 404

    store_cached_live_sim_portfolio_curve(cache_key, portfolio)
    response = jsonify({"portfolio": portfolio})
    response.headers["X-NeuralTrend-Portfolio-Cache"] = "MISS"
    response.headers["Cache-Control"] = "private, no-store, max-age=0"
    return response


@app.route("/live-simulations/<int:simulation_id>", methods=["GET"])
@login_required
def get_live_simulation(simulation_id):
    sim = LiveSimulation.query.filter_by(
        id=simulation_id,
        user_id=current_user.id
    ).first()

    if not sim:
        return jsonify({"error": "Simulation not found."}), 404

    access_error = require_live_simulation_ticker_access(sim.ticker)
    if access_error:
        return access_error

    curve_only = str(request.args.get("curve_only", "")).strip().lower() in {
        "1", "true", "yes"
    }

    if curve_only:
        response = jsonify({
            "curve": live_simulation_curve_payload(sim)
        })
        response.headers["Cache-Control"] = "private, no-store, max-age=0"
        return response

    skip_refresh = str(request.args.get("skip_refresh", "")).strip().lower() in {
        "1", "true", "yes"
    }

    if not skip_refresh and live_sim_can_update_for_user(current_user, sim):
        try:
            updated_sim = update_live_simulation_from_csv(
                sim,
                current_user,
            )

            if updated_sim is None:
                return jsonify({
                    "error": "Simulation not found."
                }), 404

            sim = updated_sim

        except Exception:
            app.logger.exception(
                "Live simulation detail refresh failed: simulation_id=%s",
                sim.id,
            )
            return jsonify({
                "error": (
                    "The simulation could not be refreshed safely. "
                    "No partial update was saved."
                )
            }), 503

    return jsonify({
        "simulation": live_simulation_detail(sim)
    })

@app.route("/live-simulations/<int:simulation_id>", methods=["DELETE"])
@login_required
def delete_live_simulation(simulation_id):
    sim = LiveSimulation.query.filter_by(
        id=simulation_id,
        user_id=current_user.id
    ).first()

    if not sim:
        return jsonify({"error": "Simulation not found."}), 404

    db.session.delete(sim)
    db.session.commit()

    return jsonify({
        "message": "Simulation deleted."
    })

@app.route("/live-simulations/<int:simulation_id>", methods=["PATCH"])
@login_required
def rename_live_simulation(simulation_id):
    sim = LiveSimulation.query.filter_by(
        id=simulation_id,
        user_id=current_user.id
    ).first()

    if not sim:
        return jsonify({"error": "Simulation not found."}), 404

    data = get_json_object()
    if data is None:
        return jsonify({"error": "A JSON object is required."}), 400

    new_name, name_error = validate_simulation_name(data.get("name", ""))
    if name_error:
        return jsonify({"error": name_error}), 400

    sim.name = new_name
    db.session.commit()

    return jsonify({
        "message": "Simulation renamed.",
        "simulation": live_simulation_summary(sim)
    })

@app.route("/live-simulations/<int:simulation_id>/status", methods=["PATCH"])
@login_required
def update_live_simulation_status(simulation_id):
    sim = LiveSimulation.query.filter_by(
        id=simulation_id,
        user_id=current_user.id
    ).first()

    if not sim:
        return jsonify({"error": "Simulation not found."}), 404

    data = get_json_object()
    if data is None:
        return jsonify({"error": "A JSON object is required."}), 400

    new_status = str(data.get("status", "")).strip().lower()

    allowed_statuses = {"active", "paused", "archived"}

    if new_status not in allowed_statuses:
        return jsonify({
            "error": "Invalid status. Use active, paused, or archived."
        }), 400

    if new_status == "active" and not can_access_live_sim_ticker_for_user(current_user, sim.ticker):
        return jsonify({
            "error": (
                "This simulation uses a Pro-only asset. It is frozen while your account is on the Free plan. "
                "Upgrade to Pro to resume it."
            ),
            "upgrade_required": True
        }), 403

    # Archived simulations become open again when changed to active or paused.
    # Re-check the plan limit so archive/reactivate cannot bypass the 5/100 cap.
    if sim.status == "archived" and new_status in {"active", "paused"}:
        # Serialize reactivation against creation/other reactivation requests.
        db.session.query(User.id).filter(
            User.id == current_user.id
        ).with_for_update().one()

        open_count = (
            LiveSimulation.query
            .filter(
                LiveSimulation.user_id == current_user.id,
                LiveSimulation.id != sim.id,
                LiveSimulation.status.in_(["active", "paused"]),
            )
            .count()
        )
        user_limit = get_live_simulation_limit_for_user(current_user)
        if user_limit is not None and open_count >= user_limit:
            return jsonify({
                "error": f"Simulation limit reached. Your current limit is {user_limit}.",
                "upgrade_required": not is_paid_user(current_user),
                "limit": user_limit,
            }), 403

    sim.status = new_status
    db.session.commit()

    return jsonify({
        "message": f"Simulation status updated to {new_status}.",
        "simulation": live_simulation_summary(sim)
    })

@app.route("/request-password-reset", methods=["POST"])
@limiter.limit("3 per minute")
def request_password_reset():
    data = get_json_object() or {}
    email = normalize_email(data.get("email"))

    user = None
    if email and not validate_email_address(email):
        user = User.query.filter_by(email=email).first()

    if user:
        token = generate_reset_token(user)
        db.session.commit()

        reset_url = f"{BASE_URL}/reset-password/{token}"

        msg = Message(
            subject="Reset your NeuralTrend password",
            sender=app.config["MAIL_USERNAME"],
            recipients=[user.email],
        )

        msg.body = f"""Hi,

Use the secure link below to reset your NeuralTrend password:

{reset_url}

The link expires in 1 hour. Requesting a new reset email invalidates every older reset link.

If you did not request this change, you can ignore this email.

NeuralTrend
"""

        try:
            mail.send(msg)
            app.logger.info("Password reset email sent for user_id=%s", user.id)
        except Exception:
            app.logger.exception(
                "Could not send password reset email for user_id=%s",
                user.id,
            )

    # Use the same response whether or not the account exists.
    return jsonify({
        "message": (
            "If an account exists for that email, a password-reset link "
            "has been sent."
        )
    })


@app.route("/reset-password/<token>", methods=["GET"])
@limiter.limit("10 per minute")
def begin_password_reset(token):
    result = confirm_reset_token(token)

    if not result:
        clear_password_reset_session()
        session["password_reset_invalid"] = True
    else:
        user, token_hash = result
        establish_password_reset_session(user, token_hash)

    response = redirect(url_for("reset_password"))
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.route("/reset-password", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def reset_password():
    invalid_link = bool(session.pop("password_reset_invalid", False))
    user = get_password_reset_user_from_session()

    if not user:
        response = render_template(
            "reset_password.html",
            invalid=True,
            success=False,
            error=(
                "This password-reset link is invalid, expired, already used, "
                "or was replaced by a newer request."
            ),
        )
        rendered = app.make_response((response, 400))
        rendered.headers["Cache-Control"] = "no-store, max-age=0"
        rendered.headers["Pragma"] = "no-cache"
        rendered.headers["Referrer-Policy"] = "same-origin"
        return rendered

    error = None

    if request.method == "POST":
        new_password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        error = validate_password(new_password)

        if not error and new_password != confirm_password:
            error = "The two password entries do not match."

        if not error and bcrypt.check_password_hash(user.password_hash, new_password):
            error = "Choose a password different from your current password."

        if not error:
            reset_user_id = user.id
            reset_user_email = user.email

            user.password_hash = bcrypt.generate_password_hash(
                new_password
            ).decode("utf-8")
            user.password_reset_token_hash = None
            user.password_reset_requested_at = None
            user.password_changed_at = datetime.utcnow()
            user.auth_version = int(user.auth_version or 1) + 1
            user.failed_attempts = 0
            user.locked_until = None
            db.session.commit()

            if current_user.is_authenticated and current_user.id == reset_user_id:
                logout_user()
                session.pop("auth_version", None)

            clear_password_reset_session()
            send_password_changed_email(reset_user_email)

            response = render_template(
                "reset_password.html",
                invalid=False,
                success=True,
                error=None,
            )
            rendered = app.make_response(response)
            rendered.headers["Cache-Control"] = "no-store, max-age=0"
            rendered.headers["Pragma"] = "no-cache"
            rendered.headers["Referrer-Policy"] = "same-origin"
            return rendered

    response = render_template(
        "reset_password.html",
        invalid=invalid_link,
        success=False,
        error=error,
        masked_email=mask_email_for_display(user.email),
        password_min_length=PASSWORD_MIN_LENGTH,
        password_max_length=PASSWORD_MAX_LENGTH,
    )
    rendered = app.make_response((response, 400 if error else 200))
    rendered.headers["Cache-Control"] = "no-store, max-age=0"
    rendered.headers["Pragma"] = "no-cache"
    rendered.headers["Referrer-Policy"] = "same-origin"
    return rendered


def mask_email_for_display(email):
    clean_email = normalize_email(email)

    if "@" not in clean_email:
        return "your account"

    local_part, domain = clean_email.split("@", 1)

    if len(local_part) <= 2:
        masked_local = local_part[:1] + "*"
    else:
        masked_local = local_part[:1] + ("*" * min(len(local_part) - 2, 8)) + local_part[-1:]

    return f"{masked_local}@{domain}"


def render_delete_confirmation_page(
    *,
    token,
    user=None,
    error=None,
    deleted=False,
    canceled_subscription_count=0,
    status_code=200,
):
    response = app.make_response(render_template(
        "confirm_delete.html",
        token=token,
        masked_email=(
            mask_email_for_display(user.email)
            if user is not None else "your account"
        ),
        has_stripe_billing=bool(
            user is not None
            and (
                getattr(user, "stripe_customer_id", None)
                or getattr(user, "stripe_subscription_id", None)
                or getattr(user, "pending_checkout_session_id", None)
            )
        ),
        can_confirm=(user is not None and not deleted),
        error=error,
        deleted=deleted,
        canceled_subscription_count=canceled_subscription_count,
    ))
    response.status_code = status_code
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    # Keep the deletion token private from external sites while allowing the
    # same-origin POST referrer required by Flask-WTF HTTPS CSRF validation.
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    return response


@app.route("/request-delete-account", methods=["POST"])
@login_required
@limiter.limit("2 per minute")
def request_delete_account():
    user = current_user

    token = generate_delete_token(user.email)
    delete_url = f"{BASE_URL}/confirm-delete/{token}"

    msg = Message(
        subject="Confirm your NeuralTrend account deletion",
        sender=app.config["MAIL_USERNAME"],
        recipients=[user.email]
    )

    msg.body = f"""
    A request was made to delete your NeuralTrend account.

    Open the secure confirmation page below:

    {delete_url}

    Opening the link does not delete the account. To complete deletion, you must
    enter the account password and type DELETE. If the account has an active
    NeuralTrend Pro subscription, final confirmation will cancel it immediately
    before the NeuralTrend account is removed.

    This link expires in 1 hour. If you did not request deletion, ignore this email.

    NeuralTrend
    """

    try:
        mail.send(msg)
        app.logger.info("Account deletion email sent for user_id=%s", user.id)
    except Exception:
        app.logger.exception(
            "Could not send account deletion email for user_id=%s",
            user.id,
        )
        return jsonify({
            "error": "Could not send the confirmation email. Please try again."
        }), 502

    return jsonify({
        "message": (
            "Check your email for the final confirmation page. "
            "Nothing has been deleted yet."
        )
    })


@app.route("/confirm-delete/<token>", methods=["GET", "POST"])
@limiter.limit("10 per hour")
def confirm_delete(token):
    email = confirm_delete_token(token)

    if not email:
        return render_delete_confirmation_page(
            token=token,
            error="This deletion link is invalid or has expired.",
            status_code=400,
        )

    user = User.query.filter_by(email=normalize_email(email)).first()

    if not user:
        return render_delete_confirmation_page(
            token=token,
            deleted=True,
            status_code=200,
        )

    if request.method == "GET":
        return render_delete_confirmation_page(
            token=token,
            user=user,
        )

    password = request.form.get("password", "")
    confirmation = request.form.get("confirmation", "").strip()

    if confirmation != "DELETE":
        return render_delete_confirmation_page(
            token=token,
            user=user,
            error="Type DELETE exactly to confirm permanent account deletion.",
            status_code=400,
        )

    if not password or not bcrypt.check_password_hash(user.password_hash, password):
        return render_delete_confirmation_page(
            token=token,
            user=user,
            error="The password is incorrect.",
            status_code=400,
        )

    user_id = user.id

    try:
        billing_result = cancel_stripe_billing_for_account_deletion(user)
    except stripe.error.StripeError:
        db.session.rollback()
        app.logger.exception(
            "Stripe prevented safe account deletion for user_id=%s",
            user_id,
        )
        return render_delete_confirmation_page(
            token=token,
            user=user,
            error=(
                "We could not safely cancel the billing connection, so your "
                "account was not deleted. Please try again or contact support."
            ),
            status_code=502,
        )
    except Exception:
        db.session.rollback()
        app.logger.exception(
            "Account deletion billing cleanup failed for user_id=%s",
            user_id,
        )
        return render_delete_confirmation_page(
            token=token,
            user=user,
            error=(
                "We could not safely complete account deletion. Your account "
                "has not been removed. Please try again or contact support."
            ),
            status_code=500,
        )

    canceled_subscription_ids = billing_result["canceled_subscription_ids"]

    try:
        user = (
            User.query
            .filter_by(id=user_id)
            .with_for_update()
            .first()
        )

        if not user:
            return render_delete_confirmation_page(
                token=token,
                deleted=True,
                canceled_subscription_count=len(canceled_subscription_ids),
            )

        # Save the Stripe cleanup state first. If the final database deletion
        # fails, the account remains but recurring billing has still stopped.
        user.subscription_type = "free"
        user.subscription_status = (
            "canceled" if canceled_subscription_ids else "inactive"
        )
        user.stripe_subscription_id = None
        user.pending_checkout_session_id = None
        user.checkout_attempt_id = None
        user.checkout_attempt_started_at = None
        user.subscription_updated_at = datetime.utcnow()
        db.session.commit()

        if current_user.is_authenticated and current_user.id == user_id:
            logout_user()

        db.session.delete(user)
        db.session.commit()

    except Exception:
        db.session.rollback()
        app.logger.exception(
            "Database account deletion failed after billing cleanup for user_id=%s",
            user_id,
        )
        return render_delete_confirmation_page(
            token=token,
            user=user,
            error=(
                "Billing was stopped, but the website account could not be "
                "removed. Please retry this confirmation or contact support."
            ),
            status_code=500,
        )

    app.logger.info(
        "Account permanently deleted: user_id=%s canceled_subscriptions=%s",
        user_id,
        len(canceled_subscription_ids),
    )

    return render_delete_confirmation_page(
        token=token,
        deleted=True,
        canceled_subscription_count=len(canceled_subscription_ids),
    )

@app.route("/")
def index():
    return render_template(
        "index.html",
        forward_record=build_forward_record_home_summary(),
    )


@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        supported_tickers=get_supported_tickers_for_user(current_user),
        forward_record=build_forward_record_home_summary(),
    )

@app.route("/subscription")
def subscription():
    return render_template(
        "subscription.html",
        monthly_checkout_available=bool(STRIPE_PRO_MONTHLY_PRICE_ID),
        annual_checkout_available=bool(STRIPE_PRO_ANNUAL_PRICE_ID),
    )


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/methodology")
def methodology():
    return render_template(
        "methodology.html",
        coverage=build_methodology_coverage(),
        featured_performance=build_methodology_featured_performance(),
        change_log=load_public_model_change_log(),
        forward_record=build_forward_record_home_summary(),
    )


def resolve_forward_record_selection(record_mode):
    assets = get_forward_record_assets(record_mode)
    tickers = [asset.ticker for asset in assets]
    requested = normalize_ticker(request.args.get("ticker"))
    if requested and requested not in tickers:
        abort(404)

    selected_ticker = requested
    if not selected_ticker:
        active_tickers = [
            asset.ticker for asset in assets if asset.status == "active"
        ]
        if "BTC-USD" in active_tickers:
            selected_ticker = "BTC-USD"
        elif active_tickers:
            selected_ticker = active_tickers[0]
        elif tickers:
            selected_ticker = tickers[0]
    return assets, selected_ticker


@app.route("/performance")
def performance():
    public_enabled = public_forward_record_enabled()
    if public_enabled:
        record_assets, selected_ticker = resolve_forward_record_selection("public")
    else:
        record_assets, selected_ticker = [], None

    performance_record = (
        build_forward_record_performance(selected_ticker, "public")
        if selected_ticker else None
    )

    response = app.make_response(render_template(
        "performance.html",
        record_assets=record_assets,
        selected_ticker=selected_ticker,
        record=performance_record,
        record_mode="public",
        record_enabled=public_enabled,
        is_admin_preview=False,
    ))
    response.headers["Cache-Control"] = "public, max-age=300"
    if not public_enabled or not record_assets:
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
    return response


@app.route("/admin/forward-record")
@login_required
def admin_forward_record():
    if not is_admin_user(current_user):
        abort(404)

    record_assets, selected_ticker = resolve_forward_record_selection("sandbox")
    performance_record = (
        build_forward_record_performance(selected_ticker, "sandbox")
        if selected_ticker else None
    )

    response = app.make_response(render_template(
        "performance.html",
        record_assets=record_assets,
        selected_ticker=selected_ticker,
        record=performance_record,
        record_mode="sandbox",
        record_enabled=True,
        is_admin_preview=True,
    ))
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    return response


@app.route("/data")
def data():
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    return jsonify({
        "dates": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "index": df["Index"].tolist()
    })

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route("/terms")
def terms():
    return render_template("terms.html")


@app.route("/risk-disclaimer")
def risk_disclaimer():
    return render_template("risk_disclaimer.html")


@app.route("/refund-policy")
def refund_policy():
    return render_template("refund_policy.html")

@app.route('/market')
def market():
    return render_template('market.html')

@app.route('/knowledge')
def knowledge():
    return render_template('knowledge.html')

@app.route("/robots.txt")
def robots_txt():
    content = """User-agent: *
Allow: /

Sitemap: https://neuraltrend.org/sitemap.xml
"""
    return Response(content, mimetype="text/plain")

@app.route("/sitemap.xml")
def sitemap_xml():
    today = datetime.utcnow().date().isoformat()

    pages = [
        {
            "loc": "https://neuraltrend.org/",
            "lastmod": today,
            "changefreq": "daily",
            "priority": "1.0"
        },
        {
            "loc": "https://neuraltrend.org/dashboard",
            "lastmod": today,
            "changefreq": "daily",
            "priority": "0.9"
        },
        {
            "loc": "https://neuraltrend.org/resources",
            "lastmod": today,
            "changefreq": "monthly",
            "priority": "0.8"
        },
        {
            "loc": "https://neuraltrend.org/subscription",
            "lastmod": today,
            "changefreq": "weekly",
            "priority": "0.8"
        },
        {
            "loc": "https://neuraltrend.org/methodology",
            "lastmod": today,
            "changefreq": "weekly",
            "priority": "0.9"
        },
        {
            "loc": "https://neuraltrend.org/privacy",
            "lastmod": today,
            "changefreq": "monthly",
            "priority": "0.3"
        },
        {
            "loc": "https://neuraltrend.org/terms",
            "lastmod": today,
            "changefreq": "monthly",
            "priority": "0.3"
        },
        {
            "loc": "https://neuraltrend.org/risk-disclaimer",
            "lastmod": today,
            "changefreq": "monthly",
            "priority": "0.3"
        },
        {
            "loc": "https://neuraltrend.org/refund-policy",
            "lastmod": today,
            "changefreq": "monthly",
            "priority": "0.3"
        },
        {
            "loc": "https://neuraltrend.org/ai-crypto-trading-signals",
            "lastmod": today,
            "changefreq": "weekly",
            "priority": "0.9"
        },
        {
            "loc": "https://neuraltrend.org/crypto-paper-trading-simulator",
            "lastmod": today,
            "changefreq": "weekly",
            "priority": "0.9"
        },
        {
            "loc": "https://neuraltrend.org/buy-and-hold-vs-ai-strategy",
            "lastmod": today,
            "changefreq": "weekly",
            "priority": "0.9"
        },
    ]

    if build_forward_record_home_summary().get("started"):
        pages.insert(3, {
            "loc": "https://neuraltrend.org/performance",
            "lastmod": today,
            "changefreq": "daily",
            "priority": "0.9",
        })

    url_entries = []

    for page in pages:
        url_entries.append(f"""
    <url>
        <loc>{page["loc"]}</loc>
        <lastmod>{page["lastmod"]}</lastmod>
        <changefreq>{page["changefreq"]}</changefreq>
        <priority>{page["priority"]}</priority>
    </url>""")

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{''.join(url_entries)}
</urlset>
"""

    return Response(xml, mimetype="application/xml")

@app.route('/ads.txt')
def ads_txt():
    return send_from_directory(os.path.dirname(__file__), 'ads.txt')

def get_backtest_date_bounds(ticker):
    """Return the usable calendar-date bounds for an asset backtest.

    Start dates may run from the asset's first available market-data date
    through one calendar day before its latest available date. End dates may
    run from one calendar day after the selected start through the asset's
    latest available date.
    """
    signals_df = load_epoch_csv_for_ticker(ticker)
    if signals_df is None or len(signals_df) < 2:
        raise ValueError("Not enough market data to create a backtest date range.")

    coverage_start = signals_df.index.min().date()
    coverage_end = signals_df.index.max().date()
    if coverage_start >= coverage_end:
        raise ValueError("Not enough market data to create a backtest date range.")

    return coverage_start, coverage_end


@app.route('/backtest/date-range', methods=['GET'])
def backtest_date_range():
    ticker = normalize_ticker(request.args.get("ticker"))

    support_error = require_supported_ticker_or_400(ticker)
    if support_error:
        return support_error

    visibility_error = require_ticker_visible_or_404(ticker)
    if visibility_error:
        return visibility_error

    try:
        coverage_start, coverage_end = get_backtest_date_bounds(ticker)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Could not load backtest date range ticker=%s", ticker)
        return jsonify({"error": "Backtest date range is temporarily unavailable."}), 503

    response = jsonify({
        "ticker": ticker,
        "coverage_start": coverage_start.isoformat(),
        "coverage_end": coverage_end.isoformat(),
        "start_min": coverage_start.isoformat(),
        "start_max": (coverage_end - timedelta(days=1)).isoformat(),
        "end_min": (coverage_start + timedelta(days=1)).isoformat(),
        "end_max": coverage_end.isoformat(),
    })
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.route('/backtest', methods=['POST'])
def backtest():
    ticker = normalize_ticker(request.form.get("ticker"))

    access_error = require_signal_access_or_403(ticker)
    if access_error:
        return access_error

    try:
        initial_cash = parse_finite_number(
            request.form.get("cash"),
            "Initial cash",
            minimum=MIN_INITIAL_CASH,
            maximum=MAX_INITIAL_CASH,
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    start_date_text = (request.form.get("start") or "").strip()
    end_date_text = (request.form.get("end") or "").strip()
    
    # Position size per signal: 100%, 50%, 25%, etc.
    try:
        position_fraction = parse_position_fraction(
            request.form.get("dca_pct", "100_pct")
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        start_date = datetime.strptime(start_date_text, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Start date must use YYYY-MM-DD."}), 400

    try:
        end_date = datetime.strptime(end_date_text, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "End date must use YYYY-MM-DD."}), 400

    if start_date >= end_date:
        return jsonify({"error": "End date must be after the start date."}), 400

    try:
        raw_signals_df = load_epoch_csv_for_ticker(ticker)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Could not load backtest market data ticker=%s", ticker)
        return jsonify({"error": "Market data is temporarily unavailable."}), 503

    if raw_signals_df is None or len(raw_signals_df) < 2:
        return jsonify({"error": "Not enough market data to run a backtest."}), 400

    coverage_start = raw_signals_df.index.min().date()
    coverage_end = raw_signals_df.index.max().date()
    latest_start_date = coverage_end - timedelta(days=1)
    earliest_end_date = coverage_start + timedelta(days=1)

    if start_date < coverage_start or start_date > latest_start_date:
        return jsonify({
            "error": (
                f"Start date must be between {coverage_start.isoformat()} "
                f"and {latest_start_date.isoformat()}."
            )
        }), 400

    if end_date < earliest_end_date or end_date > coverage_end:
        return jsonify({
            "error": (
                f"End date must be between {earliest_end_date.isoformat()} "
                f"and {coverage_end.isoformat()}."
            )
        }), 400

    signals_df = raw_signals_df.reset_index()
    
    # Filter for the exact user-selected calendar date range.
    mask = (
        (signals_df["Date"] >= pd.to_datetime(start_date)) &
        (signals_df["Date"] <= pd.to_datetime(end_date))
    )
    
    signals_df = signals_df.loc[mask].copy()
    
    # Set Date as index
    signals_df.set_index("Date", inplace=True)
    
    # Convert Close and signal to numeric explicitly
    signals_df["Close"] = pd.to_numeric(signals_df["Close"], errors="coerce")
    signals_df["epoch_signal"] = pd.to_numeric(signals_df["epoch_signal"], errors="coerce")
    signals_df = signals_df.dropna(subset=["Close", "epoch_signal"])
    
    if len(signals_df) < 2:
        return jsonify({"error": "Not enough data in selected date range"}), 400
    
    # Same transaction-cost rule used by EpochSignaler/live simulation
    transaction_cost = get_transaction_cost_rate(ticker)
    
    cash = initial_cash
    position = 0.0
    equity_curve = []
    exposure_flags = []
    executed_buy_dates = []
    executed_sell_dates = []
    
    for date, row in signals_df.iterrows():
        price = float(row['Close'])
        signal = int(row['epoch_signal'])
    
        # BUY signal:
        # Invest selected % of available cash, including transaction cost.
        if signal == 1 and cash > 0:
            cash_allocation = cash * position_fraction
    
            # Treat cash_allocation as total cash used, including transaction cost.
            gross_buy_budget = cash_allocation / (1 + transaction_cost)
            raw_quantity = gross_buy_budget / price
    
            shares_to_buy = normalize_live_quantity_for_buy(
                ticker=ticker,
                raw_quantity=raw_quantity,
                price=price,
                cash_allocation=cash_allocation,
                transaction_cost_rate=transaction_cost
            )
    
            if shares_to_buy > 0:
                gross_amount = shares_to_buy * price
                fee = gross_amount * transaction_cost
                total_cash_used = gross_amount + fee
    
                if total_cash_used <= cash + 1e-9:
                    position += shares_to_buy
                    cash -= total_cash_used
                    executed_buy_dates.append(date)
    
        # SELL signal:
        # Sell selected % of current position, subtracting transaction cost.
        elif signal == -1 and position > 0:
            raw_quantity = position * position_fraction
    
            shares_to_sell = normalize_live_quantity_for_sell(
                ticker=ticker,
                raw_quantity=raw_quantity,
                current_position=position
            )
    
            if shares_to_sell > 0:
                gross_amount = shares_to_sell * price
                fee = gross_amount * transaction_cost
                net_cash_received = gross_amount - fee
    
                position -= shares_to_sell
    
                if position < 1e-12:
                    position = 0.0
    
                cash += net_cash_received
                executed_sell_dates.append(date)
    
        # Mark-to-market equity after today's action
        equity = cash + position * price
        equity_curve.append((date, equity))
        exposure_flags.append(position > 0)
        
    eq_df = pd.DataFrame(equity_curve, columns=['Date', 'Equity']).set_index('Date')

    # Buy & Hold benchmark with entry transaction cost and residual cash.
    prices = signals_df['Close'].to_numpy().flatten().astype(float)
    benchmark_quantity, benchmark_cash_balance, equity_curve = (
        build_buy_and_hold_benchmark(
            initial_cash=initial_cash,
            prices=prices,
            ticker=ticker,
            transaction_cost_rate=transaction_cost,
            enforce_whole_stock_shares=True,
        )
    )

    final_value = float(equity_curve[-1])
    buy_hold_growth_factor = float(final_value / initial_cash)

    strategy_equity_values = (
        eq_df['Equity'].to_numpy().flatten().astype(float).tolist()
    )
    sharpe_ratio = calculate_sharpe_from_equity_curve(
        strategy_equity_values,
        ticker,
    )
    strategy_max_drawdown = calculate_max_drawdown(
        [initial_cash, *strategy_equity_values]
    )
    buy_hold_max_drawdown = calculate_max_drawdown(
        [initial_cash, *equity_curve]
    )
    strategy_annualized_volatility = calculate_annualized_volatility(
        strategy_equity_values,
        ticker,
    )
    buy_hold_annualized_volatility = calculate_annualized_volatility(
        equity_curve,
        ticker,
    )
    strategy_market_exposure = calculate_market_exposure(exposure_flags)
    
    strategy_final_value = float(strategy_equity_values[-1])
    strategy_growth_factor = strategy_final_value / initial_cash
    buy_hold_period_return = buy_hold_growth_factor - 1
    strategy_period_return = strategy_growth_factor - 1

    dates = signals_df.index.strftime('%Y-%m-%d').tolist()

    results = {
        'ticker': ticker,
        'position_size_pct': position_fraction * 100,
        'transaction_cost_rate': transaction_cost,
        'valuation_policy': 'mark_to_market',
        'final_value': final_value,
        'final_value_epoch': strategy_final_value,
        'buy_hold_growth_factor': buy_hold_growth_factor,
        'strategy_growth_factor': strategy_growth_factor,
        'buy_hold_period_return': buy_hold_period_return,
        'strategy_period_return': strategy_period_return,
        'return_spread': strategy_period_return - buy_hold_period_return,
        'sharpe_ratio': sharpe_ratio,
        'strategy_max_drawdown': strategy_max_drawdown,
        'buy_hold_max_drawdown': buy_hold_max_drawdown,
        'strategy_annualized_volatility': strategy_annualized_volatility,
        'buy_hold_annualized_volatility': buy_hold_annualized_volatility,
        'strategy_market_exposure': strategy_market_exposure,
        'benchmark_cash_balance': benchmark_cash_balance,
        'ending_cash_balance': float(cash),
        'ending_position_quantity': float(position),
        'ending_position_open': bool(position > 0),
        'equity_curve': equity_curve,
        'epoch_equity_curve': strategy_equity_values,
        'dates': dates,
        # These are actual executed trades, not every BUY/SELL signal row.
        'executed_buy_dates': [d.strftime("%Y-%m-%d") for d in executed_buy_dates],
        'executed_sell_dates': [d.strftime("%Y-%m-%d") for d in executed_sell_dates],
        'executed_trade_count': len(executed_buy_dates) + len(executed_sell_dates),
    }

    return jsonify(results)

@app.route('/equity', methods=['POST'])
def equity():

    ticker = normalize_ticker(request.form.get("ticker"))

    access_error = require_signal_access_or_403(ticker)
    if access_error:
        return access_error
    
    duration_str = (request.form.get("duration") or "5y").strip()

    # Convert duration to days
    try:
        period_days = duration_to_days(duration_str)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        signals_df = load_epoch_csv_for_ticker(ticker)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Could not load equity market data ticker=%s", ticker)
        return jsonify({"error": "Market data is temporarily unavailable."}), 503

    if len(signals_df) < 2:
        return jsonify({"error": "Not enough data"}), 400

    # ---------------------------------------------------
    # Slice from duration ago until the latest row. MAX keeps every available
    # validated row for the selected asset.
    # ---------------------------------------------------
    signals_df = slice_epoch_data_for_period(signals_df, period_days)

    if len(signals_df) < 2:
        return jsonify({"error": "Not enough data in selected duration"}), 400

    # ---------------------------------------------------
    # Transaction cost
    # ---------------------------------------------------
    transaction_cost = get_transaction_cost_rate(ticker)

    # ---------------------------------------------------
    # Strategy Simulation (start cash = 1)
    # ---------------------------------------------------
    cash = 1.0
    position = 0.0
    epoch_equity_curve = []
    exposure_flags = []
    executed_buy_dates = []
    executed_sell_dates = []

    for date, row in signals_df.iterrows():
        price = row['Close']
        signal = row['epoch_signal']

        if signal == 1 and cash > 0:
            gross_buy_budget = cash / (1 + transaction_cost)
            position = gross_buy_budget / price
            cash = 0.0
            executed_buy_dates.append(date)

        elif signal == -1 and position > 0:
            cash = (position * price) * (1 - transaction_cost)
            position = 0.0
            executed_sell_dates.append(date)

        equity = cash + position * price
        epoch_equity_curve.append(equity)
        exposure_flags.append(position > 0)

    # ---------------------------------------------------
    # Buy & Hold Curve (start = 1, entry cost included)
    # ---------------------------------------------------
    prices = signals_df['Close'].to_numpy().astype(float)
    
    # Normalized previews intentionally allow fractional notional, including
    # for stocks, so the $1 base remains comparable across asset prices.
    benchmark_quantity, benchmark_cash_balance, buy_hold_curve = (
        build_buy_and_hold_benchmark(
            initial_cash=1.0,
            prices=prices,
            ticker=ticker,
            transaction_cost_rate=transaction_cost,
            enforce_whole_stock_shares=False,
        )
    )

    # ---------------------------------------------------
    # Final Metrics
    # ---------------------------------------------------
    final_value_bh = buy_hold_curve[-1]
    final_value_epoch = epoch_equity_curve[-1]

    sharpe_ratio = calculate_sharpe_from_equity_curve(
        epoch_equity_curve,
        ticker,
    )
    strategy_max_drawdown = calculate_max_drawdown([1.0, *epoch_equity_curve])
    buy_hold_max_drawdown = calculate_max_drawdown([1.0, *buy_hold_curve])
    strategy_annualized_volatility = calculate_annualized_volatility(
        epoch_equity_curve,
        ticker,
    )
    buy_hold_annualized_volatility = calculate_annualized_volatility(
        buy_hold_curve,
        ticker,
    )
    strategy_market_exposure = calculate_market_exposure(exposure_flags)

    dates = signals_df.index.strftime('%Y-%m-%d').tolist()

    buy_hold_period_return = final_value_bh - 1
    strategy_period_return = final_value_epoch - 1

    results = {
        'ticker': ticker,
        'transaction_cost_rate': transaction_cost,
        'valuation_policy': 'mark_to_market',
        'final_value': final_value_bh,
        'final_value_epoch': final_value_epoch,
        'buy_hold_growth_factor': final_value_bh,
        'strategy_growth_factor': final_value_epoch,
        'buy_hold_period_return': buy_hold_period_return,
        'strategy_period_return': strategy_period_return,
        'return_spread': strategy_period_return - buy_hold_period_return,
        'sharpe_ratio': sharpe_ratio,
        'strategy_max_drawdown': strategy_max_drawdown,
        'buy_hold_max_drawdown': buy_hold_max_drawdown,
        'strategy_annualized_volatility': strategy_annualized_volatility,
        'buy_hold_annualized_volatility': buy_hold_annualized_volatility,
        'strategy_market_exposure': strategy_market_exposure,
        'benchmark_cash_balance': benchmark_cash_balance,
        'ending_cash_balance': float(cash),
        'ending_position_quantity': float(position),
        'ending_position_open': bool(position > 0),
        'equity_curve': buy_hold_curve,
        'epoch_equity_curve': epoch_equity_curve,
        'dates': dates,
        # These are actual executed trades, not every BUY/SELL signal row.
        'executed_buy_dates': [d.strftime("%Y-%m-%d") for d in executed_buy_dates],
        'executed_sell_dates': [d.strftime("%Y-%m-%d") for d in executed_sell_dates],
        'executed_trade_count': len(executed_buy_dates) + len(executed_sell_dates),
        'coverage_start': signals_df.index.min().date().isoformat(),
        'coverage_end': signals_df.index.max().date().isoformat(),
        **build_data_freshness_metadata(
            ticker,
            signals_df.index.max().date(),
        ),
    }

    return jsonify(results)

def mask_signal_summary_row_for_user(row, user):
    """
    Returns a user-safe signal-summary row.

    Public/free users:
    - see full signals only for BTC-USD, ETH-USD, SOL-USD, XRP-USD
    - see ticker + return columns for all other assets
    - do not receive hidden signal values from the backend

    Paid users:
    - see all signal columns for all assets
    """
    ticker = row.get("ticker")
    full_access = can_view_full_signals_for_ticker(user, ticker)

    safe_row = {
        "ticker": ticker,

        # Return columns stay visible to everyone
        "buy_hold_period_return": row.get("buy_hold_period_return"),
        "strategy_period_return": row.get("strategy_period_return"),
        "return_spread": row.get("return_spread"),
        "outperformance_ratio": row.get("outperformance_ratio"),
        "alpha": row.get("alpha"),
        "alpha_prob": row.get("alpha_prob"),
        "strategy_avg_return": row.get("strategy_avg_return"),
        "strategy_profit_prob": row.get("strategy_profit_prob"),
        "recommended_days": row.get("recommended_days"),
        "stat_window": row.get("stat_window"),
        "strategy_max_drawdown": row.get("strategy_max_drawdown"),
        "buy_hold_max_drawdown": row.get("buy_hold_max_drawdown"),
        "strategy_annualized_volatility": row.get("strategy_annualized_volatility"),
        "buy_hold_annualized_volatility": row.get("buy_hold_annualized_volatility"),
        "sharpe_ratio": row.get("sharpe_ratio"),
        "strategy_market_exposure": row.get("strategy_market_exposure"),
        "executed_trade_count": row.get("executed_trade_count"),
        "observation_count": row.get("observation_count"),
        "coverage_start": row.get("coverage_start"),
        "coverage_end": row.get("coverage_end"),

        # Frontend uses this to show locked/blurred cells
        "signals_locked": not full_access,
    }

    safe_row.update(
        build_data_freshness_metadata(
            ticker,
            row.get("data_through") or row.get("coverage_end"),
        )
    )

    if full_access:
        safe_row.update({
            "today_signal": row.get("today_signal"),
            "yesterday_signal": row.get("yesterday_signal"),
            "last_week_signal": row.get("last_week_signal"),
            "last_month_signal": row.get("last_month_signal"),
        })
    else:
        safe_row.update({
            "today_signal": None,
            "yesterday_signal": None,
            "last_week_signal": None,
            "last_month_signal": None,
        })

    return safe_row

# Cached version that invalidates when CSV files change
@lru_cache(maxsize=1)
def compute_signals_summary_cached(csv_version, stat_csv_version, period_days):
    results = []

    for t in get_all_signal_board_tickers():
        try:
            sigs = compute_signals_for_ticker(t, period_days, csv_version)
            if not sigs:
                continue

            stat_values = get_signal_stat_for_horizon(t, period_days)
            results.append({
                'ticker': t,
                'today_signal': sigs['today'],
                'yesterday_signal': sigs['yesterday'],
                'last_week_signal': sigs['last_week'],
                'last_month_signal': sigs['last_month'],
                'buy_hold_period_return': sigs['buy_hold_period_return'],
                'strategy_period_return': sigs['strategy_period_return'],
                'return_spread': sigs['return_spread'],
                'outperformance_ratio': sigs['outperformance_ratio'],
                'alpha': stat_values['alpha'],
                'alpha_prob': stat_values['alpha_prob'],
                'strategy_avg_return': stat_values['strategy_avg_return'],
                'strategy_profit_prob': stat_values['strategy_profit_prob'],
                'recommended_days': stat_values['recommended_days'],
                'stat_window': stat_values['stat_window'],
                'strategy_max_drawdown': sigs['strategy_max_drawdown'],
                'buy_hold_max_drawdown': sigs['buy_hold_max_drawdown'],
                'strategy_annualized_volatility': sigs['strategy_annualized_volatility'],
                'buy_hold_annualized_volatility': sigs['buy_hold_annualized_volatility'],
                'sharpe_ratio': sigs['sharpe_ratio'],
                'strategy_market_exposure': sigs['strategy_market_exposure'],
                'executed_trade_count': sigs['executed_trade_count'],
                'observation_count': sigs['observation_count'],
                'coverage_start': sigs['coverage_start'],
                'coverage_end': sigs['coverage_end'],
                'data_through': sigs['data_through'],
            })
        except Exception:
            app.logger.exception("Skipping signal summary ticker=%s", t)

    # Do not cache an all-empty result. A supported production universe cannot
    # legitimately have zero rows; this indicates a transient file/read/update
    # problem. Raising here prevents lru_cache from preserving that bad snapshot.
    if not results:
        raise RuntimeError("No signal-board assets could be computed.")

    return results

@app.route('/signals/summary')
def signals_summary():
    duration_str = (request.args.get("duration") or "5y").strip()
    
    try:
        period_days = duration_to_days(duration_str)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    csv_version = get_csv_version()
    stat_csv_version = get_stat_csv_version()

    # Full internal cached data. Stat files are independently versioned because
    # they may be generated/updated without touching the epoch market CSVs.
    try:
        raw_results = compute_signals_summary_cached(
            csv_version,
            stat_csv_version,
            period_days,
        )
    except Exception:
        app.logger.exception(
            "Signal summary temporarily unavailable duration=%s",
            duration_str,
        )
        response = jsonify({
            "error": "Signal data is temporarily unavailable. Please retry."
        })
        response.status_code = 503
        response.headers["Cache-Control"] = "private, no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Vary"] = "Cookie"
        return response

    # Apply admin-only and retired-asset visibility before masking. Fetch the
    # lifecycle set once rather than issuing one database query per asset.
    if is_admin_user(current_user):
        visible_results = raw_results
    else:
        inactive_tickers = get_retired_public_ticker_set()
        visible_results = [
            row for row in raw_results
            if normalize_ticker(row.get("ticker")) not in ADMIN_ONLY_TICKERS
            and normalize_ticker(row.get("ticker")) not in inactive_tickers
        ]
    
    # User-safe data returned to frontend
    safe_results = [
        mask_signal_summary_row_for_user(row, current_user)
        for row in visible_results
    ]
    
    response = jsonify(safe_results)
    response.headers["Cache-Control"] = "private, no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Vary"] = "Cookie"
    return response

def stripe_to_dict(obj):
    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "_to_dict_recursive"):
        return obj._to_dict_recursive()

    if hasattr(obj, "to_dict_recursive"):
        return obj.to_dict_recursive()

    return dict(obj)


BLOCKING_STRIPE_SUBSCRIPTION_STATUSES = {
    "active",
    "trialing",
    "past_due",
    "unpaid",
    "incomplete",
    "paused",
}

CHECKOUT_ATTEMPT_TTL = timedelta(minutes=15)
VALID_PRO_BILLING_INTERVALS = frozenset({"monthly", "annual"})


def configured_pro_price_ids():
    return {
        price_id
        for price_id in STRIPE_PRO_PRICE_IDS_BY_INTERVAL.values()
        if price_id
    }


def get_pro_price_id_for_interval(billing_interval):
    if billing_interval not in VALID_PRO_BILLING_INTERVALS:
        return None

    return STRIPE_PRO_PRICE_IDS_BY_INTERVAL.get(billing_interval)


def checkout_session_billing_interval(checkout_session):
    checkout_session = stripe_to_dict(checkout_session)
    metadata = checkout_session.get("metadata") or {}
    interval = str(metadata.get("billing_interval") or "").strip().lower()

    # Checkout Sessions created before the annual option existed did not store
    # an interval. Those sessions used the monthly Pro Price.
    if interval in VALID_PRO_BILLING_INTERVALS:
        return interval

    return "monthly"


def stripe_subscription_uses_pro_price(subscription):
    subscription = stripe_to_dict(subscription)
    items = ((subscription.get("items") or {}).get("data") or [])
    allowed_price_ids = configured_pro_price_ids()

    return bool(allowed_price_ids) and any(
        (item.get("price") or {}).get("id") in allowed_price_ids
        for item in items
    )


def pro_billing_interval_for_price_id(price_id):
    for interval, configured_price_id in STRIPE_PRO_PRICE_IDS_BY_INTERVAL.items():
        if configured_price_id and price_id == configured_price_id:
            return interval

    return None


def pro_subscription_item(subscription):
    subscription = stripe_to_dict(subscription)
    items = ((subscription.get("items") or {}).get("data") or [])

    for item in items:
        price_id = (item.get("price") or {}).get("id")
        interval = pro_billing_interval_for_price_id(price_id)

        if interval:
            return item, interval

    return None, None


def subscription_period_end_timestamp(subscription, item=None):
    subscription = stripe_to_dict(subscription)
    item = stripe_to_dict(item) if item else {}

    return (
        subscription.get("cancel_at")
        or subscription.get("current_period_end")
        or item.get("current_period_end")
    )


def subscription_cancellation_pending(subscription):
    subscription = stripe_to_dict(subscription)
    return bool(
        subscription.get("cancel_at_period_end")
        or subscription.get("cancel_at")
    )


def get_current_user_pro_subscription(user):
    customer_id = getattr(user, "stripe_customer_id", None)

    if not customer_id or not stripe.api_key:
        return None

    stored_subscription_id = getattr(user, "stripe_subscription_id", None)

    if stored_subscription_id:
        try:
            subscription = stripe.Subscription.retrieve(stored_subscription_id)
            subscription = stripe_to_dict(subscription)

            if (
                subscription.get("customer") == customer_id
                and subscription.get("status") in BLOCKING_STRIPE_SUBSCRIPTION_STATUSES
                and stripe_subscription_uses_pro_price(subscription)
            ):
                return subscription

        except stripe.error.InvalidRequestError:
            app.logger.warning(
                "Stored Stripe subscription was not found while loading billing state: user_id=%s subscription_id=%s",
                user.id,
                stored_subscription_id,
            )

    return find_existing_blocking_pro_subscription(customer_id)


def subscription_management_payload(user, subscription=None):
    if subscription is None:
        subscription = get_current_user_pro_subscription(user)

    if not subscription:
        return {
            "is_paid": False,
            "status": getattr(user, "subscription_status", None) or "inactive",
            "current_interval": None,
            "cancel_at_period_end": False,
            "period_end": None,
            "portal_configuration_ready": bool(
                STRIPE_BILLING_PORTAL_CONFIGURATION_ID
            ),
        }

    item, interval = pro_subscription_item(subscription)
    period_end = subscription_period_end_timestamp(subscription, item)

    return {
        "is_paid": subscription.get("status") in PAID_SUBSCRIPTION_STATUSES,
        "status": subscription.get("status") or "inactive",
        "current_interval": interval,
        "cancel_at_period_end": subscription_cancellation_pending(subscription),
        "period_end": period_end,
        "portal_configuration_ready": bool(
            STRIPE_BILLING_PORTAL_CONFIGURATION_ID
        ),
    }


def find_existing_blocking_pro_subscription(customer_id):
    subscriptions = stripe.Subscription.list(
        customer=customer_id,
        status="all",
        limit=100,
    )
    subscriptions = stripe_to_dict(subscriptions)

    for subscription in subscriptions.get("data", []):
        if (
            subscription.get("status") in BLOCKING_STRIPE_SUBSCRIPTION_STATUSES
            and stripe_subscription_uses_pro_price(subscription)
        ):
            return subscription

    return None


def retrieve_open_checkout_session(session_id):
    if not session_id:
        return None

    try:
        checkout_session = stripe.checkout.Session.retrieve(session_id)
    except stripe.error.InvalidRequestError:
        return None

    checkout_session = stripe_to_dict(checkout_session)

    if (
        checkout_session.get("status") == "open"
        and checkout_session.get("url")
    ):
        return checkout_session

    return None


def cancel_stripe_subscription_immediately(subscription_id):
    """Cancel a Stripe subscription now without creating prorations."""
    cancel_method = getattr(stripe.Subscription, "cancel", None)

    if callable(cancel_method):
        # Stripe defaults invoice_now and prorate to false for immediate
        # cancellation, which avoids creating a proration invoice.
        return cancel_method(subscription_id)

    # Compatibility fallback for older stripe-python resource objects.
    subscription = stripe.Subscription.retrieve(subscription_id)
    instance_cancel = getattr(subscription, "cancel", None)

    if callable(instance_cancel):
        return instance_cancel()

    instance_delete = getattr(subscription, "delete", None)

    if callable(instance_delete):
        return instance_delete()

    raise RuntimeError(
        "The installed Stripe SDK does not expose subscription cancellation."
    )


def cancel_stripe_billing_for_account_deletion(user):
    """
    Stop all Stripe billing paths before deleting a local account.

    - Expires an unfinished Checkout Session so it cannot be completed later.
    - Cancels every non-terminal subscription belonging to the Stripe customer.
    - Does not delete the Stripe customer or financial records.
    """
    result = {
        "expired_checkout_session": False,
        "canceled_subscription_ids": [],
    }

    customer_id = getattr(user, "stripe_customer_id", None)
    stored_subscription_id = getattr(user, "stripe_subscription_id", None)
    pending_checkout_session_id = getattr(
        user,
        "pending_checkout_session_id",
        None,
    )

    if not any([
        customer_id,
        stored_subscription_id,
        pending_checkout_session_id,
    ]):
        return result

    if not stripe.api_key:
        raise RuntimeError(
            "Stripe is not configured, so billing safety cannot be verified."
        )

    if pending_checkout_session_id:
        try:
            checkout_session = stripe.checkout.Session.retrieve(
                pending_checkout_session_id
            )
            checkout_session = stripe_to_dict(checkout_session)

            if checkout_session.get("status") == "open":
                stripe.checkout.Session.expire(pending_checkout_session_id)
                result["expired_checkout_session"] = True

        except stripe.error.InvalidRequestError:
            # A missing/expired Session cannot accept a payment in this mode.
            app.logger.warning(
                "Pending Checkout Session was not found during deletion: user_id=%s session_id=%s",
                user.id,
                pending_checkout_session_id,
            )

    subscription_ids_to_cancel = set()

    if stored_subscription_id:
        try:
            subscription = stripe.Subscription.retrieve(stored_subscription_id)
            subscription = stripe_to_dict(subscription)
            subscription_customer_id = subscription.get("customer")

            if (
                customer_id
                and subscription_customer_id
                and subscription_customer_id != customer_id
            ):
                raise RuntimeError(
                    "Stored Stripe subscription does not belong to the saved customer."
                )

            if subscription.get("status") in BLOCKING_STRIPE_SUBSCRIPTION_STATUSES:
                subscription_ids_to_cancel.add(subscription.get("id"))

        except stripe.error.InvalidRequestError:
            app.logger.warning(
                "Stored Stripe subscription was not found during deletion: user_id=%s subscription_id=%s",
                user.id,
                stored_subscription_id,
            )

    if customer_id:
        subscriptions = stripe.Subscription.list(
            customer=customer_id,
            status="all",
            limit=100,
        )
        subscriptions = stripe_to_dict(subscriptions)

        for subscription in subscriptions.get("data", []):
            if subscription.get("status") in BLOCKING_STRIPE_SUBSCRIPTION_STATUSES:
                subscription_id = subscription.get("id")

                if subscription_id:
                    subscription_ids_to_cancel.add(subscription_id)

    for subscription_id in sorted(subscription_ids_to_cancel):
        cancel_stripe_subscription_immediately(subscription_id)
        result["canceled_subscription_ids"].append(subscription_id)

    return result


@app.route("/create-checkout-session", methods=["POST"])
@login_required
@limiter.limit("5 per minute")
def create_checkout_session():
    payload = request.get_json(silent=True) or {}
    billing_interval = str(
        payload.get("billing_interval") or "monthly"
    ).strip().lower()

    if billing_interval not in VALID_PRO_BILLING_INTERVALS:
        return jsonify({
            "error": "Choose either monthly or annual billing."
        }), 400

    selected_price_id = get_pro_price_id_for_interval(billing_interval)

    if not stripe.api_key or not selected_price_id:
        return jsonify({
            "error": (
                "Annual checkout is temporarily unavailable."
                if billing_interval == "annual"
                else "Checkout is temporarily unavailable."
            )
        }), 503

    try:
        now = datetime.utcnow()

        user = (
            User.query
            .filter_by(id=current_user.id)
            .with_for_update()
            .one()
        )

        if is_paid_user(user):
            db.session.commit()
            return jsonify({
                "error": "A Pro subscription is already active.",
                "manage_billing": True,
            }), 409

        pending_checkout_session_id = user.pending_checkout_session_id
        open_session = retrieve_open_checkout_session(
            pending_checkout_session_id
        )

        if pending_checkout_session_id and not open_session:
            # The stored Session expired, completed, or no longer exists. Use a
            # fresh attempt token so Stripe idempotency cannot return an old,
            # unusable Checkout Session.
            user.pending_checkout_session_id = None
            user.checkout_attempt_id = secrets.token_urlsafe(24)
            user.checkout_attempt_started_at = now

        if open_session:
            open_interval = checkout_session_billing_interval(open_session)

            if open_interval == billing_interval:
                db.session.commit()
                return jsonify({
                    "url": open_session["url"],
                    "reused": True,
                    "billing_interval": billing_interval,
                })

            # A user changed the billing choice while an earlier Checkout
            # Session was still open. Expire the old Session so the selected
            # interval cannot be silently replaced by a stale checkout.
            stripe.checkout.Session.expire(open_session["id"])
            user.pending_checkout_session_id = None
            user.checkout_attempt_id = secrets.token_urlsafe(24)
            user.checkout_attempt_started_at = now
            db.session.commit()

        user.pending_checkout_session_id = None

        if user.stripe_customer_id:
            existing_subscription = find_existing_blocking_pro_subscription(
                user.stripe_customer_id
            )

            if existing_subscription:
                status = existing_subscription.get("status") or "inactive"

                user.stripe_subscription_id = existing_subscription.get("id")
                user.subscription_status = status
                user.subscription_type = (
                    "pro" if status in PAID_SUBSCRIPTION_STATUSES else "free"
                )
                user.subscription_updated_at = now
                db.session.commit()

                return jsonify({
                    "error": "A subscription already exists for this account.",
                    "manage_billing": True,
                    "subscription_status": status,
                }), 409

        attempt_is_fresh = (
            user.checkout_attempt_id
            and user.checkout_attempt_started_at
            and now - user.checkout_attempt_started_at < CHECKOUT_ATTEMPT_TTL
        )

        if not attempt_is_fresh:
            user.checkout_attempt_id = secrets.token_urlsafe(24)
            user.checkout_attempt_started_at = now

        attempt_id = user.checkout_attempt_id
        customer_id = user.stripe_customer_id
        user_id = user.id
        user_email = user.email

        # Save/reuse the attempt token before calling Stripe. Concurrent retries
        # therefore use the same Stripe idempotency keys.
        db.session.commit()

        if not customer_id:
            customer = stripe.Customer.create(
                email=user_email,
                metadata={"user_id": str(user_id)},
                idempotency_key=f"nt-customer-{attempt_id}",
            )
            customer_id = customer.id

            user = (
                User.query
                .filter_by(id=user_id)
                .with_for_update()
                .one()
            )

            if user.stripe_customer_id:
                customer_id = user.stripe_customer_id
            else:
                user.stripe_customer_id = customer_id

            db.session.commit()

        # Check Stripe again after customer creation, immediately before Checkout.
        existing_subscription = find_existing_blocking_pro_subscription(customer_id)

        if existing_subscription:
            status = existing_subscription.get("status") or "inactive"

            user = (
                User.query
                .filter_by(id=user_id)
                .with_for_update()
                .one()
            )
            user.stripe_subscription_id = existing_subscription.get("id")
            user.subscription_status = status
            user.subscription_type = (
                "pro" if status in PAID_SUBSCRIPTION_STATUSES else "free"
            )
            user.subscription_updated_at = datetime.utcnow()
            db.session.commit()

            return jsonify({
                "error": "A subscription already exists for this account.",
                "manage_billing": True,
                "subscription_status": status,
            }), 409

        checkout_session = stripe.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            payment_method_types=["card"],
            line_items=[
                {
                    "price": selected_price_id,
                    "quantity": 1,
                }
            ],
            success_url=f"{BASE_URL}/dashboard?checkout=success",
            cancel_url=f"{BASE_URL}/subscription?checkout=cancelled",
            client_reference_id=str(user_id),
            metadata={
                "user_id": str(user_id),
                "billing_interval": billing_interval,
            },
            subscription_data={
                "metadata": {
                    "user_id": str(user_id),
                    "billing_interval": billing_interval,
                }
            },
            idempotency_key=(
                f"nt-checkout-{attempt_id}-{billing_interval}"
            ),
        )

        user = (
            User.query
            .filter_by(id=user_id)
            .with_for_update()
            .one()
        )
        user.stripe_customer_id = customer_id
        user.pending_checkout_session_id = checkout_session.id
        db.session.commit()

        return jsonify({
            "url": checkout_session.url,
            "billing_interval": billing_interval,
        })

    except stripe.error.StripeError:
        db.session.rollback()
        app.logger.exception(
            "Stripe checkout failed for user_id=%s",
            current_user.id,
        )
        return jsonify({
            "error": "Could not start checkout. Please try again."
        }), 502

    except Exception:
        db.session.rollback()
        app.logger.exception(
            "Checkout failed for user_id=%s",
            current_user.id,
        )
        return jsonify({
            "error": "Could not start checkout. Please try again."
        }), 500


@app.route("/subscription-state", methods=["GET"])
@login_required
@limiter.limit("30 per minute")
def subscription_state():
    if not stripe.api_key:
        return jsonify({
            "error": "Billing is temporarily unavailable."
        }), 503

    try:
        return jsonify(subscription_management_payload(current_user))

    except stripe.error.StripeError:
        app.logger.exception(
            "Could not load Stripe subscription state for user_id=%s",
            current_user.id,
        )
        return jsonify({
            "error": "Could not load billing details. Please try again."
        }), 502


@app.route("/billing-portal", methods=["POST"])
@login_required
@limiter.limit("10 per minute")
def billing_portal():
    if not current_user.stripe_customer_id:
        return jsonify({"error": "No billing customer found."}), 400

    payload = request.get_json(silent=True) or {}
    action = str(payload.get("action") or "manage").strip().lower()

    if action not in {"manage", "switch", "cancel", "resume"}:
        return jsonify({"error": "Unsupported billing action."}), 400

    try:
        session_parameters = {
            "customer": current_user.stripe_customer_id,
            "return_url": f"{BASE_URL}/subscription",
        }

        if STRIPE_BILLING_PORTAL_CONFIGURATION_ID:
            session_parameters["configuration"] = (
                STRIPE_BILLING_PORTAL_CONFIGURATION_ID
            )

        if action == "manage":
            portal_session = stripe.billing_portal.Session.create(
                **session_parameters
            )
            return jsonify({"url": portal_session.url})

        subscription = get_current_user_pro_subscription(current_user)

        if not subscription:
            return jsonify({
                "error": "No active Pro subscription was found."
            }), 409

        subscription_id = subscription.get("id")
        item, current_interval = pro_subscription_item(subscription)

        if action == "resume":
            if not subscription_cancellation_pending(subscription):
                return jsonify({
                    "message": "Your Pro subscription is already active.",
                    "state": subscription_management_payload(
                        current_user,
                        subscription,
                    ),
                })

            updated_subscription = stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=False,
            )
            updated_subscription = stripe_to_dict(updated_subscription)

            return jsonify({
                "message": "Your Pro subscription will continue.",
                "state": subscription_management_payload(
                    current_user,
                    updated_subscription,
                ),
            })

        if not STRIPE_BILLING_PORTAL_CONFIGURATION_ID:
            return jsonify({
                "error": (
                    "Plan switching is not configured yet. "
                    "Open Manage Billing or contact support."
                ),
                "configuration_required": True,
            }), 503

        if action == "cancel":
            session_parameters["flow_data"] = {
                "type": "subscription_cancel",
                "subscription_cancel": {
                    "subscription": subscription_id,
                },
                "after_completion": {
                    "type": "redirect",
                    "redirect": {
                        "return_url": (
                            f"{BASE_URL}/subscription?billing=cancelled"
                        ),
                    },
                },
            }

        if action == "switch":
            target_interval = str(
                payload.get("billing_interval") or ""
            ).strip().lower()

            if target_interval not in VALID_PRO_BILLING_INTERVALS:
                return jsonify({
                    "error": "Choose either monthly or annual billing."
                }), 400

            target_price_id = get_pro_price_id_for_interval(target_interval)

            if not target_price_id:
                return jsonify({
                    "error": "That billing option is temporarily unavailable."
                }), 503

            if not item or not item.get("id"):
                return jsonify({
                    "error": "The Pro subscription item could not be identified."
                }), 409

            if current_interval == target_interval:
                if subscription_cancellation_pending(subscription):
                    return jsonify({
                        "error": (
                            "This plan is scheduled to end. Keep Pro first, "
                            "then choose a different billing schedule."
                        ),
                        "resume_required": True,
                    }), 409

                return jsonify({
                    "message": "That is already your current billing plan.",
                    "no_change": True,
                })

            session_parameters["flow_data"] = {
                "type": "subscription_update_confirm",
                "subscription_update_confirm": {
                    "subscription": subscription_id,
                    "items": [
                        {
                            "id": item.get("id"),
                            "price": target_price_id,
                            "quantity": int(item.get("quantity") or 1),
                        }
                    ],
                },
                "after_completion": {
                    "type": "redirect",
                    "redirect": {
                        "return_url": (
                            f"{BASE_URL}/subscription?billing=updated"
                        ),
                    },
                },
            }

        portal_session = stripe.billing_portal.Session.create(
            **session_parameters
        )

        return jsonify({
            "url": portal_session.url,
            "action": action,
        })

    except stripe.error.StripeError:
        app.logger.exception(
            "Stripe billing portal failed for user_id=%s action=%s",
            current_user.id,
            action,
        )
        return jsonify({
            "error": "Could not open billing management. Please try again."
        }), 502

    except Exception:
        app.logger.exception(
            "Billing management failed for user_id=%s action=%s",
            current_user.id,
            action,
        )
        return jsonify({
            "error": "Could not manage billing. Please try again."
        }), 500

def stripe_api_key_livemode(api_key):
    """Return True for live keys, False for test keys, or None if unknown."""
    key = str(api_key or "")

    if key.startswith(("sk_live_", "rk_live_")):
        return True

    if key.startswith(("sk_test_", "rk_test_")):
        return False

    return None


STRIPE_EXPECTED_LIVEMODE = stripe_api_key_livemode(stripe.api_key)
STRIPE_WEBHOOK_PROCESSING_LEASE = timedelta(minutes=5)
STRIPE_HANDLED_WEBHOOK_EVENTS = {
    "checkout.session.completed",
    "checkout.session.expired",
    "customer.subscription.created",
    "customer.subscription.updated",
    "customer.subscription.deleted",
}


def stripe_id(value):
    """Return an ID whether Stripe supplied a string or an expanded object."""
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return value.get("id")

    if value is None:
        return None

    return getattr(value, "id", None)


def stripe_timestamp_to_datetime(value):
    try:
        return datetime.utcfromtimestamp(int(value))
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def stripe_object_matches_expected_mode(obj):
    obj = stripe_to_dict(obj)
    object_livemode = obj.get("livemode")

    if object_livemode is None or STRIPE_EXPECTED_LIVEMODE is None:
        return True

    return bool(object_livemode) == STRIPE_EXPECTED_LIVEMODE


def find_user_for_stripe_event(customer_id=None, metadata=None, client_reference_id=None):
    """
    Resolve a Stripe event only through immutable identifiers.

    Email fallback is intentionally not used because an address can be deleted
    and later registered by a different local account.
    """
    metadata = metadata or {}
    user_by_customer = None
    user_by_reference = None

    if customer_id:
        user_by_customer = User.query.filter_by(
            stripe_customer_id=customer_id
        ).first()

    reference_id = metadata.get("user_id") or client_reference_id

    if reference_id:
        try:
            user_by_reference = User.query.get(int(reference_id))
        except (TypeError, ValueError):
            app.logger.warning(
                "Stripe event contained an invalid user reference: %r",
                reference_id,
            )

    if (
        user_by_customer
        and user_by_reference
        and user_by_customer.id != user_by_reference.id
    ):
        app.logger.error(
            "Stripe event identity conflict: customer_id=%s customer_user_id=%s metadata_user_id=%s",
            customer_id,
            user_by_customer.id,
            user_by_reference.id,
        )
        return None

    user = user_by_customer or user_by_reference

    if not user:
        return None

    if (
        customer_id
        and user.stripe_customer_id
        and user.stripe_customer_id != customer_id
    ):
        app.logger.error(
            "Stripe customer mismatch for user_id=%s saved=%s event=%s",
            user.id,
            user.stripe_customer_id,
            customer_id,
        )
        return None

    return user


def list_customer_pro_subscriptions(customer_id):
    subscriptions = stripe.Subscription.list(
        customer=customer_id,
        status="all",
        limit=100,
    )
    subscriptions = stripe_to_dict(subscriptions)

    return [
        subscription
        for subscription in subscriptions.get("data", [])
        if (
            stripe_object_matches_expected_mode(subscription)
            and stripe_subscription_uses_pro_price(subscription)
        )
    ]


def select_authoritative_pro_subscription(subscriptions):
    """Choose the subscription that should define the current access state."""
    status_rank = {
        "active": 100,
        "trialing": 90,
        "past_due": 80,
        "unpaid": 70,
        "paused": 60,
        "incomplete": 50,
        "canceled": 20,
        "incomplete_expired": 10,
    }

    if not subscriptions:
        return None

    return max(
        subscriptions,
        key=lambda subscription: (
            status_rank.get(subscription.get("status"), 0),
            int(subscription.get("created") or 0),
        ),
    )


def update_user_stripe_event_audit(user, event):
    event_created_at = stripe_timestamp_to_datetime(event.get("created"))
    saved_created_at = getattr(user, "stripe_last_event_created_at", None)

    # Keep the greatest Stripe event timestamp for audit purposes. We still
    # reconcile against Stripe's current subscription state even for an older
    # delivery, so out-of-order events cannot restore stale access.
    if (
        event_created_at
        and (saved_created_at is None or event_created_at >= saved_created_at)
    ):
        user.stripe_last_event_created_at = event_created_at
        user.stripe_last_event_id = event.get("id")


def sync_user_pro_access_from_stripe(user, customer_id, event, fallback_status="inactive"):
    """
    Reconcile access from Stripe's current state instead of trusting event order.
    """
    if user.stripe_customer_id and user.stripe_customer_id != customer_id:
        raise RuntimeError("Stripe customer does not match the local user.")

    subscriptions = list_customer_pro_subscriptions(customer_id)
    selected = select_authoritative_pro_subscription(subscriptions)

    user.stripe_customer_id = customer_id
    user.subscription_updated_at = datetime.utcnow()
    update_user_stripe_event_audit(user, event)

    if selected:
        status = selected.get("status") or fallback_status
        user.stripe_subscription_id = selected.get("id")
        user.subscription_status = status
        user.subscription_type = (
            "pro" if status in PAID_SUBSCRIPTION_STATUSES else "free"
        )
    else:
        user.stripe_subscription_id = None
        user.subscription_type = "free"
        user.subscription_status = fallback_status or "inactive"

    db.session.commit()

    if not is_paid_user(user):
        freeze_locked_live_simulations_for_user(user)
        enforce_free_live_simulation_limit_for_user(user)

    return selected


def begin_stripe_webhook_event(event):
    """
    Claim a Stripe event for processing.

    Processed/ignored events are acknowledged immediately. Failed events may be
    retried. A short lease lets a later retry recover if a worker crashed while
    the event was marked as processing.
    """
    event_id = event.get("id")

    if not event_id:
        raise ValueError("Stripe event is missing its ID.")

    now = datetime.utcnow()
    existing = StripeWebhookEvent.query.filter_by(
        stripe_event_id=event_id
    ).first()

    if existing:
        if existing.processing_status in {"processed", "ignored"}:
            return existing, "duplicate"

        lease_is_active = (
            existing.processing_status == "processing"
            and existing.processing_started_at
            and now - existing.processing_started_at < STRIPE_WEBHOOK_PROCESSING_LEASE
        )

        if lease_is_active:
            return existing, "processing"

        existing.processing_status = "processing"
        existing.processing_started_at = now
        existing.processed_at = None
        existing.error_message = None
        db.session.commit()
        return existing, "claimed"

    data_object = ((event.get("data") or {}).get("object") or {})
    data_object = stripe_to_dict(data_object)

    record = StripeWebhookEvent(
        stripe_event_id=event_id,
        event_type=event.get("type") or "unknown",
        stripe_object_id=stripe_id(data_object.get("id")),
        livemode=bool(event.get("livemode")),
        event_created_at=stripe_timestamp_to_datetime(event.get("created")),
        processing_status="processing",
        processing_started_at=now,
    )
    db.session.add(record)

    try:
        db.session.commit()
        return record, "claimed"
    except IntegrityError:
        # A concurrent delivery inserted the same event first.
        db.session.rollback()
        existing = StripeWebhookEvent.query.filter_by(
            stripe_event_id=event_id
        ).first()

        if existing and existing.processing_status in {"processed", "ignored"}:
            return existing, "duplicate"

        return existing, "processing"


def finish_stripe_webhook_event(record, status, error_message=None):
    record.processing_status = status
    record.processed_at = datetime.utcnow()
    record.error_message = (
        str(error_message)[:1000] if error_message else None
    )
    db.session.commit()


@app.route("/stripe/webhook", methods=["POST"])
@limiter.exempt
@csrf.exempt
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature")

    if (
        not stripe.api_key
        or not configured_pro_price_ids()
        or not STRIPE_WEBHOOK_SECRET
    ):
        app.logger.error(
            "Stripe webhook configuration is incomplete: api_key=%s monthly_price=%s annual_price=%s webhook_secret=%s",
            bool(stripe.api_key),
            bool(STRIPE_PRO_MONTHLY_PRICE_ID),
            bool(STRIPE_PRO_ANNUAL_PRICE_ID),
            bool(STRIPE_WEBHOOK_SECRET),
        )
        return "Webhook unavailable", 503

    try:
        event = stripe.Webhook.construct_event(
            payload,
            sig_header,
            STRIPE_WEBHOOK_SECRET,
        )
        event = stripe_to_dict(event)

    except ValueError:
        app.logger.warning("Stripe webhook rejected an invalid payload.")
        return "Invalid payload", 400

    except stripe.error.SignatureVerificationError:
        app.logger.warning("Stripe webhook rejected an invalid signature.")
        return "Invalid signature", 400

    except Exception:
        app.logger.exception("Stripe webhook construction failed.")
        return "Webhook construction failed", 400

    if STRIPE_EXPECTED_LIVEMODE is None:
        app.logger.error(
            "Stripe key mode could not be determined from the configured API key."
        )
        return "Stripe environment is not configured safely", 503

    if bool(event.get("livemode")) != STRIPE_EXPECTED_LIVEMODE:
        app.logger.error(
            "Stripe webhook mode mismatch: event_id=%s event_livemode=%s expected_livemode=%s",
            event.get("id"),
            event.get("livemode"),
            STRIPE_EXPECTED_LIVEMODE,
        )
        return "Stripe environment mismatch", 400

    try:
        record, claim_status = begin_stripe_webhook_event(event)
    except Exception:
        db.session.rollback()
        app.logger.exception(
            "Could not record Stripe webhook event_id=%s",
            event.get("id"),
        )
        return "Webhook event recording failed", 500

    if claim_status == "duplicate":
        return "Already processed", 200

    if claim_status == "processing":
        # A non-2xx response asks Stripe to retry later if the current worker
        # crashes before completing the event.
        return "Event is already processing", 409

    event_type = event.get("type")
    data_object = stripe_to_dict(((event.get("data") or {}).get("object") or {}))

    try:
        if event_type not in STRIPE_HANDLED_WEBHOOK_EVENTS:
            finish_stripe_webhook_event(record, "ignored")
            return "Ignored", 200

        if event_type == "checkout.session.expired":
            customer_id = stripe_id(data_object.get("customer"))
            metadata = data_object.get("metadata") or {}
            user = find_user_for_stripe_event(
                customer_id=customer_id,
                metadata=metadata,
                client_reference_id=data_object.get("client_reference_id"),
            )

            if user and user.pending_checkout_session_id == data_object.get("id"):
                user.pending_checkout_session_id = None
                update_user_stripe_event_audit(user, event)
                db.session.commit()

            finish_stripe_webhook_event(record, "processed")
            return "OK", 200

        if event_type == "checkout.session.completed":
            customer_id = stripe_id(data_object.get("customer"))
            subscription_id = stripe_id(data_object.get("subscription"))
            metadata = data_object.get("metadata") or {}

            if (
                data_object.get("mode") != "subscription"
                or data_object.get("status") != "complete"
                or data_object.get("payment_status") not in {"paid", "no_payment_required"}
                or not customer_id
                or not subscription_id
            ):
                finish_stripe_webhook_event(
                    record,
                    "ignored",
                    "Checkout Session did not represent a completed subscription payment.",
                )
                return "Ignored", 200

            user = find_user_for_stripe_event(
                customer_id=customer_id,
                metadata=metadata,
                client_reference_id=data_object.get("client_reference_id"),
            )

            if not user:
                finish_stripe_webhook_event(
                    record,
                    "ignored",
                    "No unambiguous local user matched the Checkout Session.",
                )
                return "Ignored", 200

            if (
                user.pending_checkout_session_id
                and user.pending_checkout_session_id != data_object.get("id")
            ):
                finish_stripe_webhook_event(
                    record,
                    "ignored",
                    "Checkout Session did not match the user's pending Session.",
                )
                return "Ignored", 200

            subscription = stripe.Subscription.retrieve(subscription_id)
            subscription = stripe_to_dict(subscription)

            if (
                stripe_id(subscription.get("customer")) != customer_id
                or not stripe_object_matches_expected_mode(subscription)
                or not stripe_subscription_uses_pro_price(subscription)
            ):
                finish_stripe_webhook_event(
                    record,
                    "ignored",
                    "Subscription customer, environment, or Pro Price validation failed.",
                )
                return "Ignored", 200

            user.pending_checkout_session_id = None
            db.session.commit()
            sync_user_pro_access_from_stripe(
                user,
                customer_id,
                event,
                fallback_status=subscription.get("status") or "inactive",
            )

            finish_stripe_webhook_event(record, "processed")
            return "OK", 200

        # Subscription lifecycle events are accepted only when they refer to
        # the configured NeuralTrend Pro Price. We then retrieve the customer's
        # current Stripe state, which makes event delivery order irrelevant.
        customer_id = stripe_id(data_object.get("customer"))
        subscription_id = stripe_id(data_object.get("id"))
        metadata = data_object.get("metadata") or {}

        if (
            not customer_id
            or not subscription_id
            or not stripe_object_matches_expected_mode(data_object)
            or not stripe_subscription_uses_pro_price(data_object)
        ):
            finish_stripe_webhook_event(
                record,
                "ignored",
                "Subscription event did not match a configured Pro Price or environment.",
            )
            return "Ignored", 200

        user = find_user_for_stripe_event(
            customer_id=customer_id,
            metadata=metadata,
        )

        if not user:
            finish_stripe_webhook_event(
                record,
                "ignored",
                "No unambiguous local user matched the subscription event.",
            )
            return "Ignored", 200

        sync_user_pro_access_from_stripe(
            user,
            customer_id,
            event,
            fallback_status=data_object.get("status") or "inactive",
        )

        finish_stripe_webhook_event(record, "processed")
        return "OK", 200

    except Exception as error:
        db.session.rollback()
        app.logger.exception(
            "Stripe webhook processing failed: event_id=%s event_type=%s",
            event.get("id"),
            event_type,
        )

        try:
            fresh_record = StripeWebhookEvent.query.filter_by(
                stripe_event_id=event.get("id")
            ).first()

            if fresh_record:
                finish_stripe_webhook_event(
                    fresh_record,
                    "failed",
                    error,
                )
        except Exception:
            db.session.rollback()
            app.logger.exception(
                "Could not mark Stripe event as failed: event_id=%s",
                event.get("id"),
            )

        return "Webhook processing failed", 500

