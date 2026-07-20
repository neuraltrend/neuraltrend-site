from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    Response,
    session,
    redirect,
    url_for,
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
from flask_wtf.csrf import CSRFError
from sqlalchemy.exc import IntegrityError

app = Flask(__name__)

# ✅ Fix proxy handling (Render-safe)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

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

# Stripe
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_PRO_PRICE_ID = os.environ.get("STRIPE_PRO_PRICE_ID")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")

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
            "/resend-verification",
            "/request-password-reset",
            "/request-delete-account",
            "/live-simulations",
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
        print("Verification email sent to:", user_email)
    except Exception as e:
        print("EMAIL ERROR (verify):", str(e))

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
    delta = parse_duration(duration_str)

    if isinstance(delta, timedelta):
        return delta.days

    # The dashboard uses calendar slicing; these values are bounded approximations
    # used only to select a window from local market CSVs.
    return delta.years * 365 + delta.months * 30 + delta.days


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

cache = {}  # simple in-memory cache per ticker

SUPPORTED_TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'NVDA', 'AAPL', 'GOOGL', 'MSFT', "1INCH-USD", "3ULL-USD", "AAVE-USD","ABBV", "ACE-USD",
               "ACH-USD", "ADA-USD", "AERO-USD", "AEVO-USD", "AGI-USD", "AIOZ-USD", "AIT-USD", "AIXBT-USD", "AKT-USD", "ALEPH-USD",
               "ALGO-USD", "ALI-USD", "ALPH-USD", "ALT-USD", "ALU-USD", "ALVA-USD", "AMP-USD", 'AMZN', "ANKR-USD", "ANON-USD", "ANYONE-USD", "APT-USD",
               "APU-USD", "AR-USD", "ARB-USD", "ARC-USD", "ASML", "ASTR-USD", "ATLAS-USD", "ATOM-USD", "AURY-USD", "AUTOS-USD", "AVAX-USD", 'AVGO', 
               "AXL-USD", "AXS-USD", "BAI-USD", "BAL-USD", "BAND-USD", "BANANA-USD", "BASEDAI-USD", "BAZED-USD",
               "BCUT-USD", "BEAM-USD", "BGB-USD", "BIGTIME-USD", "BLUR-USD", "BNB-USD", "BNT-USD", "BONK-USD", "BRETT-USD", 'BRKB', 
               "BYTES-USD", "CAKE-USD", "CELO-USD", "CERE-USD", "CETUS-USD", "CFG-USD", "CGPT-USD", "CHAPZ-USD", "CHAT-USD", "CHEX-USD", "CHZ-USD", 
               "COMP-USD", "COST", "COTI-USD", "CPOOL-USD", "CREDI-USD", "CREO-USD", "CRO-USD", "CROWN-USD", "CRU-USD", "CRV-USD", "CTC-USD", "CVC-USD",
               "DARK-USD", "DCK-USD", "DEVVE-USD", "DEXE-USD", "DIMO-USD", "DIO-USD", "DOGE-USD", "DOME-USD", "DOMI-USD", "DOT-USD", "DRIFT-USD", 
               "DSYNC-USD", "DYDX-USD", "DYM-USD", "EDU-USD", "ENA-USD", "ENJ-USD", "ENQAI-USD", "FAR-USD", "FET-USD", "FIDA-USD", 
               "FIL-USD", "FLIP-USD", "FLOW-USD", "FLR-USD", "FLUX-USD", "FOXY-USD", "FUELX-USD", "FYN-USD", "GEAR-USD", "GFAL-USD",
               "GHX-USD", "GLQ-USD", "GMEE-USD", "GMRX-USD", "GMT-USD", "GMX-USD", "GODS-USD", "GPU-USD", "GRIFFAIN-USD", "GRT-USD",
               "GSWIFT-USD", "GTAI-USD", "GTC-USD", "HASHAI-USD", "HBAR-USD", "HEART-USD", "HELLO-USD", "HNT-USD", "HONEY-USD",
               "HXD-USD", "HYPC-USD", "HYPE-USD", "IAG-USD", "ICP-USD", "ILV-USD", "IMX-USD", "INJ-USD", "INSP-USD", "IOTX-USD", "IVPAY-USD",
               "JASMY-USD", "JNJ", "JOE-USD", "JST-USD", "JTO-USD", "JUP-USD", "KARATE-USD", "KARRAT-USD", "KAS-USD", "KATA-USD", "KOMPETE-USD",
               "KRL-USD", "LAI-USD", "LEO-USD", "LFNTY-USD", "LIKE-USD", "LINK-USD", "LMWR-USD", "LPT-USD", "LRC-USD", "LTC-USD", "MA",
               "MAGIC-USD", "MASK-USD", "MAVIA-USD", "MBS-USD", 'META', "METIS-USD", "MEW-USD", "MINA-USD", "ML-USD", "MLN-USD", "MNDE-USD",
               "MNT-USD", "MOODENG-USD", "MPLX-USD", "MU", "MUBI-USD", "MXM-USD", "MYRIA-USD", "MYRO-USD", "NAKA-USD", 
               "NEAR-USD", "NEON-USD", "NEURAL-USD", "NEXO-USD", "NMT-USD", "NOS-USD", "NTRN-USD", "NU-USD", "NXRA-USD", "OGN-USD", "OKB-USD",
               "OLAS-USD", "OMG-USD", "ONDO-USD", "OP-USD", "ORAI-USD", "ORCA-USD", "ORCL", "ORDI-USD", "OTK-USD", "OXT-USD", "PAAL-USD", 
               "PAID-USD", "PANDORA-USD", "PDA-USD", "PENDLE-USD", "PENG-USD", "PENGU-USD", "PEPE-USD", "PERP-USD", "PHA-USD", 
               "PIN-USD", "PIXEL-USD", "POL-USD", "POLS-USD", "POLYX-USD", "PORTAL-USD", "PRIME-USD", "PROPC-USD", "PYR-USD", 
               "PYTH-USD", "QANX-USD", "QI-USD", "QNT-USD", "RAY-USD", "RARE-USD", "RARI-USD", "RDT-USD", "REN-USD", "RENDER-USD", 
               "REQ-USD", "RIO-USD", "RLB-USD", "RMRK-USD", "RON-USD", "ROOT-USD", "RSC-USD", "RSR-USD", "RSS3-USD",
               "RUNE-USD", "SAFE-USD", "SC-USD", "SEI-USD", "SENATE-USD", "SERSH-USD", "SHDW-USD", "SHIB-USD", 
               "SHIDO-USD", "SHRAP-USD", "SIDUS-USD", "SIPHER-USD", "SKL-USD", "SNS-USD", "SPEC-USD", "SPELL-USD", "SRM-USD", "SSV-USD", 
               "STEP-USD", "STG-USD", "STORJ-USD", "STRK-USD", "SUI-USD", "SUNDOG-USD", "SUPER-USD", "TAI-USD", "TAO-USD", 'TCEHY', 
               "TET-USD", "TFUEL-USD", "THETA-USD", "TLOS-USD", "TON-USD", "TRAC-USD", "TRIAS-USD", "TRU-USD", 'TSLA', 'TSM', "TURBO-USD",
               "UNI-USD", "UNIBOT-USD", "UOS-USD", 'V', "VAI-USD", "VET-USD", "VIA-USD", "VIRTUAL-USD", "VOO", "VR-USD", "VRA-USD",
               "WAXP-USD", "WHALES-USD", "WIF-USD", "WIFI-USD", "WILD-USD", "WINR-USD", "WLD-USD", "WMTX-USD", "XAI-USD", 
               "XCAD-USD", "XLM-USD", "XMR-USD", "XOM", "XTZ-USD", "XYO-USD", "YGG-USD", "ZBCN-USD", "ZEN-USD", "ZEREBRO-USD", "ZETA-USD", 
               "ZIG-USD", "ZKJ-USD", "ZRX-USD"]

TOP_FREE_TICKERS = {"BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"}
ADMIN_ONLY_TICKERS = {"GOOGL", "MU", "TSM", "ASML", "JNJ", "AVGO", "AMZN", "BRKB"}
ALL_SUPPORTED_TICKERS = frozenset(
    str(ticker).strip().upper()
    for ticker in [*SUPPORTED_TICKERS, *ADMIN_ONLY_TICKERS]
)

FREE_LIVE_SIMULATION_LIMIT = 5
PAID_LIVE_SIMULATION_LIMIT = 100

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


def is_admin_only_ticker(ticker):
    return normalize_ticker(ticker) in ADMIN_ONLY_TICKERS

def can_user_see_ticker(user, ticker):
    ticker = normalize_ticker(ticker)

    if ticker not in ALL_SUPPORTED_TICKERS:
        return False

    if is_admin_only_ticker(ticker) and not is_admin_user(user):
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

    return [
        ticker for ticker in SUPPORTED_TICKERS
        if not is_admin_only_ticker(ticker)
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

    if not isinstance(period_days, int) or period_days < 1 or period_days > 3650:
        raise ValueError("Unsupported signal period.")

    effective_csv_version = csv_version if csv_version is not None else get_csv_version()
    cache_key = (ticker, period_days, effective_csv_version)
    if cache_key in cache:
        return cache[cache_key]

    # -------------------------------
    # Load validated full data first
    # -------------------------------
    df = load_epoch_csv_for_ticker(ticker)

    if len(df) < 2:
        print(f"{ticker}: not enough data")
        return None

    # -------------------------------
    # Determine asset type
    # -------------------------------
    is_crypto = ticker.endswith("-USD")

    if is_crypto:
        # calendar slicing
        start_date = datetime.today().date() - pd.Timedelta(days=period_days)
        df = df[df.index >= pd.to_datetime(start_date)].copy()
        transaction_cost = 0.01 # 1% per transaction (per side)
    else:
        # stock → use trading days
        trading_days_per_year = 252
        trading_days = int(period_days * (trading_days_per_year / 365))
        df = df.tail(trading_days).copy()
        transaction_cost = 0.001 # 0.1% per transaction (per side)

    if len(df) < 2:
        return None

    # -------------------------------
    # Buy & Hold return
    # -------------------------------
    bh_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) * (1 - transaction_cost) ** 2

    # -------------------------------
    # Strategy return (cash-based)
    # -------------------------------
    cash = 1.0
    shares = 0.0

    for i in range(len(df)):
        sig = df['epoch_signal'].iloc[i]
        price = df['Close'].iloc[i]

        if sig == 1 and shares == 0:
            shares = cash / price
            shares *= (1 - transaction_cost)
            cash = 0

        elif sig == -1 and shares > 0:
            cash = shares * price
            cash *= (1 - transaction_cost)
            shares = 0

    # Final liquidation
    if shares > 0:
        cash = shares * df['Close'].iloc[-1]
        cash *= (1 - transaction_cost)

    strategy_return = cash

    # -------------------------------
    # Outperformance (relative multiple)
    # -------------------------------
    if bh_return and bh_return != 0:
        outperformance = strategy_return / bh_return
    else:
        outperformance = None

    output = {
        'today': int(df['epoch_signal'].iloc[-1]),
        'yesterday': int(df['epoch_signal'].iloc[-2]) if len(df) >= 2 else int(df['epoch_signal'].iloc[-1]),
        'last_week': int(df['epoch_signal'].iloc[-8]) if len(df) >= 8 else int(df['epoch_signal'].iloc[-1]),
        'last_month': int(df['epoch_signal'].iloc[-31]) if len(df) >= 31 else int(df['epoch_signal'].iloc[-1]),
        'buy_hold_annual_return': bh_return - 1,
        'strategy_annual_return': strategy_return - 1,
        'outperformance': outperformance
    }

    cache[cache_key] = output
    return output

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

def live_simulation_summary(sim):
    latest = get_latest_equity_point(sim.id)
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

    outperformance = (
        strategy_value / benchmark_value
        if benchmark_value and benchmark_value != 0 else None
    )

    trade_count = LiveSimulationTrade.query.filter_by(
        simulation_id=sim.id
    ).count()

    data = sim.to_dict()
    data.update({
        "latest_strategy_value": strategy_value,
        "latest_benchmark_value": benchmark_value,
        "strategy_return": strategy_return,
        "benchmark_return": benchmark_return,
        "outperformance": outperformance,
        "trade_count": trade_count,
        "latest_equity_date": latest.equity_date.isoformat() if latest else None,
        "latest_signal": int(latest.signal) if latest else None,
        "latest_close_price": float(latest.close_price) if latest else None,
        "latest_csv_date": latest_csv_date.isoformat() if latest_csv_date else None,
        "is_current_with_csv": (
            sim.last_processed_date == latest_csv_date
            if sim.last_processed_date and latest_csv_date else False
        ),
        "horizon_returns": build_live_sim_horizon_returns(sim),
    })

    return data

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

    summary = live_simulation_summary(sim)

    summary.update({
        "dates": [p.equity_date.isoformat() for p in equity_points],
        "strategy_curve": [p.strategy_value for p in equity_points],
        "benchmark_curve": [p.benchmark_value for p in equity_points],
        "signals": [p.signal for p in equity_points],
        "close_prices": [p.close_price for p in equity_points],
        "trades": [t.to_dict() for t in trades],
    })

    return summary

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

def update_live_simulation_from_csv(sim, user=None):
    """
    Reads fresh CSV rows and updates one simulation from its last processed date.

    Logic:
    - If never processed, process rows from start_date onward.
    - If already processed, process only rows after last_processed_date.
    - BUY: invest selected % of available cash.
    - SELL: sell selected % of current position.
    - HOLD: do nothing.
    """
    if sim.status != "active":
        return sim

    owner = user or getattr(sim, "user", None)

    if owner is None and getattr(sim, "user_id", None):
        owner = User.query.get(sim.user_id)

    if owner is not None and not can_access_live_sim_ticker_for_user(owner, sim.ticker):
        return sim

    df = load_epoch_csv_for_ticker(sim.ticker)

    if sim.last_processed_date:
        new_rows = df[df.index.date > sim.last_processed_date]
    else:
        new_rows = df[df.index.date >= sim.start_date]

    if new_rows.empty:
        return sim

    position_fraction = sim.position_size_pct / 100.0
    transaction_cost_rate = sim.transaction_cost_rate

    for date_index, row in new_rows.iterrows():
        equity_date = date_index.date()
        price = float(row["Close"])
        signal = int(row["epoch_signal"])

        trade_executed = False

        # --------------------
        # BUY
        # --------------------
        if signal == 1 and sim.cash_balance > 0:
            cash_allocation = sim.cash_balance * position_fraction

            # Treat cash_allocation as total cash used, including transaction cost.
            gross_buy_budget = cash_allocation / (1 + transaction_cost_rate)
            raw_quantity = gross_buy_budget / price

            quantity = normalize_live_quantity_for_buy(
                ticker=sim.ticker,
                raw_quantity=raw_quantity,
                price=price,
                cash_allocation=cash_allocation,
                transaction_cost_rate=transaction_cost_rate
            )

            if quantity > 0:
                gross_amount = quantity * price
                transaction_cost = gross_amount * transaction_cost_rate
                total_cash_used = gross_amount + transaction_cost

                if total_cash_used <= sim.cash_balance + 1e-9:
                    sim.position_quantity += quantity
                    sim.cash_balance -= total_cash_used

                    db.session.add(LiveSimulationTrade(
                        simulation_id=sim.id,
                        trade_date=equity_date,
                        ticker=sim.ticker,
                        signal=1,
                        price=price,
                        quantity=quantity,
                        gross_amount=gross_amount,
                        transaction_cost=transaction_cost,
                        cash_after=sim.cash_balance,
                        position_after=sim.position_quantity
                    ))

                    trade_executed = True

        # --------------------
        # SELL
        # --------------------
        elif signal == -1 and sim.position_quantity > 0:
            raw_quantity = sim.position_quantity * position_fraction

            quantity = normalize_live_quantity_for_sell(
                ticker=sim.ticker,
                raw_quantity=raw_quantity,
                current_position=sim.position_quantity
            )

            if quantity > 0:
                gross_amount = quantity * price
                transaction_cost = gross_amount * transaction_cost_rate
                net_cash_received = gross_amount - transaction_cost

                sim.position_quantity -= quantity

                if sim.position_quantity < 1e-12:
                    sim.position_quantity = 0.0

                sim.cash_balance += net_cash_received

                db.session.add(LiveSimulationTrade(
                    simulation_id=sim.id,
                    trade_date=equity_date,
                    ticker=sim.ticker,
                    signal=-1,
                    price=price,
                    quantity=quantity,
                    gross_amount=gross_amount,
                    transaction_cost=transaction_cost,
                    cash_after=sim.cash_balance,
                    position_after=sim.position_quantity
                ))

                trade_executed = True

        # --------------------
        # Daily equity point
        # --------------------
        strategy_value = sim.cash_balance + (sim.position_quantity * price)
        benchmark_value = sim.benchmark_quantity * price

        existing_point = LiveSimulationEquity.query.filter_by(
            simulation_id=sim.id,
            equity_date=equity_date
        ).first()

        if existing_point:
            existing_point.signal = signal
            existing_point.close_price = price
            existing_point.cash_balance = sim.cash_balance
            existing_point.position_quantity = sim.position_quantity
            existing_point.strategy_value = strategy_value
            existing_point.benchmark_value = benchmark_value
        else:
            db.session.add(LiveSimulationEquity(
                simulation_id=sim.id,
                equity_date=equity_date,
                ticker=sim.ticker,
                signal=signal,
                close_price=price,
                cash_balance=sim.cash_balance,
                position_quantity=sim.position_quantity,
                strategy_value=strategy_value,
                benchmark_value=benchmark_value
            ))

        sim.last_processed_date = equity_date

    db.session.commit()
    return sim

LIVE_SIM_HORIZON_DAYS = {
    "1d": 1,
    "1w": 7,
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "since_start": None,
}

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


def build_live_sim_horizon_returns(sim):
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

        alpha_return = None
        if strategy_return is not None and benchmark_return is not None:
            alpha_return = strategy_return - benchmark_return

        alpha_change = strategy_change - benchmark_change

        output[horizon_key] = {
            "base_date": base_date,
            "latest_date": latest_point.equity_date.isoformat() if latest_point.equity_date else None,

            "strategy_return": strategy_return,
            "benchmark_return": benchmark_return,
            "alpha_return": alpha_return,

            "strategy_change": strategy_change,
            "benchmark_change": benchmark_change,
            "alpha_change": alpha_change,

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
    except Exception as e:
        print(f"Could not read latest CSV date for {ticker}:", str(e))
        return None

# --------------------
# Routes
# --------------------

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

@app.route("/me")
def me():
    if current_user.is_authenticated:
        return jsonify({
            "email": current_user.email,
            "subscription_type": current_user.subscription_type,
            "subscription_status": current_user.subscription_status,
            "is_paid": is_paid_user(current_user),
            "is_admin": is_admin_user(current_user),
            "live_simulation_limit": get_live_simulation_limit_for_user(current_user)
        })

    return jsonify({
        "email": None,
        "subscription_type": "anonymous",
        "subscription_status": "none",
        "is_paid": False,
        "live_simulation_limit": 0
    })

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

    for sim in sims:
        if not live_sim_can_update_for_user(current_user, sim):
            continue

        try:
            update_live_simulation_from_csv(sim, current_user)
        except Exception as e:
            print(f"Live simulation update error for sim {sim.id}:", str(e))

    sims = query.order_by(LiveSimulation.created_at.desc()).all()

    active_count = visible_live_simulation_query().filter_by(status="active").count()
    paused_count = visible_live_simulation_query().filter_by(status="paused").count()
    archived_count = visible_live_simulation_query().filter_by(status="archived").count()

    open_count = active_count + paused_count
    all_count = open_count + archived_count

    user_limit = get_live_simulation_limit_for_user(current_user)

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
        "simulations": [live_simulation_summary(sim) for sim in sims],
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
            data.get("initial_cash", 10000),
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
    # invest initial cash at the start date, including entry transaction cost.
    benchmark_gross_budget = initial_cash / (1 + transaction_cost_rate)
    
    if is_crypto_ticker(ticker):
        benchmark_quantity = benchmark_gross_budget / latest_price
    else:
        # Stocks: whole-share benchmark only.
        benchmark_quantity = float(int(benchmark_gross_budget / latest_price))
    
        if benchmark_quantity < 1:
            return jsonify({
                "error": "Initial cash is too small to buy at least one whole share for the buy-and-hold benchmark."
            }), 400

    if not name:
        name = f"{ticker} {position_size_pct:.0f}% Live Simulation"

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
        start_date=latest_date,
        last_processed_date=None,
        status="active"
    )

    db.session.add(sim)
    db.session.commit()

    try:
        update_live_simulation_from_csv(sim)
    except Exception as e:
        print("Live simulation initial update error:", str(e))
        return jsonify({
            "error": "Simulation created, but its initial market-data update failed."
        }), 500

    return jsonify({
        "message": "Live simulation created.",
        "simulation": live_simulation_detail(sim)
    }), 201

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

    if live_sim_can_update_for_user(current_user, sim):
        try:
            update_live_simulation_from_csv(sim, current_user)
        except Exception as e:
            print(f"Live simulation update error for sim {sim.id}:", str(e))

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
        supported_tickers=get_supported_tickers_for_user(current_user)
    )

@app.route("/subscription")
def subscription():
    return render_template("subscription.html")

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
            "loc": "https://neuraltrend.org/subscription",
            "lastmod": today,
            "changefreq": "weekly",
            "priority": "0.8"
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
    duration = (request.form.get("duration") or "").strip()
    
    # Position size per signal: 100%, 50%, 25%, etc.
    try:
        position_fraction = parse_position_fraction(
            request.form.get("dca_pct", "100_pct")
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Parse and bound dates/horizon.
    try:
        start_date = datetime.strptime(start_date_text, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Start date must use YYYY-MM-DD."}), 400

    today = datetime.utcnow().date()
    if start_date > today:
        return jsonify({"error": "Start date cannot be in the future."}), 400

    try:
        delta = parse_duration(duration)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    end_date = min(start_date + delta, today)

    try:
        signals_df = load_epoch_csv_for_ticker(ticker).reset_index()
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Could not load backtest market data ticker=%s", ticker)
        return jsonify({"error": "Market data is temporarily unavailable."}), 503
    
    # Filter for the desired period
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
        return jsonify({"error": "Not enough data in selected duration"}), 400
    
    # Same transaction-cost rule used by EpochSignaler/live simulation
    transaction_cost = get_transaction_cost_rate(ticker)
    
    cash = initial_cash
    position = 0.0
    equity_curve = []
    
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
    
        # Mark-to-market equity after today's action
        equity = cash + position * price
        equity_curve.append((date, equity))
        
    eq_df = pd.DataFrame(equity_curve, columns=['Date', 'Equity']).set_index('Date')

     # --- Extract buy/sell points ---
    buy_dates = signals_df.index[signals_df['epoch_signal'] == 1]
    sell_dates = signals_df.index[signals_df['epoch_signal'] == -1]
    buy_prices = eq_df.loc[buy_dates, 'Equity']
    sell_prices = eq_df.loc[sell_dates, 'Equity']

    # Buy & Hold benchmark with entry transaction cost included
    prices = signals_df['Close'].to_numpy().flatten().astype(float)
    
    first_price = float(prices[0])
    
    benchmark_gross_budget = initial_cash / (1 + transaction_cost)
    
    if is_crypto_ticker(ticker):
        benchmark_quantity = benchmark_gross_budget / first_price
    else:
        benchmark_quantity = float(int(benchmark_gross_budget / first_price))
    
    equity_curve = (prices * benchmark_quantity).tolist()
    
    final_value = float(equity_curve[-1])
    profit_factor = float(final_value / initial_cash)

    returns = signals_df["Close"].pct_change().dropna()
    risk_free_rate_annual = 0.01
    risk_free_rate_daily = (1 + risk_free_rate_annual) ** (1/252) - 1
    excess_returns = returns - risk_free_rate_daily
    
    std = excess_returns.std()
    
    if len(excess_returns) < 2 or std == 0 or pd.isna(std):
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float((excess_returns.mean() / std) * (252 ** 0.5))
    
    dates = signals_df.index.strftime('%Y-%m-%d').tolist()

    results = {
        'ticker': ticker,
        'position_size_pct': position_fraction * 100,
        'transaction_cost_rate': transaction_cost,
        'final_value': final_value,
        'final_value_epoch': float(eq_df['Equity'].to_numpy().flatten().astype(float).tolist()[-1]),
        'profit_factor': profit_factor,
        'profit_factor_epoch': float(eq_df['Equity'].to_numpy().flatten().astype(float).tolist()[-1])/initial_cash,
        'sharpe_ratio': sharpe_ratio,
        'equity_curve': equity_curve,
        'epoch_equity_curve': eq_df['Equity'].to_numpy().flatten().astype(float).tolist(),
        'dates': dates,
        'buy_dates': [d.strftime("%Y-%m-%d") for d in buy_dates],
        'buy_prices': buy_prices.tolist() if isinstance(buy_prices, pd.Series) else buy_prices,
        'sell_dates': [d.strftime("%Y-%m-%d") for d in sell_dates],
        'sell_prices': sell_prices.tolist() if isinstance(sell_prices, pd.Series) else sell_prices,
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
    # Slice from duration ago until today
    # ---------------------------------------------------
    end_date = signals_df.index.max()
    start_date = end_date - pd.Timedelta(days=period_days)
    signals_df = signals_df[signals_df.index >= start_date].copy()

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

    for date, row in signals_df.iterrows():
        price = row['Close']
        signal = row['epoch_signal']

        if signal == 1 and cash > 0:
            position = (cash / price) * (1 - transaction_cost)
            cash = 0

        elif signal == -1 and position > 0:
            cash = (position * price) * (1 - transaction_cost)
            position = 0

        equity = cash + position * price
        epoch_equity_curve.append(equity)

    # Final liquidation
    if position > 0:
        cash = (position * signals_df['Close'].iloc[-1]) * (1 - transaction_cost)
        epoch_equity_curve[-1] = cash

    # ---------------------------------------------------
    # Buy & Hold Curve (start = 1, entry cost included)
    # ---------------------------------------------------
    prices = signals_df['Close'].to_numpy().astype(float)
    
    first_price = float(prices[0])
    
    # Treat the initial $1 as total cash used, including entry transaction cost.
    benchmark_gross_budget = 1.0 / (1 + transaction_cost)
    benchmark_quantity = benchmark_gross_budget / first_price
    
    buy_hold_curve = (prices * benchmark_quantity).tolist()

    # ---------------------------------------------------
    # Buy/Sell markers
    # ---------------------------------------------------
    buy_dates = signals_df.index[signals_df['epoch_signal'] == 1]
    sell_dates = signals_df.index[signals_df['epoch_signal'] == -1]

    buy_prices = [
        epoch_equity_curve[signals_df.index.get_loc(d)]
        for d in buy_dates if d in signals_df.index
    ]

    sell_prices = [
        epoch_equity_curve[signals_df.index.get_loc(d)]
        for d in sell_dates if d in signals_df.index
    ]

    # ---------------------------------------------------
    # Final Metrics
    # ---------------------------------------------------
    final_value_bh = buy_hold_curve[-1]
    final_value_epoch = epoch_equity_curve[-1]

    returns = signals_df['Close'].pct_change().dropna()
    sharpe_ratio = float((returns.mean() / returns.std()) * (252 ** 0.5)) if returns.std() != 0 else 0.0

    dates = signals_df.index.strftime('%Y-%m-%d').tolist()

    results = {
        'ticker': ticker,
        'transaction_cost_rate': transaction_cost,
        'final_value': final_value_bh,
        'final_value_epoch': final_value_epoch,
        'profit_factor': final_value_bh,
        'profit_factor_epoch': final_value_epoch,
        'sharpe_ratio': sharpe_ratio,
        'equity_curve': buy_hold_curve,
        'epoch_equity_curve': epoch_equity_curve,
        'dates': dates,
        'buy_dates': [d.strftime("%Y-%m-%d") for d in buy_dates],
        'buy_prices': buy_prices,
        'sell_dates': [d.strftime("%Y-%m-%d") for d in sell_dates],
        'sell_prices': sell_prices,
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
        "buy_hold_annual_return": row.get("buy_hold_annual_return"),
        "strategy_annual_return": row.get("strategy_annual_return"),
        "outperformance": row.get("outperformance"),

        # Frontend uses this to show locked/blurred cells
        "signals_locked": not full_access,
    }

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
def compute_signals_summary_cached(csv_version, period_days):
    results = []

    for t in get_all_signal_board_tickers():
        try:
            sigs = compute_signals_for_ticker(t, period_days, csv_version)
            results.append({
                'ticker': t,
                'today_signal': sigs['today'],
                'yesterday_signal': sigs['yesterday'],
                'last_week_signal': sigs['last_week'],
                'last_month_signal': sigs['last_month'],
                'buy_hold_annual_return': sigs['buy_hold_annual_return'],
                'strategy_annual_return': sigs['strategy_annual_return'],
                'outperformance': sigs['outperformance'],
            })
        except Exception as e:
            print(f"Skipping {t}: {e}")

    return results

@app.route('/signals/summary')
def signals_summary():
    duration_str = (request.args.get("duration") or "5y").strip()
    
    try:
        period_days = duration_to_days(duration_str)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    csv_version = get_csv_version()

    # Full internal cached data
    raw_results = compute_signals_summary_cached(csv_version, period_days)

    # Apply admin-only ticker visibility BEFORE masking.
    visible_results = [
        row for row in raw_results
        if can_user_see_ticker(current_user, row.get("ticker"))
    ]
    
    # User-safe data returned to frontend
    safe_results = [
        mask_signal_summary_row_for_user(row, current_user)
        for row in visible_results
    ]
    
    return jsonify(safe_results)

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


def stripe_subscription_uses_pro_price(subscription):
    subscription = stripe_to_dict(subscription)
    items = ((subscription.get("items") or {}).get("data") or [])

    return any(
        (item.get("price") or {}).get("id") == STRIPE_PRO_PRICE_ID
        for item in items
    )


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
    if not stripe.api_key or not STRIPE_PRO_PRICE_ID:
        return jsonify({"error": "Checkout is temporarily unavailable."}), 503

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

        open_session = retrieve_open_checkout_session(
            user.pending_checkout_session_id
        )

        if open_session:
            db.session.commit()
            return jsonify({
                "url": open_session["url"],
                "reused": True,
            })

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
                    "price": STRIPE_PRO_PRICE_ID,
                    "quantity": 1,
                }
            ],
            success_url=f"{BASE_URL}/?checkout=success",
            cancel_url=f"{BASE_URL}/subscription?checkout=cancelled",
            client_reference_id=str(user_id),
            metadata={
                "user_id": str(user_id),
            },
            subscription_data={
                "metadata": {
                    "user_id": str(user_id),
                }
            },
            idempotency_key=f"nt-checkout-{attempt_id}",
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

        return jsonify({"url": checkout_session.url})

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


@app.route("/billing-portal", methods=["POST"])
@login_required
def billing_portal():
    if not current_user.stripe_customer_id:
        return jsonify({"error": "No billing customer found."}), 400

    try:
        portal_session = stripe.billing_portal.Session.create(
            customer=current_user.stripe_customer_id,
            return_url=f"{BASE_URL}/subscription"
        )

        return jsonify({"url": portal_session.url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

    if not stripe.api_key or not STRIPE_PRO_PRICE_ID or not STRIPE_WEBHOOK_SECRET:
        app.logger.error(
            "Stripe webhook configuration is incomplete: api_key=%s price_id=%s webhook_secret=%s",
            bool(stripe.api_key),
            bool(STRIPE_PRO_PRICE_ID),
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
                "Subscription event did not match the configured Pro Price or environment.",
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

