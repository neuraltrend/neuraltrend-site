from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_login import login_user, logout_user, login_required, current_user
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import yfinance as yf
import pandas as pd
import numpy as np
import os
from functools import lru_cache
from extensions import db, bcrypt, login_manager
from models import User, LiveSimulation, LiveSimulationTrade, LiveSimulationEquity
from itsdangerous import URLSafeTimedSerializer
from flask_mail import Mail, Message
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

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

# 🔐 REQUIRED FOR SESSIONS
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# 🔒 Cookie security (recommended)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = True  # important on Render HTTPS

db.init_app(app)
bcrypt.init_app(app)
login_manager.init_app(app)
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
    return User.query.get(int(user_id))
    
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Too many requests. Please slow down and try again shortly."
    }), 429

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

    verify_url = f"https://neuraltrend.org/verify/{token}"

    msg = Message(
        subject="Verify your NeuralTrend account",
        sender=app.config["MAIL_USERNAME"],
        recipients=[user_email]
    )

    msg.body = f"""
    Click the link to verify your account:

    {verify_url}
    """

    try:
        mail.send(msg)
        print("Verification email sent to:", user_email)
    except Exception as e:
        print("EMAIL ERROR (verify):", str(e))

def normalize_email(email):
    return email.strip().lower()

def generate_reset_token(email):
    return get_serializer().dumps(email, salt="password-reset")

def confirm_reset_token(token, expiration=3600):
    try:
        email = get_serializer().loads(
            token,
            salt="password-reset",
            max_age=expiration
        )
    except Exception:
        return None
    return email

def generate_delete_token(email):
    return get_serializer().dumps(email, salt="delete-account")

def confirm_delete_token(token, expiration=3600):
    try:
        email = get_serializer().loads(token, salt="delete-account", max_age=expiration)
    except Exception:
        return None
    return email

def parse_duration(duration: str):
    """Return a relativedelta or timedelta from strings like '1mo','3mo','6mo','1yr','10d','2w'."""
    s = duration.strip().lower()
    if s.endswith("mo") or s.endswith("m"):   # support '1m' or '1mo'
        return relativedelta(months=int(s.rstrip('mo').rstrip('m')))
    if s.endswith("yr") or s.endswith("y"):
        return relativedelta(years=int(s.rstrip('yr').rstrip('y')))
    if s.endswith("w"):
        return timedelta(weeks=int(s[:-1]))
    if s.endswith("d"):
        return timedelta(days=int(s[:-1]))
    raise ValueError(f"Unsupported duration: {duration}")

def duration_to_days(duration_str: str):
    delta = parse_duration(duration_str)

    if isinstance(delta, timedelta):
        return delta.days
    elif isinstance(delta, relativedelta):
        # Approximate 1 month = 30 days, 1 year = 365 days
        return delta.years * 365 + delta.months * 30 + delta.days
    else:
        raise ValueError(f"Unsupported delta type: {type(delta)}")

def parse_position_fraction(value):
    """
    Converts values like '100_pct', '50_pct', '25_pct', '100%', or '50'
    into decimal fractions: 1.0, 0.5, 0.25.
    """
    if value is None:
        return 1.0

    s = str(value).strip().lower()
    s = s.replace("_pct", "")
    s = s.replace("pct", "")
    s = s.replace("%", "")

    try:
        pct = float(s)
    except ValueError:
        raise ValueError(f"Unsupported position size: {value}")

    if pct <= 0 or pct > 100:
        raise ValueError("Position size must be greater than 0 and less than or equal to 100.")

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
LIVE_SIMULATION_LIMIT = 100

def compute_signals_for_ticker(ticker, period_days=365*10):
    cache_key = (ticker, period_days)
    if cache_key in cache:
        return cache[cache_key]

    base_symbol = ticker.split('-')[0]
    csv_filename = f"epoch_{base_symbol}.csv"
    csv_path = os.path.join(app.root_path, 'data', csv_filename)

    # -------------------------------
    # Load full data first
    # -------------------------------
    df = pd.read_csv(csv_path, usecols=['Date', 'Close', 'epoch_signal'], parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['epoch_signal'] = pd.to_numeric(df['epoch_signal'], errors='coerce')
    df = df.dropna()

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
    base_symbol = ticker.split("-")[0]
    csv_filename = f"epoch_{base_symbol}.csv"
    csv_path = os.path.join(app.root_path, "data", csv_filename)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No CSV found for {ticker}: {csv_filename}")

    df = pd.read_csv(
        csv_path,
        usecols=["Date", "Close", "epoch_signal"],
        parse_dates=["Date"]
    )

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["epoch_signal"] = pd.to_numeric(df["epoch_signal"], errors="coerce")

    df = df.dropna(subset=["Date", "Close", "epoch_signal"]).copy()
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)

    if len(df) < 1:
        raise ValueError(f"CSV for {ticker} has no valid rows.")

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


def live_simulation_summary(sim):
    latest = get_latest_equity_point(sim.id)

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


def update_live_simulation_from_csv(sim):
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

# --------------------
# Routes
# --------------------

# @app.route("/init-db")
# def init_db():
#     with app.app_context():
#         db.create_all()
#     return "DB initialized"

# @app.route("/admin/init-live-sim-tables/<token>")
# def init_live_sim_tables(token):
#     expected_token = os.environ.get("ADMIN_INIT_TOKEN")

#     if not expected_token or token != expected_token:
#         return "Unauthorized", 403

#     db.create_all()

#     return "Live simulation tables initialized."

@app.route("/signup", methods=["POST"])
@limiter.limit("3 per minute")
def signup():    
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    email = normalize_email(data.get("email"))
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    existing_user = User.query.filter_by(email=email).first()

    if existing_user:
        return jsonify({"error": "User already exists"}), 400

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
    data = request.get_json()

    email = normalize_email(data.get("email"))
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

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

    # ✅ Successful login → reset counters
    user.failed_attempts = 0
    user.locked_until = None
    db.session.commit()

    # 🔒 Require verification
    if not user.is_verified:
        return jsonify({"error": "Please verify your email first"}), 403

    login_user(user)

    return jsonify({
        "message": "Logged in successfully",
        "user_id": user.id,
        "email": user.email
    })

@app.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out"})

@app.route("/me")
def me():
    if current_user.is_authenticated:
        return jsonify({
            "email": current_user.email
        })

    return jsonify({
        "email": None
    })

# --------------------
# Live simulation API
# --------------------

@app.route("/live-simulations", methods=["GET"])
@login_required
def list_live_simulations():
    sims = (
        LiveSimulation.query
        .filter_by(user_id=current_user.id, status="active")
        .order_by(LiveSimulation.created_at.desc())
        .all()
    )

    for sim in sims:
        try:
            update_live_simulation_from_csv(sim)
        except Exception as e:
            print(f"Live simulation update error for sim {sim.id}:", str(e))

    sims = (
        LiveSimulation.query
        .filter_by(user_id=current_user.id, status="active")
        .order_by(LiveSimulation.created_at.desc())
        .all()
    )

    return jsonify({
        "limit": LIVE_SIMULATION_LIMIT,
        "count": len(sims),
        "simulations": [live_simulation_summary(sim) for sim in sims]
    })

@app.route("/live-simulations", methods=["POST"])
@login_required
def create_live_simulation():
    data = request.get_json(silent=True) or {}

    ticker = str(data.get("ticker", "BTC-USD")).strip().upper()
    name = str(data.get("name", "")).strip()

    try:
        initial_cash = float(data.get("initial_cash", 10000))
    except (TypeError, ValueError):
        return jsonify({"error": "Initial cash must be a valid number."}), 400

    if initial_cash <= 0:
        return jsonify({"error": "Initial cash must be greater than zero."}), 400

    try:
        position_fraction = parse_position_fraction(
            data.get("position_size_pct", 100)
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    position_size_pct = position_fraction * 100

    active_count = LiveSimulation.query.filter_by(
        user_id=current_user.id,
        status="active"
    ).count()

    if active_count >= LIVE_SIMULATION_LIMIT:
        return jsonify({
            "error": f"Simulation limit reached. Current limit is {LIVE_SIMULATION_LIMIT}."
        }), 403

    try:
        df = load_epoch_csv_for_ticker(ticker)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    latest_date = df.index.max().date()
    latest_price = float(df.loc[df.index.max(), "Close"])

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
            "error": "Simulation created, but initial update failed.",
            "details": str(e)
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

    try:
        update_live_simulation_from_csv(sim)
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

@app.route("/request-password-reset", methods=["POST"])
@limiter.limit("3 per minute")
def request_password_reset():
    data = request.get_json()
    email = normalize_email(data.get("email"))

    user = User.query.filter_by(email=email).first()

    if user:
        token = generate_reset_token(email)
        reset_url = f"https://neuraltrend.org/reset-password/{token}"

        msg = Message(
            subject="Reset your password",
            sender=app.config["MAIL_USERNAME"],
            recipients=[email]
        )

        msg.body = f"Reset your password:\n\n{reset_url}"

        try:
            mail.send(msg)
            print("RESET EMAIL SENT:", email)
        except Exception as e:
            print("EMAIL ERROR:", str(e))

    # Always return same message
    return jsonify({"message": "If account exists, email sent"})

@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    email = confirm_reset_token(token)

    if not email:
        return "Invalid or expired token"

    user = User.query.filter_by(email=email).first()
    if not user:
        return "User not found"

    if request.method == "POST":
        new_password = request.form.get("password")

        if not new_password:
            return "Password required"

        user.password_hash = bcrypt.generate_password_hash(new_password).decode("utf-8")
        db.session.commit()

        return "Password reset successful! You can now log in."

    return render_template("reset_password.html")

@app.route("/request-delete-account", methods=["POST"])
@login_required
@limiter.limit("2 per minute")
def request_delete_account():
    user = current_user

    token = generate_delete_token(user.email)
    delete_url = f"https://neuraltrend.org/confirm-delete/{token}"

    msg = Message(
        subject="Confirm account deletion",
        sender=app.config["MAIL_USERNAME"],
        recipients=[user.email]
    )

    msg.body = f"""
    Click the link below to permanently delete your account:

    {delete_url}

    This link expires in 1 hour.
    """

    try:
        mail.send(msg)
        print("Delete email sent to:", user.email)
    except Exception as e:
        print("EMAIL ERROR (delete):", str(e))
        return jsonify({"error": "Email failed"}), 500
    
    return jsonify({"message": "Deletion confirmation email sent"})

@app.route("/confirm-delete/<token>")
def confirm_delete(token):
    email = confirm_delete_token(token)

    if not email:
        return "Invalid or expired link"

    user = User.query.filter_by(email=email).first()

    if not user:
        return "User not found"

    logout_user()  # 🔑 IMPORTANT FIX

    db.session.delete(user)
    db.session.commit()

    return "Your account has been permanently deleted"

# @app.route("/force_delete_user")
# def force_delete_user():
#     user = User.query.filter_by(email="x@x.com").first()
#     if user:
#         db.session.delete(user)
#         db.session.commit()
#         return "Deleted"
#     return "User not found"

# https://neuraltrend.org/force_delete_user

@app.route("/")
def index():
    return render_template("index.html")

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

@app.route('/market')
def market():
    return render_template('market.html')

@app.route('/knowledge')
def knowledge():
    return render_template('knowledge.html')

@app.route('/ads.txt')
def ads_txt():
    return send_from_directory(os.path.dirname(__file__), 'ads.txt')

@app.route('/backtest', methods=['POST'])
def backtest():
    initial_cash = float(request.form['cash'])
    ticker = request.form['ticker']
    start_date = request.form['start']
    duration = request.form["duration"]    # e.g. '1mo','3mo','6mo','1yr'
    ticker_2 = []
    
    # Position size per signal: 100%, 50%, 25%, etc.
    try:
        position_fraction = parse_position_fraction(
            request.form.get("dca_pct", "100_pct")
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Parse dates
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()

    # Compute intended end and cap at today
    delta = parse_duration(duration)
    end_date_2 = start_date + delta

    end_for_download=min(end_date_2,datetime.today().date())

    base_symbol = ticker.split('-')[0]  # -> "BTC"
    
    # --- Load CSV of signals ---
    csv_filename = f"epoch_{base_symbol}.csv"
    csv_path = os.path.join(app.root_path, 'data', csv_filename)
    signals_df = pd.read_csv(csv_path, parse_dates=['Date'])
    
    # Filter for the desired period
    mask = (signals_df['Date'] >= pd.to_datetime(start_date)) & (signals_df['Date'] <= pd.to_datetime(end_date_2))
    df_filtered = signals_df.loc[mask].copy()
    
    # Optional: set Date as index
    df_filtered.set_index('Date', inplace=True)
    signals_df=df_filtered
    
    # Convert Close to float explicitly
    signals_df['Close'] = pd.to_numeric(signals_df['Close'], errors='coerce')
    signals_df = signals_df.dropna()

    cash = initial_cash
    position = 0.0
    equity_curve = []
    
    for date, row in signals_df.iterrows():
        price = row['Close']
        signal = row['epoch_signal']
    
        # BUY signal:
        # Invest a percentage of available cash.
        # 100% = use all cash
        # 50% = use half of remaining cash
        # 25% = use one quarter of remaining cash
        if signal == 1 and cash > 0:
            cash_to_invest = cash * position_fraction
            shares_to_buy = cash_to_invest / price
    
            position += shares_to_buy
            cash -= cash_to_invest
    
        # SELL signal:
        # Sell a percentage of current position.
        # 100% = sell all holdings
        # 50% = sell half of current holdings
        # 25% = sell one quarter of current holdings
        elif signal == -1 and position > 0:
            shares_to_sell = position * position_fraction
            cash_from_sale = shares_to_sell * price
    
            position -= shares_to_sell
            cash += cash_from_sale
    
        equity = cash + position * price
        equity_curve.append((date, equity))
        
    eq_df = pd.DataFrame(equity_curve, columns=['Date', 'Equity']).set_index('Date')

     # --- Extract buy/sell points ---
    buy_dates = signals_df.index[signals_df['epoch_signal'] == 1]
    sell_dates = signals_df.index[signals_df['epoch_signal'] == -1]
    buy_prices = eq_df.loc[buy_dates, 'Equity']
    sell_prices = eq_df.loc[sell_dates, 'Equity']

    equity_curve = signals_df['Close'].to_numpy().flatten().astype(float).tolist()
    equity_curve_start=equity_curve[0]
    equity_curve = np.array(equity_curve)  # convert list to numpy array
    equity_curve = equity_curve / equity_curve[0] * initial_cash
    equity_curve = equity_curve.tolist()
    final_value = float(equity_curve[-1])
    profit_factor = float(final_value / initial_cash)

    returns = signals_df['Close'].pct_change().dropna()
    risk_free_rate_annual = 0.01
    risk_free_rate_daily = (1 + risk_free_rate_annual) ** (1/252) - 1
    excess_returns = returns - risk_free_rate_daily
    sharpe_ratio = float(((excess_returns.mean() / excess_returns.std()) * (252 ** 0.5)))

    equity_curve_2=[]
    if ticker_2:
        df_2 = yf.download(ticker_2, start=start_date, end=end_date_2, interval='1d')  # FIXED
        df_2 = df_2[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        series_2 = pd.DataFrame()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df_2[col] = df_2[col].astype(float)
            values_2 = df_2[col].values
            if values_2.ndim > 1:
                values_2 = values_2.flatten()
            df_2[col] = values_2
            series_2[col] = pd.Series(values_2, index=df_2.index)

        equity_curve_2 = df_2['Close'].to_numpy().flatten().astype(float).tolist()
        equity_curve_start=equity_curve_2[0]
        equity_curve_2 = np.array(equity_curve_2)  # convert list to numpy array
        equity_curve_2 = equity_curve_2 / equity_curve_2[0] * initial_cash
        equity_curve_2 = equity_curve_2.tolist()
        final_value_2 = float(equity_curve_2[-1])
        profit_factor_2 = float(final_value_2 / initial_cash)

        returns_2 = df_2['Close'].pct_change().dropna()
        excess_returns_2 = returns_2 - risk_free_rate_daily
        sharpe_ratio_2 = float(((excess_returns_2.mean() / excess_returns_2.std()) * (252 ** 0.5)).iloc[0])
    
    dates = signals_df.index.strftime('%Y-%m-%d').tolist()

    results = {
        'ticker': ticker,
        'position_size_pct': position_fraction * 100,
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

    if ticker_2 and equity_curve_2:  # or however you check for optional input
        results.update({
            'ticker_2': ticker_2,
            'final_value_2': final_value_2,
            'profit_factor_2': profit_factor_2,
            'sharpe_ratio_2': sharpe_ratio_2,
            'equity_curve_2': equity_curve_2
        })
    else:
        results['equity_curve_2'] = []  # keep chart code safe

    return jsonify(results)

@app.route('/equity', methods=['POST'])
def equity():

    ticker = request.form['ticker']
    duration_str = request.form["duration"]  # '1w','1mo','1y', etc.

    # Convert duration to days
    try:
        period_days = duration_to_days(duration_str)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    base_symbol = ticker.split('-')[0]
    csv_filename = f"epoch_{base_symbol}.csv"
    csv_path = os.path.join(app.root_path, 'data', csv_filename)

    signals_df = pd.read_csv(csv_path, parse_dates=['Date'])
    signals_df.set_index('Date', inplace=True)

    signals_df['Close'] = pd.to_numeric(signals_df['Close'], errors='coerce')
    signals_df['epoch_signal'] = pd.to_numeric(signals_df['epoch_signal'], errors='coerce')
    signals_df = signals_df.dropna()

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
    transaction_cost = 0.01 if ticker.endswith("-USD") else 0.001

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
    # Buy & Hold Curve (start = 1)
    # ---------------------------------------------------
    prices = signals_df['Close'].to_numpy()
    buy_hold_curve = (prices / prices[0]).tolist()

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
    
@app.route('/signals', methods=['POST'])
def signals():
    ticker = request.form['ticker']
    duration_str = request.form.get('duration', '1y')  # default '10y'

    try:
        period_days = duration_to_days(duration_str)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    sigs = compute_signals_for_ticker(ticker, period_days)

    return jsonify({
        'ticker': ticker,
        'today_signal': sigs['today'],
        'yesterday_signal': sigs['yesterday'],
        'last_week_signal': sigs['last_week'],
        'last_month_signal': sigs['last_month'],
        'buy_hold_annual_return': sigs['buy_hold_annual_return'],
        'strategy_annual_return': sigs['strategy_annual_return'],
        'outperformance': sigs['outperformance'],
    })

# Cached version that invalidates when CSV files change
@lru_cache(maxsize=1)
def compute_signals_summary_cached(csv_version, period_days):
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'NVDA', 'AAPL', 'GOOGL', 'MSFT', "1INCH-USD", "3ULL-USD", "AAVE-USD","ABBV", "ACE-USD",
               "ACH-USD", "ADA-USD", "AERO-USD", "AEVO-USD", "AGI-USD", "AIOZ-USD", "AIT-USD", "AITECH-USD", "AIXBT-USD", "AKT-USD", "ALEPH-USD",
               "ALGO-USD", "ALI-USD", "ALPH-USD", "ALT-USD", "ALU-USD", "ALVA-USD", "AMP-USD", 'AMZN', "ANKR-USD", "ANON-USD", "ANYONE-USD", "APT-USD",
               "APU-USD", "AR-USD", "ARB-USD", "ARC-USD", "ASML", "ASTR-USD", "ATLAS-USD", "ATOM-USD", "AURY-USD", "AUTOS-USD", "AVAX-USD", 'AVGO', 
               "AXL-USD", "AXS-USD", "BAI-USD", "BAL-USD", "BAND-USD", "BANANA-USD", "BASEDAI-USD", "BAZED-USD", "BCB-USD",
               "BCUT-USD", "BEAM-USD", "BGB-USD", "BIGTIME-USD", "BLUR-USD", "BNB-USD", "BNT-USD", "BONK-USD", "BRETT-USD", 'BRKB', 
               "BYTES-USD", "CELO-USD", "CERE-USD", "CETUS-USD", "CFG-USD", "CGPT-USD", "CHAPZ-USD", "CHAT-USD", "CHEX-USD", "CHZ-USD", 
               "COMP-USD", "COST", "COTI-USD", "CPOOL-USD", "CREDI-USD", "CREO-USD", "CRO-USD", "CROWN-USD", "CRU-USD", "CRV-USD", "CTC-USD", "CVC-USD",
               "DARK-USD", "DCK-USD", "DEVVE-USD", "DIMO-USD", "DIO-USD", "DOGE-USD", "DOME-USD", "DOMI-USD", "DOT-USD", "DRIFT-USD", "DSYNC-USD",
               "DYDX-USD", "DYM-USD", "EDU-USD", "ENA-USD", "ENJ-USD", "ENQAI-USD", "F3-USD", "FAR-USD", "FET-USD", "FIDA-USD", 
               "FIL-USD", "FLIP-USD", "FLOW-USD", "FLR-USD", "FLUX-USD", "FOXY-USD", "FUELX-USD", "FYN-USD", "GEAR-USD", "GFAL-USD",
               "GHX-USD", "GLQ-USD", "GMEE-USD", "GMRX-USD", "GMT-USD", "GMX-USD", "GODS-USD", "GPU-USD", "GRIFFAIN-USD", "GRT-USD",
               "GSWIFT-USD", "GTAI-USD", "GTC-USD", "HASHAI-USD", "HBAR-USD", "HEART-USD", "HELLO-USD", "HNT-USD", "HONEY-USD",
               "HXD-USD", "HYPC-USD", "HYPE-USD", "IAG-USD", "ICP-USD", "ILV-USD", "IMX-USD", "INJ-USD", "INSP-USD", "IOTX-USD",
               "IVPAY-USD", "JASMY-USD", "JNJ", "JOE-USD", "JTO-USD", "JUP-USD", "KARATE-USD", "KARRAT-USD", "KAS-USD", "KATA-USD", "KOMPETE-USD",
               "KRL-USD", "LAI-USD", "LEO-USD", "LFNTY-USD", "LIKE-USD", "LINK-USD", "LMWR-USD", "LPT-USD", "LRC-USD", "LTC-USD", "MA",
               "MAGIC-USD", "MASK-USD", "MAVIA-USD", "MBS-USD", 'META', "METIS-USD", "MEW-USD", "MINA-USD", "ML-USD", "MLN-USD", "MNDE-USD",
               "MNT-USD", "MOODENG-USD", "MPLX-USD", "MU", "MUBI-USD", "MXM-USD", "MYRIA-USD", "MYRO-USD", "NAKA-USD", 
               "NEAR-USD", "NEON-USD", "NEURAL-USD", "NMT-USD", "NOS-USD", "NTRN-USD", "NU-USD", "NXRA-USD", "OCT-USD", "OGN-USD", "OKB-USD",
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
    results = []

    for t in tickers:
        try:
            sigs = compute_signals_for_ticker(t, period_days)
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
    duration_str = request.args.get('duration', '5y')  # default '5y'
    
    try:
        period_days = duration_to_days(duration_str)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    csv_version = get_csv_version()
    results = compute_signals_summary_cached(csv_version, period_days)
    return jsonify(results)
