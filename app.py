from flask import Flask, render_template, request, jsonify, send_from_directory, session
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta  # pip install python-dateutil
import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
import json
from functools import lru_cache
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import requests


app = Flask(__name__)

# 🔐 REQUIRED FOR SESSIONS
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-only-fallback")
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_ANON_KEY"]

# 🔒 Cookie security (recommended)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Data path
DATA_DIR = os.path.join(app.root_path, 'data')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "epoch_index-USD.csv")

# Ensure folder/file exist
os.makedirs(DATA_DIR, exist_ok=True)

def supabase_auth(endpoint, payload):
    return requests.post(
        f"{SUPABASE_URL}/auth/v1/{endpoint}",
        headers={
            "apikey": SUPABASE_KEY,
            "Content-Type": "application/json"
        },
        json=payload
    )

def parse_duration(duration: str):
    """Return a relativedelta or timedelta from strings like '1mo','3mo','6mo','1yr','10d','2w'."""
    s = duration.strip().lower()
    if s.endswith("mo"):
        return relativedelta(months=int(s[:-2]))
    if s.endswith("yr") or s.endswith("y"):
        return relativedelta(years=int(s.rstrip('yr').rstrip('y')))
    if s.endswith("w"):
        return timedelta(weeks=int(s[:-1]))
    if s.endswith("d"):
        return timedelta(days=int(s[:-1]))
    raise ValueError(f"Unsupported duration: {duration}")

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

def compute_signals_for_ticker(ticker):
    delta = parse_duration('3mo')
    start_date = datetime.today().date() - delta

    base_symbol = ticker.split('-')[0]
    csv_filename = f"epoch_{base_symbol}.csv"
    csv_path = os.path.join(app.root_path, 'data', csv_filename)

    signals_df = pd.read_csv(csv_path, parse_dates=['Date'])

    signals_df = signals_df[signals_df['Date'] >= pd.to_datetime(start_date)]
    signals_df.set_index('Date', inplace=True)

    signals_df['Close'] = pd.to_numeric(signals_df['Close'], errors='coerce')
    signals_df = signals_df.dropna()

    return {
        'today': int(signals_df['epoch_signal'].iloc[-1]),
        'yesterday': int(signals_df['epoch_signal'].iloc[-2]),
        'last_week': int(signals_df['epoch_signal'].iloc[-8]),
        'last_month': int(signals_df['epoch_signal'].iloc[-31]),
    }

# --------------------
# Routes
# --------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()

    r = supabase_auth("signup", {
        "email": data["username"],
        "password": data["password"]
    })

    if not r.ok:
        # 🔥 Pass Supabase error THROUGH, unchanged
        return jsonify(r.json()), r.status_code

    auth = r.json()
    session["access_token"] = auth["access_token"]

    return jsonify(username=auth["user"]["email"])

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()

    r = supabase_auth("token?grant_type=password", {
        "email": data["username"],
        "password": data["password"]
    })

    if not r.ok:
        # 🔥 DO NOT overwrite Supabase errors
        return jsonify(r.json()), r.status_code

    auth = r.json()
    session["access_token"] = auth["access_token"]

    return jsonify(username=auth["user"]["email"])

@app.route("/logout", methods=["POST"])
def logout():
    session.pop("access_token", None)
    return jsonify(success=True)

@app.route("/me")
def me():
    access_token = session.get("access_token")
    if not access_token:
        return jsonify(username=None)

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {access_token}"
    }

    r = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers=headers
    )

    if r.status_code == 200:
        user = r.json()
        return jsonify(username=user["email"])

    return jsonify(username=None)

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
    ticker_2=[]

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
    position = 0
    equity_curve = []

    for date, row in signals_df.iterrows():
        price = row['Close']
        signal = row['epoch_signal']

        if signal == 1 and cash > 0:  # Buy
            position = cash / price
            cash = 0
        elif signal == -1 and position > 0:  # Sell
            cash = position * price
            position = 0

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

@app.route('/signals', methods=['POST'])
def signals():
    ticker = request.form['ticker']
    sigs = compute_signals_for_ticker(ticker)

    return jsonify({
        'ticker': ticker,
        'today_signal': sigs['today'],
        'yesterday_signal': sigs['yesterday'],
        'last_week_signal': sigs['last_week'],
        'last_month_signal': sigs['last_month'],
    })

# Cached version that invalidates when CSV files change
@lru_cache(maxsize=1)
def compute_signals_summary_cached(csv_version):
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', "DOGE-USD", "ADA-USD", "1INCH-USD", "3ULL-USD", "AAVE-USD", "ACE-USD",
               "ACH-USD", "AERO-USD", "AEVO-USD", "AGI-USD", "AIOZ-USD", "AIT-USD", "AITECH-USD", "AIXBT-USD", "AKT-USD", "ALEPH-USD",
               "ALGO-USD", "ALI-USD", "ALPH-USD", "ALT-USD", "ALU-USD", "ALVA-USD", "AMP-USD", "ANKR-USD", "ANON-USD", "ANYONE-USD",
               "APU-USD", "AR-USD", "ARB-USD", "ARC-USD", "ASTR-USD", "ATLAS-USD", "ATOM-USD", "AURY-USD", "AUTOS-USD", "AVAX-USD", 
               "AXL-USD", "AXS-USD", "BAG-USD", "BAI-USD", "BAL-USD", "BAND-USD", "BANANA-USD", "BASEDAI-USD", "BAZED-USD", "BCB-USD",
               "BCUBE-USD", "BCUT-USD", "BEAM-USD", "BIGTIME-USD", "BLUR-USD", "BNT-USD", "BONK-USD", "BRETT-USD", "BXBT-USD", "BYTES-USD", 
               "CELO-USD", "CERE-USD", "CETUS-USD", "CFG-USD", "CGPT-USD", "CHAPZ-USD", "CHAT-USD", "CHEX-USD", "CHZ-USD", "COMAI-USD", 
               "COMP-USD", "COTI-USD", "CPOOL-USD", "CREDI-USD", "CREO-USD", "CROWN-USD", "CRU-USD", "CRV-USD", "CTC-USD", "CVC-USD",
               "DARK-USD", "DCK-USD", "DEVVE-USD", "DIMO-USD", "DIO-USD", "DOME-USD", "DOMI-USD", "DOT-USD", "DRIFT-USD", "DSYNC-USD",
               "DYDX-USD", "DYM-USD", "EDU-USD", "ENA-USD", "ENJ-USD", "ENQAI-USD", "F3-USD", "FAR-USD", "FET-USD", "FIDA-USD", 
               "FIL-USD", "FLIP-USD", "FLOW-USD", "FLR-USD", "FLUX-USD", "FOXY-USD", "FUELX-USD", "FYN-USD", "GEAR-USD", "GFAL-USD",
               "GHX-USD", "GLQ-USD", "GMEE-USD", "GMRX-USD", "GMT-USD", "GMX-USD", "GODS-USD", "GPU-USD", "GRIFFAIN-USD", "GRT-USD",
               "GSWIFT-USD", "GTAI-USD", "GTC-USD", "HASHAI-USD", "HBAR-USD", "HEART-USD", "HELLO-USD", "HNT-USD", "HONEY-USD",
               "HXD-USD", "HYPC-USD", "IAG-USD", "ICNX-USD", "ICP-USD", "ILV-USD", "IMX-USD", "INJ-USD", "INSP-USD", "IOTX-USD",
               "IVPAY-USD", "JASMY-USD", "JOE-USD", "JTO-USD", "JUP-USD", "KARATE-USD", "KARRAT-USD", "KAS-USD", "KATA-USD", 
               "KOMPETE-USD", "KRL-USD", "LAI-USD", "LFNTY-USD", "LIKE-USD", "LINK-USD", "LMWR-USD", "LPT-USD", "LRC-USD", "LTC-USD",
               "MAGIC-USD", "MASK-USD", "MAVIA-USD", "MBS-USD", "METIS-USD", "MEW-USD", "MINA-USD", "ML-USD", "MLN-USD", "MNDE-USD",
               "MNT-USD", "MOODENG-USD", "MOZ-USD", "MPLX-USD", "MUBI-USD", "MXM-USD", "MYRIA-USD", "MYRO-USD", "MZERO-USD", "NAKA-USD", 
               "NEAR-USD", "NEON-USD", "NEURAL-USD", "NMT-USD", "NOS-USD", "NTRN-USD", "NU-USD", "NXRA-USD", "OCT-USD", "OGN-USD", 
               "OLAS-USD", "OMG-USD", "ONDO-USD", "OP-USD", "ORAI-USD", "ORCA-USD", "ORDI-USD", "OTK-USD", "OXT-USD", "PAAL-USD", 
               "PAID-USD", "PANDORA-USD", "PDA-USD", "PENDLE-USD", "PENG-USD", "PENGU-USD", "PEPE-USD", "PERP-USD", "PHA-USD", 
               "PIN-USD", "PIXEL-USD", "POL-USD", "POLS-USD", "POLYX-USD", "PORTAL-USD", "PRIME-USD", "PROPC-USD", "PYR-USD", 
               "PYTH-USD", "QANX-USD", "QI-USD", "QNT-USD", "RAY-USD", "RARE-USD", "RARI-USD", "RDT-USD", "REN-USD", "RENDER-USD", 
               "REQ-USD", "RIO-USD", "RLB-USD", "RMRK-USD", "RON-USD", "ROOT-USD", "RSC-USD", "RSR-USD", "RSS3-USD", "RST-USD",
               "RSTK-USD", "RUNE-USD", "SAFE-USD", "SC-USD", "SCAR-USD", "SEI-USD", "SENATE-USD", "SERSH-USD", "SHDW-USD", "SHIB-USD", 
               "SHIDO-USD", "SHRAP-USD", "SIDUS-USD", "SIPHER-USD", "SKL-USD", "SNS-USD", "SPEC-USD", "SPELL-USD", "SRM-USD", "SSV-USD", 
               "STEP-USD", "STG-USD", "STORJ-USD", "STRK-USD", "SUI-USD", "SUNDOG-USD", "SUPER-USD", "SYNT-USD", "TAI-USD", "TAO-USD", 
               "TET-USD", "TFUEL-USD", "THETA-USD", "TLOS-USD", "TON-USD", "TRAC-USD", "TRIAS-USD", "TRU-USD", "TURBO-USD", "UNI-USD", 
               "UNIBOT-USD", "UOS-USD", "VAI-USD", "VET-USD", "VIA-USD", "VIRTUAL-USD", "VPP-USD", "VR-USD", "VRA-USD", "WAGMIGAMES-USD",
               "WAXP-USD", "WELT-USD", "WHALES-USD", "WIF-USD", "WIFI-USD", "WILD-USD", "WINR-USD", "WMT-USD", "XAI-USD", "XCAD-USD",
               "XLM-USD", "XTZ-USD", "XYO-USD", "YGG-USD", "ZBCN-USD", "ZEN-USD", "ZEREBRO-USD", "ZETA-USD", "ZIG-USD", "ZKJ-USD",
               "ZRX-USD"]
    results = []

    for t in tickers:
        try:
            sigs = compute_signals_for_ticker(t)
            results.append({
                'ticker': t,
                'today_signal': sigs['today'],
                'yesterday_signal': sigs['yesterday'],
                'last_week_signal': sigs['last_week'],
                'last_month_signal': sigs['last_month'],
            })
        except Exception as e:
            print(f"Skipping {t}: {e}")

    return results

@app.route('/signals/summary', methods=['GET'])
def signals_summary():
    csv_version = get_csv_version()  # Step 2 helper
    results = compute_signals_summary_cached(csv_version)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
