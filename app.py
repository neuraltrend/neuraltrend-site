from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd  # FIXED
import numpy as np

app = Flask(__name__)

# ----------------------------
# Data utilities
# ----------------------------
def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV for a ticker and ensure numeric columns.
    Returns an empty DataFrame if no data.
    """
    df = yf.download(ticker, start=start, end=end, interval='1d', progress=False)
    if df.empty:
        return df

    # Keep only standard columns we might use later
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df

def buy_and_hold_equity(close: pd.Series, cash: float) -> pd.Series:
    """
    Normalize equity to start at 'cash' and track value over time.
    """
    if close is None or len(close) == 0:
        return pd.Series(dtype=float)
    return (close / close.iloc[0]) * cash

def sma_strategy_equity(df: pd.DataFrame, cash: float, fast: int = 10, slow: int = 30):
    """
    Simple SMA crossover strategy:
      - Long when SMA(fast) > SMA(slow), flat otherwise.
      - Emits buy on cross-up and sell on cross-down.

    Returns:
      eq (pd.Series): strategy equity curve
      buys (list[dict]): [{'date': 'YYYY-MM-DD', 'y': float}, ...]
      sells (list[dict]): [{'date': 'YYYY-MM-DD', 'y': float}, ...]
      idx (pd.DatetimeIndex): index for alignment
    """
    if df is None or df.empty or 'Close' not in df.columns:
        return pd.Series(dtype=float), [], [], pd.DatetimeIndex([])

    if len(df) < max(fast, slow):
        # Not enough data to compute both SMAs
        return pd.Series(dtype=float), [], [], df.index

    work = df.copy()
    work['SMA_f'] = work['Close'].rolling(fast).mean()
    work['SMA_s'] = work['Close'].rolling(slow).mean()
    work = work.dropna()
    if work.empty:
        return pd.Series(dtype=float), [], [], df.index

    # Position: 1 if SMA_f > SMA_s, else 0
    work['pos'] = (work['SMA_f'] > work['SMA_s']).astype(int)

    # Daily returns and strategy returns (use previous day's position)
    work['ret'] = work['Close'].pct_change().fillna(0.0)
    work['strat_ret'] = work['pos'].shift(1).fillna(0.0) * work['ret']

    # Strategy equity curve
    eq = (1 + work['strat_ret']).cumprod() * cash

    # Crossovers for signals
    work['pos_prev'] = work['pos'].shift(1).fillna(work['pos'])
    crosses_up = (work['pos_prev'] == 0) & (work['pos'] == 1)
    crosses_dn = (work['pos_prev'] == 1) & (work['pos'] == 0)

    buy_dates = work.index[crosses_up]
    sell_dates = work.index[crosses_dn]

    buys = [{'date': d.strftime('%Y-%m-%d'), 'y': float(eq.loc[d])} for d in buy_dates]
    sells = [{'date': d.strftime('%Y-%m-%d'), 'y': float(eq.loc[d])} for d in sell_dates]

    return eq, buys, sells, work.index

def metrics_from_equity(eq_series):
    final_value = eq_series.iloc[-1].item()
    print(final_value)
    start_value = eq_series.iloc[0].item()
    print(start_value)
    profit_factor = final_value / start_value if start_value != 0 else np.nan
    print(profit_factor)

    # daily returns of the equity curve
    rets = eq_series.pct_change().dropna()
    # if eq_series came as a DataFrame with one column
    if isinstance(rets, pd.DataFrame):
        rets = rets.iloc[:, 0]  # take the first (and only) column
    print(rets)
    if rets.std() == 0 or rets.empty:
        print('a')
        sharpe = 0.0
        print(sharpe)
    else:
        rf_annual = 0.01
        print('b')
        rf_daily = (1 + rf_annual) ** (1/252) - 1
        print('c')
        excess = rets - rf_daily
        print('d')
        sharpe = float((excess.mean() / excess.std()) * np.sqrt(252))
        print('e')
        print(sharpe)
    return final_value, profit_factor, sharpe

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/market')
def market():
    return render_template('market.html')

@app.route('/knowledge')
def knowledge():
    return render_template('knowledge.html')

@app.route('/backtest', methods=['POST'])
def backtest():
    try:
        cash = float(request.form['cash'])
        ticker = request.form['ticker'].strip()
        start_date = request.form['start']
        end_date = request.form['end']
        ticker_2 = request.form.get('ticker_2', '').strip()  # optional
    
        # --- Primary asset
        df = download_prices(ticker, start_date, end_date)
        if df.empty:
            return jsonify({'error': f'No data for {ticker} in selected range.'}), 400

        # Buy & Hold for baseline
        eq_bh = buy_and_hold_equity(df['Close'], cash)
        print(eq_bh)
        fv_bh, pf_bh, sh_bh = metrics_from_equity(eq_bh)
        print(fv_bh, pf_bh, sh_bh)
    
        series = pd.DataFrame()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
            values = df[col].values
            if values.ndim > 1:
                values = values.flatten()
            df[col] = values
            series[col] = pd.Series(values, index=df.index)
    
        equity_curve = df['Close'].to_numpy().flatten().astype(float).tolist()
        equity_curve_start=equity_curve[0]
        equity_curve = np.array(equity_curve)  # convert list to numpy array
        equity_curve = equity_curve / equity_curve[0] * cash
        equity_curve = equity_curve.tolist()
        final_value = float(equity_curve[-1])
        profit_factor = float(final_value / cash)
    
        returns = df['Close'].pct_change().dropna()
        risk_free_rate_annual = 0.01
        risk_free_rate_daily = (1 + risk_free_rate_annual) ** (1/252) - 1
        excess_returns = returns - risk_free_rate_daily
        sharpe_ratio = float(((excess_returns.mean() / excess_returns.std()) * (252 ** 0.5)).iloc[0])
    
        equity_curve_2=[]
        if ticker_2:
            df_2 = yf.download(ticker_2, start=start_date, end=end_date, interval='1d')  # FIXED
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
            equity_curve_2 = equity_curve_2 / equity_curve_2[0] * cash
            equity_curve_2 = equity_curve_2.tolist()
            final_value_2 = float(equity_curve_2[-1])
            profit_factor_2 = float(final_value_2 / cash)
    
            returns_2 = df_2['Close'].pct_change().dropna()
            excess_returns_2 = returns_2 - risk_free_rate_daily
            sharpe_ratio_2 = float(((excess_returns_2.mean() / excess_returns_2.std()) * (252 ** 0.5)).iloc[0])
        
        dates = df.index.strftime('%Y-%m-%d').tolist()
        print(equity_curve)

        results = {
            'ticker': ticker,
            'final_value': final_value,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'equity_curve': equity_curve,
            'dates': dates,
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

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
