from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd  # FIXED
import numpy as np

app = Flask(__name__)

def download_prices(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, interval='1d', progress=False)
    if df.empty:
        return df
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    series = pd.DataFrame()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
        values = df[col].values
        if values.ndim > 1:
            values = values.flatten()
        df[col] = values
        series[col] = pd.Series(values, index=df.index)

    return df

def buy_and_hold_equity(close, cash):
    # normalize to start = cash
    eq = (close / close.iloc[0]) * cash
    return eq

def sma_strategy_equity(close, cash, fast=10, slow=30):
    """
    Simple SMA crossover: long when SMA(fast) > SMA(slow), flat otherwise.
    Emits buy at cross-up and sell at cross-down.
    """
    df = pd.DataFrame({'Close': close})
    df['SMA_f'] = df['Close'].rolling(fast).mean()
    df['SMA_s'] = df['Close'].rolling(slow).mean()
    df.dropna(inplace=True)

    # position: 1 if SMA_f > SMA_s else 0
    df['pos'] = (df['SMA_f'] > df['SMA_s']).astype(int)

    # daily returns
    df['ret'] = df['Close'].pct_change().fillna(0.0)
    # strategy returns (apply pos of previous day)
    df['strat_ret'] = df['pos'].shift(1).fillna(0.0) * df['ret']

    # equity
    eq = (1 + df['strat_ret']).cumprod() * cash

    # signals
    df['pos_prev'] = df['pos'].shift(1).fillna(df['pos'])
    crosses_up = (df['pos_prev'] == 0) & (df['pos'] == 1)
    crosses_dn = (df['pos_prev'] == 1) & (df['pos'] == 0)

    buy_dates = df.index[crosses_up]
    sell_dates = df.index[crosses_dn]

    # y-values of equity at signal times (to place markers at equity level)
    buys = [{'date': d.strftime('%Y-%m-%d'), 'y': float(eq.loc[d])} for d in buy_dates]
    sells = [{'date': d.strftime('%Y-%m-%d'), 'y': float(eq.loc[d])} for d in sell_dates]

    return eq, buys, sells, df.index

def metrics_from_equity(eq_series):
    final_value = float(eq_series.iloc[-1])
    start_value = float(eq_series.iloc[0])
    profit_factor = final_value / start_value if start_value != 0 else np.nan

    # daily returns of the equity curve
    rets = eq_series.pct_change().dropna()
    if rets.std() == 0 or rets.empty:
        sharpe = 0.0
    else:
        rf_annual = 0.01
        rf_daily = (1 + rf_annual) ** (1/252) - 1
        excess = rets - rf_daily
        sharpe = float((excess.mean() / excess.std()) * np.sqrt(252))
    return final_value, profit_factor, sharpe

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
    cash = float(request.form['cash'])
    ticker = request.form['ticker']
    start_date = request.form['start']
    end_date = request.form['end']
    # ticker_2 = request.form['ticker_2']
    ticker_2 = request.form.get('ticker_2', '').strip()  # optional
    print(ticker_2)

    # PRIMARY
    df = download_prices(ticker, start_date, end_date)
    if df.empty:
        return jsonify({'error': f'No data for {ticker} in selected range.'}), 400

    eq_bh = buy_and_hold_equity(df['Close'], cash)
    eq_strat, buys, sells, idx = sma_strategy_equity(df['Close'], cash)
    dates = [d.strftime('%Y-%m-%d') for d in eq_strat.index]
    fv, pf, sh = metrics_from_equity(eq_strat)

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
    # print(equity_curve_2)
    # results = {
    #     'final_value': final_value,
    #     'profit_factor': profit_factor,
    #     'sharpe_ratio': sharpe_ratio,
    #     'equity_curve': equity_curve,
    #     'equity_curve_2': equity_curve_2,
    #     'dates': dates,
    #     'ticker': ticker,
    #     'ticker_2': ticker_2
    # }
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

if __name__ == '__main__':
    app.run(debug=True)
