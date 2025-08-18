from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd  # FIXED
import numpy as np
import os

app = Flask(__name__)

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
    initial_cash = float(request.form['cash'])
    ticker = request.form['ticker']
    start_date = request.form['start']
    end_date = request.form['end']
    ticker_2 = request.form['ticker_2']
    # print(ticker_2)

    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')  # FIXED
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    
    # --- Load CSV of signals ---
    csv_path = os.path.join(app.root_path, 'data', 'epoch_BTC.csv')
    signals_df = pd.read_csv(csv_path, parse_dates=['Date'])
    
    # Convert Close to float explicitly
    signals_df['Close'] = pd.to_numeric(signals_df['Close'], errors='coerce')
    signals_df = signals_df.dropna()
    
    # print(signals_df)

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
        # else hold

        equity = cash + position * price
        equity_curve.append((date, equity))

    eq_df = pd.DataFrame(equity_curve, columns=['Date', 'Equity']).set_index('Date')
    # print(eq_df)

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
    equity_curve = equity_curve / equity_curve[0] * initial_cash
    # print(equity_curve)
    equity_curve = equity_curve.tolist()
    final_value = float(equity_curve[-1])
    profit_factor = float(final_value / initial_cash)
    # print(equity_curve)

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
        equity_curve_2 = equity_curve_2 / equity_curve_2[0] * initial_cash
        equity_curve_2 = equity_curve_2.tolist()
        final_value_2 = float(equity_curve_2[-1])
        profit_factor_2 = float(final_value_2 / initial_cash)

        returns_2 = df_2['Close'].pct_change().dropna()
        excess_returns_2 = returns_2 - risk_free_rate_daily
        sharpe_ratio_2 = float(((excess_returns_2.mean() / excess_returns_2.std()) * (252 ** 0.5)).iloc[0])
    
    dates = df.index.strftime('%Y-%m-%d').tolist()
    print(dates)
    print(eq_df)
    print(equity_curve)
    print(equity_curve_2)
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
        'epoch_equity_curve': eq_df,
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
