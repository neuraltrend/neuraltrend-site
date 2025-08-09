from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd  # FIXED
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/backtest', methods=['POST'])
def backtest():
    cash = float(request.form['cash'])
    ticker = request.form['ticker']  # FIXED
    start_date = request.form['start']
    end_date = request.form['end']

    final_value = cash * 1.25
    profit_factor = 1.45
    sharpe_ratio = 1.75

    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')  # FIXED
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

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
    print(equity_curve)
    print(equity_curve_start)
    print(cash)
    equity_curve = np.array(equity_curve)  # convert list to numpy array
    equity_curve = equity_curve / equity_curve[0] * cash
    equity_curve = equity_curve.tolist()
    profit_factor = float(final_value / cash)
    final_value = float(equity_curve[-1])
    
    returns = df['Close'].pct_change().dropna()

    risk_free_rate_annual = 0.01
    risk_free_rate_daily = (1 + risk_free_rate_annual) ** (1/252) - 1

    excess_returns = returns - risk_free_rate_daily

    sharpe_ratio = float((excess_returns.mean() / excess_returns.std()) * (252 ** 0.5))



    print("Equity curve length:", len(equity_curve))
    print("First few values:", equity_curve[:5])
    
    dates = df.index.strftime('%Y-%m-%d').tolist()

    results = {
        'final_value': final_value,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'equity_curve': equity_curve,
        'dates': dates
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
