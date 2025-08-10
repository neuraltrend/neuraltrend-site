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
    start_date = request.form['start']
    end_date = request.form['end']

    # Dummy backtest logic
    final_value = cash * 1.25
    profit_factor = 1.45
    sharpe_ratio = 1.75
    equity_curve = [cash, cash * 1.05, cash * 1.10, final_value]

    results = {
        'final_value': final_value,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'equity_curve': equity_curve
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
