
    
    let equityChart = null;

    window.backtestLogScale = window.backtestLogScale || false;
        
    document.getElementById('backtest-form').addEventListener('submit', function(e) {
        e.preventDefault();
    
        const formData = new FormData(this);
        const resultsDiv = document.getElementById('results');
    
        fetch('/backtest', {
            method: 'POST',
            body: formData
        })
        .then(async response => {
            const data = await response.json();
    
            if (!response.ok) {
                if (data.upgrade_required) {
                    resultsDiv.innerHTML = `
                        <div class="nt-pro-lock-card">
                            <div class="nt-pro-lock-icon">🔒</div>
    
                            <h3>Pro backtest locked</h3>
    
                            <p>
                                Backtesting for ${escapeHTML(data.ticker || "this asset")}
                                is available with NeuralTrend Pro.
                            </p>
    
                            <button type="button" class="nt-pro-upgrade-btn"
                            data-nt-go-subscription>
                                Upgrade to Pro
                            </button>
                        </div>
                    `;
                } else {
                    resultsDiv.innerHTML = `
                        <div style="color: var(--nt-negative); font-weight: 700;">
                            ${escapeHTML(data.error || "Backtest failed.")}
                        </div>
                    `;
                }
    
                return;
            }
    
            renderResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
    
            resultsDiv.innerHTML = `
                <div style="color: var(--nt-negative); font-weight: 700;">
                    Backtest failed. Please try again.
                </div>
            `;
        });
    });

    function formatBacktestCurrency(value) {
        if (value === null || value === undefined || Number.isNaN(value)) {
            return "—";
        }
    
        return "$" + Number(value).toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });
    }
    
    function formatBacktestReturnFromGrowthFactor(growthFactor) {
        if (
            growthFactor === null ||
            growthFactor === undefined ||
            Number.isNaN(growthFactor)
        ) {
            return `<span class="return-value return-neutral">—</span>`;
        }
    
        const returnValue = growthFactor - 1;
        const percent = returnValue * 100;
        const sign = percent > 0 ? "+" : "";
    
        let cls = "return-neutral";
        if (returnValue > 0) cls = "return-positive";
        if (returnValue < 0) cls = "return-negative";
    
        return `
            <span class="return-value ${cls}">
                ${sign}${percent.toFixed(1)}%
            </span>
        `;
    }
    
    function formatBacktestNumber(value, decimals = 2) {
        if (value === null || value === undefined || Number.isNaN(value)) {
            return "—";
        }
    
        return Number(value).toFixed(decimals);
    }

    function renderResults({
        ticker,
        final_value,
        final_value_epoch,
        buy_hold_growth_factor,
        strategy_growth_factor,
        return_spread,
        sharpe_ratio,
        transaction_cost_rate = 0,
        valuation_policy = "mark_to_market",
        ending_position_open = false,
        equity_curve,
        epoch_equity_curve,
        dates,
        executed_buy_dates = [],
        executed_sell_dates = []
    }) {
        const resultsDiv = document.getElementById('results');
    
        const aiReturnHTML = formatBacktestReturnFromGrowthFactor(strategy_growth_factor);
        const buyHoldReturnHTML = formatBacktestReturnFromGrowthFactor(buy_hold_growth_factor);
        
        resultsDiv.innerHTML = `
            <div class="nt-backtest-results">
        
                <div class="nt-backtest-results-header">
                    <div>
                        <div class="nt-section-eyebrow">Backtest Results</div>
                        <h3 class="nt-backtest-results-title">${escapeHTML(ticker)} Backtest Results</h3>
                        <p class="nt-backtest-results-subtitle">
                            AI strategy performance compared with buy & hold. Transaction cost: ${(transaction_cost_rate * 100).toFixed(2)}% per executed trade. ${valuation_policy === "mark_to_market" ? "Open positions are marked to market at the final date." : ""} ${ending_position_open ? "The strategy ends with an open position." : "The strategy ends in cash."}
                        </p>
                    </div>
        
                </div>
        
                <div class="nt-backtest-results-grid">
        
                    <div class="nt-result-card">
                        <div class="nt-result-label">AI Strategy Final Value</div>
                        <div class="nt-result-value">${formatBacktestCurrency(final_value_epoch)}</div>
                        <div class="nt-result-sub">EpochSignaler result</div>
                    </div>
        
                    <div class="nt-result-card">
                        <div class="nt-result-label">Buy & Hold Final Value</div>
                        <div class="nt-result-value">${formatBacktestCurrency(final_value)}</div>
                        <div class="nt-result-sub">Benchmark result</div>
                    </div>
        
                    <div class="nt-result-card">
                        <div class="nt-result-label">Sharpe Ratio</div>
                        <div class="nt-result-value">${formatBacktestNumber(sharpe_ratio, 2)}</div>
                        <div class="nt-result-sub">Based on selected period</div>
                    </div>
        
                    <div class="nt-result-card">
                        <div class="nt-result-label">AI Strategy Return</div>
                        <div class="nt-result-value">${aiReturnHTML}</div>
                        <div class="nt-result-sub">Compared to initial cash</div>
                    </div>
        
                    <div class="nt-result-card">
                        <div class="nt-result-label">Buy & Hold Return</div>
                        <div class="nt-result-value">${buyHoldReturnHTML}</div>
                        <div class="nt-result-sub">Compared to initial cash</div>
                    </div>
        
                    <div class="nt-result-card">
                        <div class="nt-result-label">AI Return Spread</div>
                        <div class="nt-result-value">${formatPointSpread(return_spread)}</div>
                        <div class="nt-result-sub">AI return minus Buy & Hold return</div>
                    </div>
        
                </div>
        
                <div class="nt-backtest-chart-card">

                    <div class="nt-backtest-chart-header">
                        <div class="nt-backtest-chart-title">
                            Equity Curve: AI Strategy vs Buy & Hold
                        </div>
                
                        <div class="nt-backtest-chart-actions">
                            <span class="nt-result-pill">
                                Scale: 
                                <strong id="backtest-scale-mode">
                                    ${window.backtestLogScale ? "Log" : "Linear"}
                                </strong>
                            </span>
                
                            <button id="backtest-scale-toggle-btn" class="nt-scale-btn" type="button">
                                ${window.backtestLogScale ? "Normal Scale" : "Log Scale"}
                            </button>
                        </div>
                    </div>
                
                    <div class="nt-backtest-chart-shell">
                        <canvas id="equity-chart"></canvas>
                    </div>
                
                </div>
        
            </div>
        `;
    
        const ctx = document.getElementById('equity-chart').getContext('2d');
    
        // --- Compute dynamic offset for markers ---
        const equity_min = Math.min(...epoch_equity_curve);
        const equity_max = Math.max(...epoch_equity_curve);
        const offset = (equity_max - equity_min) * 0.04; // 4% of range
    
        const adjustedBuySignals = dates.map((d, i) => executed_buy_dates.includes(d) ? epoch_equity_curve[i] - offset : null);
        const adjustedSellSignals = dates.map((d, i) => executed_sell_dates.includes(d) ? epoch_equity_curve[i] + offset : null);
    
        const datasets = [
            {
                label: `${ticker} EpochSignaler Equity Curve`,
                data: epoch_equity_curve,
                borderColor: NT_COLORS.chartAI,
                backgroundColor: NT_COLORS.chartAI,
                pointStyle: 'line',
                pointRadius: 0,
                borderWidth: 2,
                fill: false
            },
            {
                label: `${ticker} Buy&Hold Equity Curve`,
                data: equity_curve,
                borderColor: NT_COLORS.chartBuyHold,
                backgroundColor: NT_COLORS.chartBuyHold,
                pointStyle: 'line',
                borderDash: [5, 5],
                pointRadius: 0,
                borderWidth: 2,
                fill: false
            },
            {
                label: 'Executed BUY',
                data: adjustedBuySignals,
                pointStyle: 'triangle',
                pointRadius: 10,
                backgroundColor: NT_COLORS.buy,
                borderColor: NT_COLORS.buy,
                showLine: false
            },
            {
                label: 'Executed SELL',
                data: adjustedSellSignals,
                pointStyle: 'triangle',
                rotation: 180,
                pointRadius: 10,
                backgroundColor: NT_COLORS.sell,
                borderColor: NT_COLORS.sell,
                showLine: false
            }
        ];
    
        // new Chart(ctx, {
        if (equityChart) {
            equityChart.destroy();
        }
        
        equityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: datasets
            },
            options: ntChartOptions({
                logScale: window.backtestLogScale,
                yTitle: "Equity ($)",
                yPrefix: "$",
                legendAlign: "end"
            })
        });
    }
