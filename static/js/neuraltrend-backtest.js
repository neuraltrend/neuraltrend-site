
    
    let equityChart = null;

    window.backtestLogScale = window.backtestLogScale || false;
        
    document.getElementById('backtest-form').addEventListener('submit', function(e) {
        e.preventDefault();

        if (
            typeof window.validateBacktestDateRange === "function" &&
            !window.validateBacktestDateRange()
        ) {
            this.reportValidity();
            return;
        }

        if (!this.checkValidity()) {
            this.reportValidity();
            return;
        }

        const normalizedDates = typeof window.getBacktestIsoDates === "function"
            ? window.getBacktestIsoDates()
            : {
                start: document.getElementById("start-iso")?.value || "",
                end: document.getElementById("end-iso")?.value || ""
            };
        const startDate = normalizedDates.start || "";
        const endDate = normalizedDates.end || "";
        if (!startDate || !endDate || startDate >= endDate) {
            const endInput = document.getElementById('end');
            if (endInput) {
                endInput.setCustomValidity('End date must be after the start date.');
                endInput.reportValidity();
                endInput.setCustomValidity('');
            }
            return;
        }
    
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

    function formatBacktestPercent(value, decimals = 1, colorize = false, signed = true) {
        if (value === null || value === undefined) {
            return colorize
                ? `<span class="return-value return-neutral">—</span>`
                : "—";
        }

        const numberValue = Number(value);
        if (!Number.isFinite(numberValue)) {
            return colorize
                ? `<span class="return-value return-neutral">—</span>`
                : "—";
        }

        const percent = numberValue * 100;
        const sign = signed && percent > 0 ? "+" : "";
        const text = `${sign}${percent.toFixed(decimals)}%`;

        if (!colorize) return text;

        let cls = "return-neutral";
        if (numberValue > 0) cls = "return-positive";
        if (numberValue < 0) cls = "return-negative";

        return `<span class="return-value ${cls}">${text}</span>`;
    }

    function renderResults({
        ticker,
        final_value,
        final_value_epoch,
        buy_hold_growth_factor,
        strategy_growth_factor,
        return_spread,
        sharpe_ratio,
        strategy_max_drawdown,
        buy_hold_max_drawdown,
        strategy_annualized_volatility,
        buy_hold_annualized_volatility,
        strategy_market_exposure,
        executed_trade_count = 0,
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
        
                <section class="nt-backtest-summary-panel" aria-label="Backtest performance and risk summary">
                    <div class="nt-backtest-summary-grid">

                        <div class="nt-result-card nt-backtest-summary-card">
                            <div class="nt-result-label">AI Strategy Final Value</div>
                            <div class="nt-result-value">${formatBacktestCurrency(final_value_epoch)}</div>
                            <div class="nt-result-sub">EpochSignaler result</div>
                        </div>

                        <div class="nt-result-card nt-backtest-summary-card">
                            <div class="nt-result-label">Buy & Hold Final Value</div>
                            <div class="nt-result-value">${formatBacktestCurrency(final_value)}</div>
                            <div class="nt-result-sub">Benchmark result</div>
                        </div>

                        <div class="nt-result-card nt-backtest-summary-card">
                            <div class="nt-result-label">Sharpe Ratio</div>
                            <div class="nt-result-value">${formatBacktestNumber(sharpe_ratio, 2)}</div>
                            <div class="nt-result-sub">Based on selected period</div>
                        </div>

                        <div class="nt-result-card nt-backtest-summary-card">
                            <div class="nt-result-label">AI Strategy Return</div>
                            <div class="nt-result-value">${aiReturnHTML}</div>
                            <div class="nt-result-sub">Compared to initial cash</div>
                        </div>

                        <div class="nt-result-card nt-backtest-summary-card">
                            <div class="nt-result-label">Buy & Hold Return</div>
                            <div class="nt-result-value">${buyHoldReturnHTML}</div>
                            <div class="nt-result-sub">Compared to initial cash</div>
                        </div>

                        <div class="nt-result-card nt-backtest-summary-card">
                            <div class="nt-result-label">AI Return Spread</div>
                            <div class="nt-result-value">${formatPointSpread(return_spread)}</div>
                            <div class="nt-result-sub">AI minus Buy & Hold</div>
                        </div>

                        <div class="nt-result-card nt-backtest-summary-card">
                            <div class="nt-result-label">AI Max Drawdown</div>
                            <div class="nt-result-value">${formatBacktestPercent(strategy_max_drawdown, 1, true)}</div>
                            <div class="nt-result-sub">Strategy equity curve</div>
                        </div>

                        <div class="nt-result-card nt-backtest-summary-card">
                            <div class="nt-result-label">Buy &amp; Hold Max Drawdown</div>
                            <div class="nt-result-value">${formatBacktestPercent(buy_hold_max_drawdown, 1, true)}</div>
                            <div class="nt-result-sub">Benchmark equity curve</div>
                        </div>

                        <div class="nt-result-card nt-backtest-summary-card">
                            <div class="nt-result-label">AI Annualized Volatility</div>
                            <div class="nt-result-value">${formatBacktestPercent(strategy_annualized_volatility, 1, false, false)}</div>
                            <div class="nt-result-sub">Daily strategy returns</div>
                        </div>

                        <div class="nt-result-card nt-backtest-summary-card">
                            <div class="nt-result-label">B&amp;H Annualized Volatility</div>
                            <div class="nt-result-value">${formatBacktestPercent(buy_hold_annualized_volatility, 1, false, false)}</div>
                            <div class="nt-result-sub">Daily benchmark returns</div>
                        </div>

                        <div class="nt-result-card nt-backtest-summary-card">
                            <div class="nt-result-label">Market Exposure</div>
                            <div class="nt-result-value">${formatBacktestPercent(strategy_market_exposure, 1, false, false)}</div>
                            <div class="nt-result-sub">Days holding a position</div>
                        </div>

                        <div class="nt-result-card nt-backtest-summary-card">
                            <div class="nt-result-label">Executed Trades</div>
                            <div class="nt-result-value">${Number(executed_trade_count || 0).toLocaleString()}</div>
                            <div class="nt-result-sub">BUY and SELL executions</div>
                        </div>

                    </div>
                </section>
        
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
    
        // Keep execution markers on the actual strategy-equity observation.
        // A fixed dollar offset looks reasonable on a linear axis but becomes
        // badly distorted after switching to logarithmic scale. Exact y-values
        // are scale-independent and stay attached to the trade they represent.
        const strategyEquityByDate = new Map();
        dates.forEach((dateValue, index) => {
            const equityValue = Number(epoch_equity_curve[index]);
            if (dateValue && Number.isFinite(equityValue) && equityValue > 0) {
                strategyEquityByDate.set(dateValue, equityValue);
            }
        });

        function executionMarkerPoints(executedDates) {
            return Array.from(new Set(executedDates || []))
                .map(dateValue => {
                    const equityValue = strategyEquityByDate.get(dateValue);
                    return Number.isFinite(equityValue)
                        ? {x: dateValue, y: equityValue}
                        : null;
                })
                .filter(Boolean);
        }

        const adjustedBuySignals = executionMarkerPoints(executed_buy_dates);
        const adjustedSellSignals = executionMarkerPoints(executed_sell_dates);
    
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
                type: 'scatter',
                label: 'Executed BUY',
                data: adjustedBuySignals,
                pointStyle: 'triangle',
                pointRadius: 6,
                pointHoverRadius: 8,
                pointBorderWidth: 1.25,
                backgroundColor: NT_COLORS.buy,
                borderColor: "#ffffff",
                showLine: false,
                order: 20
            },
            {
                type: 'scatter',
                label: 'Executed SELL',
                data: adjustedSellSignals,
                pointStyle: 'triangle',
                rotation: 180,
                pointRadius: 6,
                pointHoverRadius: 8,
                pointBorderWidth: 1.25,
                backgroundColor: NT_COLORS.sell,
                borderColor: "#ffffff",
                showLine: false,
                order: 20
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
