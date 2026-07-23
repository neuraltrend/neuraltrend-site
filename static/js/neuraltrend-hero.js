
/* ===============================
   Homepage BTC performance hero
=============================== */

(function () {
    let ntHeroPerformanceChart = null;
    let ntHeroRequestController = null;

    let ntHeroLogScale = false;
    let ntHeroLastData = null;
    let ntHeroLastDuration = "5y";

    let ntHeroTicker = "BTC-USD";
    const DEFAULT_HERO_DURATION = "5y";

    function heroAssetLabel(ticker) {
        const labels = {
            "BTC-USD": "BTC",
            "ETH-USD": "ETH",
            "SOL-USD": "SOL",
            "XRP-USD": "XRP"
        };
    
        return labels[ticker] || ticker.replace("-USD", "");
    }

    function parseHeroDate(value) {
        const rawValue = String(value || "").trim();

        // Prevent YYYY-MM-DD from shifting one day because of UTC parsing.
        const match = rawValue.match(/^(\d{4})-(\d{2})-(\d{2})$/);

        if (match) {
            return new Date(
                Number(match[1]),
                Number(match[2]) - 1,
                Number(match[3])
            );
        }

        return new Date(rawValue);
    }

    function formatHeroAxisDate(value, duration) {
        const date = parseHeroDate(value);

        if (Number.isNaN(date.getTime())) {
            return String(value || "");
        }

        if (duration === "1y") {
            return date.toLocaleDateString(undefined, {
                month: "short",
                year: "2-digit"
            });
        }

        return date.toLocaleDateString(undefined, {
            month: "short",
            year: "2-digit"
        });
    }

    function formatHeroTooltipDate(value) {
        const date = parseHeroDate(value);

        if (Number.isNaN(date.getTime())) {
            return String(value || "");
        }

        return date.toLocaleDateString(undefined, {
            year: "numeric",
            month: "short",
            day: "numeric"
        });
    }

    function heroDurationLabel(duration) {
        const labels = {
            "3mo": "3M",
            "6mo": "6M",
            "1y": "1Y",
            "3y": "3Y",
            "5y": "5Y",
            "10y": "10Y",
            "max": "MAX"
        };
    
        return labels[duration] || String(duration || "").toUpperCase();
    }

    function formatHeroPercent(value) {
        const numberValue = Number(value);

        if (!Number.isFinite(numberValue)) {
            return "—";
        }

        const sign = numberValue > 0 ? "+" : "";

        return `${sign}${numberValue.toFixed(0)}%`;
    }

    function normalizeHeroCurve(curve) {
        if (!Array.isArray(curve) || curve.length === 0) {
            return [];
        }

        const firstValidValue = curve.find(value =>
            Number.isFinite(Number(value)) && Number(value) !== 0
        );

        const baseValue = Number(firstValidValue);

        if (!Number.isFinite(baseValue) || baseValue === 0) {
            return [];
        }

        /*
          Convert each equity value into percentage return from the
          beginning of the selected horizon.

          Example:
          1.00 → 0%
          1.50 → +50%
          0.80 → -20%
        */
        return curve.map(value => {
            const numericValue = Number(value);

            if (!Number.isFinite(numericValue)) {
                return null;
            }

            return ((numericValue / baseValue) - 1) * 100;
        });
    }

    function normalizeHeroEquityCurve(curve) {
        if (!Array.isArray(curve) || curve.length === 0) {
            return [];
        }
    
        const firstValidValue = curve.find(value =>
            Number.isFinite(Number(value)) && Number(value) > 0
        );
    
        const baseValue = Number(firstValidValue);
    
        if (!Number.isFinite(baseValue) || baseValue <= 0) {
            return [];
        }
    
        return curve.map(value => {
            const numericValue = Number(value);
    
            if (!Number.isFinite(numericValue) || numericValue <= 0) {
                return null;
            }
    
            return numericValue / baseValue;
        });
    }

    function setHeroLoadingState(message, isError = false) {
        const loadingElement = document.getElementById(
            "nt-market-hero-chart-loading"
        );

        if (!loadingElement) {
            return;
        }

        loadingElement.textContent = message;
        loadingElement.style.display = "flex";
        loadingElement.style.color = isError
            ? "rgba(248, 113, 113, 0.92)"
            : "rgba(255, 255, 255, 0.58)";
    }

    function hideHeroLoadingState() {
        const loadingElement = document.getElementById(
            "nt-market-hero-chart-loading"
        );

        if (loadingElement) {
            loadingElement.style.display = "none";
        }
    }

    function setActiveHeroDuration(duration) {
        document
            .querySelectorAll(".nt-market-hero-period")
            .forEach(button => {
                const isActive =
                    button.dataset.heroDuration === duration;

                button.classList.toggle("active", isActive);
                button.setAttribute(
                    "aria-pressed",
                    isActive ? "true" : "false"
                );
            });
    }

    function getHeroFinalValidValue(values) {
        if (!Array.isArray(values)) {
            return null;
        }
    
        for (let index = values.length - 1; index >= 0; index -= 1) {
            const value = Number(values[index]);
    
            if (Number.isFinite(value)) {
                return value;
            }
        }
    
        return null;
    }
    
    function setHeroMetricValue(elementId, value) {
        const element = document.getElementById(elementId);
    
        if (!element) {
            return;
        }
    
        if (!Number.isFinite(value)) {
            element.textContent = "—";
            return;
        }
    
        const sign = value > 0 ? "+" : "";
    
        element.textContent =
            `${sign}${value.toLocaleString(undefined, {
                minimumFractionDigits: 1,
                maximumFractionDigits: 1
            })}%`;
    }

    function setHeroSpreadValue(elementId, percentagePoints) {
        const element = document.getElementById(elementId);

        if (!element) return;

        if (!Number.isFinite(percentagePoints)) {
            element.textContent = "—";
            return;
        }

        const sign = percentagePoints > 0 ? "+" : "";
        element.textContent = `${sign}${percentagePoints.toFixed(1)} pts`;
    }
    
    function updateHeroPerformanceMetrics(
        strategyReturns,
        buyHoldReturns,
        duration
    ) {
        const strategyReturn =
            getHeroFinalValidValue(strategyReturns);
    
        const buyHoldReturn =
            getHeroFinalValidValue(buyHoldReturns);
    
        const returnSpreadPoints =
            Number.isFinite(strategyReturn) &&
            Number.isFinite(buyHoldReturn)
                ? strategyReturn - buyHoldReturn
                : null;
    
        setHeroMetricValue(
            "nt-hero-strategy-return",
            strategyReturn
        );
    
        setHeroMetricValue(
            "nt-hero-buyhold-return",
            buyHoldReturn
        );
    
        setHeroSpreadValue(
            "nt-hero-outperformance",
            returnSpreadPoints
        );
    
        document
            .querySelectorAll(".nt-hero-metric-horizon")
            .forEach(element => {
                element.textContent =
                    heroDurationLabel(duration);
            });
    
        const alphaCard = document.querySelector(
            ".nt-market-hero-metric-alpha"
        );
    
        if (alphaCard) {
            alphaCard.classList.toggle(
                "is-negative",
                Number.isFinite(returnSpreadPoints) &&
                returnSpreadPoints < 0
            );
        }
    }

    function updateHeroDataFreshness(data) {
        const element = document.getElementById("nt-hero-data-freshness");
        if (!element) return;

        const status = typeof ntSafeFreshnessStatus === "function"
            ? ntSafeFreshnessStatus(data?.freshness_status)
            : "unknown";
        const label = String(data?.freshness_label || "Unknown");
        const dataDate = typeof ntFormatDataDate === "function"
            ? ntFormatDataDate(data?.data_through, true)
            : String(data?.data_through || "—");
        const updated = typeof ntFormatUtcDataTimestamp === "function"
            ? ntFormatUtcDataTimestamp(data?.site_data_updated_at_utc)
            : String(data?.site_data_updated_at_utc || "—");

        element.className = `nt-market-hero-freshness nt-freshness-${status}`;
        element.textContent = `Data through ${dataDate} · ${label}`;
        element.title = `${String(data?.freshness_message || "Freshness unavailable.")} Site data file updated ${updated}.`;
    }

    function updateHeroAssetText() {
        const asset = heroAssetLabel(ntHeroTicker);

        const title = document.getElementById(
            "nt-market-hero-performance-title"
        );
        
        if (title) {
            title.textContent =
                `${heroDurationLabel(ntHeroLastDuration)} performance ` +
                `of AI strategy vs Buy & Hold for ${asset}`;
        }
    
        const legend = document.getElementById(
            "nt-hero-buyhold-legend"
        );
    
        if (legend) {
            legend.textContent = `Buy & Hold (${asset})`;
        }
    
        const metricSubtitle = document.getElementById(
            "nt-hero-buyhold-metric-subtitle"
        );
    
        if (metricSubtitle) {
            metricSubtitle.textContent = `Buy & Hold ${asset}`;
        }
    
        const canvas = document.getElementById(
            "nt-market-hero-chart"
        );
    
        if (canvas) {
            canvas.setAttribute(
                "aria-label",
                `${asset} NeuralTrend strategy compared with buy and hold`
            );
        }
    }

    function renderHeroPerformanceChart(data, duration) {
        ntHeroLastData = data;
        ntHeroLastDuration = duration;
        
        updateHeroAssetText();
        updateHeroDataFreshness(data);
        const canvas = document.getElementById(
            "nt-market-hero-chart"
        );

        if (!canvas) {
            return;
        }

        const dates = Array.isArray(data.dates)
            ? data.dates
            : [];

        const strategyReturns = normalizeHeroCurve(
            data.epoch_equity_curve
        );
        
        const buyHoldReturns = normalizeHeroCurve(
            data.equity_curve
        );
        
        const strategyEquity = normalizeHeroEquityCurve(
            data.epoch_equity_curve
        );
        
        const buyHoldEquity = normalizeHeroEquityCurve(
            data.equity_curve
        );
        
        const strategyDisplayData = ntHeroLogScale
            ? strategyEquity
            : strategyReturns;
        
        const buyHoldDisplayData = ntHeroLogScale
            ? buyHoldEquity
            : buyHoldReturns;

        if (
            dates.length < 2 ||
            strategyDisplayData.length !== dates.length ||
            buyHoldDisplayData.length !== dates.length
        ) {
            throw new Error("Incomplete equity data received.");
        }
        
        updateHeroPerformanceMetrics(
            strategyReturns,
            buyHoldReturns,
            duration
        );

        if (ntHeroPerformanceChart) {
            ntHeroPerformanceChart.destroy();
            ntHeroPerformanceChart = null;
        }

        const context = canvas.getContext("2d");

        /*
          Adds a restrained glow only to the AI strategy curve.
          It does not affect the Buy & Hold line, axes, or labels.
        */
        const ntHeroStrategyGlow = {
            id: "ntHeroStrategyGlow",
        
            beforeDatasetDraw(chart, args) {
                if (args.index !== 0) {
                    return;
                }
        
                const ctx = chart.ctx;
        
                ctx.save();
                ctx.shadowColor = "rgba(96, 165, 250, 0.58)";
                ctx.shadowBlur = 10;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 0;
            },
        
            afterDatasetDraw(chart, args) {
                if (args.index !== 0) {
                    return;
                }
        
                chart.ctx.restore();
            }
        };
        
        ntHeroPerformanceChart = new Chart(context, {
            type: "line",

            plugins: [ntHeroStrategyGlow],

            data: {
                labels: dates,

                datasets: [
                    {
                        label: "NeuralTrend Strategy",
                        data: strategyDisplayData,

                        borderColor: "#6BB6FF",
                        backgroundColor: "rgba(107, 182, 255, 0.10)",
                        
                        borderWidth: 3.2,
                        borderCapStyle: "round",
                        borderJoinStyle: "round",
                        
                        pointRadius: 0,
                        pointHoverRadius: 4.5,
                        pointHoverBackgroundColor: "#FFFFFF",
                        pointHoverBorderColor: "#60A5FA",
                        pointHoverBorderWidth: 2,
                        
                        tension: 0.24,
                        cubicInterpolationMode: "monotone",

                        /*
                          Fill the space between this curve and dataset 1.

                          Green:
                          strategy is visually above Buy & Hold.

                          Red:
                          strategy is visually below Buy & Hold.
                        */
                        fill: {
                            target: 1,
                            above: "rgba(34, 197, 94, 0.21)",
                            below: "rgba(239, 68, 68, 0.18)"
                        }
                    },

                    {
                        label: `Buy & Hold (${heroAssetLabel(ntHeroTicker)})`,
                        data: buyHoldDisplayData,

                        borderColor: "rgba(216, 180, 254, 0.66)",
                        backgroundColor: "transparent",
                        
                        borderWidth: 1.55,
                        borderDash: [7, 6],
                        borderCapStyle: "round",
                        borderJoinStyle: "round",
                        
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        pointHoverBackgroundColor: "#FFFFFF",
                        pointHoverBorderColor: "#C084FC",
                        pointHoverBorderWidth: 2,
                        
                        tension: 0.24,
                        cubicInterpolationMode: "monotone",
                        fill: false
                    }
                ]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false,

                animation: {
                    duration: 1050,
                    easing: "easeOutCubic"
                },

                animations: {
                x: {
                    duration: 1050,
                    easing: "easeOutCubic"
                },
            
                y: {
                    duration: 1050,
                    easing: "easeOutCubic"
                }
            },

                interaction: {
                    mode: "index",
                    intersect: false
                },

                layout: {
                    padding: {
                        top: 12,
                        right: 12,
                        bottom: 5,
                        left: 4
                    }
                },

                plugins: {
                    /*
                      The hero has its own custom legend underneath,
                      so Chart.js's built-in legend is hidden.
                    */
                    legend: {
                        display: false
                    },

                    tooltip: {
                        mode: "index",
                        intersect: false,

                        backgroundColor: "rgba(2, 6, 23, 0.97)",
                        titleColor: "#F8FAFC",
                        bodyColor: "#E2E8F0",
                        
                        borderColor: "rgba(148, 163, 184, 0.28)",
                        borderWidth: 1,
                        
                        cornerRadius: 12,
                        padding: 13,
                        displayColors: true,
                        boxWidth: 9,
                        boxHeight: 9,
                        boxPadding: 6,
                        
                        titleFont: {
                            size: 12,
                            weight: "700"
                        },
                        
                        bodyFont: {
                            size: 12,
                            weight: "700"
                        },
                        
                        footerFont: {
                            size: 11,
                            weight: "600"
                        },
                        
                        caretSize: 6,
                        caretPadding: 9,

                        callbacks: {
                            title(items) {
                                return formatHeroTooltipDate(
                                    items?.[0]?.label
                                );
                            },

                            label(context) {
                                const label =
                                    context.dataset.label || "";
                            
                                const value =
                                    context.parsed?.y;
                            
                                if (ntHeroLogScale) {
                                    if (!Number.isFinite(Number(value))) {
                                        return `${label}: —`;
                                    }
                            
                                    return `${label}: $1 → $${Number(value).toFixed(2)}`;
                                }
                            
                                return `${label}: ${formatHeroPercent(value)}`;
                            },

                            footer(items) {
                                if (!Array.isArray(items) || items.length < 2) {
                                    return "";
                                }
                            
                                const strategyValue = Number(items[0]?.parsed?.y);
                                const buyHoldValue = Number(items[1]?.parsed?.y);
                            
                                if (
                                    !Number.isFinite(strategyValue) ||
                                    !Number.isFinite(buyHoldValue)
                                ) {
                                    return "";
                                }
                            
                                const spreadPoints = ntHeroLogScale
                                    ? (strategyValue - buyHoldValue) * 100
                                    : strategyValue - buyHoldValue;
                            
                                const sign = spreadPoints > 0 ? "+" : "";
                            
                                return `AI return spread: ${sign}${spreadPoints.toFixed(1)} pts`;
                            }
                        }
                    }
                },

                scales: {
                    x: {
                        grid: {
                            color: "rgba(148, 163, 184, 0.055)",
                            drawTicks: false,
                            lineWidth: 1
                        },

                        border: {
                            display: false
                        },

                        ticks: {
                            autoSkip: true,
                            maxTicksLimit:
                                window.innerWidth < 700 ? 4 : 6,

                            maxRotation: 0,
                            minRotation: 0,
                            padding: 9,

                            color: "rgba(255, 255, 255, 0.44)",

                            font: {
                                size: 10,
                                weight: "700"
                            },

                            callback(value) {
                                const label =
                                    this.getLabelForValue(value);

                                return formatHeroAxisDate(
                                    label,
                                    duration
                                );
                            }
                        }
                    },

                    y: {
                        type: ntHeroLogScale
                            ? "logarithmic"
                            : "linear",
                        
                        grid: {
                            color(context) {
                                return Number(context.tick?.value) === 0
                                    ? "rgba(226, 232, 240, 0.22)"
                                    : "rgba(148, 163, 184, 0.065)";
                            },
                        
                            lineWidth(context) {
                                return Number(context.tick?.value) === 0
                                    ? 1.3
                                    : 1;
                            },
                        
                            drawTicks: false
                        },

                        border: {
                            display: false
                        },

                        ticks: {
                            maxTicksLimit:
                                window.innerWidth < 700 ? 4 : 5,

                            padding: 8,

                            color: "rgba(255, 255, 255, 0.44)",

                            font: {
                                size: 10,
                                weight: "700"
                            },

                            callback(value) {
                                if (ntHeroLogScale) {
                                    const numericValue = Number(value);
                            
                                    if (!Number.isFinite(numericValue)) {
                                        return "";
                                    }
                            
                                    return `$${numericValue.toFixed(
                                        numericValue < 10 ? 1 : 0
                                    )}`;
                                }
                            
                                return formatHeroPercent(value);
                            }
                        }
                    }
                }
            }
        });

        hideHeroLoadingState();
    }

    async function loadHeroPerformanceChart(
        duration = DEFAULT_HERO_DURATION
    ) {
        setActiveHeroDuration(duration);
        setHeroLoadingState(
            `Loading ${heroAssetLabel(ntHeroTicker)} performance…`
        );

        /*
          Cancel the prior request when users switch horizons quickly.
        */
        if (ntHeroRequestController) {
            ntHeroRequestController.abort();
        }

        ntHeroRequestController = new AbortController();

        const formData = new FormData();
        formData.append("ticker", ntHeroTicker);
        formData.append("duration", duration);

        try {
            const response = await fetch("/equity", {
                method: "POST",
                body: formData,
                signal: ntHeroRequestController.signal
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.error ||
                    `Could not load ${heroAssetLabel(ntHeroTicker)} performance.`
                );
            }

            renderHeroPerformanceChart(data, duration);

        } catch (error) {
            if (error.name === "AbortError") {
                return;
            }

            console.error("Hero performance chart error:", error);

            setHeroLoadingState(
                `${heroAssetLabel(ntHeroTicker)} performance is temporarily unavailable.`,
                true
            );
        }
    }

    function initializeHeroPerformanceChart() {
        const canvas = document.getElementById(
            "nt-market-hero-chart"
        );

        if (!canvas) {
            return;
        }

        document
            .querySelectorAll(".nt-market-hero-period")
            .forEach(button => {
                button.addEventListener("click", function () {
                    const duration =
                        this.dataset.heroDuration;

                    if (!duration) {
                        return;
                    }

                    loadHeroPerformanceChart(duration);
                });
            });

        document
            .querySelectorAll(".nt-market-hero-asset-btn")
            .forEach(button => {
                button.addEventListener("click", function () {
                    const ticker =
                        this.dataset.heroTicker;
        
                    if (!ticker || ticker === ntHeroTicker) {
                        return;
                    }
        
                    ntHeroTicker = ticker;
        
                    document
                        .querySelectorAll(".nt-market-hero-asset-btn")
                        .forEach(assetButton => {
                            const isSelected =
                                assetButton.dataset.heroTicker === ticker;
        
                            assetButton.classList.toggle(
                                "active",
                                isSelected
                            );
        
                            assetButton.setAttribute(
                                "aria-pressed",
                                isSelected ? "true" : "false"
                            );
                        });
        
                    /*
                      Discard old cached data because it belongs
                      to the previously selected asset.
                    */
                    ntHeroLastData = null;
        
                    loadHeroPerformanceChart(
                        ntHeroLastDuration || DEFAULT_HERO_DURATION
                    );
                });
            });

        const moreAssetsButton = document.getElementById(
            "nt-market-hero-more-assets"
        );
        
        if (moreAssetsButton) {
            moreAssetsButton.addEventListener("click", function () {
        
                /*
                  Activate EpochSignaler in case another model
                  was previously selected.
                */
                const epochModelButton = document.querySelector(
                    '.nt-model-tab[data-model-target="epochsignaler-section"]'
                );
        
                if (
                    epochModelButton &&
                    !epochModelButton.classList.contains("active")
                ) {
                    epochModelButton.click();
                }
        
                /*
                  Activate Signal Overview in case the user
                  previously selected Live Simulation or Backtest.
                */
                const overviewButton = document.querySelector(
                    '.nt-epoch-tool-tab[data-epoch-tool-target="epoch-tool-overview"]'
                );
        
                if (
                    overviewButton &&
                    !overviewButton.classList.contains("active")
                ) {
                    overviewButton.click();
                }
        
                /*
                  Wait briefly for the panel to become visible,
                  then scroll directly to the signal board.
                */
                window.setTimeout(() => {
                    const signalBoard = document.querySelector(
                        ".nt-signal-board-shell"
                    );
        
                    if (signalBoard) {
                        signalBoard.scrollIntoView({
                            behavior: "smooth",
                            block: "start"
                        });
                    }
                }, 80);
            });
        }

        const scaleToggle = document.getElementById(
            "nt-market-hero-scale-toggle"
        );
        
        if (scaleToggle) {
            scaleToggle.addEventListener("click", function () {
                ntHeroLogScale = !ntHeroLogScale;
        
                this.classList.toggle(
                    "active",
                    ntHeroLogScale
                );
        
                this.setAttribute(
                    "aria-pressed",
                    ntHeroLogScale ? "true" : "false"
                );
        
                if (ntHeroLastData) {
                    renderHeroPerformanceChart(
                        ntHeroLastData,
                        ntHeroLastDuration
                    );
                }
            });
        }

        loadHeroPerformanceChart(DEFAULT_HERO_DURATION);
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            initializeHeroPerformanceChart
        );
    } else {
        initializeHeroPerformanceChart();
    }
})();
