
    function ntAltParseDate(value) {
        const rawValue = String(value || "").trim();
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
    
    function ntAltFormatDate(value) {
        const date = ntAltParseDate(value);
    
        if (Number.isNaN(date.getTime())) {
            return String(value || "—");
        }
    
        return date.toLocaleDateString(undefined, {
            year: "numeric",
            month: "short",
            day: "numeric"
        });
    }
    
    function altIndexerScoreFromRaw(value) {
        const numericValue = Number(value);

        if (!Number.isFinite(numericValue)) {
            return null;
        }

        // The stored AltIndexer series uses a normalized -1 to +1 scale.
        // Present it to users as an intuitive integer score from 0 to 100.
        return Math.round(Math.max(0, Math.min(100, (numericValue + 1) * 50)));
    }

    function normalizeAltIndexerRows(data) {
        const dates = Array.isArray(data.dates) ? data.dates : [];
        const values = Array.isArray(data.index) ? data.index : [];
    
        return dates
            .map((dateValue, index) => {
                const rawValue = Number(values[index]);
                const score = altIndexerScoreFromRaw(rawValue);
                const dateObj = ntAltParseDate(dateValue);
    
                return {
                    date: String(dateValue || ""),
                    dateObj,
                    rawValue,
                    value: score
                };
            })
            .filter(row =>
                row.date &&
                Number.isFinite(row.rawValue) &&
                Number.isInteger(row.value) &&
                !Number.isNaN(row.dateObj.getTime())
            )
            .sort((a, b) => a.dateObj - b.dateObj);
    }
    
    function altIndexerPointForDaysBack(rows, daysBack) {
        if (!rows.length) return null;
    
        const latest = rows[rows.length - 1];
    
        if (daysBack === 0) {
            return latest;
        }
    
        const targetDate = new Date(latest.dateObj);
        targetDate.setDate(targetDate.getDate() - daysBack);
    
        for (let i = rows.length - 1; i >= 0; i--) {
            if (rows[i].dateObj <= targetDate) {
                return rows[i];
            }
        }
    
        return rows[0];
    }
    
    function altIndexerZone(value) {
        const numericValue = Number(value);

        if (!Number.isFinite(numericValue)) {
            return {
                label: "No data",
                valueClass: "nt-alt-zone-neutral",
                iconClass: "nt-alt-icon-neutral",
                icon: "◎"
            };
        }

        // Five market-regime bands on the displayed 0–100 scale.
        // Boundary values remain in the lower band, except 80 which begins
        // Strong Buy: 0–20, >20–40, >40–60, >60–<80, and 80–100.
        if (numericValue <= 20) {
            return {
                label: "Strong Sell",
                valueClass: "nt-alt-zone-strong-sell",
                iconClass: "nt-alt-icon-strong-sell",
                icon: "↓↓"
            };
        }

        if (numericValue <= 40) {
            return {
                label: "Sell",
                valueClass: "nt-alt-zone-sell",
                iconClass: "nt-alt-icon-sell",
                icon: "↘"
            };
        }

        if (numericValue <= 60) {
            return {
                label: "Neutral",
                valueClass: "nt-alt-zone-neutral",
                iconClass: "nt-alt-icon-neutral",
                icon: "◎"
            };
        }

        if (numericValue < 80) {
            return {
                label: "Buy",
                valueClass: "nt-alt-zone-buy",
                iconClass: "nt-alt-icon-buy",
                icon: "↗"
            };
        }

        return {
            label: "Strong Buy",
            valueClass: "nt-alt-zone-strong-buy",
            iconClass: "nt-alt-icon-strong-buy",
            icon: "↑↑"
        };
    }

    function altIndexerFormatValue(value) {
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
            return "—";
        }
    
        return String(Math.round(Number(value)));
    }
    
    function updateAltIndexerMetric(metricId, point, fallbackIcon) {
        const iconEl = document.getElementById(`${metricId}-icon`);
        const valueEl = document.getElementById(`${metricId}-value`);
        const subEl = document.getElementById(`${metricId}-sub`);
    
        if (!iconEl || !valueEl || !subEl) return;
    
        if (!point) {
            iconEl.className = "nt-card-icon nt-icon-neutral";
            iconEl.textContent = fallbackIcon || "◎";
            valueEl.className = "nt-metric-value return-neutral";
            valueEl.textContent = "—";
            subEl.textContent = "No data";
            return;
        }
    
        const zone = altIndexerZone(point.value);
    
        iconEl.className = `nt-card-icon ${zone.iconClass}`;
        iconEl.textContent = fallbackIcon || zone.icon;
    
        valueEl.className = `nt-metric-value ${zone.valueClass}`;
        valueEl.textContent = altIndexerFormatValue(point.value);
    
        subEl.textContent = `${zone.label} · ${ntAltFormatDate(point.date)}`;
    }
    
    function updateAltIndexerMetrics(rows) {
        const todayPoint = altIndexerPointForDaysBack(rows, 0);
        const yesterdayPoint = altIndexerPointForDaysBack(rows, 1);
        const lastWeekPoint = altIndexerPointForDaysBack(rows, 7);
        const lastMonthPoint = altIndexerPointForDaysBack(rows, 30);
    
        updateAltIndexerMetric("altindexer-today", todayPoint, "◎");
        updateAltIndexerMetric("altindexer-yesterday", yesterdayPoint, "↩");
        updateAltIndexerMetric("altindexer-lastweek", lastWeekPoint, "1W");
        updateAltIndexerMetric("altindexer-lastmonth", lastMonthPoint, "1M");
    
        const regimeEl = document.getElementById("altindexer-regime-label");
        const latestDateEl = document.getElementById("altindexer-latest-date");
    
        if (todayPoint && regimeEl) {
            const zone = altIndexerZone(todayPoint.value);
            regimeEl.textContent = zone.label;
            regimeEl.className = zone.valueClass;
        }
    
        if (todayPoint && latestDateEl) {
            latestDateEl.textContent = ntAltFormatDate(todayPoint.date);
        }
    }
    
    fetch("/data")
        .then(response => response.json())
        .then(data => {
            const rows = normalizeAltIndexerRows(data);
    
            updateAltIndexerMetrics(rows);
    
            const trace = {
                x: rows.map(row => row.date),
                y: rows.map(row => row.value),
                type: "scatter",
                mode: "lines",
                name: "AltIndexer Score",
                line: { width: 2, color: "#60a5fa" },
                hovertemplate: "%{y}<extra></extra>"
            };
    
            const layout = {
                title: {
                    text: "AltIndexer",
                    x: 0.5
                },
    
                xaxis: {
                    title: "Date",
                    range: [
                        new Date(new Date().setFullYear(new Date().getFullYear() - 1)),
                        new Date()
                    ],
                    rangeselector: {
                        buttons: [
                            {count: 7, label: "1W", step: "day", stepmode: "backward"},
                            {count: 1, label: "1M", step: "month", stepmode: "backward"},
                            {count: 3, label: "3M", step: "month", stepmode: "backward"},
                            {count: 6, label: "6M", step: "month", stepmode: "backward"},
                            {count: 1, label: "1Y", step: "year", stepmode: "backward"},
                            {count: 3, label: "3Y", step: "year", stepmode: "backward"},
                            {count: 5, label: "5Y", step: "year", stepmode: "backward"},
                            {step: "all", label: "ALL"}
                        ]
                    },
                    type: "date"
                },
    
                yaxis: {
                    title: "AltIndexer Score",
                    range: [0, 100],
                    tickmode: "array",
                    tickvals: [0, 20, 40, 60, 80, 100],
                    ticktext: ["0", "20", "40", "60", "80", "100"],
                    zeroline: false,
                    fixedrange: false
                },
    
                shapes: [
                    {
                        type: "rect",
                        xref: "paper",
                        yref: "y",
                        x0: 0,
                        x1: 1,
                        y0: 0,
                        y1: 20,
                        fillcolor: "rgba(239, 68, 68, 0.38)",
                        line: { width: 0 },
                        layer: "below"
                    },
                    {
                        type: "rect",
                        xref: "paper",
                        yref: "y",
                        x0: 0,
                        x1: 1,
                        y0: 20,
                        y1: 40,
                        fillcolor: "rgba(220, 38, 38, 0.14)",
                        line: { width: 0 },
                        layer: "below"
                    },
                    {
                        type: "rect",
                        xref: "paper",
                        yref: "y",
                        x0: 0,
                        x1: 1,
                        y0: 40,
                        y1: 60,
                        fillcolor: "rgba(255, 255, 255, 0.96)",
                        line: { width: 0 },
                        layer: "below"
                    },
                    {
                        type: "rect",
                        xref: "paper",
                        yref: "y",
                        x0: 0,
                        x1: 1,
                        y0: 60,
                        y1: 80,
                        fillcolor: "rgba(34, 197, 94, 0.15)",
                        line: { width: 0 },
                        layer: "below"
                    },
                    {
                        type: "rect",
                        xref: "paper",
                        yref: "y",
                        x0: 0,
                        x1: 1,
                        y0: 80,
                        y1: 100,
                        fillcolor: "rgba(34, 197, 94, 0.36)",
                        line: { width: 0 },
                        layer: "below"
                    },
                    {
                        type: "line", xref: "paper", yref: "y",
                        x0: 0, x1: 1, y0: 20, y1: 20,
                        line: { color: "rgba(148, 163, 184, 0.55)", width: 1, dash: "dot" },
                        layer: "below"
                    },
                    {
                        type: "line", xref: "paper", yref: "y",
                        x0: 0, x1: 1, y0: 40, y1: 40,
                        line: { color: "rgba(148, 163, 184, 0.55)", width: 1, dash: "dot" },
                        layer: "below"
                    },
                    {
                        type: "line", xref: "paper", yref: "y",
                        x0: 0, x1: 1, y0: 60, y1: 60,
                        line: { color: "rgba(148, 163, 184, 0.55)", width: 1, dash: "dot" },
                        layer: "below"
                    },
                    {
                        type: "line", xref: "paper", yref: "y",
                        x0: 0, x1: 1, y0: 80, y1: 80,
                        line: { color: "rgba(148, 163, 184, 0.55)", width: 1, dash: "dot" },
                        layer: "below"
                    }
                ],

                annotations: [
                    {
                        xref: "paper", yref: "y", x: 0.5, y: 10,
                        text: "<b>Strong Sell</b>", showarrow: false,
                        font: { size: 23, color: "rgba(255, 255, 255, 0.96)" }
                    },
                    {
                        xref: "paper", yref: "y", x: 0.5, y: 30,
                        text: "<b>Sell</b>", showarrow: false,
                        font: { size: 23, color: "rgba(153, 27, 27, 0.92)" }
                    },
                    {
                        xref: "paper", yref: "y", x: 0.5, y: 50,
                        text: "<b>Neutral</b>", showarrow: false,
                        font: { size: 23, color: "rgba(100, 116, 139, 0.92)" }
                    },
                    {
                        xref: "paper", yref: "y", x: 0.5, y: 70,
                        text: "<b>Buy</b>", showarrow: false,
                        font: { size: 23, color: "rgba(21, 128, 61, 0.94)" }
                    },
                    {
                        xref: "paper", yref: "y", x: 0.5, y: 90,
                        text: "<b>Strong Buy</b>", showarrow: false,
                        font: { size: 23, color: "rgba(255, 255, 255, 0.96)" }
                    }
                ],

                template: "plotly_dark",
                hovermode: "x unified",
                margin: { t: 60, l: 50, r: 30, b: 50 }
            };
    
            Plotly.newPlot("epoch-index-chart", [trace], layout, {
                responsive: true,
                displaylogo: false
            });
        })
        .catch(error => {
            console.error("AltIndexer data error:", error);
        });
