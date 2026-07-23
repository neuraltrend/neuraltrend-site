
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
    
    function normalizeAltIndexerRows(data) {
        const dates = Array.isArray(data.dates) ? data.dates : [];
        const values = Array.isArray(data.index) ? data.index : [];
    
        return dates
            .map((dateValue, index) => {
                const numericValue = Number(values[index]);
                const dateObj = ntAltParseDate(dateValue);
    
                return {
                    date: String(dateValue || ""),
                    dateObj,
                    value: numericValue
                };
            })
            .filter(row =>
                row.date &&
                Number.isFinite(row.value) &&
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
        if (value > 0) {
            return {
                label: "Buy Zone",
                valueClass: "return-positive",
                iconClass: "nt-icon-green",
                icon: "↗"
            };
        }
    
        if (value < 0) {
            return {
                label: "Sell Zone",
                valueClass: "return-negative",
                iconClass: "nt-icon-neutral",
                icon: "↘"
            };
        }
    
        return {
            label: "Neutral",
            valueClass: "return-neutral",
            iconClass: "nt-icon-neutral",
            icon: "◎"
        };
    }
    
    function altIndexerFormatValue(value) {
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
            return "—";
        }
    
        const numericValue = Number(value);
        const sign = numericValue > 0 ? "+" : "";
    
        return `${sign}${numericValue.toFixed(3)}`;
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
        updateAltIndexerMetric("altindexer-lastweek", lastWeekPoint, "7D");
        updateAltIndexerMetric("altindexer-lastmonth", lastMonthPoint, "30D");
    
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
                name: "Index",
                line: { width: 2 }
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
                    title: "Index",
                    zeroline: true,
                    range: [-1, 1]
                },
    
                shapes: [
                    {
                        type: "rect",
                        xref: "paper",
                        yref: "y",
                        x0: 0,
                        x1: 1,
                        y0: 0,
                        y1: 1,
                        fillcolor: "rgba(34, 197, 94, 0.12)",
                        line: { width: 0 },
                        layer: "below"
                    },
                    {
                        type: "rect",
                        xref: "paper",
                        yref: "y",
                        x0: 0,
                        x1: 1,
                        y0: -1,
                        y1: 0,
                        fillcolor: "rgba(239, 68, 68, 0.12)",
                        line: { width: 0 },
                        layer: "below"
                    }
                ],
    
                annotations: [
                    {
                        xref: "paper",
                        yref: "y",
                        x: 0.5,
                        y: 0.5,
                        text: "Buy Zone",
                        showarrow: false,
                        font: {
                            size: 40,
                            color: "rgba(200, 200, 200, 0.6)"
                        }
                    },
                    {
                        xref: "paper",
                        yref: "y",
                        x: 0.5,
                        y: -0.5,
                        text: "Sell Zone",
                        showarrow: false,
                        font: {
                            size: 40,
                            color: "rgba(200, 200, 200, 0.6)"
                        }
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
