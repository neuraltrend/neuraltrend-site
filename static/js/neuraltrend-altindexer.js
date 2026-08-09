
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


    function altIndexerFormatMonthYear(value) {
        const date = value instanceof Date ? value : ntAltParseDate(value);

        if (!date || Number.isNaN(date.getTime())) {
            return "—";
        }

        return date.toLocaleDateString(undefined, {
            year: "numeric",
            month: "short"
        });
    }

    function altIndexerDurationDays(startDate, endDate) {
        if (!(startDate instanceof Date) || !(endDate instanceof Date)) {
            return 1;
        }

        const dayMs = 24 * 60 * 60 * 1000;
        const elapsed = Math.floor((endDate.getTime() - startDate.getTime()) / dayMs);
        return Math.max(1, elapsed + 1);
    }

    function altIndexerFormatDuration(days) {
        const safeDays = Math.max(1, Math.round(Number(days) || 1));

        if (safeDays < 14) {
            return `${safeDays} ${safeDays === 1 ? "day" : "days"}`;
        }

        if (safeDays < 60) {
            const weeks = Math.max(2, Math.round(safeDays / 7));
            return `${weeks} ${weeks === 1 ? "week" : "weeks"}`;
        }

        const months = Math.max(2, Math.round(safeDays / 30.4375));
        return `${months} ${months === 1 ? "month" : "months"}`;
    }

    function altIndexerFormatPointChange(change) {
        const rounded = Math.round(Number(change) || 0);
        const sign = rounded > 0 ? "+" : "";
        const unit = Math.abs(rounded) === 1 ? "pt" : "pts";
        return `${sign}${rounded} ${unit}`;
    }

    function altIndexerSummaryChangeClass(change) {
        if (change > 0) return "nt-alt-summary-change-positive";
        if (change < 0) return "nt-alt-summary-change-negative";
        return "nt-alt-summary-change-flat";
    }

    function updateAltIndexerMovementSummary(rows) {
        const summaryEl = document.getElementById("altindexer-movement-summary");
        if (!summaryEl) return;

        if (!Array.isArray(rows) || !rows.length) {
            summaryEl.innerHTML = `
                <span class="nt-alt-summary-label">Market regime summary</span>
                <span class="nt-alt-summary-placeholder">No movement history available.</span>
            `;
            return;
        }

        const latest = rows[rows.length - 1];
        const currentZone = altIndexerZone(latest.value);

        let currentStartIndex = rows.length - 1;
        while (currentStartIndex > 0) {
            const priorZone = altIndexerZone(rows[currentStartIndex - 1].value);
            if (priorZone.label !== currentZone.label) break;
            currentStartIndex -= 1;
        }

        const currentStart = rows[currentStartIndex];
        const durationDays = altIndexerDurationDays(currentStart.dateObj, latest.dateObj);
        const durationText = altIndexerFormatDuration(durationDays);

        const previousPoint = currentStartIndex > 0 ? rows[currentStartIndex - 1] : null;
        const previousZone = previousPoint ? altIndexerZone(previousPoint.value) : null;

        const monthPoint = altIndexerPointForDaysBack(rows, 30);
        const monthChange = monthPoint ? latest.value - monthPoint.value : 0;
        const monthChangeText = monthPoint ? altIndexerFormatPointChange(monthChange) : "—";
        const monthChangeClass = monthPoint
            ? altIndexerSummaryChangeClass(monthChange)
            : "nt-alt-summary-change-flat";

        const previousMarkup = previousZone
            ? `<strong class="${previousZone.valueClass}">${previousZone.label}</strong>`
            : `<strong class="nt-alt-zone-neutral">—</strong>`;

        summaryEl.innerHTML = `
            <span class="nt-alt-summary-label">Market regime summary</span>
            <span class="nt-alt-summary-item">
                <strong class="${currentZone.valueClass}">${currentZone.label}</strong>
                for ${durationText}
            </span>
            <span class="nt-alt-summary-separator" aria-hidden="true">·</span>
            <span class="nt-alt-summary-item">Previous regime: ${previousMarkup}</span>
            <span class="nt-alt-summary-separator" aria-hidden="true">·</span>
            <span class="nt-alt-summary-item">
                Entered <strong class="${currentZone.valueClass}">${currentZone.label}</strong>:
                ${altIndexerFormatMonthYear(currentStart.dateObj)}
            </span>
            <span class="nt-alt-summary-separator" aria-hidden="true">·</span>
            <span class="nt-alt-summary-item">
                1M change:
                <strong class="${monthChangeClass}">${monthChangeText}</strong>
            </span>
        `;
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
            subEl.className = "nt-metric-sub nt-alt-zone-neutral";
            subEl.textContent = "No data";
            return;
        }
    
        const zone = altIndexerZone(point.value);
    
        iconEl.className = `nt-card-icon ${zone.iconClass}`;
        iconEl.textContent = fallbackIcon || zone.icon;
    
        valueEl.className = `nt-metric-value ${zone.valueClass}`;
        valueEl.textContent = altIndexerFormatValue(point.value);
    
        subEl.className = `nt-metric-sub ${zone.valueClass}`;
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
            updateAltIndexerMovementSummary(rows);

            const latestRow = rows.length ? rows[rows.length - 1] : null;
            const oneYearStart = latestRow
                ? new Date(latestRow.dateObj.getFullYear() - 1, latestRow.dateObj.getMonth(), latestRow.dateObj.getDate())
                : new Date(new Date().setFullYear(new Date().getFullYear() - 1));
            const initialRangeEnd = latestRow ? latestRow.dateObj : new Date();
            const initialLabelDate = new Date((oneYearStart.getTime() + initialRangeEnd.getTime()) / 2);

            const zoneLabelTrace = {
                x: [initialLabelDate, initialLabelDate, initialLabelDate, initialLabelDate, initialLabelDate],
                y: [10, 30, 50, 70, 90],
                type: "scatter",
                mode: "text",
                text: ["<b>Strong Sell</b>", "<b>Sell</b>", "<b>Neutral</b>", "<b>Buy</b>", "<b>Strong Buy</b>"],
                textfont: { size: 23, color: "rgba(255, 255, 255, 0.96)" },
                hoverinfo: "skip",
                showlegend: false,
                cliponaxis: true
            };
    
            const trace = {
                x: rows.map(row => row.date),
                y: rows.map(row => row.value),
                type: "scatter",
                mode: "lines",
                name: "AltIndexer Score",
                line: { width: 3, color: "#111827" },
                hovertemplate: "%{y}<extra></extra>"
            };
    
            const layout = {
                title: {
                    text: "AltIndexer",
                    x: 0.5
                },
    
                xaxis: {
                    title: "Date",
                    range: [oneYearStart, initialRangeEnd],
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
                        fillcolor: "rgba(153, 27, 27, 0.80)",
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
                        fillcolor: "rgba(220, 38, 38, 0.58)",
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
                        fillcolor: "rgba(100, 116, 139, 0.58)",
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
                        fillcolor: "rgba(22, 163, 74, 0.50)",
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
                        fillcolor: "rgba(22, 101, 52, 0.78)",
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

                template: "plotly_dark",
                hovermode: "x unified",
                margin: { t: 60, l: 50, r: 30, b: 50 }
            };
    
            Plotly.newPlot("epoch-index-chart", [zoneLabelTrace, trace], layout, {
                responsive: true,
                displaylogo: false
            }).then(plotEl => {
                const updateZoneLabelPosition = eventData => {
                    if (!eventData) return;

                    let startValue = eventData["xaxis.range[0]"];
                    let endValue = eventData["xaxis.range[1]"];

                    if ((!startValue || !endValue) && Array.isArray(eventData["xaxis.range"])) {
                        [startValue, endValue] = eventData["xaxis.range"];
                    }

                    if ((!startValue || !endValue) && eventData["xaxis.autorange"]) {
                        startValue = rows.length ? rows[0].date : null;
                        endValue = rows.length ? rows[rows.length - 1].date : null;
                    }

                    const startDate = ntAltParseDate(startValue);
                    const endDate = ntAltParseDate(endValue);
                    if (Number.isNaN(startDate.getTime()) || Number.isNaN(endDate.getTime())) return;

                    const midpoint = new Date((startDate.getTime() + endDate.getTime()) / 2);
                    Plotly.restyle(plotEl, {
                        x: [[midpoint, midpoint, midpoint, midpoint, midpoint]]
                    }, [0]);
                };

                plotEl.on("plotly_relayout", updateZoneLabelPosition);
            });
        })
        .catch(error => {
            console.error("AltIndexer data error:", error);
        });
