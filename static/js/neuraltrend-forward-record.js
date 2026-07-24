(() => {
    "use strict";

    const tickerSelect = document.getElementById("forward-record-ticker");
    if (tickerSelect) {
        tickerSelect.addEventListener("change", () => {
            const ticker = String(tickerSelect.value || "").trim();
            if (!ticker) return;
            const url = new URL(window.location.href);
            url.searchParams.set("ticker", ticker);
            window.location.assign(url.toString());
        });
    }

    const canvas = document.getElementById("forward-record-chart");
    const payloadNode = document.getElementById("forward-record-chart-data");
    if (!canvas || !payloadNode || typeof Chart === "undefined") return;

    let payload;
    try {
        payload = JSON.parse(payloadNode.textContent || "{}");
    } catch (error) {
        console.error("Could not parse Forward Record chart data.", error);
        return;
    }

    const dates = Array.isArray(payload.dates) ? payload.dates : [];
    const strategy = Array.isArray(payload.strategy) ? payload.strategy : [];
    const benchmark = Array.isArray(payload.benchmark) ? payload.benchmark : [];
    if (!dates.length || dates.length !== strategy.length || dates.length !== benchmark.length) return;

    const isSandbox = Boolean(document.querySelector(".nt-forward-page.is-sandbox"));

    new Chart(canvas, {
        type: "line",
        data: {
            labels: dates,
            datasets: [
                {
                    label: isSandbox
                        ? "NeuralTrend sandbox strategy"
                        : "NeuralTrend forward strategy",
                    data: strategy,
                    borderWidth: 2.5,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    tension: 0.18,
                },
                {
                    label: "Buy & Hold",
                    data: benchmark,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    borderDash: [6, 5],
                    tension: 0.18,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: "index",
            },
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        usePointStyle: true,
                        boxWidth: 8,
                        padding: 20,
                    },
                },
                tooltip: {
                    callbacks: {
                        label(context) {
                            const value = Number(context.parsed.y);
                            const formatted = Number.isFinite(value)
                                ? value.toLocaleString(undefined, {
                                    style: "currency",
                                    currency: "USD",
                                    maximumFractionDigits: 0,
                                })
                                : "—";
                            return `${context.dataset.label}: ${formatted}`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { maxTicksLimit: 8 },
                },
                y: {
                    ticks: {
                        callback(value) {
                            return Number(value).toLocaleString(undefined, {
                                style: "currency",
                                currency: "USD",
                                maximumFractionDigits: 0,
                            });
                        },
                    },
                },
            },
        },
    });
})();
