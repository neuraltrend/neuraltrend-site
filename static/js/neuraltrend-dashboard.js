
    /* ===============================
       1. HELPERS
    ================================ */
    
    let allSignals = [];
    let defaultOrder = [];
    let assetTypeFilter = "crypto";
    let isLogScale = false;
    let signalSummaryRequestId = 0;
    let equityPreviewRequestId = 0;
    let signalBoardAuthRefreshPromise = null;
    let lastSignalBoardAuthSignature = null;
    let signalSummaryIsLoading = false;

    const NT_COLORS = {
        positive: "#16A34A",
        negative: "#DC2626",
        neutral: "#64748B",
        dark: "#0F172A",
        chartAI: "#2563EB",
        chartBuyHold: "#9333EA",
        buy: "#16A34A",
        sell: "#DC2626"
    };

    const NT_SIGNAL_BOARD_COLUMNS = [
        {key: "ticker", group: "ticker", weight: 1.03, alwaysVisible: true},
        {key: "today_signal", group: "signals", weight: 0.68, alwaysVisible: true},
        {key: "yesterday_signal", group: "signals", weight: 0.72},
        {key: "last_week_signal", group: "signals", weight: 0.72},
        {key: "last_month_signal", group: "signals", weight: 0.74},
        {key: "buy_hold_period_return", group: "recent", weight: 0.74},
        {key: "strategy_period_return", group: "recent", weight: 0.68},
        {key: "outperformance_ratio", group: "recent", weight: 0.78},
        {key: "alpha", group: "average", weight: 1.02},
        {key: "alpha_prob", group: "average", weight: 1.16},
        {key: "strategy_avg_return", group: "average", weight: 0.76},
        {key: "strategy_profit_prob", group: "average", weight: 0.68},
        {key: "recommended_days", group: "recommended", weight: 0.54}
    ];

    const ntVisibleSignalBoardColumns = new Set(
        NT_SIGNAL_BOARD_COLUMNS.map(column => column.key)
    );

    function ntIsSmallScreen() {
        return window.innerWidth < 700;
    }
    
    function ntTrimZeros(value) {
        return String(value).replace(/\.0+$|(\.\d*[1-9])0+$/, "$1");
    }
    
    function ntFormatCompactNumber(value, prefix = "") {
        const numberValue = Number(value);
    
        if (!Number.isFinite(numberValue)) {
            return "—";
        }
    
        const absValue = Math.abs(numberValue);
    
        if (absValue >= 1_000_000) {
            return prefix + ntTrimZeros((numberValue / 1_000_000).toFixed(1)) + "M";
        }
    
        if (absValue >= 1_000) {
            return prefix + ntTrimZeros((numberValue / 1_000).toFixed(1)) + "k";
        }
    
        if (absValue >= 100) {
            return prefix + ntTrimZeros(numberValue.toFixed(0));
        }
    
        if (absValue >= 10) {
            return prefix + ntTrimZeros(numberValue.toFixed(1));
        }
    
        if (absValue >= 1) {
            return prefix + ntTrimZeros(numberValue.toFixed(2));
        }
    
        return prefix + ntTrimZeros(numberValue.toFixed(3));
    }
    
    function ntParseChartDate(value) {
        const rawValue = String(value || "").trim();
    
        /*
          Important:
          Date-only strings like "2026-07-06" are parsed as UTC by JavaScript.
          In some time zones, that displays as the previous day.
          So we manually parse YYYY-MM-DD as a local date.
        */
        const match = rawValue.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    
        if (match) {
            const year = Number(match[1]);
            const month = Number(match[2]) - 1;
            const day = Number(match[3]);
    
            return new Date(year, month, day);
        }
    
        return new Date(rawValue);
    }
    
    function ntFormatDateForAxis(value) {
        const rawValue = String(value || "").trim();
        const date = ntParseChartDate(rawValue);
    
        if (Number.isNaN(date.getTime())) {
            return rawValue.slice(0, 10);
        }
    
        const duration = document.getElementById("period-select")?.value || "5y";
    
        const shortHorizon = ["1w", "1mo", "3mo", "6mo"].includes(duration);
    
        return date.toLocaleDateString(undefined, shortHorizon ? {
            month: "short",
            day: "numeric"
        } : {
            month: "short",
            year: "numeric"
        });
    }
    
    function ntFormatDateForTooltip(value) {
        const rawValue = String(value || "").trim();
        const date = ntParseChartDate(rawValue);
    
        if (Number.isNaN(date.getTime())) {
            return rawValue;
        }
    
        return date.toLocaleDateString(undefined, {
            year: "numeric",
            month: "short",
            day: "numeric"
        });
    }
    
    function ntChartOptions({
        logScale = false,
        yTitle = "",
        yPrefix = "",
        xTitle = "",
        legendAlign = "end"
    } = {}) {
        const smallScreen = ntIsSmallScreen();
    
        return {
            responsive: true,
            maintainAspectRatio: false,
    
            layout: {
                padding: {
                    top: 8,
                    right: 10,
                    bottom: 4,
                    left: 4
                }
            },
    
            interaction: {
                mode: "index",
                intersect: false
            },
    
            elements: {
                line: {
                    tension: 0.18
                },
                point: {
                    hoverRadius: 4
                }
            },
    
            plugins: {
                legend: {
                    position: "top",
                    align: legendAlign,
                    labels: {
                        usePointStyle: true,
                        boxWidth: 8,
                        boxHeight: 8,
                        padding: 14,
                        color: NT_COLORS.neutral,
                        font: {
                            size: 11,
                            weight: "700"
                        }
                    }
                },
    
                tooltip: {
                    mode: "index",
                    intersect: false,
                    backgroundColor: "rgba(15, 23, 42, 0.92)",
                    titleColor: "#ffffff",
                    bodyColor: "#ffffff",
                    borderColor: "rgba(148, 163, 184, 0.25)",
                    borderWidth: 1,
                    padding: 10,
                    displayColors: true,
                    callbacks: {
                        title: function(items) {
                            return ntFormatDateForTooltip(items?.[0]?.label);
                        }
                    }
                }
            },
    
            scales: {
                x: {
                    display: true,
                    grid: {
                        color: "rgba(148, 163, 184, 0.14)",
                        drawTicks: false
                    },
                    border: {
                        display: false
                    },
                    title: {
                        display: Boolean(xTitle),
                        text: xTitle,
                        color: NT_COLORS.neutral,
                        font: {
                            size: 11,
                            weight: "700"
                        }
                    },
                    ticks: {
                        autoSkip: true,
                        maxTicksLimit: smallScreen ? 4 : 7,
                        maxRotation: 0,
                        minRotation: 0,
                        padding: 10,
                        color: NT_COLORS.neutral,
                        font: {
                            size: 11,
                            weight: "700"
                        },
                        callback: function(value) {
                            const label = this.getLabelForValue
                                ? this.getLabelForValue(value)
                                : value;
    
                            return ntFormatDateForAxis(label);
                        }
                    }
                },
    
                y: {
                    display: true,
                    type: logScale ? "logarithmic" : "linear",
                    grid: {
                        color: "rgba(148, 163, 184, 0.18)",
                        drawTicks: false
                    },
                    border: {
                        display: false
                    },
                    title: {
                        display: Boolean(yTitle),
                        text: yTitle,
                        color: NT_COLORS.neutral,
                        font: {
                            size: 11,
                            weight: "700"
                        }
                    },
                    ticks: {
                        maxTicksLimit: smallScreen ? 4 : 6,
                        padding: 8,
                        color: NT_COLORS.neutral,
                        font: {
                            size: 11,
                            weight: "700"
                        },
                        callback: function(value) {
                            return ntFormatCompactNumber(value, yPrefix);
                        }
                    }
                }
            }
        };
    }
    
    let signalFilters = {
        today_signal: null,
        yesterday_signal: null,
        last_week_signal: null,
        last_month_signal: null
    };

    function clearSignalBoardFilters() {

        // Reset signal filter state
        signalFilters = {
            today_signal: null,
            yesterday_signal: null,
            last_week_signal: null,
            last_month_signal: null
        };
    
        // Reset signal filter dropdowns
        document.querySelectorAll(".signal-filter").forEach(select => {
            select.value = "";
            select.classList.remove("nt-filter-active");
        });
    
        // Reset search bar
        const searchInput = document.getElementById("ticker-search");
        if (searchInput) {
            searchInput.value = "";
            searchInput.blur();
        }
    
        // Reset sorting
        currentSort = {
            key: null,
            direction: 1
        };
    
        updateSortIndicators();

        // Reset optional Board Columns visibility to the default all-visible view.
        resetSignalBoardColumns();
    
        // Reset asset type to Crypto
        assetTypeFilter = "crypto";
    
        const assetSelect = document.getElementById("asset-type-filter");
        if (assetSelect) {
            assetSelect.value = "crypto";
        }
    
        if (typeof syncAssetPills === "function") {
            syncAssetPills();
        }
    
        // Reset return horizon to 5Y
        const periodSelect = document.getElementById("period-select");
        const horizonChanged = periodSelect && periodSelect.value !== "5y";
    
        if (periodSelect) {
            periodSelect.value = "5y";
        }
    
        if (typeof syncPeriodPills === "function") {
            syncPeriodPills();
        }
    
        // If horizon changed, reload data for 5Y.
        // Otherwise, just reapply local filters.
        if (horizonChanged && periodSelect) {
            periodSelect.dispatchEvent(new Event("change", { bubbles: true }));
        } else {
            applyAllFilters();
        }
    }

    function escapeHTML(value) {
        return String(value).replace(/[&<>"']/g, function (char) {
            return {
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                '"': "&quot;",
                "'": "&#039;"
            }[char];
        });
    }

    function clearSuspiciousTickerSearchAutofill() {
        const searchInput = document.getElementById("ticker-search");
    
        if (!searchInput) return;
    
        const value = (searchInput.value || "").trim();
    
        // Browser/password-manager autofill sometimes puts login email here.
        const looksLikeEmail = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
    
        if (looksLikeEmail) {
            searchInput.value = "";
    
            if (typeof applyAllFilters === "function") {
                applyAllFilters();
            }
        }
    }
    
    function plainSignalText(value) {
        if (value === 1) return "BUY";
        if (value === -1) return "SELL";
        return "HOLD";
    }
    
    function signalChipClass(value) {
        if (value === 1) return "chip-buy";
        if (value === -1) return "chip-sell";
        return "chip-hold";
    }
    
    function assetTypeLabel(value) {
        if (value === "all") return "All Assets";
        if (value === "crypto") return "Crypto";
        if (value === "stock") return "Stocks";
        if (value === "watchlist") return "My Watchlist";
        return value;
    }
    
    function sortLabel(key) {
        if (key === "buy_hold_period_return") return "B&H Return";
        if (key === "strategy_period_return") return "AI Return";
        if (key === "outperformance_ratio") return "Return Ratio";
        if (key === "alpha") return "Avg. Return Ratio";
        if (key === "alpha_prob") return "Outperformance Prob.";
        if (key === "strategy_avg_return") return "AI Avg. Return";
        if (key === "strategy_profit_prob") return "Profit Prob.";
        if (key === "recommended_days") return "Min. Days";
        return "";
    }
    
    function ntSignalBoardColumnDefinition(key) {
        return NT_SIGNAL_BOARD_COLUMNS.find(column => column.key === key) || null;
    }

    function ntVisibleSignalBoardDefinitions() {
        return NT_SIGNAL_BOARD_COLUMNS.filter(column =>
            column.alwaysVisible || ntVisibleSignalBoardColumns.has(column.key)
        );
    }

    function syncBoardColumnChooserUI() {
        const visibleDefinitions = ntVisibleSignalBoardDefinitions();
        const total = NT_SIGNAL_BOARD_COLUMNS.length;
        const toggleLabel = document.getElementById("board-columns-toggle-label");

        document.querySelectorAll("[data-board-column-toggle]").forEach(input => {
            input.checked = ntVisibleSignalBoardColumns.has(input.dataset.boardColumnToggle);
        });

        if (toggleLabel) {
            toggleLabel.textContent = visibleDefinitions.length === total
                ? "All columns"
                : `${visibleDefinitions.length}/${total} shown`;
        }
    }

    function applySignalBoardColumnLayout() {
        const tableCard = document.querySelector("#epoch-tool-overview .nt-signal-table-card");
        const header = document.querySelector("#epoch-tool-overview .nt-signal-header");
        const groupHeader = document.querySelector("#epoch-tool-overview .nt-signal-group-header");
        if (!tableCard || !header || !groupHeader) return;

        const visibleDefinitions = ntVisibleSignalBoardDefinitions();
        const visibleKeys = new Set(visibleDefinitions.map(column => column.key));
        const gridTemplate = visibleDefinitions
            .map(column => `minmax(0, ${column.weight}fr)`)
            .join(" ");

        tableCard.style.setProperty("--nt-signal-columns", gridTemplate);

        const applyVisibilityToCells = cells => {
            NT_SIGNAL_BOARD_COLUMNS.forEach((column, index) => {
                const cell = cells[index];
                if (!cell) return;
                cell.classList.toggle("nt-board-column-hidden", !visibleKeys.has(column.key));
            });
        };

        applyVisibilityToCells(Array.from(header.children));
        document.querySelectorAll("#signal-board .signal-row").forEach(row => {
            applyVisibilityToCells(Array.from(row.children));
        });

        ["ticker", "signals", "recent", "average", "recommended"].forEach(groupName => {
            const groupElement = groupHeader.querySelector(`[data-board-group="${groupName}"]`);
            if (!groupElement) return;

            const groupColumns = visibleDefinitions.filter(column => column.group === groupName);
            if (!groupColumns.length) {
                groupElement.classList.add("nt-board-column-hidden");
                return;
            }

            groupElement.classList.remove("nt-board-column-hidden");
            const firstIndex = visibleDefinitions.indexOf(groupColumns[0]) + 1;
            const lastIndex = visibleDefinitions.indexOf(groupColumns[groupColumns.length - 1]) + 2;
            groupElement.style.gridColumn = `${firstIndex} / ${lastIndex}`;
        });

        syncBoardColumnChooserUI();
        requestAnimationFrame(syncSignalBoardScrollbarCompensation);
    }

    function closeBoardColumnChooser() {
        const toggle = document.getElementById("board-columns-toggle");
        const menu = document.getElementById("board-columns-menu");
        if (!toggle || !menu) return;
        menu.hidden = true;
        toggle.setAttribute("aria-expanded", "false");
    }

    function setSignalBoardColumnVisibility(key, visible) {
        const column = ntSignalBoardColumnDefinition(key);
        if (!column || column.alwaysVisible) return;

        if (visible) {
            ntVisibleSignalBoardColumns.add(key);
        } else {
            ntVisibleSignalBoardColumns.delete(key);

            // Never leave an invisible signal filter active.
            if (Object.prototype.hasOwnProperty.call(signalFilters, key)) {
                signalFilters[key] = null;
                const filter = document.querySelector(`.signal-filter[data-key="${key}"]`);
                if (filter) {
                    filter.value = "";
                    filter.classList.remove("nt-filter-active");
                }
            }

            // Never keep sorting by a metric that the user has hidden.
            if (currentSort.key === key) {
                currentSort = {key: null, direction: 1};
                updateSortIndicators();
            }
        }

        applySignalBoardColumnLayout();
        applyAllFilters();
    }

    function resetSignalBoardColumns() {
        ntVisibleSignalBoardColumns.clear();
        NT_SIGNAL_BOARD_COLUMNS.forEach(column => ntVisibleSignalBoardColumns.add(column.key));
        applySignalBoardColumnLayout();
    }

    function initializeBoardColumnChooser() {
        const root = document.getElementById("board-columns-control");
        const toggle = document.getElementById("board-columns-toggle");
        const menu = document.getElementById("board-columns-menu");
        if (!root || !toggle || !menu) return;

        toggle.addEventListener("click", event => {
            event.preventDefault();
            const willOpen = menu.hidden;
            menu.hidden = !willOpen;
            toggle.setAttribute("aria-expanded", willOpen ? "true" : "false");
        });

        menu.querySelectorAll("[data-board-column-toggle]").forEach(input => {
            input.addEventListener("change", () => {
                setSignalBoardColumnVisibility(input.dataset.boardColumnToggle, input.checked);
            });
        });

        document.addEventListener("click", event => {
            if (!root.contains(event.target)) closeBoardColumnChooser();
        });

        document.addEventListener("keydown", event => {
            if (event.key === "Escape") closeBoardColumnChooser();
        });

        syncBoardColumnChooserUI();
        applySignalBoardColumnLayout();
    }

    function updateActiveFilterChips(count) {
        const row = document.getElementById("active-filter-row");
        const chipsContainer = document.getElementById("active-filter-chips");
    
        if (!row || !chipsContainer) return;
    
        const chips = [];
    
        const duration = document.getElementById("period-select")?.value || "5y";
        chips.push({
            label: "Return Horizon",
            value: durationLabel(duration),
            className: ""
        });
    
        chips.push({
            label: "Asset Type",
            value: assetTypeLabel(assetTypeFilter),
            className: ""
        });
    
        const visibleCount = Number.isFinite(count) ? count : allSignals.length;
        chips.push({
            label: "Assets Shown",
            value: String(visibleCount),
            className: ""
        });

        chips.push({
            label: "Assumptions",
            value: "$1 normalized · costs on executed trades · open positions marked to market",
            className: "chip-assumption"
        });
    
        const searchQuery = document.getElementById("ticker-search")?.value.trim();
        if (searchQuery) {
            chips.push({
                label: "Search",
                value: searchQuery,
                className: ""
            });
        }
    
        const signalLabels = {
            today_signal: "Today",
            yesterday_signal: "Yesterday",
            last_week_signal: "Last Week",
            last_month_signal: "Last Month"
        };
    
        Object.keys(signalFilters).forEach(key => {
            const value = signalFilters[key];
    
            if (value !== null) {
                chips.push({
                    label: signalLabels[key],
                    value: plainSignalText(value),
                    className: signalChipClass(value)
                });
            }
        });
    
        if (currentSort.key) {
            const direction = currentSort.direction === 1 ? "Ascending" : "Descending";
    
            chips.push({
                label: "Sort",
                value: `${sortLabel(currentSort.key)} (${direction})`,
                className: ""
            });
        }
    
        chipsContainer.innerHTML = chips.map(chip => `
            <span class="nt-active-chip ${chip.className}">
                <span class="nt-active-chip-label">${escapeHTML(chip.label)}:</span>
                <strong class="nt-active-chip-value">${escapeHTML(chip.value)}</strong>
            </span>
        `).join("");
    }

    function syncBacktestTicker(ticker) {
        const tickerSelect = document.getElementById("ticker");
        if (!tickerSelect || !ticker) return;
    
        const optionExists = Array.from(tickerSelect.options)
            .some(option => option.value === ticker);
    
        if (optionExists) {
            tickerSelect.value = ticker;
        }
    }

    const DEFAULT_LIVE_SIM_INITIAL_CASH = 10000;

    function parseLiveSimulationCashInput(rawValue) {
        const text = String(rawValue ?? "").trim();

        if (!text) {
            return DEFAULT_LIVE_SIM_INITIAL_CASH;
        }

        // Accept convenient user formats such as 100, 100$, 100 $, $100,
        // and comma-separated values such as $10,000.
        const normalized = text.replace(/[\s,$]/g, "");

        if (!/^\d+(?:\.\d+)?$/.test(normalized)) {
            return null;
        }

        const value = Number(normalized);
        return Number.isFinite(value) && value > 0 ? value : null;
    }

    function liveSimulationCashNamePart(value) {
        const numericValue = Number(value);
        if (!Number.isFinite(numericValue)) return String(DEFAULT_LIVE_SIM_INITIAL_CASH);
        if (Number.isInteger(numericValue)) return String(numericValue);
        return String(numericValue).replace(/0+$/, "").replace(/\.$/, "");
    }

    function defaultLiveSimulationName(ticker) {
        const positionSelect = document.getElementById("live-sim-position");
        const cashInput = document.getElementById("live-sim-cash");
        const positionPct = positionSelect ? positionSelect.value : "50";
        const parsedCash = parseLiveSimulationCashInput(cashInput ? cashInput.value : "");
        const cashValue = parsedCash === null ? DEFAULT_LIVE_SIM_INITIAL_CASH : parsedCash;

        return `${ticker}_${liveSimulationCashNamePart(cashValue)}_${positionPct}%`;
    }

    function refreshAutoLiveSimulationName() {
        const tickerInput = document.getElementById("live-sim-ticker");
        const nameInput = document.getElementById("live-sim-name");

        if (!tickerInput || !nameInput) return;

        const shouldAutoUpdate =
            !nameInput.value.trim() ||
            nameInput.dataset.autoName === "true" ||
            nameInput.value.includes("Live Simulation");

        if (!shouldAutoUpdate) return;

        const cashInput = document.getElementById("live-sim-cash");
        const rawCash = cashInput ? String(cashInput.value || "").trim() : "";
        const parsedCash = parseLiveSimulationCashInput(rawCash);

        // While the user is midway through typing an invalid cash string,
        // preserve the current auto-name rather than flashing back to $10,000.
        if (rawCash && parsedCash === null) return;

        nameInput.value = defaultLiveSimulationName(tickerInput.value || "BTC-USD");
        nameInput.dataset.autoName = "true";
    }
    
    function syncLiveSimulationTicker(ticker) {
        if (!ticker) return;
    
        const normalizedTicker = normalizeTickerForCompare(ticker);
        const select = document.getElementById("live-sim-ticker");
    
        if (!select) return;
    
        renderLiveSimTickerOptions(ticker);
    
        const matchingOption = Array.from(select.options).find(option =>
            normalizeTickerForCompare(option.value) === normalizedTicker
        );
    
        if (matchingOption) {
            select.value = matchingOption.value;
        }
    
        const selectedTicker = select.value || ticker;
    
        updateLiveSimTickerSelectionUI(selectedTicker, true);
    }
    
    function applyAllFilters() {
        // Do not let browser autofill, restored controls, or filter events replace
        // a real Loading/Updating state while a fresh summary is still in flight.
        // This was the main cause of the board briefly showing the misleading
        // “No assets match…” message during login/account refreshes.
        if (signalSummaryIsLoading) {
            return;
        }
    
        let filtered = [...defaultOrder];
    
        // 1️⃣ Search filter
        const query = document.getElementById('ticker-search').value.toLowerCase();
        if (query) {
            filtered = filtered.filter(item =>
                item.ticker.toLowerCase().includes(query)
            );
        }
    
        // 2️⃣ Asset type / watchlist filter
        if (assetTypeFilter !== "all") {

            filtered = filtered.filter(item => {

                const isCrypto = item.ticker.endsWith("-USD");

                if (assetTypeFilter === "crypto") return isCrypto;
                if (assetTypeFilter === "stock") return !isCrypto;
                if (assetTypeFilter === "watchlist") {
                    return Boolean(
                        window.neuralTrendWatchlistHasTicker
                        && window.neuralTrendWatchlistHasTicker(item.ticker)
                    );
                }

                return true;
            });
        }
    
        // 2️⃣ Signal filters (AND logic)
        Object.keys(signalFilters).forEach(key => {
            const val = signalFilters[key];
            if (val !== null) {
                filtered = filtered.filter(item => item[key] === val);
            }
        });
    
        // 3️⃣ Sorting
        if (currentSort.key) {
            filtered.sort((a, b) => {
                const rawA = a[currentSort.key];
                const rawB = b[currentSort.key];
                const valA = Number(rawA);
                const valB = Number(rawB);
                const hasA = rawA !== null && rawA !== undefined && rawA !== "" && Number.isFinite(valA);
                const hasB = rawB !== null && rawB !== undefined && rawB !== "" && Number.isFinite(valB);

                // Keep unavailable statistics at the bottom in either sort direction.
                if (!hasA && !hasB) return 0;
                if (!hasA) return 1;
                if (!hasB) return -1;

                return (valA - valB) * currentSort.direction;
            });
        }

        // allSignals/defaultOrder remain the full server response. Filtering is
        // presentation-only; do not replace the canonical source with a subset.
        renderBoard(filtered);
        updateAssumptionsBar(filtered.length);
        updateActiveFilterChips(filtered.length);
    }
    
    function getReturnClass(value) {
        if (value === null || value === undefined || Number.isNaN(value)) {
            return "return-neutral";
        }
    
        if (value > 0) return "return-positive";
        if (value < 0) return "return-negative";
    
        return "return-neutral";
    }
    
    
    function formatSignal(signal, locked = false) {
        if (locked) {
            return `<span class="signal-badge signal-locked">🔒 Pro</span>`;
        }
    
        if (signal === 1) {
            return `<span class="signal-badge signal-buy">▲ BUY</span>`;
        }
    
        if (signal === -1) {
            return `<span class="signal-badge signal-sell">▼ SELL</span>`;
        }
    
        return `<span class="signal-badge signal-hold">— HOLD</span>`;
    }
    
    function formatPercent(value) {
        if (value === null || value === undefined || Number.isNaN(value)) {
            return `<span class="return-value return-neutral">—</span>`;
        }
    
        const percent = value * 100;
        const sign = percent > 0 ? "+" : "";
    
        return `
            <span class="return-value ${getReturnClass(value)}">
                ${sign}${percent.toFixed(1)}%
            </span>
        `;
    }
    
    function formatPointSpread(value) {
        if (value === null || value === undefined || Number.isNaN(value)) {
            return `<span class="return-value return-neutral">—</span>`;
        }
    
        const points = Number(value) * 100;
        const sign = points > 0 ? "+" : "";
    
        return `
            <span class="return-value ${getReturnClass(Number(value))}">
                ${sign}${points.toFixed(1)} pts
            </span>
        `;
    }

    function ntStatValueClass(value, neutralPoint) {
        const numericValue = Number(value);
        if (!Number.isFinite(numericValue)) return "return-neutral";
        if (numericValue > neutralPoint) return "return-positive";
        if (numericValue < neutralPoint) return "return-negative";
        return "return-neutral";
    }

    function formatOutperformanceRatio(value) {
        if (value === null || value === undefined || value === "") {
            return `<span class="nt-stat-unavailable">Not available</span>`;
        }

        const numericValue = Number(value);
        if (!Number.isFinite(numericValue)) {
            return `<span class="nt-stat-unavailable">Not available</span>`;
        }

        return `
            <span class="return-value ${ntStatValueClass(numericValue, 1)}">
                ${numericValue.toFixed(3)}×
            </span>
        `;
    }

    function formatOutperformanceProbability(value) {
        if (value === null || value === undefined || value === "") {
            return `<span class="nt-stat-unavailable">Not available</span>`;
        }

        const numericValue = Number(value);
        if (!Number.isFinite(numericValue)) {
            return `<span class="nt-stat-unavailable">Not available</span>`;
        }

        const probability = Math.max(0, Math.min(1, numericValue));
        return `
            <span class="return-value return-neutral">
                ${(probability * 100).toFixed(1)}%
            </span>
        `;
    }

    function formatRecommendedDays(value) {
        if (value === null || value === undefined || value === "") {
            return `<span class="nt-stat-unavailable">Not available</span>`;
        }

        const numericValue = Number(value);
        if (!Number.isFinite(numericValue) || numericValue < 1) {
            return `<span class="nt-stat-unavailable">Not available</span>`;
        }

        return `<span class="return-value return-neutral">${Math.round(numericValue)}d</span>`;
    }

    function signalStatWindowTitle(item, duration) {
        if (item?.stat_window === null || item?.stat_window === undefined || item?.stat_window === "") {
            return "Model statistics not available for this asset.";
        }

        const statWindow = Number(item.stat_window);
        if (!Number.isFinite(statWindow)) return "Model statistics not available for this asset.";

        const requestedDays = {
            "1w": 7,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "3y": 1095,
            "5y": 1825,
            "10y": 3650
        }[duration];

        if (duration === "max" || (Number.isFinite(requestedDays) && statWindow < requestedDays)) {
            return `Using the longest available statistics window (${statWindow.toLocaleString()} days).`;
        }

        return `Statistics window: ${statWindow.toLocaleString()} days.`;
    }

    function formatMetricNumber(value, decimals = 2) {
        if (value === null || value === undefined) return "—";
        const numberValue = Number(value);
        if (!Number.isFinite(numberValue)) return "—";
        return numberValue.toFixed(decimals);
    }

    function formatMetricPercentText(value, decimals = 1, signed = true) {
        if (value === null || value === undefined) return "—";
        const numberValue = Number(value);
        if (!Number.isFinite(numberValue)) return "—";
        const percent = numberValue * 100;
        const sign = signed && percent > 0 ? "+" : "";
        return `${sign}${percent.toFixed(decimals)}%`;
    }


    function ntSafeFreshnessStatus(value) {
        const status = String(value || "unknown").toLowerCase();
        return ["current", "delayed", "stale", "unknown"].includes(status)
            ? status
            : "unknown";
    }

    function ntFormatDataDate(value, short = false) {
        if (!value) return "—";
        const date = ntParseChartDate(value);
        if (Number.isNaN(date.getTime())) return String(value).slice(0, 10) || "—";
        return date.toLocaleDateString(undefined, short ? {
            month: "short",
            day: "numeric"
        } : {
            year: "numeric",
            month: "short",
            day: "numeric"
        });
    }

    function ntFormatUtcDataTimestamp(value) {
        if (!value) return "—";
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return "—";
        return new Intl.DateTimeFormat(undefined, {
            year: "numeric",
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
            timeZone: "UTC",
            timeZoneName: "short"
        }).format(date);
    }

    function ntApplyFreshnessBadge(element, data, compact = false) {
        if (!element) return;
        const status = ntSafeFreshnessStatus(data?.freshness_status);
        const label = String(data?.freshness_label || "Unknown");
        element.className = compact
            ? `nt-inline-freshness nt-freshness-${status}`
            : `nt-freshness-badge nt-freshness-${status}`;
        element.textContent = label;
        element.title = String(
            data?.freshness_message ||
            "Freshness could not be determined for this asset."
        );
    }

    function ntFreshnessCompactText(data) {
        const label = String(data?.freshness_label || "Unknown");
        const dateText = ntFormatDataDate(data?.data_through, true);
        return `${label} · ${dateText}`;
    }

    function durationLabel(value) {
        const labels = {
            "1w": "1W",
            "1mo": "1M",
            "3mo": "3M",
            "6mo": "6M",
            "1y": "1Y",
            "3y": "3Y",
            "5y": "5Y",
            "10y": "10Y",
            "max": "MAX",
            "1yr": "1Y",
            "3yr": "3Y",
            "5yr": "5Y"
        };
    
        return labels[value] || value.toUpperCase();
    }

    function updateSignalStatHeaderCopy() {
        const duration = document.getElementById("period-select")?.value || "5y";
        const horizon = durationLabel(duration);

        const recentGroup = document.getElementById("recent-statistics-group-label");
        const averageGroup = document.getElementById("average-statistics-group-label");
        if (recentGroup) recentGroup.textContent = `Recent ${horizon} Statistics`;
        if (averageGroup) averageGroup.textContent = `Average ${horizon} Statistics`;

        const buyHoldTooltip = document.querySelector('[data-stat-tooltip="buy-hold-return"]');
        const aiReturnTooltip = document.querySelector('[data-stat-tooltip="ai-return"]');
        const returnRatioTooltip = document.querySelector('[data-stat-tooltip="outperformance-ratio"]');
        const alphaTooltip = document.querySelector('[data-stat-tooltip="alpha"]');
        const outperformanceProbTooltip = document.querySelector('[data-stat-tooltip="alpha-prob"]');
        const strategyAvgReturnTooltip = document.querySelector('[data-stat-tooltip="strategy-avg-return"]');
        const strategyProfitProbTooltip = document.querySelector('[data-stat-tooltip="strategy-profit-prob"]');
        const recommendedDaysTooltip = document.querySelector('[data-stat-tooltip="recommended-days"]');

        if (recommendedDaysTooltip) {
            recommendedDaysTooltip.textContent = "Recommended minimum number of days to follow the strategy for profitability and outperformance";
        }

        if (duration === "max") {
            if (buyHoldTooltip) {
                buyHoldTooltip.textContent = "return of Buy and Hold strategy over the longest available time horizon";
            }
            if (aiReturnTooltip) {
                aiReturnTooltip.textContent = "return of AI strategy over the longest available time horizon";
            }
            if (returnRatioTooltip) {
                returnRatioTooltip.textContent = "Final value of the AI strategy divided by the final value of Buy & Hold over the longest available time horizon. Above 1.00× means AI outperformed; below 1.00× means Buy & Hold outperformed.";
            }
            if (alphaTooltip) {
                alphaTooltip.textContent = "AI strategy final value divided by Buy & Hold final value, averaged over all historical periods at the longest available statistics horizon.";
            }
            if (outperformanceProbTooltip) {
                outperformanceProbTooltip.textContent = "Probability of AI strategy outperforming Buy and Hold return over the longest available statistics horizon.";
            }
            if (strategyAvgReturnTooltip) {
                strategyAvgReturnTooltip.textContent = "AI strategy final value divided by its initial value, averaged over all historical periods at the longest available statistics horizon";
            }
            if (strategyProfitProbTooltip) {
                strategyProfitProbTooltip.textContent = "Probability of AI strategy being in profit over the longest available statistics horizon";
            }
            return;
        }

        if (buyHoldTooltip) {
            buyHoldTooltip.textContent = `return of Buy and Hold strategy over the last ${horizon}`;
        }
        if (aiReturnTooltip) {
            aiReturnTooltip.textContent = `return of AI strategy over the last ${horizon}`;
        }
        if (returnRatioTooltip) {
            returnRatioTooltip.textContent = `Final value of the AI strategy divided by the final value of Buy & Hold over the last ${horizon}. Above 1.00× means AI outperformed; below 1.00× means Buy & Hold outperformed.`;
        }
        if (alphaTooltip) {
            alphaTooltip.textContent = `AI strategy final value divided by Buy & Hold final value, averaged over all historical ${horizon} periods.`;
        }
        if (outperformanceProbTooltip) {
            outperformanceProbTooltip.textContent = `Probability of AI strategy outperforming Buy and Hold return in a ${horizon} period.`;
        }
        if (strategyAvgReturnTooltip) {
            strategyAvgReturnTooltip.textContent = `AI strategy final value divided by its initial value, averaged over all ${horizon} historical periods`;
        }
        if (strategyProfitProbTooltip) {
            strategyProfitProbTooltip.textContent = `Probability of AI strategy being in profit in a ${horizon} period`;
        }
    }

    function initializeSignalStatInfoTooltips() {
        const roots = Array.from(document.querySelectorAll("[data-stat-info-root]"));
        if (!roots.length) return;

        function setOpen(root, open) {
            const button = root.querySelector("[data-stat-info-button]");
            root.classList.toggle("is-open", Boolean(open));
            if (button) button.setAttribute("aria-expanded", open ? "true" : "false");
        }

        roots.forEach(root => {
            const button = root.querySelector("[data-stat-info-button]");
            if (!button) return;

            button.addEventListener("click", event => {
                event.preventDefault();
                event.stopPropagation();
                const shouldOpen = !root.classList.contains("is-open");
                roots.forEach(otherRoot => setOpen(otherRoot, false));
                setOpen(root, shouldOpen);
            });

            // The info button sits inside a sortable header. Keep keyboard
            // interaction with the tooltip from triggering the column sort.
            button.addEventListener("keydown", event => {
                event.stopPropagation();
            });
        });

        document.addEventListener("click", event => {
            roots.forEach(root => {
                if (!root.contains(event.target)) setOpen(root, false);
            });
        });

        document.addEventListener("keydown", event => {
            if (event.key !== "Escape") return;
            roots.forEach(root => setOpen(root, false));
        });
    }

    function updateAssumptionsBar(count) {
        const horizonEl = document.getElementById("assumption-horizon");
        const countEl = document.getElementById("assumption-asset-count");
    
        // If the separate assumptions bar was removed, do nothing.
        if (!horizonEl && !countEl) return;
    
        const duration = document.getElementById("period-select")?.value || "5y";
    
        if (horizonEl) {
            horizonEl.textContent = durationLabel(duration);
        }
    
        if (countEl) {
            const safeCount = Number.isFinite(count) ? count : allSignals.length;
            countEl.textContent = safeCount.toString();
        }
    }
    
    function normalizedDollarText(returnValue) {
        if (returnValue === null || returnValue === undefined || Number.isNaN(returnValue)) {
            return "Normalized $1 → —";
        }
    
        const finalValue = 1 + returnValue;
    
        return "Normalized $1 → $" + finalValue.toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });
    }
    
    function signalSubText(signal) {
        if (signal === 1) return "Bullish";
        if (signal === -1) return "Bearish";
        return "Neutral";
    }
    
    function updateSelectedPerformanceCards(ticker) {
        const panel = document.getElementById("selected-performance-panel");
        if (!panel) return;
    
        const selectedTicker = ticker || currentTicker || "BTC-USD";
    
        const item =
            defaultOrder.find(row => row.ticker === selectedTicker) ||
            allSignals.find(row => row.ticker === selectedTicker);
    
        if (!item) return;
    
        const duration = document.getElementById("period-select")?.value || "5y";
        const horizon = durationLabel(duration);
    
        document.querySelectorAll(".metric-horizon-text").forEach(el => {
            el.textContent = horizon;
        });
    
        document.getElementById("metric-strategy-return").innerHTML =
            formatPercent(item.strategy_period_return);
    
        document.getElementById("metric-buyhold-return").innerHTML =
            formatPercent(item.buy_hold_period_return);
    
        document.getElementById("metric-outperformance").innerHTML =
            formatOutperformanceRatio(item.outperformance_ratio);
    
        document.getElementById("metric-today-signal").innerHTML =
        formatSignal(item.today_signal, item.signals_locked);
    
        document.getElementById("metric-strategy-sub").textContent =
            normalizedDollarText(item.strategy_period_return);
    
        document.getElementById("metric-buyhold-sub").textContent =
            normalizedDollarText(item.buy_hold_period_return);
    
        document.getElementById("metric-signal-sub").textContent =
            item.signals_locked ? "Pro signal locked" : signalSubText(item.today_signal);

        document.getElementById("metric-strategy-max-drawdown").innerHTML =
            formatPercent(item.strategy_max_drawdown);

        document.getElementById("metric-buyhold-max-drawdown").innerHTML =
            formatPercent(item.buy_hold_max_drawdown);

        document.getElementById("metric-strategy-sharpe").textContent =
            formatMetricNumber(item.sharpe_ratio, 2);

        document.getElementById("metric-market-exposure").textContent =
            formatMetricPercentText(item.strategy_market_exposure, 1, false);

        const tradeCount = Number(item.executed_trade_count);
        document.getElementById("metric-trade-count").textContent =
            Number.isFinite(tradeCount)
                ? `${tradeCount.toLocaleString()} executed trades`
                : "— executed trades";

        const dataThroughEl = document.getElementById("metric-data-through");
        if (dataThroughEl) {
            dataThroughEl.textContent = ntFormatDataDate(item.data_through);
        }

        const siteUpdatedEl = document.getElementById("metric-site-data-updated");
        if (siteUpdatedEl) {
            siteUpdatedEl.textContent = ntFormatUtcDataTimestamp(
                item.site_data_updated_at_utc
            );
        }

        ntApplyFreshnessBadge(
            document.getElementById("metric-freshness-badge"),
            item
        );
    }
    
    let currentSort = {
        key: null,
        direction: 1   // 1 = ascending, -1 = descending
    };
    
    function sortBy(key) {
    
        if (currentSort.key !== key) {
            currentSort.key = key;
            currentSort.direction = 1;
        }
        else {
            if (currentSort.direction === 1) {
                currentSort.direction = -1;
            }
            else {
                currentSort.key = null;
                currentSort.direction = 1;
                updateSortIndicators();
                applyAllFilters();
                return;
            }
        }
    
        updateSortIndicators();
        applyAllFilters();
    }
    
    function updateSortIndicators() {
        const indicators = {
            buy_hold_period_return: document.getElementById("buy-hold-sort-indicator"),
            strategy_period_return: document.getElementById("strategy-sort-indicator"),
            outperformance_ratio: document.getElementById("outperformance-sort-indicator"),
            alpha: document.getElementById("alpha-sort-indicator"),
            alpha_prob: document.getElementById("alpha-prob-sort-indicator"),
            strategy_avg_return: document.getElementById("strategy-avg-return-sort-indicator"),
            strategy_profit_prob: document.getElementById("strategy-profit-prob-sort-indicator"),
            recommended_days: document.getElementById("recommended-days-sort-indicator")
        };

        Object.values(indicators).forEach(element => {
            if (element) element.textContent = "▲▼";
        });

        if (!currentSort.key) return;

        const activeIndicator = indicators[currentSort.key];
        if (activeIndicator) {
            activeIndicator.textContent = currentSort.direction === 1 ? "▲" : "▼";
        }
    }
    
    function signalBoardStateMarkup(kind, message) {
        if (kind === "loading") {
            return `
                <div class="nt-board-state nt-board-state-loading" role="status" aria-live="polite" aria-label="${escapeHTML(message)}">
                    <div class="nt-board-loading-copy">
                        <span class="nt-board-loading-spinner" aria-hidden="true"></span>
                        <span>${escapeHTML(message)}</span>
                    </div>
                    ${Array.from({ length: 5 }, () => `
                        <div class="nt-board-skeleton-row" aria-hidden="true">
                            <span class="nt-board-skeleton nt-board-skeleton-wide"></span>
                            <span class="nt-board-skeleton"></span>
                            <span class="nt-board-skeleton"></span>
                            <span class="nt-board-skeleton"></span>
                        </div>
                    `).join("")}
                    <span class="nt-sr-only">${escapeHTML(message)}</span>
                </div>
            `;
        }

        const safeKind = ["empty", "error", "updating"].includes(kind) ? kind : "empty";
        return `
            <div class="nt-board-state nt-board-state-${safeKind}" role="status">
                <span class="nt-board-state-icon" aria-hidden="true">${safeKind === "error" ? "!" : "◇"}</span>
                <span>${escapeHTML(message)}</span>
            </div>
        `;
    }

    function syncSignalBoardScrollbarCompensation() {
        const tableCard = document.querySelector("#epoch-tool-overview .nt-signal-table-card");
        const boardEl = document.getElementById("signal-board");
        if (!tableCard || !boardEl) return;

        // The data area scrolls vertically while the two header rows remain fixed.
        // Reserve exactly the browser's real scrollbar width in the headers so
        // every grid track stays aligned from Ticker through Recommended Days.
        const scrollbarWidth = Math.max(0, boardEl.offsetWidth - boardEl.clientWidth);
        tableCard.style.setProperty("--nt-board-scrollbar-width", `${scrollbarWidth}px`);
    }

    function renderBoard(data) {
        if (!Array.isArray(data) || data.length === 0) {
            board.innerHTML = signalBoardStateMarkup(
                "empty",
                "No assets match the current search and filters."
            );
            requestAnimationFrame(syncSignalBoardScrollbarCompensation);
            return;
        }

        const duration = document.getElementById("period-select")?.value || "5y";

        board.innerHTML = data.map(item => {
            const safeTicker = escapeHTML(item.ticker || "");
            const isWatched = typeof window.neuralTrendWatchlistHasTicker === "function"
                ? window.neuralTrendWatchlistHasTicker(item.ticker)
                : false;
            const watchlistAction = isWatched ? "Remove from watchlist" : "Add to watchlist";

            return `
                <div
                    class="signal-row"
                    data-ticker="${safeTicker}"
                    role="button"
                    tabindex="0"
                >
                    <div class="signal-ticker-cell">
                        <button
                            type="button"
                            class="nt-row-watchlist-star${isWatched ? " is-watching" : ""}"
                            data-watchlist-row-toggle
                            data-ticker="${safeTicker}"
                            aria-pressed="${isWatched ? "true" : "false"}"
                            aria-label="${watchlistAction} ${safeTicker}"
                            title="${watchlistAction}"
                        >
                            <svg
                                class="nt-row-watchlist-star-icon"
                                viewBox="0 0 24 24"
                                aria-hidden="true"
                                focusable="false"
                            >
                                <path d="M12 2.75l2.85 5.78 6.38.93-4.62 4.5 1.09 6.35L12 17.32l-5.7 2.99 1.09-6.35-4.62-4.5 6.38-.93L12 2.75z"></path>
                            </svg>
                        </button>

                        <div class="signal-ticker-stack">
                            <div class="signal-ticker">${safeTicker}</div>
                            <div
                                class="signal-freshness-meta nt-freshness-${ntSafeFreshnessStatus(item.freshness_status)}"
                                title="${escapeHTML(item.freshness_message || "Freshness unavailable.")}"
                            >
                                <span class="signal-freshness-dot" aria-hidden="true"></span>
                                <span>${escapeHTML(ntFreshnessCompactText(item))}</span>
                            </div>
                        </div>
                    </div>
        
                    <div class="signal-cell-left">${formatSignal(item.today_signal, item.signals_locked)}</div>
                    <div class="signal-cell-left">${formatSignal(item.yesterday_signal, item.signals_locked)}</div>
                    <div class="signal-cell-left">${formatSignal(item.last_week_signal, item.signals_locked)}</div>
                    <div class="signal-cell-left">${formatSignal(item.last_month_signal, item.signals_locked)}</div>
        
                    <div class="signal-cell-right">${formatPercent(item.buy_hold_period_return)}</div>
                    <div class="signal-cell-right">${formatPercent(item.strategy_period_return)}</div>
                    <div class="signal-cell-right">${formatOutperformanceRatio(item.outperformance_ratio)}</div>
                    <div class="signal-cell-right nt-stat-value-cell" title="${escapeHTML(signalStatWindowTitle(item, duration))}">${formatOutperformanceRatio(item.alpha)}</div>
                    <div class="signal-cell-right nt-stat-value-cell" title="${escapeHTML(signalStatWindowTitle(item, duration))}">${formatOutperformanceProbability(item.alpha_prob)}</div>
                    <div class="signal-cell-right nt-stat-value-cell" title="${escapeHTML(signalStatWindowTitle(item, duration))}">${formatOutperformanceRatio(item.strategy_avg_return)}</div>
                    <div class="signal-cell-right nt-stat-value-cell" title="${escapeHTML(signalStatWindowTitle(item, duration))}">${formatOutperformanceProbability(item.strategy_profit_prob)}</div>
                    <div class="signal-cell-right nt-recommended-days-cell">${formatRecommendedDays(item.recommended_days)}</div>
                </div>
            `;
        }).join("");

        // Newly rendered rows inherit the current Board Columns selection.
        applySignalBoardColumnLayout();
        requestAnimationFrame(syncSignalBoardScrollbarCompensation);
        highlightSelectedTicker();
    }

    function highlightSelectedTicker() {
        if (!currentTicker) return;
    
        document.querySelectorAll("#signal-board .signal-row").forEach(row => {
            row.classList.toggle("selected", row.dataset.ticker === currentTicker);
        });
    }
    
    /* ===============================
       2. QUICK SIGNAL BOARD (AUTO LOAD)
    ================================ */
    
    const board = document.getElementById('signal-board');

    function activateSignalBoardRow(event) {
        if (event.target.closest("[data-watchlist-row-toggle]")) return;

        const row = event.target.closest(".signal-row[data-ticker]");
        if (!row || !board.contains(row)) return;

        if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") {
            return;
        }

        if (event.type === "keydown") {
            event.preventDefault();
        }

        loadTicker(row.dataset.ticker);
    }

    board.addEventListener("click", activateSignalBoardRow);
    board.addEventListener("keydown", activateSignalBoardRow);
    
    board.innerHTML = signalBoardStateMarkup("loading", "Loading signals…");
    
    function signalBoardDelay(ms) {
        return new Promise(resolve => window.setTimeout(resolve, ms));
    }

    async function loadSummary() {
        const duration = document.getElementById('period-select')?.value || "5y";
        const requestId = ++signalSummaryRequestId;
        signalSummaryIsLoading = true;

        try {
            let data = null;
            let lastError = null;
            const retryDelays = [0, 650, 1600];

            for (let attempt = 0; attempt < retryDelays.length; attempt += 1) {
                if (retryDelays[attempt] > 0) {
                    if (requestId === signalSummaryRequestId) {
                        board.innerHTML = signalBoardStateMarkup(
                            "updating",
                            "Signal data is updating… retrying automatically."
                        );
                    }
                    await signalBoardDelay(retryDelays[attempt]);
                }

                // A newer horizon/account request supersedes this one.
                if (
                    requestId !== signalSummaryRequestId ||
                    (document.getElementById('period-select')?.value || "5y") !== duration
                ) {
                    return;
                }

                try {
                    const response = await fetch(
                        `/signals/summary?duration=${duration}&_=${Date.now()}`,
                        {
                            cache: "no-store",
                            credentials: "same-origin"
                        }
                    );

                    let payload = null;
                    try {
                        payload = await response.json();
                    } catch (_parseError) {
                        throw new Error(`Signal summary returned HTTP ${response.status}.`);
                    }

                    if (!response.ok) {
                        throw new Error(
                            payload?.error || `Signal summary returned HTTP ${response.status}.`
                        );
                    }

                    if (!Array.isArray(payload)) {
                        throw new Error(
                            payload?.error || "Signal summary response was not a list."
                        );
                    }

                    // The application normally has hundreds of supported assets.
                    // An empty unfiltered server payload therefore means the data
                    // layer was temporarily unavailable, not that user filters
                    // matched zero rows. Never convert that condition into the
                    // misleading “No assets match…” board state.
                    if (payload.length === 0) {
                        throw new Error("Signal summary was temporarily empty.");
                    }

                    data = payload;
                    break;
                } catch (error) {
                    lastError = error;
                }
            }

            // Ignore stale responses after a horizon/account change.
            if (
                requestId !== signalSummaryRequestId ||
                (document.getElementById('period-select')?.value || "5y") !== duration
            ) {
                return;
            }

            if (!data) {
                throw lastError || new Error("Signal summary could not be loaded.");
            }

            allSignals = data;
            defaultOrder = [...data];

            // Mark the request complete before applying local filters; otherwise
            // the loading guard would intentionally keep the status message.
            signalSummaryIsLoading = false;

            updateSortIndicators();
            applyAllFilters();

            // Auto-load BTC-USD preview only once
            if (!currentTicker) {
                const btcExists = data.find(item => item.ticker === "BTC-USD");
                if (btcExists) {
                    loadTicker("BTC-USD");
                }
            } else {
                updateSelectedPerformanceCards(currentTicker);
            }
        } catch (err) {
            if (requestId !== signalSummaryRequestId) return;

            signalSummaryIsLoading = false;
            board.innerHTML = signalBoardStateMarkup(
                "error",
                "Signal data is temporarily unavailable. Please try again shortly."
            );
            console.error(err);
        } finally {
            if (requestId === signalSummaryRequestId) {
                signalSummaryIsLoading = false;
            }
        }
    }

    function refreshSignalBoardForCurrentUser() {
        // Login/account updates can reach this function through both the global
        // site-shell callback and the neuralTrendUserUpdated event. Coalesce
        // overlapping calls so the board refreshes exactly once with the new
        // authenticated session instead of issuing competing requests.
        if (signalBoardAuthRefreshPromise) {
            return signalBoardAuthRefreshPromise;
        }

        signalBoardAuthRefreshPromise = (async () => {
            const board = document.getElementById("signal-board");

            if (board) {
                board.innerHTML = signalBoardStateMarkup(
                    "updating",
                    "Updating your account access…"
                );
            }

            try {
                await loadSummary();

                if (currentTicker) {
                    await loadTicker(currentTicker);
                }
            } catch (error) {
                console.error("Could not refresh signal board after account update:", error);
            } finally {
                signalBoardAuthRefreshPromise = null;
            }
        })();

        return signalBoardAuthRefreshPromise;
    }

    // Make the refresh hook unambiguously available to the global login shell
    // and checkout-status code, regardless of how the browser scopes this file.
    window.refreshSignalBoardForCurrentUser = refreshSignalBoardForCurrentUser;

    function signalBoardAuthSignature(userData) {
        if (!userData || !userData.email) {
            return "anonymous";
        }

        return [
            String(userData.email).toLowerCase(),
            `paid:${Boolean(userData.is_paid)}`,
            `admin:${Boolean(userData.is_admin)}`
        ].join("|");
    }

    // The first user-state event simply establishes the page's starting state.
    // Any later login/account-access change immediately reloads the user-masked
    // signal payload, which removes Pro locks without requiring a page refresh.
    document.addEventListener("neuralTrendUserUpdated", function(event) {
        const nextSignature = signalBoardAuthSignature(event.detail);

        if (lastSignalBoardAuthSignature === null) {
            lastSignalBoardAuthSignature = nextSignature;
            return;
        }

        if (nextSignature === lastSignalBoardAuthSignature) {
            return;
        }

        lastSignalBoardAuthSignature = nextSignature;
        refreshSignalBoardForCurrentUser();
    });
    
    // Load on page start
    initializeSignalStatInfoTooltips();
    updateSignalStatHeaderCopy();
    initializeBoardColumnChooser();
    syncSignalBoardScrollbarCompensation();
    window.addEventListener("resize", function () {
        applySignalBoardColumnLayout();
        syncSignalBoardScrollbarCompensation();
    });

    if (typeof ResizeObserver !== "undefined") {
        const signalBoardResizeObserver = new ResizeObserver(syncSignalBoardScrollbarCompensation);
        signalBoardResizeObserver.observe(board);
    }

    loadSummary();
    
    document.querySelectorAll('.signal-filter').forEach(select => {
        select.addEventListener('change', function() {
            const key = this.dataset.key;
            const value = this.value;
    
            signalFilters[key] = value === "" ? null : parseInt(value);
    
            this.classList.toggle("nt-filter-active", value !== "");
    
            applyAllFilters();
        });
    });
    
    document.getElementById('period-select')
        .addEventListener('change', function() {
    
            updateSignalStatHeaderCopy();

            board.innerHTML = signalBoardStateMarkup(
                "loading",
                "Updating the selected horizon…"
            );
    
            // Load the summary for the new horizon first, then refresh the
            // selected asset preview. This prevents a transient state where the
            // equity chart shows the new window while the metric cards still
            // display returns from the previous horizon.
            loadSummary().then(() => {
                if (currentTicker) {
                    loadTicker(currentTicker);
                }
            });
            updateAssumptionsBar();
        });

    /* ===============================
       Return horizon pill buttons
    ================================ */
    
    function syncPeriodPills() {
        const select = document.getElementById("period-select");
        const group = document.getElementById("period-pill-group");
    
        if (!select || !group) return;
    
        group.querySelectorAll(".period-pill").forEach(btn => {
            const isActive = btn.dataset.value === select.value;
            btn.classList.toggle("active", isActive);
            btn.setAttribute("aria-pressed", isActive ? "true" : "false");
        });
    }
    
    document.querySelectorAll("#period-pill-group .period-pill").forEach(btn => {
        btn.addEventListener("click", function () {
            const select = document.getElementById("period-select");
            if (!select) return;
    
            const newValue = this.dataset.value;
    
            if (!newValue) return;
    
            if (select.value === newValue) {
                return;
            }
    
            select.value = newValue;
            syncPeriodPills();
    
            select.dispatchEvent(new Event("change", { bubbles: true }));
        });
    });
    
    // Initial visual sync on page load
    syncPeriodPills();
    
    document.getElementById('ticker-search')
        .addEventListener('input', function() {
            applyAllFilters();
        });

    window.addEventListener("pageshow", function() {
        setTimeout(clearSuspiciousTickerSearchAutofill, 100);
    });
    
    document.addEventListener("neuralTrendUserUpdated", function() {
        setTimeout(clearSuspiciousTickerSearchAutofill, 100);
    });
    
    document.getElementById('asset-type-filter')
        .addEventListener('change', function() {
    
            assetTypeFilter = this.value;
            applyAllFilters();
        });

    /* ===============================
       Asset type pill buttons
    ================================ */
    
    function syncAssetPills() {
        const select = document.getElementById("asset-type-filter");
        const group = document.getElementById("asset-pill-group");
    
        if (!select || !group) return;
    
        group.querySelectorAll(".asset-pill").forEach(btn => {
            const isActive = btn.dataset.value === select.value;
            btn.classList.toggle("active", isActive);
            btn.setAttribute("aria-pressed", isActive ? "true" : "false");
        });
    }
    
    document.querySelectorAll("#asset-pill-group .asset-pill").forEach(btn => {
        btn.addEventListener("click", function () {
            const select = document.getElementById("asset-type-filter");
            if (!select) return;
    
            const newValue = this.dataset.value;

            if (newValue === "watchlist" && !window.neuralTrendCurrentUser?.email) {
                if (typeof openNeuralTrendLoginModal === "function") {
                    openNeuralTrendLoginModal("Log in to save assets and view your watchlist.");
                }
                return;
            }
    
            if (!newValue) return;
    
            if (select.value === newValue) {
                return;
            }
    
            select.value = newValue;
            syncAssetPills();
    
            select.dispatchEvent(new Event("change", { bubbles: true }));
        });
    });
    
    // Initial visual sync
    syncAssetPills();
    
    document.getElementById("reset-signal-board")
        .addEventListener("click", function () {
            clearSignalBoardFilters();
        });
    
    /* ===============================
       3. LOAD TICKER FROM BOARD CLICK
    ================================ */
    
    let currentTicker = null;
    let previewChart = null;   // 🔥 prevents duplicate charts
    
    function loadTicker(ticker) {

        currentTicker = ticker;
        const requestId = ++equityPreviewRequestId;
        document.dispatchEvent(new CustomEvent("neuralTrendTickerSelected", {
            detail: { ticker }
        }));
    
        highlightSelectedTicker();
        updateSelectedPerformanceCards(ticker);
        syncBacktestTicker(ticker);
        syncLiveSimulationTicker(ticker);
    
        const duration = document.getElementById('period-select').value;
    
        const formData = new FormData();
        formData.append('ticker', ticker);
        formData.append('duration', duration);
    
        fetch('/equity', {
            method: 'POST',
            body: formData
        })
        .then(async response => {
            const data = await response.json();

            // Keep the preview tied to the same ticker + horizon that initiated
            // this request. Slow older responses must never replace newer data.
            if (
                requestId !== equityPreviewRequestId ||
                currentTicker !== ticker ||
                document.getElementById('period-select').value !== duration
            ) {
                return;
            }
    
            if (!response.ok) {
                showLockedEquityPreview(data);
                return;
            }
    
            renderEquityPreview(data);
        })
        .catch(error => {
            if (requestId !== equityPreviewRequestId) return;
            console.error("Equity preview error:", error);
            showLockedEquityPreview({
                error: "Could not load equity preview.",
                ticker: ticker
            });
        });
    }
    
    /* ===============================
       5. RENDER DETAILED SIGNALS
    ================================ */
    
    const signalMarkerPlugin = {
        id: "signalMarkerPlugin",
        afterDatasetsDraw(chart) {
    
            const { ctx } = chart;
    
            chart.data.datasets.forEach((dataset, datasetIndex) => {
    
                if (!dataset.isSignal) return;
    
                const meta = chart.getDatasetMeta(datasetIndex);
    
                meta.data.forEach(point => {
    
                    if (!point || point.skip) return;
    
                    const x = point.x;
                    const y = point.y;
    
                    const offset = dataset.offsetPx || 0;
    
                    ctx.save();
    
                    ctx.translate(x, y + offset);
    
                    ctx.beginPath();
    
                    if (dataset.direction === "buy") {
                        // Up triangle
                        ctx.moveTo(0, -8);
                        ctx.lineTo(6, 6);
                        ctx.lineTo(-6, 6);
                    } else {
                        // Down triangle
                        ctx.moveTo(0, 8);
                        ctx.lineTo(6, -6);
                        ctx.lineTo(-6, -6);
                    }
    
                    ctx.closePath();
                    ctx.fillStyle = dataset.color;
                    ctx.fill();
    
                    ctx.restore();
                });
            });
        }
    };
    
    Chart.register(signalMarkerPlugin);

    function showLockedEquityPreview(data = {}) {
        const container = document.getElementById("equity-preview-container");
        const chartDiv = document.getElementById("equity-preview-chart");
    
        if (!container || !chartDiv) return;
    
        container.style.display = "block";

        const riskGrid = document.getElementById("equity-preview-risk-grid");
        if (riskGrid) riskGrid.style.display = "none";
    
        const ticker = data.ticker || currentTicker || "this asset";
    
        const titleEl = document.getElementById("preview-title");
        if (titleEl) {
            titleEl.innerText = `Compare AI strategy equity vs. Buy&Hold for ${ticker}`;
        }
    
        const previewTickerEl = document.getElementById("preview-selected-ticker");
        if (previewTickerEl) {
            previewTickerEl.textContent = ticker;
        }

        const previewDataThrough = document.getElementById("preview-data-through");
        if (previewDataThrough) previewDataThrough.textContent = "—";
        ntApplyFreshnessBadge(
            document.getElementById("preview-freshness-badge"),
            { freshness_status: "unknown", freshness_label: "Locked", freshness_message: "Upgrade to load this asset's data freshness." },
            true
        );
    
        chartDiv.innerHTML = `
            <div class="nt-pro-lock-card">
                <div class="nt-pro-lock-icon">🔒</div>
    
                <h3>Pro feature locked</h3>
    
                <p>
                    Full signal history, equity preview, and backtesting for ${escapeHTML(ticker)}
                    are available with NeuralTrend Pro.
                </p>
    
                <button type="button" class="nt-pro-upgrade-btn"
                            data-nt-go-subscription>
                    Upgrade to Pro
                </button>
            </div>
        `;
    }
    
    function renderEquityPreview(data) {
    
        const container = document.getElementById("equity-preview-container");
        const chartDiv = document.getElementById("equity-preview-chart");
    
        container.style.display = "block";

        const currentDuration = document.getElementById("period-select")?.value || "5y";
        const currentHorizonLabel = durationLabel(currentDuration);
        
        document.getElementById("preview-title").innerText =
            `Compare AI strategy equity vs. Buy&Hold for ${data.ticker}`;
        
        const previewTickerEl = document.getElementById("preview-selected-ticker");
        if (previewTickerEl) {
            previewTickerEl.textContent = data.ticker;
        }
        
        const previewHorizonEl = document.getElementById("preview-horizon-label");
        if (previewHorizonEl) {
            previewHorizonEl.textContent = currentHorizonLabel;
        }
        
        const previewScaleModeEl = document.getElementById("preview-scale-mode");
        if (previewScaleModeEl) {
            previewScaleModeEl.textContent = isLogScale ? "Log" : "Linear";
        }

        const previewCostLabelEl = document.getElementById("preview-cost-label");
        if (previewCostLabelEl) {
            const costPct = Number(data.transaction_cost_rate || 0) * 100;
            previewCostLabelEl.textContent = `${costPct.toFixed(2)}% per executed trade`;
        }

        const previewDataThrough = document.getElementById("preview-data-through");
        if (previewDataThrough) {
            previewDataThrough.textContent = ntFormatDataDate(data.data_through);
        }

        ntApplyFreshnessBadge(
            document.getElementById("preview-freshness-badge"),
            data,
            true
        );

        const riskGrid = document.getElementById("equity-preview-risk-grid");
        if (riskGrid) riskGrid.style.display = "grid";

        document.getElementById("preview-strategy-max-drawdown").textContent =
            formatMetricPercentText(data.strategy_max_drawdown, 1);
        document.getElementById("preview-buyhold-max-drawdown").textContent =
            formatMetricPercentText(data.buy_hold_max_drawdown, 1);
        document.getElementById("preview-strategy-sharpe").textContent =
            formatMetricNumber(data.sharpe_ratio, 2);
        document.getElementById("preview-strategy-volatility").textContent =
            formatMetricPercentText(data.strategy_annualized_volatility, 1, false);

        const exposureText = formatMetricPercentText(
            data.strategy_market_exposure,
            1,
            false
        );
        const tradeCount = Number(data.executed_trade_count);
        const tradeText = Number.isFinite(tradeCount)
            ? tradeCount.toLocaleString()
            : "—";
        document.getElementById("preview-exposure-trades").textContent =
            `${exposureText} / ${tradeText}`;
    
        chartDiv.innerHTML = `
            <canvas id="previewChartCanvas"></canvas>
        `;
        const ctx = document.getElementById("previewChartCanvas").getContext("2d");
    
        if (previewChart) {
            previewChart.destroy();
        }
    
        const equity = data.epoch_equity_curve;
        const buyHold = data.equity_curve;
        const dates = data.dates;
        const executedBuyDates = Array.isArray(data.executed_buy_dates) ? data.executed_buy_dates : [];
        const executedSellDates = Array.isArray(data.executed_sell_dates) ? data.executed_sell_dates : [];
    
        // 🔥 Dynamic offset (same as backtest)
        const equity_min = Math.min(...equity);
        const equity_max = Math.max(...equity);
        const offset = (equity_max - equity_min) * 0.04;
    
        const buySignals = dates.map((d, i) =>
        executedBuyDates.includes(d) ? equity[i] : null
        );
        
        const sellSignals = dates.map((d, i) =>
            executedSellDates.includes(d) ? equity[i] : null
        );
    
        const ntPerformanceShadePlugin = {
            id: "ntPerformanceShade",

            beforeDatasetsDraw(chart) {
                const {ctx, chartArea} = chart;
                if (!chartArea) return;

                const aiPoints = chart.getDatasetMeta(0)?.data || [];
                const buyHoldPoints = chart.getDatasetMeta(1)?.data || [];
                const pointCount = Math.min(aiPoints.length, buyHoldPoints.length);

                if (pointCount < 2) return;

                const EPSILON = 1e-7;

                const lerpPoint = (a, b, t) => ({
                    x: a.x + (b.x - a.x) * t,
                    y: a.y + (b.y - a.y) * t
                });

                const makeCurve = (start, end) => ({
                    p0: {x: start.x, y: start.y},
                    p1: {
                        x: Number.isFinite(start.cp2x) ? start.cp2x : start.x,
                        y: Number.isFinite(start.cp2y) ? start.cp2y : start.y
                    },
                    p2: {
                        x: Number.isFinite(end.cp1x) ? end.cp1x : end.x,
                        y: Number.isFinite(end.cp1y) ? end.cp1y : end.y
                    },
                    p3: {x: end.x, y: end.y}
                });

                const pointOnCurve = (curve, t) => {
                    const oneMinusT = 1 - t;
                    const oneMinusTSquared = oneMinusT * oneMinusT;
                    const tSquared = t * t;

                    return {
                        x:
                            oneMinusTSquared * oneMinusT * curve.p0.x +
                            3 * oneMinusTSquared * t * curve.p1.x +
                            3 * oneMinusT * tSquared * curve.p2.x +
                            tSquared * t * curve.p3.x,
                        y:
                            oneMinusTSquared * oneMinusT * curve.p0.y +
                            3 * oneMinusTSquared * t * curve.p1.y +
                            3 * oneMinusT * tSquared * curve.p2.y +
                            tSquared * t * curve.p3.y
                    };
                };

                const splitCurve = (curve, t) => {
                    const p01 = lerpPoint(curve.p0, curve.p1, t);
                    const p12 = lerpPoint(curve.p1, curve.p2, t);
                    const p23 = lerpPoint(curve.p2, curve.p3, t);
                    const p012 = lerpPoint(p01, p12, t);
                    const p123 = lerpPoint(p12, p23, t);
                    const p0123 = lerpPoint(p012, p123, t);

                    return [
                        {p0: curve.p0, p1: p01, p2: p012, p3: p0123},
                        {p0: p0123, p1: p123, p2: p23, p3: curve.p3}
                    ];
                };

                const sliceCurve = (curve, startT, endT) => {
                    let sliced = curve;
                    let upperBound = endT;

                    if (endT < 1 - EPSILON) {
                        sliced = splitCurve(sliced, endT)[0];
                    } else {
                        upperBound = 1;
                    }

                    if (startT > EPSILON) {
                        sliced = splitCurve(sliced, startT / upperBound)[1];
                    }

                    return sliced;
                };

                const cubicValue = (a, b, c, d, t) =>
                    ((a * t + b) * t + c) * t + d;

                const uniqueSorted = (values) => values
                    .filter((value) => value > EPSILON && value < 1 - EPSILON)
                    .sort((a, b) => a - b)
                    .filter((value, index, sorted) =>
                        index === 0 || Math.abs(value - sorted[index - 1]) > 1e-5
                    );

                /*
                  Find every crossover inside one rendered curve segment.
                  The two equity lines use monotone interpolation below, so
                  corresponding Bezier segments share the same x progression.
                */
                const findCrossings = (aiCurve, buyHoldCurve) => {
                    const d0 = aiCurve.p0.y - buyHoldCurve.p0.y;
                    const d1 = aiCurve.p1.y - buyHoldCurve.p1.y;
                    const d2 = aiCurve.p2.y - buyHoldCurve.p2.y;
                    const d3 = aiCurve.p3.y - buyHoldCurve.p3.y;

                    const a = -d0 + 3 * d1 - 3 * d2 + d3;
                    const b = 3 * d0 - 6 * d1 + 3 * d2;
                    const c = -3 * d0 + 3 * d1;
                    const d = d0;

                    const evaluate = (t) => cubicValue(a, b, c, d, t);
                    const boundaries = [0, 1];

                    // Add derivative roots so each tested interval is monotonic.
                    const derivativeA = 3 * a;
                    const derivativeB = 2 * b;
                    const derivativeC = c;

                    if (Math.abs(derivativeA) < EPSILON) {
                        if (Math.abs(derivativeB) >= EPSILON) {
                            const root = -derivativeC / derivativeB;
                            if (root > 0 && root < 1) boundaries.push(root);
                        }
                    } else {
                        const discriminant =
                            derivativeB * derivativeB -
                            4 * derivativeA * derivativeC;

                        if (discriminant >= 0) {
                            const sqrtDiscriminant = Math.sqrt(discriminant);
                            const root1 = (-derivativeB - sqrtDiscriminant) /
                                (2 * derivativeA);
                            const root2 = (-derivativeB + sqrtDiscriminant) /
                                (2 * derivativeA);

                            if (root1 > 0 && root1 < 1) boundaries.push(root1);
                            if (root2 > 0 && root2 < 1) boundaries.push(root2);
                        }
                    }

                    boundaries.sort((left, right) => left - right);

                    const roots = [];
                    for (let i = 0; i < boundaries.length; i++) {
                        const boundary = boundaries[i];
                        if (
                            boundary > EPSILON &&
                            boundary < 1 - EPSILON &&
                            Math.abs(evaluate(boundary)) < 1e-5
                        ) {
                            roots.push(boundary);
                        }
                    }

                    for (let i = 1; i < boundaries.length; i++) {
                        let left = boundaries[i - 1];
                        let right = boundaries[i];
                        let leftValue = evaluate(left);
                        const rightValue = evaluate(right);

                        if (leftValue * rightValue >= 0) continue;

                        for (let iteration = 0; iteration < 45; iteration++) {
                            const middle = (left + right) / 2;
                            const middleValue = evaluate(middle);

                            if (Math.abs(middleValue) < 1e-7) {
                                left = middle;
                                right = middle;
                                break;
                            }

                            if (leftValue * middleValue <= 0) {
                                right = middle;
                            } else {
                                left = middle;
                                leftValue = middleValue;
                            }
                        }

                        roots.push((left + right) / 2);
                    }

                    return uniqueSorted(roots);
                };

                ctx.save();
                ctx.beginPath();
                ctx.rect(
                    chartArea.left,
                    chartArea.top,
                    chartArea.right - chartArea.left,
                    chartArea.bottom - chartArea.top
                );
                ctx.clip();

                for (let i = 1; i < pointCount; i++) {
                    const aiStart = aiPoints[i - 1];
                    const aiEnd = aiPoints[i];
                    const buyHoldStart = buyHoldPoints[i - 1];
                    const buyHoldEnd = buyHoldPoints[i];

                    if (
                        aiStart.skip || aiEnd.skip ||
                        buyHoldStart.skip || buyHoldEnd.skip
                    ) {
                        continue;
                    }

                    const aiCurve = makeCurve(aiStart, aiEnd);
                    const buyHoldCurve = makeCurve(buyHoldStart, buyHoldEnd);
                    const crossings = findCrossings(aiCurve, buyHoldCurve);
                    const intervalBounds = [0, ...crossings, 1];

                    for (let j = 1; j < intervalBounds.length; j++) {
                        const startT = intervalBounds[j - 1];
                        const endT = intervalBounds[j];

                        if (endT - startT <= EPSILON) continue;

                        const middleT = (startT + endT) / 2;
                        const aiMiddle = pointOnCurve(aiCurve, middleT);
                        const buyHoldMiddle = pointOnCurve(buyHoldCurve, middleT);
                        const aiSlice = sliceCurve(aiCurve, startT, endT);
                        const buyHoldSlice = sliceCurve(
                            buyHoldCurve,
                            startT,
                            endT
                        );

                        ctx.beginPath();
                        ctx.moveTo(aiSlice.p0.x, aiSlice.p0.y);
                        ctx.bezierCurveTo(
                            aiSlice.p1.x,
                            aiSlice.p1.y,
                            aiSlice.p2.x,
                            aiSlice.p2.y,
                            aiSlice.p3.x,
                            aiSlice.p3.y
                        );
                        ctx.lineTo(buyHoldSlice.p3.x, buyHoldSlice.p3.y);
                        ctx.bezierCurveTo(
                            buyHoldSlice.p2.x,
                            buyHoldSlice.p2.y,
                            buyHoldSlice.p1.x,
                            buyHoldSlice.p1.y,
                            buyHoldSlice.p0.x,
                            buyHoldSlice.p0.y
                        );
                        ctx.closePath();

                        // Canvas y-coordinates decrease as values rise.
                        ctx.fillStyle = aiMiddle.y <= buyHoldMiddle.y
                            ? "rgba(22, 163, 74, 0.10)"
                            : "rgba(220, 38, 38, 0.10)";
                        ctx.fill();
                    }
                }

                ctx.restore();
            }
        };

        previewChart = new Chart(ctx, {
            type: "line",
            plugins: [ntPerformanceShadePlugin],
            data: {
                labels: dates,
                datasets: [
                    {
                        label: "EpochSignaler",
                        data: equity,
                        borderColor: NT_COLORS.chartAI,
                        backgroundColor: NT_COLORS.chartAI,
                        pointStyle: "line",
                        pointRadius: 0,
                        borderWidth: 2,
                        cubicInterpolationMode: "monotone",
                        fill: false
                    },
                    {
                        label: "Buy & Hold",
                        data: buyHold,
                        borderColor: NT_COLORS.chartBuyHold,
                        backgroundColor: NT_COLORS.chartBuyHold,
                        borderDash: [5, 5],
                        pointStyle: "line",
                        pointRadius: 0,
                        borderWidth: 2,
                        cubicInterpolationMode: "monotone",
                        fill: false
                    },
                    {
                        label: "Executed BUY",
                        data: buySignals,
                        showLine: false,
                        pointRadius: 0,
                        pointStyle: "triangle",
                        backgroundColor: NT_COLORS.buy,
                        borderColor: NT_COLORS.buy,
                        isSignal: true,
                        direction: "buy",
                        color: NT_COLORS.buy,
                        offsetPx: 11
                    },
                    {
                        label: "Executed SELL",
                        data: sellSignals,
                        showLine: false,
                        pointRadius: 0,
                        pointStyle: "triangle",
                        rotation: 180,
                        backgroundColor: NT_COLORS.sell,
                        borderColor: NT_COLORS.sell,
                        isSignal: true,
                        direction: "sell",
                        color: NT_COLORS.sell,
                        offsetPx: -10
                    }
                ]
            },
            options: ntChartOptions({
                logScale: isLogScale,
                yTitle: "$1 normalized equity",
                yPrefix: "",
                legendAlign: "end"
            })
        });
    
        const toggleBtn = document.getElementById("scale-toggle-btn");
        
        toggleBtn.innerText = isLogScale
            ? "Normal Scale"
            : "Log Scale";
    }

    /* ===============================
       Live Simulation UI
    ================================ */

    let liveSimChart = null;
    let selectedLiveSimulationId = null;
    let recentlyCreatedLiveSimulationId = null;
    let liveSimulationsCache = [];
    let liveSimStatusFilter = "open";
    
    let liveSimLoggedIn = false;
    let liveSimUserEmail = null;

    let liveSimBoardSortKey = "created_at";
    let liveSimBoardSortDirection = "desc";
    let liveSimBoardSignalFilter = "all";
    let liveSimBoardPositionFilter = "all";
    let liveSimBoardAssetTypeFilter = "all";

    let liveSimBoardStrategyDisplayMode = "percent";
    let liveSimBoardBenchmarkDisplayMode = "percent";
    let liveSimBoardSpreadDisplayMode = "percent";
    let liveSimBoardSearchQuery = "";

    let liveSimBoardHorizon = "since_start";

    const FREE_LIVE_SIM_TICKERS = new Set([
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
        "XRP-USD"
    ]);
    
    let liveSimUserIsPaid = false;

    function getVisibleLiveSimTickerOptions() {
        const allOptions = getLiveSimTickerOptions();
    
        if (liveSimUserIsPaid) {
            return allOptions;
        }
    
        return allOptions.filter(option =>
            FREE_LIVE_SIM_TICKERS.has(normalizeTickerForCompare(option.value))
        );
    }
    
    function liveSimMoney(value) {
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
            return "—";
        }
    
        return "$" + Number(value).toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });
    }
    
    function liveSimPercent(value) {
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
            return "—";
        }
    
        const pct = Number(value) * 100;
        const sign = pct > 0 ? "+" : "";
    
        return `${sign}${pct.toFixed(2)}%`;
    }
    
    function liveSimReturnClass(value) {
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
            return "return-neutral";
        }
    
        if (Number(value) > 0) return "return-positive";
        if (Number(value) < 0) return "return-negative";
        return "return-neutral";
    }

    function setLiveSimMessage(message, isError = false) {
        const el = document.getElementById("live-sim-message");
        if (!el) return;
    
        el.dataset.userEdited = "true";
        el.textContent = message;
        el.style.color = isError ? "var(--nt-negative)" : "var(--nt-neutral)";
    }
    
    async function liveSimFetchJSON(url, options = {}) {
        const response = await fetch(url, options);
        const text = await response.text();
    
        let data = null;
    
        try {
            data = JSON.parse(text);
        } catch (e) {
            throw new Error(`Unexpected server response (${response.status}).`);
        }
    
        if (!response.ok) {
            const error = new Error(data.error || `Request failed (${response.status}).`);
            error.data = data;
            error.status = response.status;
            throw error;
        }
    
        return data;
    }
    
    async function loadLiveSimulations() {
        const listEl = document.getElementById("live-sim-list");
        const limitEl = document.getElementById("live-sim-limit");
        const refreshBtn = document.getElementById("live-sim-refresh-btn");
    
        if (!listEl || !limitEl) return;
    
        if (refreshBtn) {
            refreshBtn.disabled = true;
            refreshBtn.textContent = "Refreshing...";
        }
    
        setLiveSimRefreshStatus("Checking latest saved simulations...");
    
        try {
            const statusQuery = encodeURIComponent(liveSimStatusFilter || "open");
            const data = await liveSimFetchJSON(`/live-simulations?status=${statusQuery}`);

            updateLiveSimFilterCounts(data);

            liveSimulationsCache = data.simulations || [];
            
            limitEl.textContent = liveSimLimitText(data);
            
            renderLiveSimulationList(liveSimulationsCache);
    
            if (data.simulations && data.simulations.length > 0) {
                const selectedExists = data.simulations.some(sim => sim.id === selectedLiveSimulationId);
                const targetId = selectedExists ? selectedLiveSimulationId : data.simulations[0].id;
                await selectLiveSimulation(targetId);
            } else {
                document.getElementById("live-sim-dashboard").style.display = "none";
            
                const listEl = document.getElementById("live-sim-list");
                if (listEl) {
                    listEl.innerHTML = `
                        <div class="nt-live-sim-small-note">
                            ${escapeHTML(liveSimEmptyStateMessage(liveSimStatusFilter))}
                        </div>
                    `;
                }
            }

            setLiveSimRefreshStatus(`Last refreshed at ${liveSimRefreshTimestamp()}.`);
    
        } catch (error) {
            resetLiveSimFilterCounts();
            
            if (isLiveSimLoginError(error)) {
                renderLiveSimulationLoggedOutState();
                setLiveSimMessage("Log in or sign up to create and save live simulations.");
                setLiveSimRefreshStatus("Login required to load saved simulations.");
            } else {
                limitEl.textContent = "Could not load simulations.";
                listEl.innerHTML = "";
                setLiveSimMessage(error.message, true);
                setLiveSimRefreshStatus("Could not refresh simulations.");
            }
        
        } finally {
            if (refreshBtn) {
                refreshBtn.disabled = false;
                refreshBtn.textContent = "Refresh";
            }
        }
    }
    
    function renderLiveSimulationList(simulations) {
        const list = document.getElementById("live-sim-list");
        if (!list) return;
    
        const allSimulations = simulations || [];
        const visibleSimulations = getFilteredAndSortedLiveSimulations(allSimulations);
    
        const positionOptions = Array.from(
            new Set(
                allSimulations
                    .map(sim => Number(sim.position_size_pct))
                    .filter(value => Number.isFinite(value))
                    .map(value => String(value.toFixed(0)))
            )
        ).sort((a, b) => Number(a) - Number(b));
    
        if (!allSimulations.length) {
            list.innerHTML = `
                <div class="nt-live-sim-board-empty">
                    No simulations found for this status.
                </div>
            `;
    
            renderLiveSimSelectedActions();
            return;
        }
    
        list.innerHTML = `
    
            <div class="nt-live-sim-board-scroll">
                <table class="nt-live-sim-board-table">
                    <thead>
                        <tr>
                            <th>
                                <button type="button" data-live-sim-board-sort="created_at">
                                    Date ${liveSimBoardSortArrow("created_at")}
                                </button>
                            </th>
    
                            <th>
                                <button type="button" data-live-sim-board-sort="name">
                                    Simulation ${liveSimBoardSortArrow("name")}
                                </button>
                            </th>
    
                            <th>
                                <button type="button" data-live-sim-board-sort="ticker">
                                    Ticker ${liveSimBoardSortArrow("ticker")}
                                </button>
                            </th>
    
                            <th>Latest Signal</th>
    
                            <th>Position</th>
    
                            <th class="numeric">
                                <button type="button" data-live-sim-board-sort="cash_position">
                                    Cash / Position ${liveSimBoardSortArrow("cash_position")}
                                </button>
                            </th>
    
                            <th class="numeric">
                                <button type="button" data-live-sim-board-sort="strategy_value">
                                    Strategy ${liveSimBoardSortArrow("strategy_value")}
                                </button>
                            </th>
    
                            <th class="numeric">
                                <button type="button" data-live-sim-board-sort="benchmark_value">
                                    B&H ${liveSimBoardSortArrow("benchmark_value")}
                                </button>
                            </th>
    
                            <th class="numeric">
                                <button type="button" data-live-sim-board-sort="return_spread">
                                    Return Spread ${liveSimBoardSortArrow("return_spread")}
                                </button>
                            </th>
                        </tr>
    
                        <tr class="nt-live-sim-board-filter-row">
                            <th></th>
                            <th></th>
                            <th></th>
    
                            <th>
                                <select id="live-sim-board-signal-filter" aria-label="Filter by latest signal">
                                    <option value="all" ${liveSimBoardSignalFilter === "all" ? "selected" : ""}>All</option>
                                    <option value="BUY" ${liveSimBoardSignalFilter === "BUY" ? "selected" : ""}>BUY</option>
                                    <option value="SELL" ${liveSimBoardSignalFilter === "SELL" ? "selected" : ""}>SELL</option>
                                    <option value="HOLD" ${liveSimBoardSignalFilter === "HOLD" ? "selected" : ""}>HOLD</option>
                                </select>
                            </th>
    
                            <th>
                                <select id="live-sim-board-position-filter" aria-label="Filter by position size">
                                    <option value="all" ${liveSimBoardPositionFilter === "all" ? "selected" : ""}>All</option>
                                    ${positionOptions.map(value => `
                                        <option value="${escapeHTML(value)}" ${liveSimBoardPositionFilter === value ? "selected" : ""}>
                                            ${escapeHTML(value)}%
                                        </option>
                                    `).join("")}
                                </select>
                            </th>
    
                            <th></th>
    
                            <th class="numeric">
                                <select id="live-sim-board-strategy-mode" aria-label="Strategy display mode">
                                    <option value="percent" ${liveSimBoardStrategyDisplayMode === "percent" ? "selected" : ""}>Return %</option>
                                    <option value="value" ${liveSimBoardStrategyDisplayMode === "value" ? "selected" : ""}>$ Value</option>
                                </select>
                            </th>
    
                            <th class="numeric">
                                <select id="live-sim-board-benchmark-mode" aria-label="Buy and Hold display mode">
                                    <option value="percent" ${liveSimBoardBenchmarkDisplayMode === "percent" ? "selected" : ""}>Return %</option>
                                    <option value="value" ${liveSimBoardBenchmarkDisplayMode === "value" ? "selected" : ""}>$ Value</option>
                                </select>
                            </th>
    
                            <th class="numeric">
                                <select id="live-sim-board-spread-mode" aria-label="Return spread display mode">
                                    <option value="percent" ${liveSimBoardSpreadDisplayMode === "percent" ? "selected" : ""}>Spread (pts)</option>
                                    <option value="value" ${liveSimBoardSpreadDisplayMode === "value" ? "selected" : ""}>$ Value</option>
                                </select>
                            </th>
                        </tr>
                    </thead>
    
                    <tbody>
                        ${
                            visibleSimulations.length
                                ? visibleSimulations.map(sim => {
                                    const simId = Number(sim.id);
                                    const isActive = Number(selectedLiveSimulationId) === simId;
                                    const signal = liveSimLatestSignalValue(sim);

                                    const strategyValue = liveSimBoardStrategyDollarForHorizon(sim);
                                    const benchmarkValue = liveSimBoardBenchmarkDollarForHorizon(sim);
                                    const valueDifference = liveSimBoardValueDifferenceForHorizon(sim);
                                    
                                    const strategyReturn = liveSimBoardStrategyReturnForHorizon(sim);
                                    const benchmarkReturn = liveSimBoardBenchmarkReturnForHorizon(sim);
                                    const returnSpread = liveSimBoardReturnSpreadForHorizon(sim);
                                    
                                    const fullName = sim.name || "Simulation";
    
                                    return `
                                        <tr 
                                            class="nt-live-sim-board-row ${isActive ? "active" : ""} ${Number(recentlyCreatedLiveSimulationId) === simId ? "just-created" : ""}"
                                            data-live-sim-board-row-id="${simId}"
                                        >
                                            <td>
                                                <div class="nt-live-sim-board-main-value">
                                                    ${escapeHTML(liveSimShortDate(sim.created_at || sim.start_date))}
                                                </div>
                                            </td>
    
                                            <td>
                                                <div class="nt-live-sim-name-tooltip-wrap">
                                                    <div class="nt-live-sim-board-name">
                                                        ${escapeHTML(fullName)}
                                                    </div>
                                                
                                                    <div class="nt-live-sim-name-tooltip">
                                                        ${escapeHTML(fullName)}
                                                    </div>
                                                </div>
                                                
                                                <div class="nt-live-sim-board-status">
                                                    ${escapeHTML(sim.status || "active")}
                                                </div>
                                            </td>
    
                                            <td>
                                                <div class="nt-live-sim-board-ticker">
                                                    ${escapeHTML(sim.ticker || "—")}
                                                </div>
                                            </td>
    
                                            <td>
                                                ${liveSimBoardSignalBadgeHTML(signal)}
                                            </td>
    
                                            <td>
                                                <div class="nt-live-sim-board-main-value">
                                                    ${Number(sim.position_size_pct).toFixed(0)}%
                                                </div>
                                                <div class="nt-live-sim-board-sub-value neutral">
                                                    per signal
                                                </div>
                                            </td>
    
                                            <td class="numeric">
                                                ${liveSimCashPositionHTML(sim)}
                                            </td>

                                            <td class="numeric">
                                                ${liveSimBoardReturnMetricHTML(strategyValue, strategyReturn, liveSimBoardStrategyDisplayMode)}
                                            </td>
                                            
                                            <td class="numeric">
                                                ${liveSimBoardReturnMetricHTML(benchmarkValue, benchmarkReturn, liveSimBoardBenchmarkDisplayMode)}
                                            </td>
                                            
                                            <td class="numeric">
                                                ${liveSimBoardSpreadMetricHTML(valueDifference, returnSpread, liveSimBoardSpreadDisplayMode)}
                                            </td>
                                        </tr>
                                    `;
                                }).join("")
                                : `
                                    <tr>
                                        <td colspan="9">
                                            <div class="nt-live-sim-board-empty">
                                                No simulations match the selected filters.
                                            </div>
                                        </td>
                                    </tr>
                                `
                        }
                    </tbody>
                </table>
            </div>
        `;

        scrollToRecentlyCreatedSimulation();
        updateLiveSimBoardActiveRow();
        renderLiveSimSelectedActions();
        syncLiveSimAssetPills();
        syncLiveSimHorizonPills();

        const liveSimSearchInput = document.getElementById("live-sim-board-search");
        if (liveSimSearchInput) {
            liveSimSearchInput.value = liveSimBoardSearchQuery;
        }
    }
    
    async function selectLiveSimulation(simId) {
        selectedLiveSimulationId = simId;
    
        try {
            const data = await liveSimFetchJSON(`/live-simulations/${simId}`);
            renderLiveSimulationDetail(data.simulation);

            document.querySelectorAll(".nt-live-sim-item").forEach(card => {
                card.classList.toggle(
                    "active",
                    Number(card.dataset.liveSimId) === Number(simId)
                );
            });
            
            updateLiveSimBoardActiveRow();
            
            renderLiveSimSelectedActions();
    
        } catch (error) {
            setLiveSimMessage(error.message, true);
        }
    }
    
    function renderLiveSimulationDetail(sim) {
        const dashboard = document.getElementById("live-sim-dashboard");
        if (!dashboard) return;
    
        dashboard.style.display = "block";
    
        document.getElementById("live-sim-detail-title").textContent =
            `${sim.name}`;
    
        document.getElementById("live-sim-detail-subtitle").textContent =
            `${sim.ticker} · ${Number(sim.position_size_pct).toFixed(0)}% per signal · Started ${sim.start_date} · ${sim.status || "active"}`;

        const liveDataThrough = document.getElementById("live-sim-data-through");
        if (liveDataThrough) {
            liveDataThrough.textContent = ntFormatDataDate(
                sim.data_through || sim.latest_csv_date
            );
        }

        const liveSimulationThrough = document.getElementById("live-sim-simulation-through");
        if (liveSimulationThrough) {
            liveSimulationThrough.textContent = ntFormatDataDate(
                sim.simulation_through || sim.latest_equity_date
            );
        }

        const liveSiteUpdated = document.getElementById("live-sim-site-data-updated");
        if (liveSiteUpdated) {
            liveSiteUpdated.textContent = ntFormatUtcDataTimestamp(
                sim.site_data_updated_at_utc
            );
        }

        ntApplyFreshnessBadge(
            document.getElementById("live-sim-freshness-badge"),
            sim
        );

        const detailSignalWrap = document.getElementById("live-sim-detail-signal-wrap");
            if (detailSignalWrap) {
                detailSignalWrap.innerHTML = liveSimLatestSignalHTML(sim);
            }
    
        document.getElementById("live-sim-strategy-value").textContent =
            liveSimMoney(sim.latest_strategy_value);
    
        document.getElementById("live-sim-benchmark-value").textContent =
            liveSimMoney(sim.latest_benchmark_value);
    
        document.getElementById("live-sim-cash-balance").textContent =
            liveSimMoney(sim.cash_balance);
    
        document.getElementById("live-sim-position-quantity").textContent =
            `Position: ${Number(sim.position_quantity).toLocaleString(undefined, {
                maximumFractionDigits: 8
            })}`;

        const valueDifference = liveSimValueDifference(sim);
        const returnSpread = liveSimReturnSpread(sim);
        const spreadClass = liveSimReturnClass(returnSpread);
        
        const spreadValueEl = document.getElementById("live-sim-spread-value");
        const spreadDetailEl = document.getElementById("live-sim-spread-detail");
        
        if (spreadValueEl) {
            spreadValueEl.textContent =
                returnSpread === null
                    ? "—"
                    : `${returnSpread > 0 ? "+" : ""}${(returnSpread * 100).toFixed(2)} pts`;
            spreadValueEl.className = `nt-live-sim-metric-value ${spreadClass}`;
        }
        
        if (spreadDetailEl) {
            spreadDetailEl.textContent =
                valueDifference === null
                    ? "—"
                    : `Value difference: ${liveSimSignedMoney(valueDifference)}`;
            spreadDetailEl.className = `nt-live-sim-metric-sub ${spreadClass}`;
        }
        
        const tradeCountPill = document.getElementById("live-sim-trade-count-pill");
        if (tradeCountPill) {
            tradeCountPill.textContent = `${sim.trade_count || 0} trades`;
        }
        
        const tradesSubtitle = document.getElementById("live-sim-trades-subtitle");
        if (tradesSubtitle) {
            tradesSubtitle.textContent = sim.latest_equity_date
                ? `Latest update: ${sim.latest_equity_date}`
                : "No equity points yet";
        }
    
        const strategyReturnEl = document.getElementById("live-sim-strategy-return");
        strategyReturnEl.textContent = `Return: ${liveSimPercent(sim.strategy_return)}`;
        strategyReturnEl.className = `nt-live-sim-metric-sub ${liveSimReturnClass(sim.strategy_return)}`;
    
        const benchmarkReturnEl = document.getElementById("live-sim-benchmark-return");
        benchmarkReturnEl.textContent = `Return: ${liveSimPercent(sim.benchmark_return)} · entry cost included · open holding marked to market`;
        benchmarkReturnEl.className = `nt-live-sim-metric-sub ${liveSimReturnClass(sim.benchmark_return)}`;
    
        renderLiveSimulationChart(sim);
        renderLiveSimulationTrades(sim.trades || []);
    }
    
    function renderLiveSimulationChart(sim) {
        const canvas = document.getElementById("live-sim-chart");
        if (!canvas) return;
    
        const ctx = canvas.getContext("2d");
    
        if (liveSimChart) {
            liveSimChart.destroy();
        }
    
        const colors = typeof NT_COLORS !== "undefined"
            ? NT_COLORS
            : {
                chartAI: "#2563EB",
                chartBuyHold: "#9333EA"
            };
    
        liveSimChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: sim.dates || [],
                datasets: [
                    {
                        label: `${sim.ticker} Live Strategy`,
                        data: sim.strategy_curve || [],
                        borderColor: colors.chartAI,
                        backgroundColor: colors.chartAI,
                        pointRadius: 2,
                        borderWidth: 2,
                        fill: false
                    },
                    {
                        label: `${sim.ticker} Buy & Hold`,
                        data: sim.benchmark_curve || [],
                        borderColor: colors.chartBuyHold,
                        backgroundColor: colors.chartBuyHold,
                        borderDash: [5, 5],
                        pointRadius: 2,
                        borderWidth: 2,
                        fill: false
                    }
                ]
            },
            options: ntChartOptions({
                logScale: false,
                yTitle: "Equity ($)",
                yPrefix: "$",
                legendAlign: "end"
            })
        });
    }
    
    function renderLiveSimulationTrades(trades) {
        const table = document.getElementById("live-sim-trades-table");
        if (!table) return;
    
        if (!trades.length) {
            table.innerHTML = `
                <div class="nt-live-sim-small-note">
                    No BUY/SELL trades yet. HOLD days do not create trades.
                </div>
            `;
            return;
        }
    
        table.innerHTML = `
            <div class="nt-live-trade-row header">
                <div>Date</div>
                <div>Signal</div>
                <div>Price</div>
                <div>Quantity</div>
                <div>Cost</div>
                <div>Cash After</div>
            </div>
    
            ${trades.slice().reverse().map(trade => {
                const signalText = trade.signal === 1 ? "BUY" : "SELL";
                const signalClass = trade.signal === 1 ? "nt-live-trade-buy" : "nt-live-trade-sell";
    
                return `
                    <div class="nt-live-trade-row">
                        <div>${escapeHTML(trade.trade_date)}</div>
                        <div class="${signalClass}">${signalText}</div>
                        <div>${liveSimMoney(trade.price)}</div>
                        <div>${Number(trade.quantity).toLocaleString(undefined, {
                            maximumFractionDigits: 8
                        })}</div>
                        <div>${liveSimMoney(trade.transaction_cost)}</div>
                        <div>${liveSimMoney(trade.cash_after)}</div>
                    </div>
                `;
            }).join("")}
        `;
    }

    function normalizeLiveSimTickerForCompare(ticker) {
        return String(ticker || "").trim().toUpperCase();
    }
    
    function findDuplicateLiveSimulation(payload) {
        const ticker = normalizeLiveSimTickerForCompare(payload.ticker);
        const positionSize = Number(payload.position_size_pct);
        const initialCash = Number(payload.initial_cash);
    
        return liveSimulationsCache.find(sim => {
            const sameTicker =
                normalizeLiveSimTickerForCompare(sim.ticker) === ticker;
    
            const samePosition =
                Number(sim.position_size_pct) === positionSize;
    
            const sameCash =
                Number(sim.initial_cash) === initialCash;
    
            return sameTicker && samePosition && sameCash;
        });
    }
    
    function findSimilarLiveSimulation(payload) {
        const ticker = normalizeLiveSimTickerForCompare(payload.ticker);
        const positionSize = Number(payload.position_size_pct);
    
        return liveSimulationsCache.find(sim => {
            const sameTicker =
                normalizeLiveSimTickerForCompare(sim.ticker) === ticker;
    
            const samePosition =
                Number(sim.position_size_pct) === positionSize;
    
            return sameTicker && samePosition;
        });
    }

    function liveSimValueDifference(sim) {
        const strategy = Number(sim.latest_strategy_value);
        const benchmark = Number(sim.latest_benchmark_value);
    
        if (Number.isNaN(strategy) || Number.isNaN(benchmark)) {
            return null;
        }
    
        return strategy - benchmark;
    }
    
    function liveSimReturnSpread(sim) {
        const explicitSpread = liveSimNumberOrNull(sim.return_spread);

        if (explicitSpread !== null) {
            return explicitSpread;
        }

        const strategyReturn = liveSimNumberOrNull(sim.strategy_return);
        const benchmarkReturn = liveSimNumberOrNull(sim.benchmark_return);

        if (strategyReturn === null || benchmarkReturn === null) {
            return null;
        }

        return strategyReturn - benchmarkReturn;
    }

    function setLiveSimRefreshStatus(message) {
        const el = document.getElementById("live-sim-refresh-status");
        if (!el) return;
    
        el.textContent = message;
    }
    
    function liveSimRefreshTimestamp() {
        const now = new Date();
    
        return now.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit"
        });
    }

    function liveSimSignalInfo(signal) {
        const value = Number(signal);
    
        if (value === 1) {
            return {
                label: "BUY",
                className: "buy"
            };
        }
    
        if (value === -1) {
            return {
                label: "SELL",
                className: "sell"
            };
        }
    
        if (value === 0) {
            return {
                label: "HOLD",
                className: "hold"
            };
        }
    
        return {
            label: "—",
            className: "unknown"
        };
    }
    
    function liveSimSignalBadgeHTML(signal) {
        const info = liveSimSignalInfo(signal);
    
        return `
            <span class="nt-live-sim-signal-badge ${info.className}">
                ${info.label}
            </span>
        `;
    }
    
    function liveSimLatestSignalHTML(sim) {
        const closePrice = sim.latest_close_price
            ? liveSimMoney(sim.latest_close_price)
            : "—";
    
        const signalDate =
            sim.latest_equity_date ||
            sim.last_processed_date ||
            "—";
    
        return `
            <div class="nt-live-sim-signal-row">
                ${liveSimSignalBadgeHTML(sim.latest_signal)}
                <span class="nt-live-sim-signal-price">
                    Latest signal · ${escapeHTML(signalDate)} · Close ${closePrice}
                </span>
            </div>
        `;
    }

    function scrollToRecentlyCreatedSimulation() {
        if (!recentlyCreatedLiveSimulationId) return;
    
        const card = document.querySelector(
            `[data-live-sim-board-row-id="${recentlyCreatedLiveSimulationId}"]`
        );
    
        if (!card) return;
    
        card.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });
    
        setTimeout(() => {
            card.classList.remove("just-created");
            recentlyCreatedLiveSimulationId = null;
        }, 2200);
    }

    // function openNeuralTrendLoginModal() {
    //     const loginModal = document.getElementById("loginModal");
    //     const emailInput = document.getElementById("emailInput");
    
    //     if (!loginModal) return;
    
    //     loginModal.style.display = "flex";
    
    //     requestAnimationFrame(() => {
    //         loginModal.style.opacity = "1";
    //     });
    
    //     if (emailInput) {
    //         setTimeout(() => emailInput.focus(), 120);
    //     }
    // }
    
    function isLiveSimLoginError(error) {
        const message = String(error?.message || "").toLowerCase();
    
        return (
            message.includes("login required") ||
            message.includes("401") ||
            message.includes("unauthorized")
        );
    }
    
    function renderLiveSimulationLoggedOutState() {
        const listEl = document.getElementById("live-sim-list");
        const limitEl = document.getElementById("live-sim-limit");
        const dashboard = document.getElementById("live-sim-dashboard");
    
        liveSimulationsCache = [];
        selectedLiveSimulationId = null;
    
        if (limitEl) {
            limitEl.textContent = "Sign in to save and track live simulations.";
        }
    
        if (dashboard) {
            dashboard.style.display = "none";
        }
    
        if (!listEl) return;
    
        listEl.innerHTML = `
            <div class="nt-live-sim-auth-empty">
                <div class="nt-live-sim-auth-kicker">
                    Account required
                </div>
    
                <div class="nt-live-sim-auth-title">
                    Log in to create live simulations
                </div>
    
                <div class="nt-live-sim-auth-text">
                    Your live simulations are saved to your account, so each portfolio can keep its own cash balance, position size, trades, equity curve, and buy-and-hold benchmark.
                </div>
    
                <div class="nt-live-sim-auth-actions">
                    <button 
                        type="button" 
                        id="live-sim-login-action" 
                        class="nt-live-sim-auth-btn"
                    >
                        Log in / Sign up
                    </button>
    
                    <span class="nt-live-sim-auth-note">
                        Simulation only. No trades are placed.
                    </span>
                </div>
            </div>
        `;
    }

    function applyLiveSimAuthState(isLoggedIn, email = null, isPaid = false) {
        liveSimLoggedIn = Boolean(isLoggedIn);
        liveSimUserEmail = email || null;
        liveSimUserIsPaid = Boolean(isPaid);
    
        const submitBtn = document.getElementById("live-sim-submit-btn");
    
        if (submitBtn) {
            submitBtn.textContent = liveSimLoggedIn
                ? "Create Simulation"
                : "Log in to Create";
        }
    
        const messageEl = document.getElementById("live-sim-message");
    
        if (messageEl && !messageEl.dataset.userEdited) {
            messageEl.textContent = liveSimLoggedIn
                ? `Signed in as ${liveSimUserEmail}. Simulations are saved to your account.`
                : "Login is required to create and track live simulations.";
    
            messageEl.style.color = "var(--nt-neutral)";
        }
    }
    
    async function refreshLiveSimAuthState() {
        try {
            const response = await fetch("/me", {
                cache: "no-store"
            });
    
            const data = await response.json();
    
            const messageEl = document.getElementById("live-sim-message");
    
            if (data && data.email) {
                if (messageEl) {
                    delete messageEl.dataset.userEdited;
                }
    
                applyLiveSimAuthState(true, data.email, Boolean(data.is_paid));
    
                const currentTicker = document.getElementById("live-sim-ticker")?.value || "BTC-USD";
                renderLiveSimTickerOptions(currentTicker);
    
                return true;
            }
    
            if (messageEl) {
                delete messageEl.dataset.userEdited;
            }
    
            applyLiveSimAuthState(false, null, false);
            renderLiveSimTickerOptions("BTC-USD");
    
            return false;
    
        } catch (error) {
            applyLiveSimAuthState(false, null, false);
            renderLiveSimTickerOptions("BTC-USD");
            return false;
        }
    }

    function liveSimTickerIsCrypto(ticker) {
        return String(ticker || "").trim().toUpperCase().endsWith("-USD");
    }
    
    function updateLiveSimAssumptionPanel(ticker) {
        const costEl = document.getElementById("live-sim-assumption-cost");
        const quantityEl = document.getElementById("live-sim-assumption-quantity");
    
        const isCrypto = liveSimTickerIsCrypto(ticker);
    
        if (costEl) {
            costEl.textContent = isCrypto
                ? "Cost: 1.00% per trade"
                : "Cost: 0.10% per trade";
        }
    
        if (quantityEl) {
            quantityEl.textContent = isCrypto
                ? "Crypto allows fractional positions. BUY and SELL can use partial quantities."
                : "Stocks use whole-share quantities. BUY and SELL quantities are rounded while respecting available cash and current holdings.";
        }
    }

    let liveSimTickerOptionsCache = [];

    function normalizeTickerForCompare(ticker) {
        return String(ticker || "").trim().toUpperCase();
    }
    
    function getLiveSimTickerOptions() {
        /*
          Priority:
          1. Use loaded signal-board data if available.
          2. Fallback to the existing Backtest #ticker dropdown.
          3. Fallback to a small safe default list.
        */
    
        const seen = new Set();
        const options = [];
    
        if (Array.isArray(defaultOrder) && defaultOrder.length > 0) {
            defaultOrder.forEach(item => {
                const value = String(item?.ticker || "").trim();
                const key = normalizeTickerForCompare(value);
    
                if (!value || seen.has(key)) return;
    
                seen.add(key);
                options.push({
                    value,
                    label: value
                });
            });
        }
    
        if (!options.length) {
            const backtestTickerSelect = document.getElementById("ticker");
    
            if (backtestTickerSelect) {
                Array.from(backtestTickerSelect.options).forEach(option => {
                    const value = String(option.value || "").trim();
                    const label = String(option.textContent || value).trim();
                    const key = normalizeTickerForCompare(value);
    
                    if (!value || seen.has(key)) return;
    
                    seen.add(key);
                    options.push({
                        value,
                        label
                    });
                });
            }
        }
    
        if (!options.length) {
            ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "AAPL", "MSFT", "GOOGL", "NVDA"].forEach(value => {
                options.push({
                    value,
                    label: value
                });
            });
        }
    
        liveSimTickerOptionsCache = options;
        return options;
    }
    
    function liveSimTickerExists(ticker) {
        const key = normalizeTickerForCompare(ticker);
    
        return getVisibleLiveSimTickerOptions().some(option =>
            normalizeTickerForCompare(option.value) === key
        );
    }
    
    function renderLiveSimTickerOptions(preferredTicker = null) {
        const select = document.getElementById("live-sim-ticker");
        const helper = document.getElementById("live-sim-ticker-helper");
    
        if (!select) return;
    
        const allOptions = getVisibleLiveSimTickerOptions();
    
        const optionElements = allOptions.map(option => {
            const optionElement = document.createElement("option");
            optionElement.value = String(option.value || "");
            optionElement.textContent = String(option.label || option.value || "");
            return optionElement;
        });

        select.replaceChildren(...optionElements);
    
        const preferredKey = normalizeTickerForCompare(preferredTicker);
        const preferredMatch = allOptions.find(option =>
            normalizeTickerForCompare(option.value) === preferredKey
        );
    
        if (preferredMatch) {
            select.value = preferredMatch.value;
        } else if (allOptions.length) {
            select.value = allOptions[0].value;
        }
    
        if (helper) {
            helper.textContent = liveSimUserIsPaid
                ? `${allOptions.length} supported assets available.`
                : "Free plan: live simulations are available for BTC-USD, ETH-USD, SOL-USD, and XRP-USD.";
        }
    }
    
    function updateLiveSimTickerSelectionUI(ticker, updateName = true) {
        if (!ticker) return;
    
        const selectedTickerLabel = document.getElementById("live-sim-selected-ticker-label");
    
        if (selectedTickerLabel) {
            selectedTickerLabel.textContent = ticker;
        }
    
        updateLiveSimAssumptionPanel(ticker);
    
        if (updateName) {
            refreshAutoLiveSimulationName();
        }
    }
    
    function initializeLiveSimTickerSelector() {
        const select = document.getElementById("live-sim-ticker");
    
        if (!select) return;
    
        renderLiveSimTickerOptions("BTC-USD");
    
        const btcOption = Array.from(select.options).find(option =>
            normalizeTickerForCompare(option.value) === "BTC-USD"
        );
    
        if (btcOption) {
            select.value = btcOption.value;
        }
    
        select.addEventListener("change", function () {
            const selectedTicker = this.value;
    
            updateLiveSimTickerSelectionUI(selectedTicker, true);
            setLiveSimMessage(`Using ${selectedTicker} for the new simulation.`);
        });
    
        updateLiveSimTickerSelectionUI(select.value || "BTC-USD", true);
    }

    async function renameLiveSimulation(simId, currentName) {
        const newName = prompt("Rename simulation:", currentName || "");
    
        if (newName === null) {
            return;
        }
    
        const cleanName = newName.trim();
    
        if (!cleanName) {
            setLiveSimMessage("Simulation name cannot be empty.", true);
            return;
        }
    
        if (cleanName.length > 120) {
            setLiveSimMessage("Simulation name must be 120 characters or fewer.", true);
            return;
        }
    
        try {
            await liveSimFetchJSON(`/live-simulations/${simId}`, {
                method: "PATCH",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    name: cleanName
                })
            });
    
            setLiveSimMessage("Simulation renamed successfully.");
            selectedLiveSimulationId = simId;
    
            await loadLiveSimulations();
    
        } catch (error) {
            setLiveSimMessage(error.message, true);
        }
    }

    function liveSimStatusBadgeHTML(status) {
        const requestedStatus = String(status || "active").toLowerCase();
        const cleanStatus = ["active", "paused", "archived"].includes(requestedStatus)
            ? requestedStatus
            : "active";
    
        return `
            <span class="nt-live-sim-status-badge ${cleanStatus}">
                ${cleanStatus}
            </span>
        `;
    }
    
    async function updateLiveSimulationStatus(simId, newStatus) {
        try {
            await liveSimFetchJSON(`/live-simulations/${simId}/status`, {
                method: "PATCH",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    status: newStatus
                })
            });
    
            selectedLiveSimulationId = simId;
    
            if (newStatus === "archived") {
                selectedLiveSimulationId = null;
                setLiveSimMessage("Simulation archived. It is hidden from the main list.");
            } else if (newStatus === "paused") {
                setLiveSimMessage("Simulation paused. It will not process new CSV rows until resumed.");
            } else {
                setLiveSimMessage("Simulation restored/resumed.");
            }
    
            await loadLiveSimulations();
    
        } catch (error) {
            setLiveSimMessage(error.message, true);
        }
    }

    function liveSimFilterLabel(filter) {
        const labels = {
            open: "Open",
            active: "Active",
            paused: "Paused",
            archived: "Archived",
            all: "All"
        };
    
        return labels[filter] || "Open";
    }
    
    function updateLiveSimFilterUI() {
        document.querySelectorAll("[data-live-sim-filter]").forEach(btn => {
            const isSelected = btn.dataset.liveSimFilter === liveSimStatusFilter;
    
            btn.classList.toggle("selected", isSelected);
            btn.setAttribute("aria-pressed", isSelected ? "true" : "false");
        });
    }
    
    function liveSimEmptyStateMessage(filter) {
        if (filter === "active") {
            return "No active simulations. Resume a paused simulation or create a new one.";
        }
    
        if (filter === "paused") {
            return "No paused simulations.";
        }
    
        if (filter === "archived") {
            return "No archived simulations yet.";
        }
    
        if (filter === "all") {
            return "No simulations yet. Create your first one above.";
        }
    
        return "No open simulations yet. Create your first one above.";
    }
    
    function liveSimLimitText(data) {
        if (data.is_admin || data.limit === null || data.limit === undefined) {
            return `${data.used_count} active/paused simulations used. Admin account: unlimited live simulations.`;
        }
                
        if (liveSimStatusFilter === "active") {
            return `${data.active_count} active simulations. ${data.paused_count} paused · ${data.archived_count} archived.`;
        }
    
        if (liveSimStatusFilter === "paused") {
            return `${data.paused_count} paused simulations. ${data.active_count} active · ${data.archived_count} archived.`;
        }
    
        if (liveSimStatusFilter === "archived") {
            return `${data.archived_count} archived simulations. ${data.open_count} open.`;
        }
    
        if (liveSimStatusFilter === "all") {
            return `${data.all_count} total simulations. ${data.open_count} open · ${data.archived_count} archived.`;
        }
    
        return `${data.used_count} of ${data.limit} active/paused simulations used. ${data.archived_count} archived.`;
    }

    function updateLiveSimFilterCounts(data) {
        const counts = {
            open: data.open_count || data.used_count || 0,
            active: data.active_count || 0,
            paused: data.paused_count || 0,
            archived: data.archived_count || 0,
            all: data.all_count || 0
        };
    
        Object.entries(counts).forEach(([key, value]) => {
            const el = document.getElementById(`live-sim-count-${key}`);
    
            if (el) {
                el.textContent = String(value);
            }
        });
    }
    
    function resetLiveSimFilterCounts() {
        ["open", "active", "paused", "archived", "all"].forEach(key => {
            const el = document.getElementById(`live-sim-count-${key}`);
    
            if (el) {
                el.textContent = "0";
            }
        });
    }

    function getSelectedLiveSimulation() {
        if (!selectedLiveSimulationId) return null;
    
        return liveSimulationsCache.find(sim =>
            Number(sim.id) === Number(selectedLiveSimulationId)
        ) || null;
    }
    
    function renderLiveSimSelectedActions() {
        const container = document.getElementById("live-sim-selected-actions");
        if (!container) return;
    
        const sim = getSelectedLiveSimulation();
    
        if (!sim) {
            container.innerHTML = `
                <div class="nt-live-sim-selected-actions-empty">
                    Select a simulation to manage it.
                </div>
            `;
            return;
        }
    
        const status = String(sim.status || "active").toLowerCase();
    
        let buttonsHTML = "";
    
        if (status === "archived") {
            buttonsHTML = `
                <button 
                    type="button" 
                    class="nt-live-sim-selected-action-btn restore"
                    data-selected-live-sim-action="restore"
                >
                    Restore
                </button>
    
                <button 
                    type="button" 
                    class="nt-live-sim-selected-action-btn delete"
                    data-selected-live-sim-action="delete"
                >
                    Delete
                </button>
            `;
        } else {
            buttonsHTML = `
                <button 
                    type="button" 
                    class="nt-live-sim-selected-action-btn rename"
                    data-selected-live-sim-action="rename"
                >
                    Rename
                </button>
    
                <button 
                    type="button" 
                    class="nt-live-sim-selected-action-btn pause"
                    data-selected-live-sim-action="${status === "paused" ? "resume" : "pause"}"
                >
                    ${status === "paused" ? "Resume" : "Pause"}
                </button>
    
                <button 
                    type="button" 
                    class="nt-live-sim-selected-action-btn archive"
                    data-selected-live-sim-action="archive"
                >
                    Archive
                </button>
    
                <button 
                    type="button" 
                    class="nt-live-sim-selected-action-btn delete"
                    data-selected-live-sim-action="delete"
                >
                    Delete
                </button>
            `;
        }
    
        container.innerHTML = `
            <div class="nt-live-sim-selected-actions-row">
                <div class="nt-live-sim-selected-actions-info">
                    <div class="nt-live-sim-selected-actions-label">
                        Selected simulation
                    </div>
    
                    <div class="nt-live-sim-selected-actions-name">
                        ${escapeHTML(sim.name)} · ${escapeHTML(sim.ticker)}
                        ${liveSimStatusBadgeHTML(status)}
                    </div>
                </div>
    
                <div class="nt-live-sim-selected-actions-buttons">
                    ${buttonsHTML}
                </div>
            </div>
        `;
    }

    function liveSimNumberOrNull(value) {
        const numberValue = Number(value);
        return Number.isFinite(numberValue) ? numberValue : null;
    }
    
    function liveSimShortDate(value) {
        if (!value) return "—";
    
        const date = new Date(value);
    
        if (Number.isNaN(date.getTime())) {
            return String(value).slice(0, 10);
        }
    
        return date.toLocaleDateString(undefined, {
            year: "numeric",
            month: "short",
            day: "numeric"
        });
    }
    
    function liveSimBaseSymbol(sim) {
        return String(sim.ticker || "").split("-")[0] || "";
    }
    
    function liveSimLatestSignalValue(sim) {
        return normalizeLiveSimSignalLabel(
            sim.latest_signal ??
            sim.latest_signal_action ??
            sim.latest_epoch_signal ??
            sim.epoch_signal ??
            sim.signal ??
            "—"
        );
    }
    
    function liveSimStrategyValue(sim) {
        return liveSimNumberOrNull(
            sim.strategy_value ??
            sim.current_value ??
            sim.latest_strategy_value
        );
    }
    
    function liveSimBenchmarkValue(sim) {
        return liveSimNumberOrNull(
            sim.benchmark_value ??
            sim.buy_hold_value ??
            sim.latest_benchmark_value
        );
    }
    
    function liveSimReturnPctFromValue(currentValue, initialCash) {
        const value = liveSimNumberOrNull(currentValue);
        const initial = liveSimNumberOrNull(initialCash);
    
        if (value === null || initial === null || initial <= 0) {
            return null;
        }
    
        /* 
           Important:
           liveSimPercent() expects decimal return, not percentage points.
    
           Correct:
           0.05  -> 5.00%
           -0.01 -> -1.00%
    
           So do NOT multiply by 100 here.
        */
        return (value / initial) - 1;
    }
    
    function liveSimStrategyReturnPct(sim) {
        const calculatedReturn = liveSimReturnPctFromValue(
            liveSimStrategyValue(sim),
            sim.initial_cash
        );
    
        if (calculatedReturn !== null) {
            return calculatedReturn;
        }
    
        return liveSimNumberOrNull(
            sim.strategy_return_pct ??
            sim.strategy_return_percent ??
            sim.current_return_pct
        );
    }
    
    function liveSimBenchmarkReturnPct(sim) {
        const calculatedReturn = liveSimReturnPctFromValue(
            liveSimBenchmarkValue(sim),
            sim.initial_cash
        );
    
        if (calculatedReturn !== null) {
            return calculatedReturn;
        }
    
        return liveSimNumberOrNull(
            sim.benchmark_return_pct ??
            sim.benchmark_return_percent ??
            sim.buy_hold_return_pct
        );
    }
    
    function liveSimBoardToneClass(value) {
        const numberValue = liveSimNumberOrNull(value);
    
        if (numberValue === null || numberValue === 0) {
            return "neutral";
        }
    
        return numberValue > 0 ? "positive" : "negative";
    }
    
    function liveSimBoardSignalBadgeHTML(signal) {
        const cleanSignal = normalizeLiveSimSignalLabel(signal);
    
        let className = "hold";
    
        if (cleanSignal === "BUY") className = "buy";
        if (cleanSignal === "SELL") className = "sell";
        if (cleanSignal === "HOLD") className = "hold";
    
        return `
            <span class="nt-live-sim-board-signal ${className}">
                ${escapeHTML(cleanSignal)}
            </span>
        `;
    }
    
    function liveSimCashPositionHTML(sim) {
        const cash = liveSimNumberOrNull(sim.cash_balance);
        const position = liveSimNumberOrNull(sim.position_quantity);
        const baseSymbol = liveSimBaseSymbol(sim);
    
        return `
            <div class="nt-live-sim-board-main-value">
                ${cash === null ? "—" : liveSimMoney(cash)}
            </div>
    
            <div class="nt-live-sim-board-sub-value neutral">
                ${position === null ? "Position: —" : `${position.toLocaleString(undefined, {
                    maximumFractionDigits: 8
                })} ${escapeHTML(baseSymbol)}`}
            </div>
        `;
    }
    
    function liveSimBoardSortValue(sim, key) {
        if (key === "created_at") {
            return Date.parse(sim.created_at || sim.start_date || "") || 0;
        }
    
        if (key === "name") {
            return String(sim.name || "").toLowerCase();
        }
    
        if (key === "ticker") {
            return String(sim.ticker || "").toLowerCase();
        }
    
        if (key === "cash_position") {
            return liveSimNumberOrNull(sim.cash_balance) ?? -Infinity;
        }

        if (key === "strategy_value") {
            if (liveSimBoardStrategyDisplayMode === "percent") {
                return liveSimBoardStrategyReturnForHorizon(sim) ?? -Infinity;
            }
        
            return liveSimBoardStrategyDollarForHorizon(sim) ?? -Infinity;
        }
        
        if (key === "benchmark_value") {
            if (liveSimBoardBenchmarkDisplayMode === "percent") {
                return liveSimBoardBenchmarkReturnForHorizon(sim) ?? -Infinity;
            }
        
            return liveSimBoardBenchmarkDollarForHorizon(sim) ?? -Infinity;
        }
        
        if (key === "return_spread") {
            if (liveSimBoardSpreadDisplayMode === "percent") {
                return liveSimBoardReturnSpreadForHorizon(sim) ?? -Infinity;
            }

            return liveSimBoardValueDifferenceForHorizon(sim) ?? -Infinity;
        }
    
        return "";
    }
    
    function getFilteredAndSortedLiveSimulations(simulations) {
        let result = simulations.slice();

        if (liveSimBoardSearchQuery.trim()) {
            const query = liveSimBoardSearchQuery.trim().toLowerCase();
        
            result = result.filter(sim => {
                const name = String(sim.name || "").toLowerCase();
                const ticker = String(sim.ticker || "").toLowerCase();
                const status = String(sim.status || "").toLowerCase();
        
                return (
                    name.includes(query) ||
                    ticker.includes(query) ||
                    status.includes(query)
                );
            });
        }
    
        if (liveSimBoardAssetTypeFilter !== "all") {
            result = result.filter(sim =>
                liveSimAssetTypeValue(sim) === liveSimBoardAssetTypeFilter
            );
        }
    
        if (liveSimBoardSignalFilter !== "all") {
            result = result.filter(sim =>
                liveSimLatestSignalValue(sim) === liveSimBoardSignalFilter
            );
        }
    
        if (liveSimBoardPositionFilter !== "all") {
            result = result.filter(sim =>
                String(Number(sim.position_size_pct).toFixed(0)) === liveSimBoardPositionFilter
            );
        }
    
        result.sort((a, b) => {
            const valueA = liveSimBoardSortValue(a, liveSimBoardSortKey);
            const valueB = liveSimBoardSortValue(b, liveSimBoardSortKey);
    
            let comparison = 0;
    
            if (typeof valueA === "string" || typeof valueB === "string") {
                comparison = String(valueA).localeCompare(String(valueB));
            } else {
                comparison = valueA - valueB;
            }
    
            return liveSimBoardSortDirection === "asc" ? comparison : -comparison;
        });
    
        return result;
    }
    
    function liveSimBoardSortArrow(key) {
        if (liveSimBoardSortKey !== key) return "↕";
        return liveSimBoardSortDirection === "asc" ? "↑" : "↓";
    }

    function normalizeLiveSimSignalLabel(signal) {
        if (signal === null || signal === undefined || signal === "") {
            return "—";
        }
    
        const rawSignal = String(signal).trim().toUpperCase();
        const numericSignal = Number(rawSignal);
    
        if (Number.isFinite(numericSignal)) {
            if (numericSignal > 0) return "BUY";
            if (numericSignal < 0) return "SELL";
            return "HOLD";
        }
    
        if (rawSignal === "BUY") return "BUY";
        if (rawSignal === "SELL") return "SELL";
        if (rawSignal === "HOLD") return "HOLD";
    
        return rawSignal || "—";
    }

    function updateLiveSimBoardActiveRow() {
        document.querySelectorAll("[data-live-sim-board-row-id]").forEach(row => {
            row.classList.toggle(
                "active",
                Number(row.dataset.liveSimBoardRowId) === Number(selectedLiveSimulationId)
            );
        });
    }
    
    function liveSimSignedMoney(value) {
        const numberValue = liveSimNumberOrNull(value);
    
        if (numberValue === null) return "—";
    
        const sign = numberValue > 0 ? "+" : "";
    
        return `${sign}${liveSimMoney(numberValue)}`;
    }
    
    function liveSimSignedPercent(value) {
        const numberValue = liveSimNumberOrNull(value);
    
        if (numberValue === null) return "—";
    
        /*
           liveSimPercent() already adds + for positive values.
           So we should not add another + here.
        */
        return liveSimPercent(numberValue);
    }

    function formatPointSpreadText(value) {
        const numberValue = liveSimNumberOrNull(value);

        if (numberValue === null) return "—";

        const points = numberValue * 100;
        const sign = points > 0 ? "+" : "";
        return `${sign}${points.toFixed(2)} pts`;
    }
    

    function liveSimAssetTypeValue(sim) {
        const assetType = String(sim.asset_type || "").trim().toLowerCase();
    
        if (assetType) {
            return assetType;
        }
    
        const ticker = String(sim.ticker || "").trim().toUpperCase();
    
        if (ticker.endsWith("-USD")) {
            return "crypto";
        }
    
        return "stock";
    }
    
    function syncLiveSimAssetPills() {
        document.querySelectorAll("[data-live-sim-asset-filter]").forEach(button => {
            button.classList.toggle(
                "active",
                button.dataset.liveSimAssetFilter === liveSimBoardAssetTypeFilter
            );
        });
    }
    
    async function resetLiveSimulationBoardView() {
        liveSimStatusFilter = "open";

        liveSimBoardSearchQuery = "";
        liveSimBoardAssetTypeFilter = "all";
        liveSimBoardSignalFilter = "all";
        liveSimBoardPositionFilter = "all";
        liveSimBoardHorizon = "since_start";
    
        liveSimBoardStrategyDisplayMode = "percent";
        liveSimBoardBenchmarkDisplayMode = "percent";
        liveSimBoardSpreadDisplayMode = "percent";
    
        liveSimBoardSortKey = "created_at";
        liveSimBoardSortDirection = "desc";   

        const liveSimSearchInput = document.getElementById("live-sim-board-search");
        if (liveSimSearchInput) {
            liveSimSearchInput.value = "";
        }
    
        updateLiveSimFilterUI();
        syncLiveSimAssetPills();
        syncLiveSimHorizonPills();
    
        await loadLiveSimulations();
    }

    function syncLiveSimHorizonPills() {
        document.querySelectorAll("[data-live-sim-horizon]").forEach(button => {
            button.classList.toggle(
                "active",
                button.dataset.liveSimHorizon === liveSimBoardHorizon
            );
        });
    }
    
    function liveSimBoardHorizonMetrics(sim) {
        const allMetrics = sim.horizon_returns || {};
        return allMetrics[liveSimBoardHorizon] || allMetrics.since_start || null;
    }
    
    function liveSimBoardStrategyReturnForHorizon(sim) {
        const metrics = liveSimBoardHorizonMetrics(sim);
    
        if (metrics && metrics.strategy_return !== null && metrics.strategy_return !== undefined) {
            return Number(metrics.strategy_return);
        }
    
        return liveSimStrategyReturnPct(sim);
    }
    
    function liveSimBoardBenchmarkReturnForHorizon(sim) {
        const metrics = liveSimBoardHorizonMetrics(sim);
    
        if (metrics && metrics.benchmark_return !== null && metrics.benchmark_return !== undefined) {
            return Number(metrics.benchmark_return);
        }
    
        return liveSimBenchmarkReturnPct(sim);
    }
    
    function liveSimBoardReturnSpreadForHorizon(sim) {
        const metrics = liveSimBoardHorizonMetrics(sim);
    
        if (metrics && metrics.return_spread !== null && metrics.return_spread !== undefined) {
            return Number(metrics.return_spread);
        }
    
        const strategyReturn = liveSimBoardStrategyReturnForHorizon(sim);
        const benchmarkReturn = liveSimBoardBenchmarkReturnForHorizon(sim);
    
        if (strategyReturn === null || benchmarkReturn === null) {
            return null;
        }
    
        return strategyReturn - benchmarkReturn;
    }
    
    function liveSimBoardStrategyDollarForHorizon(sim) {
        const metrics = liveSimBoardHorizonMetrics(sim);
    
        if (metrics && metrics.strategy_change !== null && metrics.strategy_change !== undefined) {
            return Number(metrics.strategy_change);
        }
    
        const value = liveSimStrategyValue(sim);
        const initial = liveSimNumberOrNull(sim.initial_cash);
    
        if (value === null || initial === null) return null;
    
        return value - initial;
    }
    
    function liveSimBoardBenchmarkDollarForHorizon(sim) {
        const metrics = liveSimBoardHorizonMetrics(sim);
    
        if (metrics && metrics.benchmark_change !== null && metrics.benchmark_change !== undefined) {
            return Number(metrics.benchmark_change);
        }
    
        const value = liveSimBenchmarkValue(sim);
        const initial = liveSimNumberOrNull(sim.initial_cash);
    
        if (value === null || initial === null) return null;
    
        return value - initial;
    }
    
    function liveSimBoardValueDifferenceForHorizon(sim) {
        const metrics = liveSimBoardHorizonMetrics(sim);
    
        if (metrics && metrics.value_difference !== null && metrics.value_difference !== undefined) {
            return Number(metrics.value_difference);
        }
    
        const strategyChange = liveSimBoardStrategyDollarForHorizon(sim);
        const benchmarkChange = liveSimBoardBenchmarkDollarForHorizon(sim);
    
        if (strategyChange === null || benchmarkChange === null) return null;
    
        return strategyChange - benchmarkChange;
    }
    
    function liveSimBoardSpreadMetricHTML(
        valueDifference,
        returnSpread,
        displayMode = "percent"
    ) {
        const value = liveSimNumberOrNull(valueDifference);
        const spread = liveSimNumberOrNull(returnSpread);
        const toneClass = liveSimBoardToneClass(spread);

        if (displayMode === "value") {
            return `
                <div class="nt-live-sim-board-main-value ${toneClass}">
                    ${value === null ? "—" : liveSimSignedMoney(value)}
                </div>

                <div class="nt-live-sim-board-sub-value ${toneClass}">
                    ${spread === null ? "—" : `(${formatPointSpreadText(spread)})`}
                </div>
            `;
        }

        return `
            <div class="nt-live-sim-board-main-value ${toneClass}">
                ${spread === null ? "—" : formatPointSpreadText(spread)}
            </div>

            <div class="nt-live-sim-board-sub-value ${toneClass}">
                ${value === null ? "—" : `(${liveSimSignedMoney(value)})`}
            </div>
        `;
    }

    function liveSimBoardReturnMetricHTML(value, pct, displayMode = "percent") {
        const numberValue = liveSimNumberOrNull(value);
        const pctValue = liveSimNumberOrNull(pct);
        const toneClass = liveSimBoardToneClass(pctValue);
    
        if (displayMode === "value") {
            return `
                <div class="nt-live-sim-board-main-value ${toneClass}">
                    ${numberValue === null ? "—" : liveSimSignedMoney(numberValue)}
                </div>
    
                <div class="nt-live-sim-board-sub-value ${toneClass}">
                    ${pctValue === null ? "—" : `(${liveSimSignedPercent(pctValue)})`}
                </div>
            `;
        }
    
        return `
            <div class="nt-live-sim-board-main-value ${toneClass}">
                ${pctValue === null ? "—" : liveSimSignedPercent(pctValue)}
            </div>
    
            <div class="nt-live-sim-board-sub-value ${toneClass}">
                ${numberValue === null ? "—" : `(${liveSimSignedMoney(numberValue)})`}
            </div>
        `;
    }

    document.addEventListener("click", function (e) {
        const horizonButton = e.target.closest("[data-live-sim-horizon]");
    
        if (!horizonButton) return;
    
        e.preventDefault();
    
        liveSimBoardHorizon = horizonButton.dataset.liveSimHorizon || "since_start";
    
        syncLiveSimHorizonPills();
        renderLiveSimulationList(liveSimulationsCache);
    });
    
    document.getElementById("live-sim-form")?.addEventListener("submit", async function (e) {
        e.preventDefault();
    
        const isLoggedInNow = await refreshLiveSimAuthState();
    
        if (!isLoggedInNow) {
            renderLiveSimulationLoggedOutState();
            setLiveSimMessage("Please log in or sign up before creating a live simulation.", true);
            openNeuralTrendLoginModal();
            return;
        }
    
        const cashInput = document.getElementById("live-sim-cash");
        const parsedInitialCash = parseLiveSimulationCashInput(cashInput ? cashInput.value : "");

        if (parsedInitialCash === null) {
            setLiveSimMessage("Initial cash must be a positive number, with or without a $ sign.", true);
            cashInput?.focus();
            return;
        }

        const payload = {
            name: document.getElementById("live-sim-name").value.trim(),
            ticker: document.getElementById("live-sim-ticker").value.trim(),
            initial_cash: parsedInitialCash,
            position_size_pct: Number(document.getElementById("live-sim-position").value)
        };

        if (!payload.ticker || !liveSimTickerExists(payload.ticker)) {
            setLiveSimMessage("Please choose a supported asset from the ticker dropdown.", true);
            return;
        }
    
        const duplicate = findDuplicateLiveSimulation(payload);
    
        if (duplicate) {
            const ok = confirm(
                `You already have an identical simulation:\n\n` +
                `${duplicate.name}\n` +
                `${duplicate.ticker} · ${Number(duplicate.position_size_pct).toFixed(0)}% · ${liveSimMoney(duplicate.initial_cash)}\n\n` +
                `Create another one anyway?`
            );
    
            if (!ok) {
                setLiveSimMessage("Simulation creation cancelled. Existing simulation kept.");
                return;
            }
        } else {
            const similar = findSimilarLiveSimulation(payload);
    
            if (similar) {
                const ok = confirm(
                    `You already have a similar simulation:\n\n` +
                    `${similar.name}\n` +
                    `${similar.ticker} · ${Number(similar.position_size_pct).toFixed(0)}% position size\n\n` +
                    `The new one has a different initial cash or name. Create it anyway?`
                );
    
                if (!ok) {
                    setLiveSimMessage("Simulation creation cancelled. Existing simulation kept.");
                    return;
                }
            }
        }
    
        setLiveSimMessage("Creating simulation...");
    
        try {
            const data = await liveSimFetchJSON("/live-simulations", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            selectedLiveSimulationId = data.simulation.id;
            recentlyCreatedLiveSimulationId = data.simulation.id;
            
            setLiveSimMessage("Simulation created successfully. New card highlighted below.");
            
            await loadLiveSimulations();
    
                } catch (error) {
                    if (isLiveSimLoginError(error)) {
                        renderLiveSimulationLoggedOutState();
                        setLiveSimMessage("Please log in or sign up before creating a live simulation.", true);
                        openNeuralTrendLoginModal();
                        return;
                    }
        
                    if (error.data && error.data.upgrade_required) {
                        setLiveSimMessage(error.message, true);
                        return;
                    }
        
                    setLiveSimMessage(error.message, true);
                }
            });

    document.getElementById("live-sim-position")?.addEventListener("change", function () {
        refreshAutoLiveSimulationName();
    });

    document.getElementById("live-sim-cash")?.addEventListener("input", function () {
        refreshAutoLiveSimulationName();
    });

    document.getElementById("live-sim-cash")?.addEventListener("blur", function () {
        const parsedCash = parseLiveSimulationCashInput(this.value);
        if (parsedCash === null) return;

        // Keep the placeholder behavior when the user leaves the field blank.
        // Otherwise normalize obvious currency formatting to a clean number.
        if (String(this.value || "").trim()) {
            this.value = liveSimulationCashNamePart(parsedCash);
        }
        refreshAutoLiveSimulationName();
    });

    document.getElementById("live-sim-name")?.addEventListener("input", function () {
        this.dataset.autoName = "false";
    });

    document.getElementById("live-sim-refresh-btn")?.addEventListener("click", async function () {
        setLiveSimMessage("Refreshing simulations from latest available CSV data...");
        await loadLiveSimulations();
        setLiveSimMessage("Simulations refreshed.");
    });

    document.addEventListener("click", function (e) {
        if (e.target && e.target.id === "live-sim-login-action") {
            e.preventDefault();
            openNeuralTrendLoginModal();
        }
    });

    document.querySelectorAll("[data-live-sim-filter]").forEach(btn => {
        btn.addEventListener("click", async function () {
            liveSimStatusFilter = this.dataset.liveSimFilter || "open";
    
            selectedLiveSimulationId = null;
    
            updateLiveSimFilterUI();
    
            setLiveSimMessage(`Showing ${liveSimFilterLabel(liveSimStatusFilter).toLowerCase()} simulations.`);
    
            await loadLiveSimulations();
        });
    });

    document.addEventListener("click", async function (e) {
        const btn = e.target.closest("[data-selected-live-sim-action]");
        if (!btn) return;
    
        e.preventDefault();
    
        const sim = getSelectedLiveSimulation();
    
        if (!sim) {
            setLiveSimMessage("Please select a simulation first.", true);
            return;
        }
    
        const action = btn.dataset.selectedLiveSimAction;
    
        if (action === "rename") {
            await renameLiveSimulation(sim.id, sim.name);
            return;
        }
    
        if (action === "pause") {
            await updateLiveSimulationStatus(sim.id, "paused");
            return;
        }
    
        if (action === "resume") {
            await updateLiveSimulationStatus(sim.id, "active");
            return;
        }
    
        if (action === "archive") {
            const ok = confirm(
                "Archive this simulation? It will be hidden from the main list but its history will be kept."
            );
    
            if (!ok) return;
    
            await updateLiveSimulationStatus(sim.id, "archived");
            return;
        }
    
        if (action === "restore") {
            liveSimStatusFilter = "open";
            updateLiveSimFilterUI();
    
            await updateLiveSimulationStatus(sim.id, "active");
            return;
        }
    
        if (action === "delete") {
            const ok = confirm(
                "Delete this simulation permanently? This will remove its trades and equity history."
            );
    
            if (!ok) return;
    
            try {
                await liveSimFetchJSON(`/live-simulations/${sim.id}`, {
                    method: "DELETE"
                });
    
                selectedLiveSimulationId = null;
                setLiveSimMessage("Simulation deleted.");
    
                await loadLiveSimulations();
    
            } catch (error) {
                setLiveSimMessage(error.message, true);
            }
        }
    });

    document.addEventListener("click", async function (e) {
        const sortButton = e.target.closest("[data-live-sim-board-sort]");
    
        if (sortButton) {
            e.preventDefault();
    
            const sortKey = sortButton.dataset.liveSimBoardSort;
    
            if (liveSimBoardSortKey === sortKey) {
                liveSimBoardSortDirection =
                    liveSimBoardSortDirection === "asc" ? "desc" : "asc";
            } else {
                liveSimBoardSortKey = sortKey;
                liveSimBoardSortDirection = "desc";
            }
    
            renderLiveSimulationList(liveSimulationsCache);
            return;
        }
    
        const row = e.target.closest("[data-live-sim-board-row-id]");
    
        if (row) {
            e.preventDefault();
        
            const simId = Number(row.dataset.liveSimBoardRowId);
        
            if (Number.isFinite(simId)) {
                selectedLiveSimulationId = simId;
                updateLiveSimBoardActiveRow();
        
                await selectLiveSimulation(simId);
        
                updateLiveSimBoardActiveRow();
            }
        }
    });
    
    document.addEventListener("change", function (e) {
        if (e.target && e.target.id === "live-sim-board-asset-filter") {
            liveSimBoardAssetTypeFilter = e.target.value;
            renderLiveSimulationList(liveSimulationsCache);
            return;
        }
        
        if (e.target && e.target.id === "live-sim-board-strategy-mode") {
            liveSimBoardStrategyDisplayMode = e.target.value;
            renderLiveSimulationList(liveSimulationsCache);
            return;
        }
        
        if (e.target && e.target.id === "live-sim-board-benchmark-mode") {
            liveSimBoardBenchmarkDisplayMode = e.target.value;
            renderLiveSimulationList(liveSimulationsCache);
            return;
        }
        
        if (e.target && e.target.id === "live-sim-board-spread-mode") {
            liveSimBoardSpreadDisplayMode = e.target.value;
            renderLiveSimulationList(liveSimulationsCache);
            return;
        }
        
        if (e.target && e.target.id === "live-sim-board-signal-filter") {
            liveSimBoardSignalFilter = e.target.value;
            renderLiveSimulationList(liveSimulationsCache);
            return;
        }
    
        if (e.target && e.target.id === "live-sim-board-position-filter") {
            liveSimBoardPositionFilter = e.target.value;
            renderLiveSimulationList(liveSimulationsCache);
        }
    });

    document.addEventListener("click", async function (e) {
        const resetButton = e.target.closest("#reset-live-sim-board");
    
        if (resetButton) {
            e.preventDefault();
            await resetLiveSimulationBoardView();
            return;
        }
    
        const assetButton = e.target.closest("[data-live-sim-asset-filter]");
    
        if (assetButton) {
            e.preventDefault();
    
            liveSimBoardAssetTypeFilter =
                assetButton.dataset.liveSimAssetFilter || "all";
    
            syncLiveSimAssetPills();
            renderLiveSimulationList(liveSimulationsCache);
        }
    });

    document.addEventListener("input", function (e) {
        if (e.target && e.target.id === "live-sim-board-search") {
            liveSimBoardSearchQuery = e.target.value || "";
            renderLiveSimulationList(liveSimulationsCache);
        }
    });
        
    // Auto-load saved simulations when page loads.
    initializeLiveSimTickerSelector();
    updateLiveSimAssumptionPanel("BTC-USD");
    updateLiveSimFilterUI();
    refreshLiveSimAuthState();
    loadLiveSimulations();
    
    // Global delegated click handler
    document.addEventListener("click", function (e) {

        const btn = e.target.closest("#scale-toggle-btn");
    
        if (!btn) return;
        if (!previewChart) return;
    
        isLogScale = !isLogScale;
    
        btn.innerText = isLogScale
            ? "Normal Scale"
            : "Log Scale";
    
        const previewScaleModeEl = document.getElementById("preview-scale-mode");
        if (previewScaleModeEl) {
            previewScaleModeEl.textContent = isLogScale ? "Log" : "Linear";
        }
    
        previewChart.options.scales.y.type =
            isLogScale ? "logarithmic" : "linear";
    
        previewChart.update();
    });

    document.addEventListener("click", function (e) {

        const btn = e.target.closest("#backtest-scale-toggle-btn");
    
        if (!btn) return;
    
        const chart =
            window.equityChart ||
            (typeof equityChart !== "undefined" ? equityChart : null);
    
        if (!chart) return;
    
        window.backtestLogScale = !window.backtestLogScale;
    
        btn.innerText = window.backtestLogScale
            ? "Normal Scale"
            : "Log Scale";
    
        const scaleModeEl = document.getElementById("backtest-scale-mode");
        if (scaleModeEl) {
            scaleModeEl.textContent = window.backtestLogScale ? "Log" : "Linear";
        }
    
        chart.options.scales.y.type =
            window.backtestLogScale ? "logarithmic" : "linear";
    
        chart.update();
    });
    
    // Load default ticker
    setTimeout(() => {
        loadTicker("BTC-USD");
    }, 200);

// Declarative signal-board sorting replaces inline onclick handlers.
function activateSignalSortControl(event) {
    const control = event.target.closest("[data-signal-sort-key]");
    if (!control) return;

    if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") {
        return;
    }

    if (event.type === "keydown") {
        event.preventDefault();
    }

    sortBy(control.dataset.signalSortKey);
}

document.addEventListener("click", activateSignalSortControl);
document.addEventListener("keydown", activateSignalSortControl);
