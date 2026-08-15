
    
    // Backtest date-range validation
    document.addEventListener("DOMContentLoaded", function() {
        const form = document.getElementById("backtest-form");
        const tickerSelect = document.getElementById("ticker");
        const startInput = document.getElementById("start");
        const endInput = document.getElementById("end");

        if (!form || !tickerSelect || !startInput || !endInput) return;

        let coverageStart = null;
        let coverageEnd = null;
        let boundsRequestId = 0;

        function shiftIsoDate(isoDate, days) {
            if (!isoDate) return "";
            const date = new Date(`${isoDate}T00:00:00Z`);
            if (Number.isNaN(date.getTime())) return "";
            date.setUTCDate(date.getUTCDate() + days);
            return date.toISOString().slice(0, 10);
        }

        function updateDateConstraints(changedField = null) {
            if (!coverageStart || !coverageEnd) return;

            const absoluteStartMax = shiftIsoDate(coverageEnd, -1);
            const absoluteEndMin = shiftIsoDate(coverageStart, 1);

            startInput.min = coverageStart;
            startInput.max = endInput.value
                ? shiftIsoDate(endInput.value, -1)
                : absoluteStartMax;

            endInput.min = startInput.value
                ? shiftIsoDate(startInput.value, 1)
                : absoluteEndMin;
            endInput.max = coverageEnd;

            // If changing one boundary makes the other boundary invalid, clear
            // only the now-invalid value and let the user choose a valid one.
            if (
                changedField === "start" &&
                endInput.value &&
                endInput.value <= startInput.value
            ) {
                endInput.value = "";
                endInput.min = shiftIsoDate(startInput.value, 1);
            }

            if (
                changedField === "end" &&
                startInput.value &&
                startInput.value >= endInput.value
            ) {
                startInput.value = "";
                startInput.max = shiftIsoDate(endInput.value, -1);
            }

            startInput.setCustomValidity("");
            endInput.setCustomValidity("");
        }

        async function loadBacktestDateBounds() {
            const requestId = ++boundsRequestId;
            const ticker = tickerSelect.value;

            startInput.setCustomValidity("Loading available backtest dates…");
            endInput.setCustomValidity("Loading available backtest dates…");

            try {
                const response = await fetch(
                    `/backtest/date-range?ticker=${encodeURIComponent(ticker)}`,
                    { cache: "no-store" }
                );
                const payload = await response.json();

                if (requestId !== boundsRequestId) return;
                if (!response.ok) {
                    throw new Error(payload.error || "Backtest date range is unavailable.");
                }

                coverageStart = payload.coverage_start;
                coverageEnd = payload.coverage_end;

                // Keep already selected values only when they remain inside the
                // new asset's available market-data coverage.
                if (
                    startInput.value &&
                    (startInput.value < coverageStart ||
                     startInput.value > shiftIsoDate(coverageEnd, -1))
                ) {
                    startInput.value = "";
                }

                if (
                    endInput.value &&
                    (endInput.value < shiftIsoDate(coverageStart, 1) ||
                     endInput.value > coverageEnd)
                ) {
                    endInput.value = "";
                }

                if (
                    startInput.value &&
                    endInput.value &&
                    startInput.value >= endInput.value
                ) {
                    endInput.value = "";
                }

                updateDateConstraints();
            } catch (error) {
                console.error("Could not load backtest date range:", error);
                coverageStart = null;
                coverageEnd = null;
                startInput.removeAttribute("min");
                startInput.removeAttribute("max");
                endInput.removeAttribute("min");
                endInput.removeAttribute("max");
                startInput.setCustomValidity(
                    "Available backtest dates could not be loaded for this asset."
                );
                endInput.setCustomValidity(
                    "Available backtest dates could not be loaded for this asset."
                );
            }
        }

        startInput.addEventListener("change", function() {
            updateDateConstraints("start");
        });

        endInput.addEventListener("change", function() {
            updateDateConstraints("end");
        });

        tickerSelect.addEventListener("change", loadBacktestDateBounds);

        form.addEventListener("submit", function(event) {
            if (!coverageStart || !coverageEnd) {
                event.preventDefault();
                startInput.reportValidity();
                return;
            }

            updateDateConstraints();

            if (
                !startInput.value ||
                !endInput.value ||
                startInput.value >= endInput.value
            ) {
                event.preventDefault();
                if (startInput.value && endInput.value && startInput.value >= endInput.value) {
                    endInput.setCustomValidity("End date must be after the start date.");
                }
                form.reportValidity();
                return;
            }
        });

        loadBacktestDateBounds();
    });

    /* ===============================
       EpochSignaler internal tool tabs
       Signal Overview | Backtest | Live Simulation
    ================================ */
    
    function setActiveEpochToolPanel(panelId) {
        const targetPanel = document.getElementById(panelId);
        if (!targetPanel) return;
    
        document.querySelectorAll(".nt-epoch-tool-tab").forEach(tab => {
            const isActive = tab.dataset.epochToolTarget === panelId;
    
            tab.classList.toggle("active", isActive);
            tab.setAttribute("aria-selected", isActive ? "true" : "false");
        });
    
        document.querySelectorAll(".nt-epoch-tool-panel").forEach(panel => {
            panel.classList.toggle(
                "nt-epoch-tool-panel-active",
                panel.id === panelId
            );
        });

        const toolContentCard = document.querySelector(".nt-epoch-tool-content-card");
        if (toolContentCard) {
            toolContentCard.dataset.activeTool = panelId;
        }
    
        // Resize visible charts after switching tabs
        setTimeout(() => {
            if (panelId === "epoch-tool-overview") {
                if (typeof previewChart !== "undefined" && previewChart) {
                    previewChart.resize();
                }
            }
    
            if (panelId === "epoch-tool-backtest") {
                if (typeof equityChart !== "undefined" && equityChart) {
                    equityChart.resize();
                }
            }
    
            if (panelId === "epoch-tool-live") {
                if (typeof liveSimChart !== "undefined" && liveSimChart) {
                    liveSimChart.resize();
                }
            }
        }, 80);
    }
    
    document.querySelectorAll(".nt-epoch-tool-tab").forEach(tab => {
        tab.addEventListener("click", function () {
            const panelId = this.dataset.epochToolTarget;
            setActiveEpochToolPanel(panelId);
        });
    });
    
    // Default internal EpochSignaler tab
    setActiveEpochToolPanel("epoch-tool-overview");

    /* ===============================
       Model tab behavior
    ================================ */
    
    function setActiveModelPanel(panelId) {
        const targetPanel = document.getElementById(panelId);
        if (!targetPanel) return;
    
        document.querySelectorAll(".nt-model-tab").forEach(tab => {
            const isActive = tab.dataset.modelTarget === panelId;
            tab.classList.toggle("active", isActive);
            tab.setAttribute("aria-selected", isActive ? "true" : "false");
        });
    
        document.querySelectorAll(".nt-model-panel").forEach(panel => {
            panel.classList.toggle("nt-model-panel-active", panel.id === panelId);
        });

        const modelContent = document.querySelector(".nt-model-content");
        if (modelContent) {
            modelContent.dataset.activeModel = panelId;
        }
    }
    
    document.querySelectorAll(".nt-model-tab").forEach(tab => {
        tab.addEventListener("click", function () {
            const panelId = this.dataset.modelTarget;
            setActiveModelPanel(panelId);
        });
    });
    
    function activateDashboardHash() {
        const targetId = String(window.location.hash || "").replace(/^#/, "");
        const modelTargets = new Set([
            "epochsignaler-section",
            "altindexer-section",
            "epochforecaster-section"
        ]);
        const toolTargets = new Set([
            "epoch-tool-overview",
            "epoch-tool-live",
            "epoch-tool-backtest"
        ]);

        if (modelTargets.has(targetId)) {
            setActiveModelPanel(targetId);
        } else if (toolTargets.has(targetId)) {
            setActiveModelPanel("epochsignaler-section");
            setActiveEpochToolPanel(targetId);
        } else {
            setActiveModelPanel("epochsignaler-section");
        }

        if (targetId === "signal-overview") {
            setActiveModelPanel("epochsignaler-section");
            setActiveEpochToolPanel("epoch-tool-overview");
        }

        if (targetId) {
            window.setTimeout(() => {
                document.getElementById(targetId)?.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }, 100);
        }
    }

    activateDashboardHash();
    window.addEventListener("hashchange", activateDashboardHash);
