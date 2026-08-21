document.addEventListener("DOMContentLoaded", function() {
    const modelTabs = Array.from(document.querySelectorAll(".nt-model-tab"));
    const modelPanels = Array.from(document.querySelectorAll(".nt-model-panel"));
    const epochToolTabs = Array.from(document.querySelectorAll(".nt-epoch-tool-tab"));
    const epochToolPanels = Array.from(document.querySelectorAll(".nt-epoch-tool-panel"));

    function safeTrack(eventName, payload) {
        if (typeof ntTrack === "function") {
            ntTrack(eventName, payload);
        }
    }

    function resizeVisibleChart(panelId) {
        window.setTimeout(() => {
            try {
                if (
                    panelId === "epoch-tool-overview" &&
                    typeof previewChart !== "undefined" &&
                    previewChart
                ) {
                    previewChart.resize();
                }

                if (
                    panelId === "epoch-tool-backtest" &&
                    typeof equityChart !== "undefined" &&
                    equityChart
                ) {
                    equityChart.resize();
                }

                if (
                    panelId === "epoch-tool-live" &&
                    typeof liveSimChart !== "undefined" &&
                    liveSimChart
                ) {
                    liveSimChart.resize();
                }

                if (
                    panelId === "altindexer-section" &&
                    typeof Plotly !== "undefined"
                ) {
                    const altChart = document.getElementById("altindexer-chart");
                    if (altChart) {
                        Plotly.Plots.resize(altChart);
                    }
                }
            } catch (error) {
                console.warn("Dashboard chart resize skipped:", error);
            }
        }, 80);
    }

    function setActiveEpochToolPanel(panelId) {
        const targetPanel = document.getElementById(panelId);

        if (
            !targetPanel ||
            !targetPanel.classList.contains("nt-epoch-tool-panel")
        ) {
            return false;
        }

        epochToolTabs.forEach(tab => {
            const isActive = tab.dataset.epochToolTarget === panelId;
            tab.classList.toggle("active", isActive);
            tab.setAttribute("aria-selected", isActive ? "true" : "false");
        });

        epochToolPanels.forEach(panel => {
            panel.classList.toggle(
                "nt-epoch-tool-panel-active",
                panel.id === panelId
            );
        });

        const toolContentCard = document.querySelector(".nt-epoch-tool-content-card");
        if (toolContentCard) {
            toolContentCard.dataset.activeTool = panelId;
        }

        resizeVisibleChart(panelId);
        return true;
    }

    function setActiveModelPanel(panelId) {
        const targetPanel = document.getElementById(panelId);

        if (
            !targetPanel ||
            !targetPanel.classList.contains("nt-model-panel")
        ) {
            return false;
        }

        modelTabs.forEach(tab => {
            const isActive = tab.dataset.modelTarget === panelId;
            tab.classList.toggle("active", isActive);
            tab.setAttribute("aria-selected", isActive ? "true" : "false");
        });

        modelPanels.forEach(panel => {
            panel.classList.toggle(
                "nt-model-panel-active",
                panel.id === panelId
            );
        });

        const modelContent = document.querySelector(".nt-model-content");
        if (modelContent) {
            modelContent.dataset.activeModel = panelId;
        }

        resizeVisibleChart(panelId);
        return true;
    }

    /*
      Dashboard navigation belongs here rather than inside Backtest-specific
      code. This prevents future changes to the Backtest form from disabling
      EpochSignaler / AltIndexer / EpochForecaster or the three tool tabs.
    */
    modelTabs.forEach(tab => {
        tab.addEventListener("click", function() {
            const panelId = this.dataset.modelTarget;
            if (!panelId) return;

            setActiveModelPanel(panelId);

            safeTrack("model_tab_clicked", {
                model: panelId
            });
        });
    });

    epochToolTabs.forEach(tab => {
        tab.addEventListener("click", function() {
            const panelId = this.dataset.epochToolTarget;
            if (!panelId) return;

            setActiveModelPanel("epochsignaler-section");
            setActiveEpochToolPanel(panelId);

            safeTrack("epoch_tool_tab_clicked", {
                tool: panelId
            });
        });
    });

    function activateDashboardHash({scroll = true} = {}) {
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
        } else if (targetId === "signal-overview") {
            setActiveModelPanel("epochsignaler-section");
            setActiveEpochToolPanel("epoch-tool-overview");
        } else {
            setActiveModelPanel("epochsignaler-section");
            setActiveEpochToolPanel("epoch-tool-overview");
        }

        if (scroll && targetId) {
            window.setTimeout(() => {
                document.getElementById(targetId)?.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }, 100);
        }
    }

    window.ntSetActiveModelPanel = setActiveModelPanel;
    window.ntSetActiveEpochToolPanel = setActiveEpochToolPanel;

    activateDashboardHash({
        scroll: Boolean(window.location.hash)
    });

    window.addEventListener("hashchange", function() {
        activateDashboardHash({scroll: true});
    });
});
