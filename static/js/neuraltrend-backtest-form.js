
    
    // Form Validation 
    document.addEventListener("DOMContentLoaded", function() {
        const startInput = document.getElementById("start");
    
        if (!startInput) return;
    
        const maxDateObj = new Date();
    
        const yyyy = maxDateObj.getFullYear();
        const mm = String(maxDateObj.getMonth() + 1).padStart(2, "0");
        const dd = String(maxDateObj.getDate()).padStart(2, "0");
    
        const maxDate = `${yyyy}-${mm}-${dd}`;
    
        if (startInput) {
            startInput.setAttribute("max", maxDate);
        }
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
