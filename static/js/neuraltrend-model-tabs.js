
document.addEventListener("DOMContentLoaded", function() {

    // Product model tabs: EpochSignaler, AltIndexer, EpochForecaster
    document.querySelectorAll(".nt-model-tab").forEach(tab => {
        tab.addEventListener("click", function() {
            ntTrack("model_tab_clicked", {
                model: this.dataset.modelTarget || "unknown"
            });
        });
    });

    // EpochSignaler internal tabs: Overview, Backtest, Live Simulation
    document.querySelectorAll(".nt-epoch-tool-tab").forEach(tab => {
        tab.addEventListener("click", function() {
            ntTrack("epoch_tool_tab_clicked", {
                tool: this.dataset.epochToolTarget || "unknown"
            });
        });
    });

    // Signal board row click
    const signalBoard = document.getElementById("signal-board");
    if (signalBoard) {
        signalBoard.addEventListener("click", function(event) {
            const row = event.target.closest(".signal-row");
            if (!row) return;

            ntTrack("signal_row_clicked", {
                ticker: row.dataset.ticker || "unknown"
            });
        });
    }

    // Ticker search — track only when user leaves/searches, not every keypress
    const tickerSearch = document.getElementById("ticker-search");
    if (tickerSearch) {
        tickerSearch.addEventListener("change", function() {
            const query = this.value.trim();

            if (query) {
                ntTrack("ticker_search_used");
            }
        });
    }

    // Return horizon buttons
    document.querySelectorAll(".period-pill").forEach(button => {
        button.addEventListener("click", function() {
            ntTrack("return_horizon_changed", {
                horizon: this.dataset.value || this.dataset.liveSimHorizon || "unknown"
            });
        });
    });

    // Asset filter buttons
    document.querySelectorAll(".asset-pill").forEach(button => {
        button.addEventListener("click", function() {
            ntTrack("asset_filter_changed", {
                asset_filter: this.dataset.value || this.dataset.liveSimAssetFilter || "unknown"
            });
        });
    });

    // Backtest submit
    const backtestForm = document.getElementById("backtest-form");
    if (backtestForm) {
        backtestForm.addEventListener("submit", function() {
            ntTrack("backtest_submitted", {
                ticker: document.getElementById("ticker")?.value || "unknown",
                duration: document.getElementById("duration")?.value || "unknown"
            });
        });
    }

    // Live simulation submit
    const liveSimForm = document.getElementById("live-sim-form");
    if (liveSimForm) {
        liveSimForm.addEventListener("submit", function() {
            ntTrack("live_simulation_submitted", {
                ticker: document.getElementById("live-sim-ticker")?.value || "unknown"
            });
        });
    }

    // Contact form
    const contactForm = document.querySelector(".nt-contact-form");
    if (contactForm) {
        contactForm.addEventListener("submit", function() {
            ntTrack("contact_form_submitted");
        });
    }

    // Pro card on homepage
    const proCardButton = document.getElementById("pro-card-action-btn");
    if (proCardButton) {
        proCardButton.addEventListener("click", function() {
            ntTrack("homepage_pro_card_clicked");
        });
    }

});
