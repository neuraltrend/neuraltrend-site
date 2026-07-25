(() => {
    "use strict";

    const state = {
        items: new Map(),
        selectedTicker: "BTC-USD",
        user: window.neuralTrendCurrentUser || null,
        isPaid: false,
        limit: 0,
        usedCount: 0,
        loading: false
    };

    function normalizeTicker(value) {
        return String(value || "").trim().toUpperCase();
    }

    function currentItem() {
        return state.items.get(normalizeTicker(state.selectedTicker)) || null;
    }

    function isLoggedIn() {
        return Boolean(state.user && state.user.email);
    }

    function setMessage(message, isError = false) {
        const element = document.getElementById("selected-watchlist-message");
        if (!element) return;

        element.textContent = message || "";
        element.classList.toggle("is-error", Boolean(isError));
        element.classList.toggle("is-success", Boolean(message) && !isError);
    }

    async function fetchJSON(url, options = {}) {
        const response = await fetch(url, options);
        const text = await response.text();
        let data = {};

        try {
            data = text ? JSON.parse(text) : {};
        } catch (error) {
            throw new Error(`Unexpected server response (${response.status}).`);
        }

        if (!response.ok) {
            const requestError = new Error(data.error || `Request failed (${response.status}).`);
            requestError.status = response.status;
            requestError.data = data;
            throw requestError;
        }

        return data;
    }

    function updateCountBadge() {
        const countElement = document.getElementById("watchlist-pill-count");
        if (countElement) {
            countElement.textContent = String(state.items.size);
        }
    }

    function syncSignalBoardStars() {
        document.querySelectorAll("[data-watchlist-row-toggle]").forEach(button => {
            const ticker = normalizeTicker(button.dataset.ticker);
            const watching = state.items.has(ticker);
            const action = watching ? "Remove from watchlist" : "Add to watchlist";
            button.classList.toggle("is-watching", watching);
            button.classList.toggle("is-loading", state.loading);
            button.disabled = state.loading;
            button.setAttribute("aria-pressed", watching ? "true" : "false");
            button.setAttribute("aria-label", `${action} ${ticker}`);
            button.title = action;
        });
    }

    function updateSelectedControls() {
        const button = document.getElementById("selected-watchlist-toggle");
        const star = document.getElementById("selected-watchlist-star");
        const label = document.getElementById("selected-watchlist-label");
        const alertToggle = document.getElementById("selected-watchlist-alert-toggle");
        const alertNote = document.getElementById("selected-watchlist-alert-note");

        if (!button || !star || !label || !alertToggle || !alertNote) return;

        const item = currentItem();
        const watching = Boolean(item);

        button.disabled = state.loading;
        button.classList.toggle("is-watching", watching);
        button.setAttribute("aria-pressed", watching ? "true" : "false");

        if (!isLoggedIn()) {
            star.textContent = "☆";
            label.textContent = "Log in to save";
            alertToggle.checked = false;
            alertToggle.disabled = true;
            alertNote.textContent = "Log in to create a personal watchlist.";
            return;
        }

        star.textContent = watching ? "★" : "☆";
        label.textContent = watching ? "Watching" : "Add to Watchlist";

        if (!watching) {
            alertToggle.checked = false;
            alertToggle.disabled = true;
            alertNote.textContent = state.isPaid
                ? "Add this asset first, then enable future signal-change emails."
                : "Pro feature · daily published signals, not real-time execution alerts";
            return;
        }

        alertToggle.checked = Boolean(item.email_alert_enabled);

        if (item.retired_from_public_tracking) {
            alertToggle.checked = false;
            alertToggle.disabled = true;
            alertNote.textContent = "Tracking ended · alerts are disabled; the public record remains archived.";
            return;
        }

        if (item.locked) {
            alertToggle.disabled = true;
            alertNote.textContent = "This saved asset is currently locked for your plan.";
            return;
        }

        if (!state.isPaid) {
            alertToggle.disabled = true;
            alertNote.textContent = "Pro feature · watchlists remain available on the Free plan.";
            return;
        }

        alertToggle.disabled = state.loading || !item.can_enable_email_alert;
        alertNote.textContent = item.email_alert_enabled
            ? "Enabled · an email is sent only when the published daily signal changes."
            : "Daily published signal changes only · not real-time execution alerts.";
    }

    function refreshDependentUI() {
        updateCountBadge();
        updateSelectedControls();

        if (typeof window.applyAllFilters === "function") {
            window.applyAllFilters();
        } else if (typeof applyAllFilters === "function") {
            applyAllFilters();
        }

        // Filters can re-render the board, so synchronize stars afterward.
        syncSignalBoardStars();
    }

    function resetWatchlistState() {
        state.items.clear();
        state.isPaid = false;
        state.limit = 0;
        state.usedCount = 0;
        state.loading = false;

        const assetSelect = document.getElementById("asset-type-filter");
        if (assetSelect?.value === "watchlist") {
            assetSelect.value = "crypto";
            if (typeof syncAssetPills === "function") syncAssetPills();
            assetSelect.dispatchEvent(new Event("change", { bubbles: true }));
        }

        setMessage("");
        refreshDependentUI();
    }

    async function loadWatchlist() {
        if (!isLoggedIn()) {
            resetWatchlistState();
            return;
        }

        state.loading = true;
        updateSelectedControls();

        try {
            const data = await fetchJSON("/watchlist", { cache: "no-store" });
            state.items = new Map(
                (data.items || []).map(item => [normalizeTicker(item.ticker), item])
            );
            state.isPaid = Boolean(data.is_paid);
            state.limit = data.limit;
            state.usedCount = Number(data.used_count || 0);
            setMessage("");
        } catch (error) {
            if (error.status === 401) {
                resetWatchlistState();
                return;
            }

            console.error("Could not load watchlist:", error);
            setMessage("Could not load your watchlist.", true);
        } finally {
            state.loading = false;
            refreshDependentUI();
        }
    }

    async function toggleWatchlistTicker(tickerValue) {
        if (!isLoggedIn()) {
            if (typeof openNeuralTrendLoginModal === "function") {
                openNeuralTrendLoginModal("Log in to save assets to your watchlist.");
            }
            return;
        }

        const ticker = normalizeTicker(tickerValue);
        if (!ticker || state.loading) return;

        const existing = state.items.get(ticker);
        state.loading = true;
        refreshDependentUI();
        setMessage(existing ? `Removing ${ticker}…` : `Adding ${ticker}…`);

        try {
            if (existing) {
                await fetchJSON(`/watchlist/${encodeURIComponent(ticker)}`, {
                    method: "DELETE"
                });
                state.items.delete(ticker);
                setMessage(`${ticker} removed from your watchlist.`);
            } else {
                const data = await fetchJSON("/watchlist", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ ticker })
                });
                state.items.set(ticker, data.item);
                setMessage(`${ticker} added to your watchlist.`);
            }
        } catch (error) {
            console.error("Could not update watchlist:", error);
            setMessage(error.message, true);
        } finally {
            state.loading = false;
            refreshDependentUI();
        }
    }

    async function toggleSelectedWatchlist() {
        return toggleWatchlistTicker(state.selectedTicker);
    }

    async function updateSelectedAlert(enabled) {
        const ticker = normalizeTicker(state.selectedTicker);
        const item = state.items.get(ticker);
        const toggle = document.getElementById("selected-watchlist-alert-toggle");

        if (!item || state.loading) {
            if (toggle) toggle.checked = Boolean(item?.email_alert_enabled);
            return;
        }

        state.loading = true;
        updateSelectedControls();
        setMessage(enabled ? `Enabling alerts for ${ticker}…` : `Disabling alerts for ${ticker}…`);

        try {
            const data = await fetchJSON(`/watchlist/${encodeURIComponent(ticker)}/alerts`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ enabled })
            });
            state.items.set(ticker, data.item);
            setMessage(data.message || "Alert preference updated.");
        } catch (error) {
            console.error("Could not update signal alert:", error);
            setMessage(error.message, true);
            if (toggle) toggle.checked = Boolean(item.email_alert_enabled);
        } finally {
            state.loading = false;
            refreshDependentUI();
        }
    }

    window.neuralTrendWatchlistHasTicker = ticker => (
        state.items.has(normalizeTicker(ticker))
    );
    window.neuralTrendRefreshWatchlist = loadWatchlist;
    window.neuralTrendSyncWatchlistStars = syncSignalBoardStars;
    window.neuralTrendToggleWatchlistTicker = toggleWatchlistTicker;

    document.addEventListener("click", event => {
        const button = event.target.closest("[data-watchlist-row-toggle]");
        if (!button) return;

        event.preventDefault();
        event.stopPropagation();
        toggleWatchlistTicker(button.dataset.ticker);
    });

    document.getElementById("selected-watchlist-toggle")?.addEventListener(
        "click",
        toggleSelectedWatchlist
    );

    document.getElementById("selected-watchlist-alert-toggle")?.addEventListener(
        "change",
        event => updateSelectedAlert(Boolean(event.target.checked))
    );

    document.addEventListener("neuralTrendTickerSelected", event => {
        state.selectedTicker = normalizeTicker(event.detail?.ticker || "BTC-USD");
        setMessage("");
        updateSelectedControls();
    });

    document.addEventListener("neuralTrendUserUpdated", event => {
        state.user = event.detail || null;
        state.isPaid = Boolean(state.user?.is_paid);
        loadWatchlist();
    });

    if (window.neuralTrendCurrentUser) {
        state.user = window.neuralTrendCurrentUser;
        state.isPaid = Boolean(state.user?.is_paid);
        loadWatchlist();
    } else {
        updateSelectedControls();
    }
})();
