(function () {
    const dashboardHashes = new Set([
        "epochsignaler-section",
        "altindexer-section",
        "epochforecaster-section",
        "epoch-tool-overview",
        "epoch-tool-live",
        "epoch-tool-backtest",
        "signal-overview",
        "signal-board"
    ]);

    const target = String(window.location.hash || "").replace(/^#/, "");
    if (dashboardHashes.has(target)) {
        const dashboardTarget = target === "signal-board"
            ? "signal-overview"
            : target;
        window.location.replace(`/dashboard#${encodeURIComponent(dashboardTarget)}`);
    }
})();
