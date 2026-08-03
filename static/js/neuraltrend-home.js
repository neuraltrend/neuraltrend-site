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
        return;
    }

    const tabs = Array.from(document.querySelectorAll("[data-home-path]"));
    const panels = Array.from(document.querySelectorAll("[data-home-panel]"));

    if (!tabs.length || !panels.length) return;

    function setOpenPath(pathName, options = {}) {
        const { scrollOnMobile = false } = options;

        tabs.forEach((tab) => {
            const isOpen = Boolean(pathName) && tab.dataset.homePath === pathName;
            tab.classList.toggle("is-active", isOpen);
            tab.setAttribute("aria-expanded", isOpen ? "true" : "false");

            const label = tab.querySelector(".nt-home-path-more");
            if (label) label.textContent = isOpen ? "See less ↑" : "See more →";
        });

        panels.forEach((panel) => {
            const isOpen = Boolean(pathName) && panel.dataset.homePanel === pathName;
            panel.hidden = !isOpen;
        });

        if (pathName && scrollOnMobile && window.matchMedia("(max-width: 860px)").matches) {
            const activePanel = panels.find((panel) => panel.dataset.homePanel === pathName);
            if (activePanel) {
                window.setTimeout(() => {
                    activePanel.scrollIntoView({ behavior: "smooth", block: "start" });
                }, 40);
            }
        }
    }

    tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            const pathName = tab.dataset.homePath;
            const isAlreadyOpen = tab.getAttribute("aria-expanded") === "true";
            setOpenPath(isAlreadyOpen ? null : pathName, { scrollOnMobile: !isAlreadyOpen });
        });
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") setOpenPath(null);
    });
})();
