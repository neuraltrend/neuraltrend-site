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

    const cards = Array.from(document.querySelectorAll('[data-home-path]'));
    const panels = Array.from(document.querySelectorAll('[data-home-panel]'));
    if (!cards.length || !panels.length) return;

    const setState = (activeName) => {
        cards.forEach((card) => {
            const name = card.getAttribute('data-home-path');
            const isActive = activeName && name === activeName;
            card.classList.toggle('is-active', Boolean(isActive));
            card.setAttribute('aria-expanded', isActive ? 'true' : 'false');
            const link = card.querySelector('.nt-home-use-card-link');
            if (link) link.textContent = isActive ? 'See less ←' : 'See more →';
        });

        panels.forEach((panel) => {
            const isActive = activeName && panel.getAttribute('data-home-panel') === activeName;
            panel.hidden = !isActive;
            panel.classList.toggle('is-active', Boolean(isActive));
        });
    };

    cards.forEach((card) => {
        card.addEventListener('click', () => {
            const name = card.getAttribute('data-home-path');
            const isOpen = card.classList.contains('is-active');
            setState(isOpen ? null : name);

            if (!isOpen) {
                const panel = document.querySelector(`[data-home-panel="${name}"]`);
                if (panel && window.matchMedia('(max-width: 900px)').matches) {
                    window.setTimeout(() => {
                        panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }, 50);
                }
            }
        });
    });
})();
