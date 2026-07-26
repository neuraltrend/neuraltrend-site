(() => {
    "use strict";

    const MOBILE_BREAKPOINT = 860;
    const body = document.body;
    const toggle = document.getElementById("nt-mobile-nav-toggle");
    const navigation = document.getElementById("nt-primary-navigation");
    const backdrop = document.getElementById("nt-mobile-nav-backdrop");
    let returnFocus = null;

    function isMobileNavigation() {
        return window.innerWidth <= MOBILE_BREAKPOINT;
    }

    function setNavigationState(open, { restoreFocus = false } = {}) {
        if (!toggle || !navigation || !backdrop) return;

        const shouldOpen = Boolean(open) && isMobileNavigation();
        body.classList.toggle("nt-mobile-nav-open", shouldOpen);
        toggle.setAttribute("aria-expanded", shouldOpen ? "true" : "false");
        toggle.setAttribute("aria-label", shouldOpen ? "Close navigation" : "Open navigation");

        if (isMobileNavigation()) {
            navigation.setAttribute("aria-hidden", shouldOpen ? "false" : "true");
            backdrop.tabIndex = shouldOpen ? 0 : -1;
        } else {
            navigation.removeAttribute("aria-hidden");
            backdrop.tabIndex = -1;
        }

        if (shouldOpen) {
            returnFocus = document.activeElement;
            window.requestAnimationFrame(() => {
                navigation.querySelector("a")?.focus();
            });
        } else if (restoreFocus && returnFocus instanceof HTMLElement) {
            returnFocus.focus();
            returnFocus = null;
        }
    }

    if (toggle && navigation && backdrop) {
        toggle.addEventListener("click", () => {
            setNavigationState(!body.classList.contains("nt-mobile-nav-open"));
        });

        backdrop.addEventListener("click", () => {
            setNavigationState(false, { restoreFocus: true });
        });

        navigation.addEventListener("click", event => {
            if (event.target.closest("a")) {
                setNavigationState(false);
            }
        });

        document.addEventListener("keydown", event => {
            if (event.key === "Escape" && body.classList.contains("nt-mobile-nav-open")) {
                setNavigationState(false, { restoreFocus: true });
            }
        });

        window.addEventListener("resize", () => {
            if (!isMobileNavigation()) {
                setNavigationState(false);
            } else if (!body.classList.contains("nt-mobile-nav-open")) {
                navigation.setAttribute("aria-hidden", "true");
            }
        }, { passive: true });

        setNavigationState(false);
    }

    const overflowSelectors = [
        ".nt-product-card-grid",
        ".nt-epoch-tool-tabs",
        ".nt-period-pills",
        ".nt-asset-pills",
        ".nt-live-sim-status-pills",
        ".nt-market-hero-assets",
        ".nt-market-hero-periods",
        ".nt-signal-table-card"
    ];

    function updateOverflowState(element) {
        const overflow = element.scrollWidth - element.clientWidth > 3;
        element.classList.toggle("nt-has-horizontal-overflow", overflow);
        element.classList.toggle("nt-at-scroll-start", element.scrollLeft <= 3);
        element.classList.toggle(
            "nt-at-scroll-end",
            element.scrollLeft + element.clientWidth >= element.scrollWidth - 3
        );
    }

    function initializeOverflowIndicators() {
        document.querySelectorAll(overflowSelectors.join(",")).forEach(element => {
            if (element.dataset.ntOverflowReady === "true") {
                updateOverflowState(element);
                return;
            }

            element.dataset.ntOverflowReady = "true";
            element.addEventListener("scroll", () => updateOverflowState(element), {
                passive: true
            });
            updateOverflowState(element);
        });
    }

    document.addEventListener("click", event => {
        const selectedControl = event.target.closest(
            ".nt-model-tab, .nt-epoch-tool-tab, .period-pill, .asset-pill"
        );
        if (!selectedControl || !isMobileNavigation()) return;

        window.setTimeout(() => {
            selectedControl.scrollIntoView({
                behavior: "smooth",
                block: "nearest",
                inline: "center"
            });
            initializeOverflowIndicators();
        }, 0);
    });

    initializeOverflowIndicators();
    window.addEventListener("resize", initializeOverflowIndicators, { passive: true });
    document.addEventListener("neuralTrendUserUpdated", initializeOverflowIndicators);

    if ("ResizeObserver" in window) {
        const observer = new ResizeObserver(initializeOverflowIndicators);
        observer.observe(document.documentElement);
    }
})();
