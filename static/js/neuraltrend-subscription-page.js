(function initializeNeuralTrendSubscriptionPage() {
    "use strict";

    const section = document.getElementById("subscription-plan");
    const monthlyButton = document.getElementById("subscription-billing-monthly");
    const annualButton = document.getElementById("subscription-billing-annual");
    const priceElement = document.getElementById("subscription-pro-price");
    const periodElement = document.getElementById("subscription-pro-price-period");
    const billingNote = document.getElementById("subscription-pro-billing-note");
    const proButton = document.getElementById("subscription-pro-action-btn");
    const freePill = document.getElementById("subscription-free-status-pill");
    const freeButton = document.getElementById("subscription-free-action-btn");
    const proPill = document.getElementById("subscription-pro-status-pill");
    const currentPlanPanel = document.getElementById("subscription-current-plan");
    const currentPlanName = document.getElementById("subscription-current-plan-name");
    const currentPlanDetail = document.getElementById("subscription-current-plan-detail");
    const manageBillingButton = document.getElementById("subscription-manage-billing-btn");
    const planChangeNote = document.getElementById("subscription-plan-change-note");

    const monthlyAvailable = section?.dataset.monthlyCheckoutAvailable === "true";
    const annualAvailable = section?.dataset.annualCheckoutAvailable === "true";

    const billingDetails = {
        monthly: {
            name: "Monthly Pro",
            price: "$9.99",
            period: "/ month",
            note: "Billed monthly. Cancel anytime through Stripe billing.",
            chooseButton: "Choose Monthly Pro",
            switchButton: "Switch to Monthly Pro"
        },
        annual: {
            name: "Annual Pro",
            price: "$99.99",
            period: "/ year",
            note: "$8.33 per month equivalent, billed as $99.99 once per year. Save $19.89 compared with monthly billing.",
            chooseButton: "Choose Annual Pro",
            switchButton: "Switch to Annual Pro"
        }
    };

    let selectedBillingInterval = monthlyAvailable ? "monthly" : "annual";
    let currentUser = {};
    let billingState = {};

    function track(eventName, payload = {}) {
        if (typeof window.ntTrack === "function") {
            window.ntTrack(eventName, payload);
        }
    }

    function goWithTracking(eventName, url, payload = {}) {
        if (typeof window.ntTrackAndGo === "function") {
            window.ntTrackAndGo(eventName, url, payload);
            return;
        }

        track(eventName, payload);
        window.location.assign(url);
    }

    function intervalIsAvailable(interval) {
        return interval === "annual" ? annualAvailable : monthlyAvailable;
    }

    function titleCaseInterval(interval) {
        return interval === "annual" ? "Annual" : "Monthly";
    }

    function formatPeriodEnd(timestamp) {
        const numericTimestamp = Number(timestamp);

        if (!Number.isFinite(numericTimestamp) || numericTimestamp <= 0) {
            return null;
        }

        return new Intl.DateTimeFormat("en-US", {
            year: "numeric",
            month: "short",
            day: "numeric"
        }).format(new Date(numericTimestamp * 1000));
    }

    function setButtonBusy(button, busy, busyText) {
        if (!button) return;

        if (busy) {
            button.dataset.previousText = button.textContent || "";
            button.disabled = true;
            button.classList.add("nt-pricing-btn-disabled");
            button.textContent = busyText;
            return;
        }

        button.disabled = false;
        button.classList.remove("nt-pricing-btn-disabled");

        if (button.dataset.previousText) {
            button.textContent = button.dataset.previousText;
            delete button.dataset.previousText;
        }
    }

    function setBillingInterval(interval, { userInitiated = false } = {}) {
        if (!billingDetails[interval] || !intervalIsAvailable(interval)) {
            return;
        }

        selectedBillingInterval = interval;
        const details = billingDetails[interval];

        monthlyButton?.classList.toggle("is-active", interval === "monthly");
        annualButton?.classList.toggle("is-active", interval === "annual");
        monthlyButton?.setAttribute(
            "aria-pressed",
            interval === "monthly" ? "true" : "false"
        );
        annualButton?.setAttribute(
            "aria-pressed",
            interval === "annual" ? "true" : "false"
        );

        if (priceElement) priceElement.textContent = details.price;
        if (periodElement) periodElement.textContent = details.period;
        if (billingNote) billingNote.textContent = details.note;

        if (userInitiated) {
            track("billing_interval_selected", {
                billing_interval: interval,
                current_interval: billingState.current_interval || null
            });
        }

        renderSubscriptionActions();
    }

    async function readJsonResponse(response) {
        const payload = await response.json().catch(() => ({}));

        if (!response.ok) {
            const error = new Error(payload.error || "The billing request failed.");
            error.payload = payload;
            error.status = response.status;
            throw error;
        }

        return payload;
    }

    async function startStripeCheckout() {
        if (!intervalIsAvailable(selectedBillingInterval)) {
            alert("That billing option is temporarily unavailable.");
            return;
        }

        setButtonBusy(proButton, true, "Opening secure checkout…");
        track("upgrade_clicked", {
            source: "subscription_page",
            billing_interval: selectedBillingInterval
        });

        try {
            const response = await fetch("/create-checkout-session", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    billing_interval: selectedBillingInterval
                })
            });
            const data = await readJsonResponse(response);

            if (!data.url) {
                throw new Error("Could not start checkout.");
            }

            goWithTracking("checkout_started", data.url, {
                billing_interval: selectedBillingInterval
            });

        } catch (error) {
            if (error.payload?.manage_billing) {
                alert(
                    error.message ||
                    "A subscription already exists. Opening billing management."
                );
                await openBillingPortal("manage");
                return;
            }

            console.error("Checkout error:", error);
            alert(error.message || "Could not start checkout. Please try again.");

        } finally {
            setButtonBusy(proButton, false);
            renderSubscriptionActions();
        }
    }

    async function openBillingPortal(action, billingInterval = null) {
        const activeButton = action === "manage" ? manageBillingButton : proButton;
        const busyText = action === "switch"
            ? "Opening plan confirmation…"
            : action === "cancel"
                ? "Opening cancellation…"
                : "Opening billing…";

        setButtonBusy(activeButton, true, busyText);

        try {
            const body = { action };

            if (billingInterval) {
                body.billing_interval = billingInterval;
            }

            const response = await fetch("/billing-portal", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body)
            });
            const data = await readJsonResponse(response);

            if (data.url) {
                goWithTracking("billing_portal_started", data.url, {
                    action,
                    billing_interval: billingInterval
                });
                return;
            }

            if (data.no_change || data.message) {
                await refreshSubscriptionPageUI();
                return;
            }

            throw new Error("Could not open billing management.");

        } catch (error) {
            console.error("Billing portal error:", error);
            alert(error.message || "Could not manage billing. Please try again.");

        } finally {
            setButtonBusy(activeButton, false);
            renderSubscriptionActions();
        }
    }

    async function resumeSubscription(options = {}) {
        const silent = Boolean(options.silent);
        const resumeButton = options.button || (
            billingState.cancel_at_period_end &&
            selectedBillingInterval === billingState.current_interval
                ? proButton
                : freeButton
        );

        setButtonBusy(resumeButton, true, "Keeping Pro…");

        try {
            const response = await fetch("/billing-portal", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: "resume" })
            });
            const data = await readJsonResponse(response);

            if (data.state) {
                billingState = data.state;
            }

            if (!silent) {
                await refreshSubscriptionPageUI();
                alert(data.message || "Your Pro subscription will continue.");
            }

            return true;

        } catch (error) {
            console.error("Resume subscription error:", error);
            alert(error.message || "Could not keep the subscription active.");
            return false;

        } finally {
            setButtonBusy(resumeButton, false);
            renderSubscriptionActions();
        }
    }

    async function switchPaidPlan() {
        const currentInterval = billingState.current_interval;

        if (!currentInterval || currentInterval === selectedBillingInterval) {
            if (billingState.cancel_at_period_end) {
                await resumeSubscription();
            }
            return;
        }

        const target = billingDetails[selectedBillingInterval];
        const periodEnd = formatPeriodEnd(billingState.period_end);
        let message = `Switch from ${titleCaseInterval(currentInterval)} Pro to ${target.name} now?\n\nStripe will show any prorated credit or charge before confirmation. The new billing period begins when the switch is confirmed.`;

        if (billingState.cancel_at_period_end) {
            message += `\n\nYour current subscription is scheduled to end${periodEnd ? ` on ${periodEnd}` : ""}. Confirming the new plan in Stripe replaces that scheduled cancellation.`;
        }

        if (!window.confirm(message)) {
            return;
        }

        if (billingState.cancel_at_period_end) {
            const resumed = await resumeSubscription({
                silent: true,
                button: proButton
            });

            if (!resumed) {
                return;
            }
        }

        await openBillingPortal("switch", selectedBillingInterval);
    }

    async function switchToFree() {
        if (billingState.cancel_at_period_end) {
            await resumeSubscription();
            return;
        }

        const periodEnd = formatPeriodEnd(billingState.period_end);
        const message = periodEnd
            ? `Switch to Free at the end of the current billing period on ${periodEnd}? You keep Pro access until then.`
            : "Switch to Free at the end of the current billing period?";

        if (!window.confirm(message)) {
            return;
        }

        await openBillingPortal("cancel");
    }

    function openLoginForFreePlan() {
        if (typeof window.openNeuralTrendLoginModal === "function") {
            window.openNeuralTrendLoginModal(
                "Create a free NeuralTrend account to start using the free plan."
            );
            return;
        }

        alert("Please log in or create an account to start using NeuralTrend.");
    }

    function renderCurrentPlanPanel() {
        const isPaid = Boolean(currentUser.is_paid && billingState.is_paid);

        if (!currentPlanPanel || !currentPlanName || !currentPlanDetail) {
            return;
        }

        currentPlanPanel.hidden = !isPaid;

        if (!isPaid) {
            return;
        }

        const currentInterval = billingState.current_interval || "monthly";
        const details = billingDetails[currentInterval] || billingDetails.monthly;
        const periodEnd = formatPeriodEnd(billingState.period_end);

        currentPlanName.textContent = details.name;

        if (billingState.cancel_at_period_end) {
            currentPlanDetail.textContent = periodEnd
                ? `Pro access ends ${periodEnd}, then the account moves to Free.`
                : "Cancellation is scheduled for the end of the current billing period.";
            currentPlanPanel.classList.add("is-cancelling");
        } else {
            currentPlanDetail.textContent = periodEnd
                ? `Next renewal: ${periodEnd}.`
                : "Subscription active.";
            currentPlanPanel.classList.remove("is-cancelling");
        }

        if (manageBillingButton) {
            manageBillingButton.hidden = false;
            manageBillingButton.onclick = () => openBillingPortal("manage");
        }
    }

    function renderSubscriptionActions() {
        if (!proButton || !freeButton || !proPill || !freePill) {
            return;
        }

        const isLoggedIn = Boolean(currentUser.email);
        const isPaid = Boolean(currentUser.is_paid && billingState.is_paid);
        const selectedDetails = billingDetails[selectedBillingInterval];
        const currentInterval = billingState.current_interval;
        const cancellationPending = Boolean(billingState.cancel_at_period_end);
        const isCurrentPaidChoice = isPaid && currentInterval === selectedBillingInterval;

        proButton.disabled = false;
        proButton.classList.remove("nt-pricing-btn-disabled");
        freeButton.disabled = false;
        freeButton.classList.remove("nt-pricing-btn-disabled");
        planChangeNote.hidden = true;

        if (!isLoggedIn) {
            proPill.hidden = true;
            proButton.textContent = selectedDetails.chooseButton;
            proButton.onclick = openLoginForFreePlan;

            freePill.hidden = true;
            freeButton.textContent = "Start Free";
            freeButton.onclick = openLoginForFreePlan;
            renderCurrentPlanPanel();
            return;
        }

        if (!isPaid) {
            proPill.hidden = true;
            proButton.textContent = selectedDetails.chooseButton;
            proButton.onclick = startStripeCheckout;

            freePill.hidden = false;
            freePill.textContent = "Current Plan";
            freeButton.textContent = "Free Plan Active";
            freeButton.disabled = true;
            freeButton.classList.add("nt-pricing-btn-disabled");
            renderCurrentPlanPanel();
            return;
        }

        proPill.hidden = false;
        proPill.textContent = cancellationPending ? "Pro — Ending" : "Pro Active";

        if (isCurrentPaidChoice) {
            if (cancellationPending) {
                proButton.textContent = `Keep ${selectedDetails.name}`;
                proButton.onclick = resumeSubscription;
            } else {
                proButton.textContent = `Current ${selectedDetails.name}`;
                proButton.disabled = true;
                proButton.classList.add("nt-pricing-btn-disabled");
                proButton.onclick = null;
            }
        } else {
            proButton.textContent = selectedDetails.switchButton;
            proButton.onclick = switchPaidPlan;
            planChangeNote.hidden = false;
        }

        if (cancellationPending) {
            const periodEnd = formatPeriodEnd(billingState.period_end);
            freePill.hidden = false;
            freePill.textContent = "Scheduled";
            freeButton.textContent = periodEnd
                ? `Keep Pro beyond ${periodEnd}`
                : "Keep Pro";
            freeButton.onclick = resumeSubscription;
        } else {
            freePill.hidden = true;
            freeButton.textContent = "Switch to Free at Renewal";
            freeButton.onclick = switchToFree;
        }

        renderCurrentPlanPanel();
    }

    async function refreshSubscriptionPageUI() {
        try {
            const meResponse = await fetch("/me", { cache: "no-store" });
            currentUser = await meResponse.json();

            if (currentUser.email && currentUser.is_paid) {
                const stateResponse = await fetch("/subscription-state", {
                    cache: "no-store"
                });
                billingState = await readJsonResponse(stateResponse);

                if (
                    billingState.current_interval &&
                    intervalIsAvailable(billingState.current_interval)
                ) {
                    setBillingInterval(billingState.current_interval);
                }
            } else {
                billingState = {
                    is_paid: false,
                    current_interval: null,
                    cancel_at_period_end: false,
                    period_end: null
                };
            }

            renderSubscriptionActions();

        } catch (error) {
            console.error("Could not refresh subscription page UI:", error);
            renderSubscriptionActions();
        }
    }

    monthlyButton?.addEventListener("click", function() {
        setBillingInterval("monthly", { userInitiated: true });
    });

    annualButton?.addEventListener("click", function() {
        setBillingInterval("annual", { userInitiated: true });
    });

    document.addEventListener("neuralTrendUserUpdated", function(event) {
        currentUser = event.detail || {};
        refreshSubscriptionPageUI();
    });

    window.addEventListener("pageshow", refreshSubscriptionPageUI);

    if (manageBillingButton) {
        manageBillingButton.onclick = () => openBillingPortal("manage");
    }

    setBillingInterval(selectedBillingInterval);
    refreshSubscriptionPageUI();
})();
