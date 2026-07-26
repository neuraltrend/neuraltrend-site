(function initializeNeuralTrendSubscriptionPage() {
    "use strict";

    const section = document.getElementById("subscription-plan");
    const monthlyButton = document.getElementById("subscription-billing-monthly");
    const annualButton = document.getElementById("subscription-billing-annual");
    const priceElement = document.getElementById("subscription-pro-price");
    const periodElement = document.getElementById("subscription-pro-price-period");
    const billingNote = document.getElementById("subscription-pro-billing-note");
    const proButton = document.getElementById("subscription-pro-action-btn");

    const monthlyAvailable = section?.dataset.monthlyCheckoutAvailable === "true";
    const annualAvailable = section?.dataset.annualCheckoutAvailable === "true";

    const billingDetails = {
        monthly: {
            price: "$9.99",
            period: "/ month",
            note: "Billed monthly. Cancel anytime through Stripe billing.",
            button: "Choose Monthly Pro"
        },
        annual: {
            price: "$99",
            period: "/ year",
            note: "$8.25 per month equivalent, billed as $99 once per year. Save $20.88 compared with monthly billing.",
            button: "Choose Annual Pro"
        }
    };

    function intervalIsAvailable(interval) {
        return interval === "annual" ? annualAvailable : monthlyAvailable;
    }

    let selectedBillingInterval = monthlyAvailable ? "monthly" : "annual";

    function setBillingInterval(interval) {
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

        if (proButton && proButton.dataset.action !== "manage-billing") {
            proButton.textContent = details.button;
        }
    }

    async function startStripeCheckoutFromSubscriptionPage() {
        const button = document.getElementById("subscription-pro-action-btn");

        if (button?.disabled) {
            return;
        }

        if (!intervalIsAvailable(selectedBillingInterval)) {
            alert("That billing option is temporarily unavailable.");
            return;
        }

        const originalButtonText = button?.textContent || billingDetails[selectedBillingInterval].button;

        ntTrack("upgrade_clicked", {
            source: "subscription_page",
            billing_interval: selectedBillingInterval
        });

        try {
            if (button) {
                button.disabled = true;
                button.classList.add("nt-pricing-btn-disabled");
                button.textContent = "Opening secure checkout…";
            }

            const meResponse = await fetch("/me", { cache: "no-store" });
            const me = await meResponse.json();

            if (!me.email) {
                ntTrack("upgrade_requires_login", {
                    source: "subscription_page",
                    billing_interval: selectedBillingInterval
                });

                if (typeof openNeuralTrendLoginModal === "function") {
                    openNeuralTrendLoginModal();
                } else {
                    alert("Please log in or sign up before upgrading.");
                }
                return;
            }

            if (me.is_paid) {
                ntTrack("billing_portal_clicked", {
                    source: "subscription_page"
                });

                await startBillingPortalFromSubscriptionPage();
                return;
            }

            const response = await fetch("/create-checkout-session", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    billing_interval: selectedBillingInterval
                })
            });

            const data = await response.json().catch(() => ({}));

            if (response.status === 409 && data.manage_billing) {
                alert(
                    data.error ||
                    "A subscription already exists for this account. Opening billing management."
                );
                await startBillingPortalFromSubscriptionPage();
                return;
            }

            if (!response.ok || !data.url) {
                ntTrack("checkout_start_failed", {
                    status: response.status,
                    billing_interval: selectedBillingInterval
                });
                alert(data.error || "Could not start checkout.");
                return;
            }

            ntTrackAndGo("checkout_started", data.url);

        } catch (error) {
            console.error("Checkout error:", error);
            alert("Could not start checkout. Please try again.");

        } finally {
            if (button) {
                button.disabled = false;
                button.classList.remove("nt-pricing-btn-disabled");

                if (button.dataset.action === "manage-billing") {
                    button.textContent = "Manage Billing";
                } else {
                    button.textContent = originalButtonText;
                    setBillingInterval(selectedBillingInterval);
                }
            }
        }
    }

    async function startBillingPortalFromSubscriptionPage() {
        try {
            const response = await fetch("/billing-portal", {
                method: "POST"
            });

            const data = await response.json();

            if (data.url) {
                ntTrackAndGo("billing_portal_started", data.url);
                return;
            }

            alert(data.error || "Could not open billing portal.");

        } catch (error) {
            console.error("Billing portal error:", error);
            alert("Could not open billing portal. Please try again.");
        }
    }

    function updateSubscriptionPageUI(user = {}) {
        const pill = document.getElementById("subscription-pro-status-pill");
        const button = document.getElementById("subscription-pro-action-btn");
        const freePill = document.getElementById("subscription-free-status-pill");
        const freeButton = document.getElementById("subscription-free-action-btn");

        if (!pill || !button) return;

        if (user.is_paid) {
            pill.hidden = false;
            pill.textContent = "Pro Active";

            button.dataset.action = "manage-billing";
            button.textContent = "Manage Billing";
            button.onclick = startBillingPortalFromSubscriptionPage;
        } else {
            pill.hidden = true;

            button.dataset.action = "checkout";
            button.onclick = startStripeCheckoutFromSubscriptionPage;
            setBillingInterval(selectedBillingInterval);
        }

        const isLoggedIn = Boolean(user.email);
        const isPaid = Boolean(user.is_paid);

        if (freePill && freeButton) {
            freeButton.disabled = false;
            freeButton.classList.remove("nt-pricing-btn-disabled");

            if (!isLoggedIn) {
                freePill.hidden = true;
                freeButton.textContent = "Start Free";
                freeButton.onclick = startFreePlanFromSubscriptionPage;
            } else if (isPaid) {
                freePill.hidden = true;
                freeButton.textContent = "Included with Pro";
                freeButton.disabled = true;
                freeButton.classList.add("nt-pricing-btn-disabled");
            } else {
                freePill.hidden = false;
                freePill.textContent = "Current Plan";

                freeButton.textContent = "Free Plan Active";
                freeButton.disabled = true;
                freeButton.classList.add("nt-pricing-btn-disabled");
            }
        }
    }

    async function refreshSubscriptionPageUI() {
        try {
            const response = await fetch("/me", { cache: "no-store" });
            const user = await response.json();
            updateSubscriptionPageUI(user);
        } catch (error) {
            console.error("Could not refresh subscription page UI:", error);
        }
    }

    async function startFreePlanFromSubscriptionPage() {
        try {
            const response = await fetch("/me", { cache: "no-store" });
            const user = await response.json();

            if (!user.email) {
                openNeuralTrendLoginModal(
                    "Create a free NeuralTrend account to start using the free plan."
                );
                return;
            }

            updateSubscriptionPageUI(user);

        } catch (error) {
            console.error("Free plan check error:", error);
            openNeuralTrendLoginModal(
                "Create a free NeuralTrend account to start using the free plan."
            );
        }
    }

    monthlyButton?.addEventListener("click", function() {
        setBillingInterval("monthly");
    });

    annualButton?.addEventListener("click", function() {
        setBillingInterval("annual");
    });

    document.addEventListener("neuralTrendUserUpdated", function(event) {
        updateSubscriptionPageUI(event.detail || {});
    });

    window.addEventListener("pageshow", refreshSubscriptionPageUI);

    const subscriptionFreeActionButton = document.getElementById(
        "subscription-free-action-btn"
    );

    if (subscriptionFreeActionButton) {
        subscriptionFreeActionButton.onclick = startFreePlanFromSubscriptionPage;
    }

    if (proButton) {
        proButton.dataset.action = "checkout";
        proButton.onclick = startStripeCheckoutFromSubscriptionPage;
    }

    setBillingInterval(selectedBillingInterval);
})();
