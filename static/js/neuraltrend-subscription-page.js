
    async function startStripeCheckoutFromSubscriptionPage() {
        const button = document.getElementById("subscription-pro-action-btn");

        if (button?.disabled) {
            return;
        }

        const originalButtonText = button?.textContent || "Upgrade to Pro";

        ntTrack("upgrade_clicked", {
            source: "subscription_page"
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
                    source: "subscription_page"
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
                method: "POST"
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
                    status: response.status
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
                button.textContent = originalButtonText;
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
            } else {
                alert(data.error || "Could not open billing portal.");
            }

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
            pill.style.display = "inline-flex";
            pill.textContent = "Pro Active";

            button.textContent = "Manage Billing";
            button.onclick = startBillingPortalFromSubscriptionPage;
        } else {
            pill.style.display = "none";

            button.textContent = "Upgrade to Pro";
            button.onclick = startStripeCheckoutFromSubscriptionPage;
        }

        const isLoggedIn = Boolean(user.email);
        const isPaid = Boolean(user.is_paid);
        
        if (freePill && freeButton) {
            freeButton.disabled = false;
            freeButton.classList.remove("nt-pricing-btn-disabled");
        
            if (!isLoggedIn) {
                freePill.style.display = "none";
                freeButton.textContent = "Start Free";
                freeButton.onclick = startFreePlanFromSubscriptionPage;
            } else if (isPaid) {
                freePill.style.display = "none";
                freeButton.textContent = "Included with Pro";
                freeButton.disabled = true;
                freeButton.classList.add("nt-pricing-btn-disabled");
            } else {
                freePill.style.display = "inline-flex";
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
                openNeuralTrendLoginModal("Create a free NeuralTrend account to start using the free plan.");
                return;
            }
    
            // Already logged in, so no signup/login modal is needed.
            updateSubscriptionPageUI(user);
    
        } catch (error) {
            console.error("Free plan check error:", error);
            openNeuralTrendLoginModal("Create a free NeuralTrend account to start using the free plan.");
        }
    }

    document.addEventListener("neuralTrendUserUpdated", function(event) {
        updateSubscriptionPageUI(event.detail || {});
    });

    window.addEventListener("pageshow", refreshSubscriptionPageUI);

// Initial handlers are assigned from JavaScript rather than inline HTML.
const subscriptionFreeActionButton = document.getElementById("subscription-free-action-btn");
const subscriptionProActionButton = document.getElementById("subscription-pro-action-btn");

if (subscriptionFreeActionButton) {
    subscriptionFreeActionButton.onclick = startFreePlanFromSubscriptionPage;
}

if (subscriptionProActionButton) {
    subscriptionProActionButton.onclick = startStripeCheckoutFromSubscriptionPage;
}
