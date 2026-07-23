
    function goToSubscriptionPage() {
        window.location.href = "/subscription";
    }
        
    function updateProSubscriptionUI(user = {}) {
        const card = document.getElementById("pro-upgrade-card");
        const kicker = document.getElementById("pro-card-kicker");
        const title = document.getElementById("pro-card-title");
        const text = document.getElementById("pro-card-text");
        const pill = document.getElementById("pro-status-pill");
        const actionBtn = document.getElementById("pro-card-action-btn");

        const pricingProPill = document.getElementById("pricing-pro-status-pill");
        const pricingProBtn = document.getElementById("pricing-pro-action-btn");

        if (!card || !title || !text || !pill || !actionBtn) return;

        const isPaid = Boolean(user.is_paid);

        card.classList.toggle("nt-pro-active-card", isPaid);

        if (isPaid) {
            if (kicker) kicker.textContent = "NeuralTrend Pro";
            title.textContent = "Pro subscription active";
            text.textContent = "Your account has full access to all supported EpochSignaler assets, equity previews, backtests, and Pro live-simulation limits.";

            pill.style.display = "inline-flex";
            pill.textContent = "Pro Active";

            actionBtn.textContent = "Manage Billing";
            actionBtn.onclick = startBillingPortal;

            if (pricingProPill) {
                pricingProPill.style.display = "inline-flex";
                pricingProPill.textContent = "Pro Active";
            }
            
            if (pricingProBtn) {
                pricingProBtn.textContent = "Manage Billing";
                pricingProBtn.onclick = startBillingPortal;
            }
        } else {
            if (kicker) kicker.textContent = "NeuralTrend Pro";
            title.textContent = "Unlock all EpochSignaler assets";
            text.textContent = "Free users can access full signals for BTC-USD, ETH-USD, SOL-USD, and XRP-USD. Pro unlocks all supported assets, full signal history, equity previews, backtests, and up to 100 live simulations.";

            pill.style.display = "none";

            actionBtn.textContent = "View Subscription Plan";
            actionBtn.onclick = goToSubscriptionPage;

            if (pricingProPill) {
                pricingProPill.style.display = "none";
            }
            
            if (pricingProBtn) {
                pricingProBtn.textContent = "Upgrade to Pro";
                pricingProBtn.onclick = startProCheckout;
            }
        }
    }

    async function refreshProSubscriptionUI() {
        try {
            const response = await fetch("/me", { cache: "no-store" });
            const data = await response.json();
            updateProSubscriptionUI(data);
        } catch (error) {
            console.error("Could not refresh subscription UI:", error);
        }
    }

    function startProCheckout() {
        goToSubscriptionPage();
    }

    async function startBillingPortal() {
        try {
            const response = await fetch("/billing-portal", {
                method: "POST"
            });

            const data = await response.json();

            if (data.url) {
                window.location.href = data.url;
            } else {
                alert(data.error || "Could not open billing portal.");
            }

        } catch (error) {
            console.error("Billing portal error:", error);
            alert("Could not open billing portal. Please try again.");
        }
    }

    document.addEventListener("neuralTrendUserUpdated", function(event) {
        updateProSubscriptionUI(event.detail || {});
    });

    window.addEventListener("pageshow", function() {
        refreshProSubscriptionUI();
    });


// Preserve the initial Free-plan action without an inline HTML handler.
const proCardActionButton = document.getElementById("pro-card-action-btn");
if (proCardActionButton) {
    proCardActionButton.onclick = goToSubscriptionPage;
}

function activateSubscriptionNavigation(event) {
    const trigger = event.target.closest("[data-nt-go-subscription]");
    if (!trigger) return;

    if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") {
        return;
    }

    if (event.type === "keydown") {
        event.preventDefault();
    }

    goToSubscriptionPage();
}

document.addEventListener("click", activateSubscriptionNavigation);
document.addEventListener("keydown", activateSubscriptionNavigation);
