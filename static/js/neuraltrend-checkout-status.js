
    function showCheckoutStatusBanner() {
        const params = new URLSearchParams(window.location.search);
        const checkoutStatus = params.get("checkout");

        if (!checkoutStatus) return;
    
        const banner = document.getElementById("checkout-status-banner");
        const title = document.getElementById("checkout-status-title");
        const text = document.getElementById("checkout-status-text");
        const closeBtn = document.getElementById("checkout-status-close");
    
        if (!banner || !title || !text) return;
    
        banner.classList.remove("success", "cancelled");
    
        if (checkoutStatus === "success") {
            ntTrack("checkout_success_return");
            
            banner.classList.add("success");
            title.textContent = "Payment successful";
            text.textContent = "Your NeuralTrend Pro subscription is being activated. If the page still shows locked assets, wait a few seconds and refresh.";
            banner.style.display = "flex";
    
            if (typeof refreshProSubscriptionUI === "function") {
                refreshProSubscriptionUI();
            }
    
            if (typeof refreshSignalBoardForCurrentUser === "function") {
                setTimeout(refreshSignalBoardForCurrentUser, 1200);
            }
        }
    
        if (checkoutStatus === "cancelled") {
            ntTrack("checkout_cancelled_return");
            
            banner.classList.add("cancelled");
            title.textContent = "Checkout cancelled";
            text.textContent = "No payment was made. You can upgrade to NeuralTrend Pro anytime.";
            banner.style.display = "flex";
        }
    
        if (closeBtn) {
            closeBtn.onclick = function() {
                banner.style.display = "none";
    
                const cleanUrl = window.location.origin + window.location.pathname;
                window.history.replaceState({}, document.title, cleanUrl);
            };
        }
    }
    
    window.addEventListener("pageshow", showCheckoutStatusBanner);
