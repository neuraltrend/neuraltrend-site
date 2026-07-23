
    window.sa_event = window.sa_event || function() {
        var args = [].slice.call(arguments);
        window.sa_event.q ? window.sa_event.q.push(args) : window.sa_event.q = [args];
    };
    
    window.ntTrack = function(eventName, metadata = {}) {
        try {
            const safeEventName = String(eventName || "")
                .trim()
                .toLowerCase()
                .replace(/[^a-z0-9_]/g, "_");
    
            if (!safeEventName) return;
    
            if (typeof window.sa_event === "function") {
                window.sa_event(safeEventName, metadata);
            }
        } catch (error) {
            console.warn("Tracking error:", error);
        }
    };
    
    window.ntTrackAndGo = function(eventName, url) {
        let hasNavigated = false;
    
        function go() {
            if (hasNavigated) return;
            hasNavigated = true;
            window.location.href = url;
        }
    
        try {
            const safeEventName = String(eventName || "")
                .trim()
                .toLowerCase()
                .replace(/[^a-z0-9_]/g, "_");
    
            if (typeof window.sa_event === "function" && safeEventName) {
                window.sa_event(safeEventName, go);
                setTimeout(go, 350);
                return;
            }
        } catch (error) {
            console.warn("Tracking redirect error:", error);
        }
    
        go();
    };

// Declarative analytics hooks replace inline onclick attributes.
document.addEventListener("click", function(event) {
    const trigger = event.target.closest("[data-nt-track-event]");
    if (!trigger) return;

    const eventName = trigger.dataset.ntTrackEvent;
    const metadata = {};

    if (trigger.dataset.ntTrackPage) {
        metadata.page = trigger.dataset.ntTrackPage;
    }

    window.ntTrack(eventName, metadata);
});
