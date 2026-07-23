
    (function installNeuralTrendCsrfFetchProtection() {
        const originalFetch = window.fetch.bind(window);
        const safeMethods = new Set(["GET", "HEAD", "OPTIONS", "TRACE"]);

        window.ntGetCsrfToken = function() {
            return document
                .querySelector('meta[name="csrf-token"]')
                ?.getAttribute("content") || "";
        };

        window.fetch = function(input, init = {}) {
            const inputIsRequest = typeof Request !== "undefined" && input instanceof Request;
            const rawUrl = inputIsRequest ? input.url : String(input);
            const url = new URL(rawUrl, window.location.href);
            const method = String(
                init.method || (inputIsRequest ? input.method : "GET")
            ).toUpperCase();

            if (url.origin !== window.location.origin || safeMethods.has(method)) {
                return originalFetch(input, init);
            }

            const headers = new Headers(inputIsRequest ? input.headers : undefined);
            new Headers(init.headers || {}).forEach((value, name) => {
                headers.set(name, value);
            });

            const csrfToken = window.ntGetCsrfToken();

            if (csrfToken && !headers.has("X-CSRFToken")) {
                headers.set("X-CSRFToken", csrfToken);
            }

            return originalFetch(input, {
                ...init,
                headers,
                credentials: init.credentials || "same-origin"
            });
        };
    })();
