
    // const loginBtn = document.getElementById("login-btn");
    const loginModal = document.getElementById("loginModal"); // FIXED
    const closeModal = document.querySelector(".close-btn");
    
    const signupBtn = document.getElementById("signupBtn");
    const signinBtn = document.getElementById("signinBtn");
    
    const emailInput = document.getElementById("emailInput");
    const passwordInput = document.getElementById("passwordInput");
    
    const authMessage = document.getElementById("authMessage");
    const userArea = document.getElementById("user-area");

    function replaceElementChildren(element, ...children) {
        if (!element) return;
        element.replaceChildren(...children);
    }

    function renderLoggedOutUserArea() {
        if (!userArea) return;

        const loginLink = document.createElement("a");
        loginLink.href = "#";
        loginLink.id = "login-btn";
        loginLink.textContent = "Login";

        replaceElementChildren(userArea, loginLink);
    }

    function renderVerificationRequiredMessage(message) {
        if (!authMessage) return;

        const messageText = document.createTextNode(
            String(message || "Please verify your account before logging in.")
        );
        const lineBreak = document.createElement("br");
        const resendButton = document.createElement("button");

        resendButton.type = "button";
        resendButton.id = "inline-resend-verification";
        resendButton.style.marginTop = "10px";
        resendButton.textContent = "Resend verification email";

        replaceElementChildren(authMessage, messageText, lineBreak, resendButton);
    }

    function createUserMenuAction(label, handler, { danger = false } = {}) {
        const action = document.createElement("div");
        action.setAttribute("role", "button");
        action.tabIndex = 0;
        action.textContent = label;

        if (danger) {
            action.style.color = "red";
        }

        action.addEventListener("click", handler);
        action.addEventListener("keydown", event => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                handler();
            }
        });

        return action;
    }

    function renderAuthenticatedUserArea(email) {
        if (!userArea) return;

        const menu = document.createElement("div");
        menu.className = "user-menu";
        menu.id = "userMenu";

        const pill = document.createElement("div");
        pill.className = "user-pill";
        pill.append(document.createTextNode(String(email || "")));

        const arrow = document.createElement("span");
        arrow.className = "arrow";
        arrow.textContent = "▼";
        pill.append(arrow);

        const dropdown = document.createElement("div");
        dropdown.className = "dropdown";
        dropdown.id = "dropdownMenu";
        dropdown.style.display = "none";

        dropdown.append(
            createUserMenuAction("Manage Subscription", manageSubscriptionFromUserMenu),
            createUserMenuAction("Logout", logout),
            createUserMenuAction("Delete Account", deleteAccount, { danger: true })
        );

        menu.append(pill, dropdown);
        replaceElementChildren(userArea, menu);
    }

    function openNeuralTrendLoginModal(message = "Create a free account or log in to continue.") {
        const loginModal = document.getElementById("loginModal");
        const emailInput = document.getElementById("emailInput");
        const authMessage = document.getElementById("authMessage");
    
        if (!loginModal) return;
    
        if (authMessage) {
            authMessage.innerText = message;
        }
    
        loginModal.style.display = "flex";
    
        requestAnimationFrame(() => {
            loginModal.style.opacity = "1";
        });
    
        if (emailInput) {
            setTimeout(() => emailInput.focus(), 120);
        }
    }
    
    window.openNeuralTrendLoginModal = openNeuralTrendLoginModal;
    
    // ------------------
    // Close modal
    // ------------------
    closeModal.addEventListener("click", function() {
        loginModal.style.opacity = "0";
        setTimeout(() => {
            loginModal.style.display = "none";
        }, 300);
    });
    
    window.addEventListener("click", function(e) {
        if (e.target === loginModal) {
            loginModal.style.opacity = "0";
            setTimeout(() => {
                loginModal.style.display = "none";
            }, 300);
        }
    });

    signupBtn.addEventListener("click", async function() {
        ntTrack("signup_clicked");
        
        const response = await fetch("/signup", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                email: emailInput.value.trim(),
                password: passwordInput.value
            })
        });
    
        const data = await response.json();
    
        if (data.message) {
            ntTrack("signup_created_pending_verification");
            authMessage.innerText = "Check your email to verify your account. If you do not see it, please check your spam folder.";
    
            // Close modal after short delay (better UX)
            setTimeout(() => {
                loginModal.style.opacity = "0";
                setTimeout(() => {
                    loginModal.style.display = "none";
                }, 300);
            }, 3500);
    
        } else {
            ntTrack("signup_failed");
            authMessage.innerText = data.error;
        }
    });
    
    // ------------------
    // Login
    // ------------------
    signinBtn.addEventListener("click", async function() {
        ntTrack("login_clicked");
        
        const email = emailInput.value;
        const password = passwordInput.value;
    
        const response = await fetch("/login", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                email: emailInput.value.trim(),
                password: passwordInput.value
            })
        });
    
        const data = await response.json();
    
        if (data.message) {
            ntTrack("login_success");
            authMessage.innerText = data.message;
            loginModal.style.opacity = "0";
            setTimeout(() => {
                loginModal.style.display = "none";
            }, 300);
            await checkUser();

            if (typeof refreshSignalBoardForCurrentUser === "function") {
                await refreshSignalBoardForCurrentUser();
            }
            
            if (typeof refreshLiveSimAuthState === "function") {
                await refreshLiveSimAuthState();
            }
            
            if (typeof loadLiveSimulations === "function") {
                await loadLiveSimulations();
            }
        } else {
            ntTrack("login_failed");
            authMessage.innerText = data.error || "Login failed.";
        
            if (data.verification_required) {
                renderVerificationRequiredMessage(data.error);
            }
        }
    });

    // ------------------
    // Manage Subscription
    // ------------------
    function manageSubscriptionFromUserMenu() {
        if (typeof ntTrack === "function") {
            ntTrack("user_menu_subscription_clicked");
        }
    
        window.location.href = "/subscription";
    }
    
    // ------------------
    // Logout
    // ------------------
    async function logout() {
        await fetch("/logout", {
            method: "POST"
        });
    
        renderLoggedOutUserArea();
        location.reload();
    }

    // ------------------
    // Password Reset
    // ------------------

    async function resendVerificationEmail(email) {
        const cleanEmail = (email || "").trim();
    
        if (!cleanEmail) {
            authMessage.innerText = "Please enter your email first.";
            return;
        }
    
        const response = await fetch("/resend-verification", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                email: cleanEmail
            })
        });
    
        const data = await response.json();
    
        authMessage.innerText = data.message || data.error || "Please check your inbox and spam folder.";
    }
    
    document.addEventListener("click", async function(e) {
        if (e.target && e.target.id === "resend-verification") {
            e.preventDefault();
            await resendVerificationEmail(emailInput.value);
        }
    
        if (e.target && e.target.id === "inline-resend-verification") {
            e.preventDefault();
            await resendVerificationEmail(emailInput.value);
        }
    });
    document.addEventListener("click", async function(e) {
        if (e.target && e.target.id === "forgot-password") {
            e.preventDefault();
    
            const email = prompt("Enter your email:");
            if (!email) return;
    
            const res = await fetch("/request-password-reset", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ email })
            });
    
            const data = await res.json();
            alert(data.message);
        }
    });

    // ------------------
    // Delete Account
    // ------------------
    async function deleteAccount() {
        const confirmDelete = confirm(
            "This will send a secure confirmation email. Nothing is deleted by " +
            "this click. On the final confirmation page, any active Pro " +
            "subscription will be canceled immediately before the account is " +
            "permanently deleted. Continue?"
        );

        if (!confirmDelete) return;

        try {
            const response = await fetch("/request-delete-account", {
                method: "POST",
                cache: "no-store"
            });

            const data = await response.json();

            if (!response.ok) {
                alert(data.error || "Could not send the deletion confirmation email.");
                return;
            }

            alert(
                data.message ||
                "Check your email. Nothing has been deleted yet."
            );

        } catch (error) {
            console.error("Account deletion request failed:", error);
            alert("Could not send the deletion confirmation email. Please try again.");
        }
    }
    
    // ------------------
    // Check current user
    // ------------------
    async function checkUser() {
        const response = await fetch("/me");
        const data = await response.json();

        window.neuralTrendCurrentUser = data;

        document.dispatchEvent(new CustomEvent("neuralTrendUserUpdated", {
            detail: data
        }));
        
        if (typeof updateProSubscriptionUI === "function") {
            updateProSubscriptionUI(data);
        }
        
        if (typeof clearSuspiciousTickerSearchAutofill === "function") {
            setTimeout(clearSuspiciousTickerSearchAutofill, 50);
        }
    
        if (data.email) {
            renderAuthenticatedUserArea(data.email);
        } else {
            renderLoggedOutUserArea();
        }
    }
    
    document.addEventListener("click", function(e) {
    
        // ✅ Handle Login button (works even after re-render)
        if (e.target && e.target.id === "login-btn") {
            e.preventDefault();
            loginModal.style.display = "flex";
            loginModal.style.opacity = "1";
            return;
        }
    
        // ✅ Dropdown logic
        const menu = document.getElementById("userMenu");
        const dropdown = document.getElementById("dropdownMenu");
    
        if (!menu || !dropdown) return;
    
        const pill = menu.querySelector(".user-pill");
    
        if (pill && pill.contains(e.target)) {
            dropdown.style.display =
                dropdown.style.display === "block" ? "none" : "block";
        } else {
            dropdown.style.display = "none";
        }
    });
    
    // Run on page load
    checkUser();
