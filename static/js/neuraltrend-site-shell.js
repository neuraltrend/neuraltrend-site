
    // const loginBtn = document.getElementById("login-btn");
    const loginModal = document.getElementById("loginModal"); // FIXED
    const closeModal = document.querySelector(".close-btn");
    
    const signupBtn = document.getElementById("signupBtn");
    const signinBtn = document.getElementById("signinBtn");
    
    const emailInput = document.getElementById("emailInput");
    const passwordInput = document.getElementById("passwordInput");
    
    const authMessage = document.getElementById("authMessage");
    const userArea = document.getElementById("user-area");

    const changePasswordModal = document.getElementById("changePasswordModal");
    const changePasswordForm = document.getElementById("change-password-form");
    const changePasswordClose = document.getElementById("change-password-close");
    const changePasswordCancel = document.getElementById("change-password-cancel");
    const changePasswordSubmit = document.getElementById("change-password-submit");
    const changePasswordMessage = document.getElementById("change-password-message");
    const changeCurrentPassword = document.getElementById("change-current-password");
    const changeNewPassword = document.getElementById("change-new-password");
    const changeConfirmPassword = document.getElementById("change-confirm-password");

    let changePasswordReturnFocus = null;

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

    function closeUserDropdown({ restoreFocus = false } = {}) {
        const menu = document.getElementById("userMenu");
        const dropdown = document.getElementById("dropdownMenu");
        const trigger = menu?.querySelector(".user-pill");

        if (!dropdown || !trigger) return;

        dropdown.hidden = true;
        menu.classList.remove("is-open");
        trigger.setAttribute("aria-expanded", "false");

        if (restoreFocus) {
            trigger.focus();
        }
    }

    function toggleUserDropdown() {
        const menu = document.getElementById("userMenu");
        const dropdown = document.getElementById("dropdownMenu");
        const trigger = menu?.querySelector(".user-pill");

        if (!dropdown || !trigger) return;

        const willOpen = dropdown.hidden;
        dropdown.hidden = !willOpen;
        menu.classList.toggle("is-open", willOpen);
        trigger.setAttribute("aria-expanded", willOpen ? "true" : "false");

        if (willOpen) {
            dropdown.querySelector("button[role='menuitem']")?.focus();
        }
    }

    function createUserMenuAction(
        label,
        handler,
        { danger = false, icon = "", description = "" } = {}
    ) {
        const action = document.createElement("button");
        action.type = "button";
        action.className = "nt-account-menu-action";
        action.setAttribute("role", "menuitem");

        if (danger) {
            action.classList.add("nt-account-menu-action-danger");
        }

        const iconWrap = document.createElement("span");
        iconWrap.className = "nt-account-menu-icon";
        iconWrap.setAttribute("aria-hidden", "true");
        iconWrap.textContent = icon;

        const copy = document.createElement("span");
        copy.className = "nt-account-menu-action-copy";

        const labelEl = document.createElement("span");
        labelEl.className = "nt-account-menu-action-label";
        labelEl.textContent = label;
        copy.append(labelEl);

        if (description) {
            const descriptionEl = document.createElement("span");
            descriptionEl.className = "nt-account-menu-action-description";
            descriptionEl.textContent = description;
            copy.append(descriptionEl);
        }

        action.append(iconWrap, copy);
        action.addEventListener("click", () => {
            closeUserDropdown();
            handler(action);
        });

        return action;
    }

    function createUserMenuDivider() {
        const divider = document.createElement("div");
        divider.className = "nt-account-menu-divider";
        divider.setAttribute("role", "separator");
        return divider;
    }

    function createUserMenuSectionLabel(label) {
        const sectionLabel = document.createElement("div");
        sectionLabel.className = "nt-account-menu-section-label";
        sectionLabel.textContent = label;
        return sectionLabel;
    }

    function renderAuthenticatedUserArea(userData) {
        if (!userArea) return;

        const email = String(userData?.email || "");
        const isPaid = Boolean(userData?.is_paid);
        const isAdmin = Boolean(userData?.is_admin);
        const planLabel = isPaid ? "Pro" : "Free";

        const menu = document.createElement("div");
        menu.className = "user-menu nt-account-menu";
        menu.id = "userMenu";

        const pill = document.createElement("button");
        pill.type = "button";
        pill.className = "user-pill nt-account-trigger";
        pill.setAttribute("aria-haspopup", "menu");
        pill.setAttribute("aria-expanded", "false");
        pill.setAttribute("aria-controls", "dropdownMenu");

        const avatar = document.createElement("span");
        avatar.className = "nt-account-trigger-avatar";
        avatar.setAttribute("aria-hidden", "true");
        avatar.textContent = (email.charAt(0) || "U").toUpperCase();

        const triggerEmail = document.createElement("span");
        triggerEmail.className = "nt-account-trigger-email";
        triggerEmail.textContent = email;

        const arrow = document.createElement("span");
        arrow.className = "arrow";
        arrow.setAttribute("aria-hidden", "true");
        arrow.textContent = "▾";

        pill.append(avatar, triggerEmail, arrow);
        pill.addEventListener("click", event => {
            event.stopPropagation();
            toggleUserDropdown();
        });

        const dropdown = document.createElement("div");
        dropdown.className = "dropdown nt-account-dropdown";
        dropdown.id = "dropdownMenu";
        dropdown.setAttribute("role", "menu");
        dropdown.setAttribute("aria-label", "Account menu");
        dropdown.hidden = true;

        const header = document.createElement("div");
        header.className = "nt-account-menu-header";

        const headerCopy = document.createElement("div");
        headerCopy.className = "nt-account-menu-header-copy";

        const headerLabel = document.createElement("span");
        headerLabel.className = "nt-account-menu-header-label";
        headerLabel.textContent = "Signed in as";

        const headerEmail = document.createElement("span");
        headerEmail.className = "nt-account-menu-email";
        headerEmail.textContent = email;
        headerEmail.title = email;

        headerCopy.append(headerLabel, headerEmail);

        const badge = document.createElement("span");
        badge.className = `nt-account-plan-badge ${isPaid ? "is-pro" : "is-free"}`;
        badge.textContent = planLabel;

        header.append(headerCopy, badge);

        const subscriptionLabel = isPaid
            ? "Manage Subscription"
            : "View Pro Plan";
        const subscriptionDescription = isPaid
            ? "Billing and plan settings"
            : "See Pro features and pricing";

        dropdown.append(
            header,
            createUserMenuDivider(),
            createUserMenuSectionLabel("Workspace"),
            createUserMenuAction(
                "Open Dashboard",
                () => { window.location.href = "/dashboard"; },
                {
                    icon: "▦",
                    description: "Signals, watchlists, backtests, and simulations"
                }
            ),
            createUserMenuDivider(),
            createUserMenuSectionLabel("Account"),
            createUserMenuAction(
                "Change Password",
                openChangePasswordModal,
                {
                    icon: "🔒",
                    description: "Update password and revoke other sessions"
                }
            ),
            createUserMenuAction(
                subscriptionLabel,
                manageSubscriptionFromUserMenu,
                {
                    icon: isPaid ? "💳" : "◇",
                    description: subscriptionDescription
                }
            )
        );

        if (isAdmin) {
            dropdown.append(
                createUserMenuDivider(),
                createUserMenuSectionLabel("Administration"),
                createUserMenuAction(
                    "Alerts & Forward Record",
                    () => { window.location.href = "/admin/signal-alerts"; },
                    {
                        icon: "✉",
                        description: "Manual alerts, sandbox testing, and public launch"
                    }
                )
            );
        }

        dropdown.append(
            createUserMenuDivider(),
            createUserMenuAction(
                "Log Out",
                logout,
                { icon: "↪", description: "Sign out of this browser" }
            ),
            createUserMenuDivider(),
            createUserMenuAction(
                "Delete Account",
                deleteAccount,
                {
                    danger: true,
                    icon: "🗑",
                    description: "Permanently remove your account"
                }
            )
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

            if (window.location.pathname === "/") {
                window.location.assign("/dashboard");
                return;
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
    // Authenticated password change
    // ------------------
    function setChangePasswordMessage(message, type = "") {
        if (!changePasswordMessage) return;

        changePasswordMessage.textContent = String(message || "");
        changePasswordMessage.className = "nt-account-modal-message";

        if (type) {
            changePasswordMessage.classList.add(`is-${type}`);
        }
    }

    function openChangePasswordModal(triggerElement) {
        if (!changePasswordModal || !changePasswordForm) return;

        const accountTrigger = document.querySelector(
            "#userMenu .user-pill"
        );

        changePasswordReturnFocus =
            accountTrigger instanceof HTMLElement
                ? accountTrigger
                : (
                    triggerElement instanceof HTMLElement
                        ? triggerElement
                        : document.activeElement
                );

        changePasswordForm.reset();
        setChangePasswordMessage("");
        changePasswordModal.classList.add("is-open");
        changePasswordModal.setAttribute("aria-hidden", "false");
        document.body.classList.add("nt-account-modal-open");

        window.requestAnimationFrame(() => {
            changeCurrentPassword?.focus();
        });
    }

    function closeChangePasswordModal({ restoreFocus = true } = {}) {
        if (!changePasswordModal) return;

        changePasswordModal.classList.remove("is-open");
        changePasswordModal.setAttribute("aria-hidden", "true");
        document.body.classList.remove("nt-account-modal-open");

        if (restoreFocus && changePasswordReturnFocus instanceof HTMLElement) {
            changePasswordReturnFocus.focus();
        }

        changePasswordReturnFocus = null;
    }

    changePasswordClose?.addEventListener("click", () => {
        closeChangePasswordModal();
    });

    changePasswordCancel?.addEventListener("click", () => {
        closeChangePasswordModal();
    });

    changePasswordModal?.addEventListener("click", event => {
        if (event.target === changePasswordModal) {
            closeChangePasswordModal();
        }
    });

    changePasswordForm?.addEventListener("submit", async event => {
        event.preventDefault();

        const currentPassword = changeCurrentPassword?.value || "";
        const newPassword = changeNewPassword?.value || "";
        const confirmPassword = changeConfirmPassword?.value || "";

        if (!currentPassword) {
            setChangePasswordMessage("Enter your current password.", "error");
            changeCurrentPassword?.focus();
            return;
        }

        if (newPassword.length < 15 || newPassword.length > 64) {
            setChangePasswordMessage(
                "The new password must contain 15–64 characters.",
                "error"
            );
            changeNewPassword?.focus();
            return;
        }

        if (newPassword !== confirmPassword) {
            setChangePasswordMessage(
                "The two new password entries do not match.",
                "error"
            );
            changeConfirmPassword?.focus();
            return;
        }

        if (changePasswordSubmit) {
            changePasswordSubmit.disabled = true;
            changePasswordSubmit.textContent = "Changing…";
        }

        setChangePasswordMessage("Updating your password…");

        try {
            const response = await fetch("/change-password", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    current_password: currentPassword,
                    new_password: newPassword,
                    confirm_password: confirmPassword
                })
            });

            const data = await response.json();

            if (!response.ok) {
                setChangePasswordMessage(
                    data.error || "Could not change your password.",
                    "error"
                );
                return;
            }

            changePasswordForm.reset();
            setChangePasswordMessage(
                data.message || "Password changed successfully.",
                "success"
            );

            if (typeof ntTrack === "function") {
                ntTrack("account_password_changed");
            }

            window.setTimeout(() => {
                closeChangePasswordModal();
            }, 1600);

        } catch (error) {
            console.error("Password change failed:", error);
            setChangePasswordMessage(
                "Could not change your password. Please try again.",
                "error"
            );
        } finally {
            if (changePasswordSubmit) {
                changePasswordSubmit.disabled = false;
                changePasswordSubmit.textContent = "Change Password";
            }
        }
    });

    document.addEventListener("keydown", event => {
        if (event.key === "Escape") {
            if (changePasswordModal?.classList.contains("is-open")) {
                event.preventDefault();
                closeChangePasswordModal();
                return;
            }

            closeUserDropdown({ restoreFocus: true });
            return;
        }

        if (
            event.key === "Tab"
            && changePasswordModal?.classList.contains("is-open")
        ) {
            const focusable = Array.from(
                changePasswordModal.querySelectorAll(
                    'button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])'
                )
            );

            if (!focusable.length) return;

            const first = focusable[0];
            const last = focusable[focusable.length - 1];

            if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first.focus();
            }
        }
    });
    
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
            renderAuthenticatedUserArea(data);
        } else {
            renderLoggedOutUserArea();
        }
    }
    
    document.addEventListener("click", function(e) {
        // Handle Login button even after the account area is re-rendered.
        if (e.target && e.target.id === "login-btn") {
            e.preventDefault();
            loginModal.style.display = "flex";
            loginModal.style.opacity = "1";
            return;
        }

        const menu = document.getElementById("userMenu");
        if (menu && !menu.contains(e.target)) {
            closeUserDropdown();
        }
    });
    
    // Run on page load
    checkUser();
