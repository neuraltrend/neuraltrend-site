# NeuralTrend Launch Readiness — Pass 1

Date: 2026-09-10

This patch is a launch-hardening candidate. Deploy to staging first and run the full CI suite before production.

## Implemented in this pass

- Fixed signal-alert unsubscribe token path redaction in operational logs.
- Raised only the Stripe webhook hard request ceiling to 1 MiB while keeping ordinary requests at 64 KiB.
- Added site-wide browser security headers and a CSP Report-Only baseline.
- Added no-store/noindex handling for sensitive/auth/admin/token routes without matching public `/methodology`.
- Added environment-backed TOTP MFA enforcement for administrator routes, with 12-hour session verification.
- Replaced persistent account lockout with IP + HMAC-per-account login throttling to avoid lockout-as-DoS.
- Made signup responses non-enumerating for existing accounts.
- Made AdSense opt-in and scoped external analytics to anonymous public pages.
- Pinned Chart.js and load chart libraries only where used.
- Updated privacy disclosures for Stripe, Formspree, Simple Analytics, optional AdSense, watchlists/alerts/simulations, and current non-custodial/non-brokerage product scope.
- Softened marketing language that could read as a performance promise or personalized action recommendation.
- Renamed `Min. Days` to `50%+ Prob. Days` to match the backend calculation and explicitly state it is not a holding-period recommendation.
- Aligned GitHub Actions with the production Python 3.11 line and moved the runtime to Python 3.11.16.
- Expanded the prelaunch checker for secret strength, admin MFA, persistent backup location, and advertising state.

## Required staging checks before production

1. Install dependencies and run `python -m pytest`.
2. Run `python tools/prelaunch_check.py` against staging/production-like environment variables.
3. Test signup, verification, login, password reset, password change, logout, and account deletion.
4. Test admin login -> MFA -> every admin page/action.
5. Test monthly and annual Stripe Checkout, webhook processing, Pro access, portal, cancellation, and failed/duplicate webhook events.
6. Test Signal Board, BUY/HOLD/SELL filters, watchlists, alerts, backtests, live simulations, Forward Record, and mobile layouts.
7. Inspect browser console for CSP Report-Only violations; do not enforce CSP until legitimate dependencies are enumerated.
8. Keep `ADSENSE_ENABLED=false` until consent/privacy requirements are intentionally implemented.
9. Verify an off-service backup and a restore drill.
10. Obtain legal review of paid securities-signal positioning and confirm commercial market-data rights.

## Production MFA configuration

Generate a secret locally:

```bash
python tools/generate_admin_totp_secret.py --email admin@example.com
```

For one admin, store `ADMIN_TOTP_SECRET` in the deployment environment. For multiple admins, use a JSON object in `ADMIN_TOTP_SECRETS`. Never commit a real TOTP secret.
