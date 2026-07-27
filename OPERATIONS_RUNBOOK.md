# NeuralTrend Operations Runbook

## Purpose

Use this runbook to detect production problems, collect useful evidence, and
recover without exposing secrets or modifying customer data unnecessarily.

## Daily quick check

1. Open `https://neuraltrend.org`.
2. Open `https://neuraltrend.org/dashboard`.
3. Open `https://neuraltrend.org/healthz` and confirm database/storage are `ok`.
4. When signed in as admin, open `/admin/operations`.
5. Investigate any error and review warnings before publishing or sending alerts.

## After every deployment

```bash
python tools/operational_check.py
python tools/recovery_check.py https://neuraltrend.org
```

Also confirm:

- GitHub Actions is green.
- Render shows the deployment as Live.
- `/healthz` returns HTTP 200.
- The feature changed in the deployment works manually.
- Render logs contain no new traceback or repeated HTTP 500 entries.

## Weekly review

Run:

```bash
python tools/operational_check.py
python tools/production_smoke_test.py https://neuraltrend.org --include-summary
```

Review:

- failed or stale Stripe webhook events;
- failed or stale signal-alert deliveries;
- persistent disk free space;
- BTC working-data freshness;
- latest Forward Record publication time;
- Render restarts and failed health checks;
- Stripe webhook delivery history.

## Log settings

Optional Render environment variables:

```text
NEURALTREND_LOG_FORMAT=json
NEURALTREND_LOG_LEVEL=INFO
NEURALTREND_SLOW_REQUEST_SECONDS=2.0
```

`text` is also supported for `NEURALTREND_LOG_FORMAT`.

Every response includes an `X-Request-ID`. When a user reports a failure, collect:

- approximate time;
- page/action;
- HTTP status from browser Network tools;
- `X-Request-ID` response header;
- first relevant Render traceback.

Logs redact common passwords, tokens, Stripe secrets, database passwords, bearer
tokens, and email addresses. Do not intentionally log request bodies or raw
third-party payloads.

## Incident: 502 or site unavailable

1. Open Render Events and identify deploy/restart/health-check activity.
2. Open Render Logs and find the first traceback before the 502.
3. Test `/healthz`.
4. Run `python tools/operational_check.py` in Render Shell.
5. If the new release caused the failure, roll back to the latest known-good
   deploy.
6. Run `python tools/recovery_check.py https://neuraltrend.org` after rollback.
7. Do not run database migrations again unless the migration state has been
   verified.

## Incident: feature failure while site is healthy

A green `/healthz` means only the database and Forward Record storage are
available. It does not prove that Stripe, email, signal calculations, or every
route works.

1. Reproduce once in a private browser window.
2. Capture browser Console and Network errors.
3. Record request URL, method, status, response body, and `X-Request-ID`.
4. Search Render logs around that request ID/time.
5. Check the matching external service when relevant:
   - Stripe logs/webhooks for billing;
   - email provider for mail;
   - working CSV for signal/backtest data.
6. Run the specific automated test locally or in GitHub Actions.

## Admin operational status

`/admin/operations` is read-only and admin-only. It shows counts and component
states but never customer emails, credentials, Stripe IDs, database URLs,
storage paths, tokens, or request bodies.

Command-line equivalent:

```bash
python tools/operational_check.py
python tools/operational_check.py --json
python tools/operational_check.py --strict
```

`--strict` returns a non-zero status for warnings as well as errors.
