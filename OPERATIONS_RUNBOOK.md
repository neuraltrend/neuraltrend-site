# NeuralTrend Operations Runbook

## Purpose

Use this runbook to detect production problems, collect useful evidence, and
recover without exposing secrets or unnecessarily modifying customer data.

## Admin operations hub

For admins, open the username menu and select **Admin Operations**.

The hub links to:

- **Health & Operations** — component status and operational counts
- **Alerts & Forward Record** — alert dispatch and publication controls
- **Backups** — create, verify, download, retain, and delete backups
- **Recovery** — latest verified backup pair and safe recovery sequence

Direct route:

```text
https://neuraltrend.org/admin/operations
```

## Every time you open the website

1. Open `https://neuraltrend.org`.
2. Open `https://neuraltrend.org/dashboard`.
3. Confirm the main data loads without repeated refreshes.
4. If signed in as admin, open **Admin Operations**.
5. Investigate warnings before sending alerts or publishing Forward Record
   changes.

## After every deployment

```bash
python tools/prelaunch_check.py
python tools/operational_check.py --strict
python tools/recovery_check.py https://neuraltrend.org
python tools/production_smoke_test.py https://neuraltrend.org --include-summary
```

Also confirm:

- GitHub Actions is green.
- Render shows the deployment as Live.
- `/healthz` returns HTTP 200.
- the changed feature works manually;
- Admin Operations opens from the username menu;
- Render logs contain no new traceback, worker timeout, or repeated 500/502.

## Weekly review

Review in **Admin Operations** and, when needed, run:

```bash
python tools/operational_check.py --strict
python tools/production_smoke_test.py https://neuraltrend.org --include-summary
```

Check:

- failed or stale Stripe webhook events;
- failed or stale signal-alert deliveries;
- persistent-disk free space;
- market-data freshness;
- latest Forward Record publication;
- latest verified database and Forward Record backups;
- backup age and retention;
- Render restarts and failed health checks;
- Stripe webhook delivery history.

## Before risky changes

Use **Admin Operations → Backups** to create both backup types before:

- database migrations;
- bulk data corrections;
- major release changes;
- Forward Record storage or lifecycle changes.

Download both backups and both checksums off Render.

## Logging settings

Optional Render environment variables:

```text
NEURALTREND_LOG_FORMAT=json
NEURALTREND_LOG_LEVEL=INFO
NEURALTREND_SLOW_REQUEST_SECONDS=2.0
```

`text` is also supported for `NEURALTREND_LOG_FORMAT`.

Every response includes an `X-Request-ID`. For a reported failure, collect:

- approximate time;
- page/action;
- HTTP method and status from browser Network tools;
- `X-Request-ID` response header;
- first relevant Render traceback.

Logs redact common passwords, tokens, Stripe secrets, database passwords, bearer
tokens, and email addresses. Do not intentionally log request bodies or raw
third-party payloads.

## Incident: 502 or site unavailable

1. Open Render Events and identify deploy, restart, or health-check activity.
2. Open Render Logs and locate the first error before the 502.
3. Test `/healthz`.
4. Run `python tools/operational_check.py --strict` in Render Shell.
5. Look for `WORKER TIMEOUT`, worker exits, OOM/SIGKILL, connection resets, or
   repeated slow `/live-simulations` requests.
6. If the release caused the failure, roll back to the latest known-good deploy.
7. Run recovery and smoke checks after rollback.
8. Do not rerun migrations until migration state is verified.

## Incident: feature failure while site is healthy

A green `/healthz` proves only the health-check dependencies. It does not prove
that Stripe, email, signal calculations, every route, or every browser flow
works.

1. Reproduce once in a private browser window.
2. Capture Console and Network errors.
3. Record URL, method, status, response body, and `X-Request-ID`.
4. Search Render logs around that request ID and time.
5. Check the relevant external service:
   - Stripe logs/webhooks for billing;
   - email provider for mail;
   - market/working CSV data for signal and backtest problems.
6. Run the matching automated test locally or in GitHub Actions.

## Incident: backup or recovery warning

1. Open **Admin Operations → Backups** and confirm both backup types exist.
2. Open **Recovery** and check verification status and age.
3. Create a fresh backup pair if either file is missing, old, or unverified.
4. Download all four files off Render.
5. Do not attempt a production restore before staging verification.

## Command-line equivalents

```bash
python tools/operational_check.py
python tools/operational_check.py --json
python tools/operational_check.py --strict
```

`--strict` returns a non-zero status for warnings as well as errors.
