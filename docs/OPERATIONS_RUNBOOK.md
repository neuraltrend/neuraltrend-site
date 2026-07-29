# NeuralTrend Operations Runbook

## 1. Purpose

This is the primary routine operating checklist. Use the other manuals for
architecture, detailed deployment, or disaster recovery.

## 2. Every time you open the site for an operational check

1. Open the homepage and confirm it loads normally.
2. Sign in and confirm the dashboard loads.
3. Check that Signal Overview data is current and plausible.
4. Open one live simulation and confirm its status/data loads.
5. Open the username menu → **Admin Operations**.
6. Review Health & Operations for warnings.
7. Review Render for unexpected deploys, restarts, or health failures.
8. Avoid repeated refreshing if a heavy page is slow; use logs and request IDs.

## 3. After every deployment

Run in Render Shell:

```bash
python tools/prelaunch_check.py
python tools/operational_check.py --strict
python tools/recovery_check.py https://neuraltrend.org
python tools/production_smoke_test.py https://neuraltrend.org --include-summary
```

Then confirm:

- GitHub Actions is green;
- Render reports the service as Live;
- `/healthz` returns HTTP 200 quickly;
- the changed feature works manually;
- Admin Operations opens from the username menu;
- no new traceback, worker timeout, restart loop, 500, or 502 appears in logs;
- backup/recovery status still points to the intended persistent directories.

## 4. Weekly review

Run:

```bash
python tools/operational_check.py --strict
python tools/production_smoke_test.py https://neuraltrend.org --include-summary
```

Review:

- failed or stale Stripe webhooks;
- failed or stale signal-alert deliveries;
- persistent-disk free space;
- database connectivity and schema state;
- market-data freshness;
- most recent Forward Record publication;
- latest verified database and Forward Record backups;
- backup age and retention;
- Render deployment/restart/health events;
- slow-request logs, especially `/live-simulations`;
- external uptime-monitor results.

## 5. Monthly maintenance

- Download a fresh verified backup pair and checksums off Render.
- Verify the downloaded files locally or on a trusted host.
- Review backup-retention configuration and disk usage.
- Review administrator membership in `ADMIN_EMAILS`.
- Review Stripe webhook failures and endpoint configuration.
- Review email-delivery failures and unsubscribe behavior.
- Review dependency/security-update alerts in GitHub.
- Perform or schedule a staging restore rehearsal periodically.
- Check that this documentation still matches the deployed architecture.

## 6. Before risky changes

Create and download both backup types before:

- database migrations;
- bulk data correction;
- authentication or subscription changes;
- Stripe webhook changes;
- Forward Record storage/publication changes;
- backup/recovery implementation changes;
- persistent-disk path changes;
- major framework or dependency upgrades.

Use **Admin Operations → Backups**. Confirm checksums before proceeding.

## 7. Forward Record routine

1. Confirm the working market CSVs are current.
2. Open Alerts & Forward Record.
3. Run the private sandbox preview.
4. Review only genuinely new dates and assets.
5. Investigate any candidate-row validation error.
6. Approve/publish only after the preview is reasonable.
7. Confirm the compact approved disk record was appended, not rebuilt.
8. Review public Methodology/Performance output.

Already-approved dates in the working CSV may change without altering published
history. The approved compact CSV remains authoritative.

## 8. Logging and diagnostic information

Optional environment variables:

```text
NEURALTREND_LOG_FORMAT=json
NEURALTREND_LOG_LEVEL=INFO
NEURALTREND_SLOW_REQUEST_SECONDS=2.0
```

For a reported failure collect:

- approximate UTC and local time;
- URL and action;
- HTTP method and status from browser Network tools;
- response body or visible error;
- `X-Request-ID` header;
- first relevant Render traceback or warning;
- deployment commit shown by Render.

Logs redact common secret classes. Do not intentionally log raw passwords,
Stripe secrets, database URLs, authorization headers, reset tokens, request
bodies, or third-party payloads.

## 9. Incident: 502 or site unavailable

1. Stop repeatedly refreshing the site.
2. Open Render Events and identify deploy, restart, or health-check activity.
3. Open Render Logs and find the first error before the 502.
4. Test `/healthz`.
5. Run:

```bash
python tools/operational_check.py --strict
```

6. Look for:
   - `WORKER TIMEOUT`;
   - worker exits;
   - SIGKILL/OOM;
   - database or Redis connection failures;
   - startup exceptions;
   - repeated slow `/live-simulations` requests;
   - health-check restart loops.
7. If the current release caused the outage, roll back code to the latest
   known-good deployment.
8. Run recovery and smoke checks after rollback.
9. Do not rerun migrations until actual migration state is confirmed.

## 10. Incident: site healthy but a feature fails

A green `/healthz` is intentionally narrow.

1. Reproduce once in a private browser window.
2. Capture Console and Network errors.
3. Record URL, method, status, response, time, and request ID.
4. Search Render logs around that request.
5. Check the relevant subsystem:
   - Stripe logs/webhooks for billing;
   - email provider and alert ledger for email;
   - working CSV data for Signal Overview;
   - approved compact CSV and metadata for Forward Record;
   - database records for account/watchlist/simulation issues.
6. Run the narrowest relevant test locally or in GitHub Actions.
7. Apply a controlled fix; do not silently edit unrelated production state.

## 11. Incident: Forward Record preview error

1. Run the preview command in Render Shell to expose the exact asset/date error.
2. Distinguish between:
   - an invalid new candidate row; and
   - approved-record integrity failure.
3. Do not treat a changed already-approved row in the working CSV as a public
   history correction. Normal preview should ignore it.
4. For approved-record digest/metadata mismatch, stop publication and preserve
   files/logs before repair.
5. After correction, rerun sandbox preview and inspect public performance.

## 12. Incident: backup or recovery warning

1. Open **Admin Operations → Backups**.
2. Confirm both backup types exist and verify.
3. Open **Recovery** and check age/status.
4. Create a new pair if either component is missing, stale, or unverified.
5. Download both backups and both checksum files off Render.
6. Do not perform a first restore attempt against production.

## 13. Incident: failed GitHub Actions test

1. Read the first failing test and traceback, not only the final summary.
2. Reproduce the exact test locally when possible.
3. Determine whether the test exposes a real regression or an obsolete
   expectation.
4. Update code and tests only when the intended behavior is clear.
5. Run the complete suite before deployment.
6. Do not bypass or disable a failing safety test merely to make CI green.

## 14. Command reference

```bash
# Full test suite
python -m pytest

# CI-style coverage
python -m pytest \
  --cov=app \
  --cov=models \
  --cov=operational_logging \
  --cov=tools \
  --cov-report=term-missing

# Configuration and dependency readiness
python tools/prelaunch_check.py

# Operational state
python tools/operational_check.py --strict

# Important route recovery check
python tools/recovery_check.py https://neuraltrend.org

# Read-only production smoke test
python tools/production_smoke_test.py https://neuraltrend.org --include-summary

# Forward Record sandbox preview
python tools/publish_forward_record.py --preview
```

## 15. Escalation record

For a material incident, record:

- incident start/end times;
- affected users/features;
- detection source;
- request IDs and relevant logs;
- deployed commit;
- database migration state;
- backups selected or created;
- actions and commands executed;
- verification results;
- root cause;
- preventive follow-up.
