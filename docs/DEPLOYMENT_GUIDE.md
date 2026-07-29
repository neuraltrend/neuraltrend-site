# NeuralTrend Deployment Guide

## 1. Release model

NeuralTrend is deployed from the Git repository to Render. GitHub Actions
provides automated compilation and test feedback. Render starts the Flask
application through Gunicorn using the repository `Procfile`.

## 2. Local and CI tests

Install dependencies:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Run the suite:

```bash
python -m pytest
```

Run the GitHub Actions-style coverage command:

```bash
python -m pytest \
  --cov=app \
  --cov=models \
  --cov=operational_logging \
  --cov=tools \
  --cov-report=term-missing
```

Tests use isolated temporary storage and mocked external actions where
configured. Production CSVs, Stripe, and real email delivery must not be changed
by the test suite.

## 3. GitHub Actions gate

The workflow under `.github/workflows/tests.yml` should compile and run pytest
on every push and pull request. Treat a failed workflow as a deployment blocker
until the failure is understood.

Warnings may not fail CI automatically, but deprecations and repeated warnings
should be scheduled for correction. Test failures and non-zero exits must be
resolved.

## 4. Pre-deployment checklist

- GitHub Actions is green.
- The exact commit to deploy is known.
- Migration requirements are understood.
- A verified PostgreSQL backup exists before schema/data changes.
- A verified Forward Record backup exists before publication/storage changes.
- High-risk backups and checksums have been downloaded off Render.
- The previous known-good Render deployment is identified.
- New/changed environment variables are documented.
- Persistent-disk mount paths are confirmed.
- Expected manual verification steps are written down.
- A rollback decision point is clear.

## 5. Render configuration

### Start command

Use the checked-in `Procfile`. Its Gunicorn configuration should provide enough
time for known requests without masking a genuinely hung worker. Do not put
expensive work into startup or the health endpoint.

### Health check

```text
Health Check Path: /healthz
```

Do not use the homepage, dashboard, or `/live-simulations` as the platform
health check.

### Core environment variables

```text
SECRET_KEY
DATABASE_URL
REDIS_URL
BASE_URL
ADMIN_EMAILS
EMAIL_USER
EMAIL_PASS
STRIPE_SECRET_KEY
STRIPE_WEBHOOK_SECRET
STRIPE_PRO_MONTHLY_PRICE_ID
STRIPE_PRO_ANNUAL_PRICE_ID
FORWARD_RECORD_STORAGE_DIR
NEURALTREND_BACKUP_DIR
NEURALTREND_BACKUP_RETENTION
```

Optional logging controls:

```text
NEURALTREND_LOG_FORMAT
NEURALTREND_LOG_LEVEL
NEURALTREND_SLOW_REQUEST_SECONDS
```

Use `.env.example` as a key-name reference only. Never commit real production
secrets.

### Persistent storage

Confirm that:

- `FORWARD_RECORD_STORAGE_DIR` is on the intended persistent disk;
- `NEURALTREND_BACKUP_DIR` is also persistent;
- the two locations are separate;
- neither path is public or inside the repository;
- the service user can read/write both directories.

## 6. Deployment sequence

1. Merge or push the approved commit.
2. Confirm GitHub Actions passes.
3. Let Render deploy the exact commit.
4. Watch build and startup logs.
5. Confirm Render reports Live.
6. Run in Render Shell:

```bash
python tools/prelaunch_check.py
python tools/operational_check.py --strict
python tools/recovery_check.py https://neuraltrend.org
python tools/production_smoke_test.py https://neuraltrend.org --include-summary
```

7. Open the site and manually test the changed feature.
8. Open **Admin Operations** and review all four sections.
9. Inspect Render logs for tracebacks, worker exits, slow requests, 500s, and
   502s.
10. Record the deployed commit and outcome.

## 7. What each post-deployment check proves

### Prelaunch

Validates configuration, database/schema access, persistent storage,
backup-directory setup when enabled, and required market-data availability. It
does not send email or call Stripe.

### Operational

Reviews live operational state including storage, market freshness, disk space,
and failed/stale webhook or alert records. `--strict` returns non-zero for
warnings as well as errors.

### Recovery

Verifies application initialization and important public routes after a deploy,
restart, rollback, or restore.

### Production smoke

Performs read-only checks against production. `--include-summary` exercises the
heavier Signal Overview path and can reveal performance or source-data issues
not covered by `/healthz`.

## 8. Manual verification

At minimum:

- homepage;
- login/logout;
- dashboard;
- Signal Overview;
- one live simulation;
- subscription page;
- the feature changed by the release;
- username menu → Admin Operations;
- administrator access control;
- public Performance and Methodology pages when publication logic changed.

For Stripe or email releases, test with a safe controlled path and inspect the
external provider logs.

## 9. Code rollback

Roll back code when:

- startup repeatedly fails;
- the service cannot remain healthy;
- a release causes widespread 500/502 responses;
- a critical account, billing, publication, or security flow regresses;
- a fix cannot be made safely within the incident window.

Procedure:

1. Identify the last known-good Render deployment/commit.
2. Preserve logs and current-state evidence.
3. Confirm whether the failed release ran a migration or changed data.
4. Use Render's rollback/redeploy mechanism for the known-good commit.
5. Do not automatically reverse database migrations.
6. Run the full post-deployment command sequence.
7. Manually verify the affected flow.
8. Document the incident.

## 10. Rollback versus restore

```text
Code rollback
  └─ changes application code only

Database restore
  └─ replaces relational application state

Forward Record restore
  └─ replaces approved disk files
```

A rollback may not repair data already changed by a faulty release. Conversely,
a data restore is destructive and should not be used merely because a code
release failed.

## 11. Migration safety

Before a migration:

- create and download a fresh database backup;
- understand whether the migration is backward-compatible;
- identify the deployed code version that expects the new schema;
- avoid rerunning a partially applied migration without checking actual state;
- test restore/migration behavior in staging for high-risk changes.

After migration, run the prelaunch and operational checks before relying on
browser testing alone.

## 12. Deployment troubleshooting

### Build failure

Read the earliest package, syntax, compile, or test error. Verify runtime and
requirements files. Do not repeatedly redeploy unchanged code.

### Startup failure

Look for missing environment variables, database connectivity, import errors,
migration/schema mismatch, disk permissions, or invalid Gunicorn configuration.

### 502 after deployment

Look for worker timeout, OOM/SIGKILL, slow request queueing, restart loops, or a
health-check failure. Test `/healthz` separately from heavy routes.

### Healthy deploy but broken feature

Capture request ID and browser Network response, then trace the relevant
subsystem rather than rolling back blindly.
