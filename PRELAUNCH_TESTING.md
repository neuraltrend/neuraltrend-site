# NeuralTrend Automated Testing and Deployment Verification

## Local or CI test suite

Install production and development dependencies:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Run all tests:

```bash
python -m pytest
```

Run the same coverage command used by GitHub Actions:

```bash
python -m pytest \
  --cov=app \
  --cov=models \
  --cov=operational_logging \
  --cov=tools \
  --cov-report=term-missing
```

The tests use temporary SQLite and Forward Record storage. Email sending is
suppressed, Stripe calls are mocked, rate limits are disabled, and production
CSV data is not modified.

## GitHub Actions

`.github/workflows/tests.yml` runs compilation and pytest on every push and pull
request. A failed workflow should block deployment until the failure is
understood.

Warnings do not fail the workflow unless configured to do so. Test failures and
non-zero exits must be fixed before deployment.

## Render prelaunch check

After migrations and environment variables are in place:

```bash
python tools/prelaunch_check.py
```

This checks configuration, database connectivity/schema, persistent Forward
Record storage, backup-directory configuration when enabled, and required
market-data availability. It does not call Stripe or send email.

## Operational check

```bash
python tools/operational_check.py --strict
```

This reviews database and storage health, failed/stale webhooks, failed/stale
alerts, disk space, and market-data freshness. `--strict` treats warnings as a
non-zero result.

## Recovery check

```bash
python tools/recovery_check.py https://neuraltrend.org
```

This verifies application initialization and important public routes after a
deployment, restart, rollback, or restore.

## Read-only production smoke test

```bash
python tools/production_smoke_test.py https://neuraltrend.org
```

Include the heavier Signal Overview calculation when appropriate:

```bash
python tools/production_smoke_test.py https://neuraltrend.org --include-summary
```

## Admin UI verification

Sign in as an admin and open the username menu → **Admin Operations**.

Verify these hub destinations:

```text
/admin/operations
/admin/signal-alerts
/admin/backups
/admin/recovery
```

Confirm:

- the operations page contains no private customer identifiers;
- alert and Forward Record controls are accessible;
- backup creation and downloads work;
- Recovery reports the latest verified pair;
- non-admin users cannot access admin routes.

## Before a major release or migration

Create both backups through **Admin Operations → Backups**, then download the
backups and checksum files off Render.

Shell alternatives remain available:

```bash
python tools/backup_database.py --output-dir /tmp/neuraltrend-backups
python tools/backup_forward_record.py --output-dir /tmp/neuraltrend-backups
```

Verify downloaded copies:

```bash
python tools/verify_database_backup.py /path/to/neuraltrend-postgres-....dump
python tools/verify_forward_record_backup.py /path/to/neuraltrend-forward-record-....tar.gz
```

## Standard post-deployment sequence

```bash
python tools/prelaunch_check.py
python tools/operational_check.py --strict
python tools/recovery_check.py https://neuraltrend.org
python tools/production_smoke_test.py https://neuraltrend.org --include-summary
```

Then perform the admin UI and changed-feature manual checks.
