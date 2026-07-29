# NeuralTrend Deployment and Rollback Checklist

## Before deployment

- GitHub Actions is green.
- The migration requirement is understood.
- A verified database backup exists before schema/data changes.
- A verified Forward Record backup exists before storage/publication changes.
- Both backups have been downloaded off Render for high-risk releases.
- The previous known-good commit/deployment is identified.
- New environment variables and persistent-disk paths are documented.
- The expected manual verification steps are known.

Backups can be created from **Admin Operations → Backups**.

## After deployment

Run in Render Shell:

```bash
python tools/prelaunch_check.py
python tools/operational_check.py --strict
python tools/recovery_check.py https://neuraltrend.org
python tools/production_smoke_test.py https://neuraltrend.org --include-summary
```

Then:

1. Confirm GitHub Actions is green.
2. Confirm Render reports the service as Live.
3. Confirm `/healthz` returns HTTP 200.
4. Open the username menu → **Admin Operations**.
5. Review Health & Operations, Backups, and Recovery.
6. Manually test the feature changed by the deployment.
7. Review Render logs for new tracebacks, worker exits, 500s, or 502s.

## Roll back code when

- the service cannot become healthy;
- repeated 500/502 responses begin after the release;
- authentication, billing, publication, or data integrity is at risk;
- a required variable or migration was omitted and cannot be fixed safely;
- a critical manual workflow fails after deployment.

## Code rollback process

1. Preserve failed-deployment logs, timestamps, and request IDs.
2. Roll back to the latest known-good Render deployment or commit.
3. Do not automatically reverse database migrations.
4. Confirm `/healthz`.
5. Run `recovery_check.py` and the smoke test.
6. Open Admin Operations and review component status.
7. Verify the affected feature manually.
8. Decide separately whether a database/data correction is needed.

## Code rollback versus data restore

A code rollback changes application code. A data restore replaces database or
persistent-file state. They are separate operations.

Never restore an older database merely because application code was rolled
back. First assess schema compatibility and the loss of newer users,
subscriptions, webhooks, watchlists, alerts, and simulations.

Use **Admin Operations → Recovery** for recovery readiness and the safe restore
sequence. Production restore remains a controlled shell/staging operation, not
a one-click browser action.
