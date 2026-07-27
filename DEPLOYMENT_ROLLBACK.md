# Deployment and Rollback Checklist

## Before deployment

- GitHub Actions is green.
- Database migration requirement is understood.
- A database backup exists for schema/data changes.
- A Forward Record backup exists for publication/storage changes.
- Rollback commit/deploy is known.
- New environment variables are documented.

## After deployment

```bash
python tools/prelaunch_check.py
python tools/operational_check.py
python tools/recovery_check.py https://neuraltrend.org
```

Then manually test the changed feature.

## Roll back when

- the service cannot become healthy;
- repeated 500/502 responses begin after the release;
- authentication, billing, publication, or data integrity is at risk;
- a required environment variable or migration was omitted and cannot be fixed
  safely in place.

## Rollback process

1. Preserve the failed deployment logs and request IDs.
2. Roll back to the latest known-good Render deployment/commit.
3. Do not automatically reverse database migrations.
4. Confirm `/healthz`.
5. Run `recovery_check.py`.
6. Verify the affected feature manually.
7. Decide separately whether a database/data correction is required.

A code rollback and a database rollback are different operations. Never restore
an older database merely because application code was rolled back without first
checking compatibility and potential loss of newer customer data.
