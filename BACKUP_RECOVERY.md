# NeuralTrend Backup and Recovery

## Recovery scope

NeuralTrend production state has two independent data components:

1. **PostgreSQL** — users, password hashes, subscriptions, simulations,
   watchlists, alert-delivery ledger, Stripe webhook ledger, and Forward Record
   lifecycle metadata.
2. **Persistent Forward Record files** — approved compact CSV files stored under
   `FORWARD_RECORD_STORAGE_DIR`.

A complete application-data recovery normally requires a verified backup of
both components from approximately the same time.

These backups do not recreate unrelated platform configuration such as Render
environment variables, custom domains, Redis settings, Stripe dashboard
configuration, DNS, or external email-provider settings. Keep those settings
documented separately.

## Preferred backup workflow

Use the username menu and open **Admin Operations → Backups**.

Create and download:

- `neuraltrend-postgres-....dump`
- `neuraltrend-postgres-....dump.sha256`
- `neuraltrend-forward-record-....tar.gz`
- `neuraltrend-forward-record-....tar.gz.sha256`

Keep at least one verified copy outside Render. A backup stored only on the same
service or persistent disk is not sufficient disaster recovery.

## Recommended schedule

At minimum:

- before every database migration or major release;
- before high-risk data corrections;
- after major Forward Record enrollment, retirement, or removal changes;
- weekly while usage is low;
- more frequently as customer activity grows.

## Understanding the files

### PostgreSQL `.dump`

The `.dump` is a PostgreSQL custom-format database archive. It is not intended
to be read in a text editor. It is restored with PostgreSQL tools such as
`pg_restore`.

### Forward Record `.tar.gz`

The archive contains approved Forward Record files and a manifest of paths,
sizes, and SHA-256 hashes. Unsafe paths and symbolic links are rejected.

### `.sha256`

The checksum file proves that the downloaded or re-uploaded backup still
matches the file created by NeuralTrend. It does not replace a restore rehearsal.

## Verify downloaded files

Database:

```bash
python tools/verify_database_backup.py /path/to/neuraltrend-postgres-....dump
```

Forward Record:

```bash
python tools/verify_forward_record_backup.py /path/to/neuraltrend-forward-record-....tar.gz
```

## Uploading local backups for recovery

Yes, downloaded backup files can later be uploaded from a local computer to a
controlled Render Shell, staging service, or another trusted host.

Recommended sequence:

1. Upload the backup and matching `.sha256` file.
2. Verify the checksum after upload.
3. Restore into temporary/staging resources first.
4. Run automated and manual checks.
5. Restore or switch production only after validation.

Do not upload backups into the Git repository or a public/static directory.
Backups contain sensitive application data.

## PostgreSQL restore rehearsal

Never make the first restore attempt against production.

1. Create a temporary empty PostgreSQL database.
2. Upload and verify the `.dump`.
3. Restore it using `pg_restore`.
4. Point a staging NeuralTrend deployment to the restored database.
5. Run migrations only if the restored backup predates required migrations.
6. Run prelaunch, recovery, and smoke checks.
7. Verify user counts, subscription state, watchlists, simulations, and admin
   workflows.

Typical command pattern:

```bash
pg_restore \
  --clean \
  --if-exists \
  --no-owner \
  --no-privileges \
  --dbname="$TARGET_DATABASE_URL" \
  /path/to/neuraltrend-postgres-....dump
```

Use a temporary target database first. Confirm the target URL before running
any destructive command.

## Forward Record restore rehearsal

Dry run:

```bash
python tools/restore_forward_record.py /path/to/neuraltrend-forward-record-....tar.gz
```

Apply to the configured directory:

```bash
python tools/restore_forward_record.py \
  /path/to/neuraltrend-forward-record-....tar.gz \
  --apply \
  --confirmation RESTORE-FORWARD-RECORD
```

The script creates a safety backup of the current Forward Record directory
before replacement. Run the operational check immediately afterward.

## Production recovery order

1. Restrict customer/admin write actions.
2. Preserve logs and identify the incident start time.
3. Select the last consistent verified database and Forward Record backups.
4. Prefer restoring into new temporary resources first.
5. Restore PostgreSQL.
6. Restore the corresponding Forward Record files.
7. Verify environment variables, Redis, disk mount, Stripe, email, and domain
   configuration.
8. Run:

```bash
python tools/prelaunch_check.py
python tools/operational_check.py --strict
python tools/recovery_check.py https://neuraltrend.org
python tools/production_smoke_test.py https://neuraltrend.org --include-summary
```

9. Manually verify login, subscriptions, watchlists, dashboard data, public
   performance, alerts, and admin publication controls.
10. Re-enable traffic and write actions only after validation.
11. Record the backup timestamps, commands, results, and incident resolution.

## Important limits

- PostgreSQL backup does not include persistent Forward Record files.
- Forward Record backup does not replace PostgreSQL lifecycle metadata.
- Checksum verification proves integrity, not business-level correctness.
- A code rollback is not a database rollback.
- Restoring an older production database can erase newer customer activity.
