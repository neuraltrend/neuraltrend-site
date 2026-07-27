# NeuralTrend Backup and Recovery

## What must be protected

NeuralTrend production state has two independent parts:

1. PostgreSQL: users, subscriptions, simulations, watchlists, alert delivery
   ledger, Stripe webhook ledger, and Forward Record lifecycle metadata.
2. Persistent Forward Record files: approved compact CSV files under
   `FORWARD_RECORD_STORAGE_DIR`.

A complete recovery requires both from approximately the same time.

## Backup schedule

Recommended minimum:

- before every database migration or major release;
- after enrolling, retiring, or removing a public Forward Record asset;
- weekly while prelaunch or low-volume;
- more frequently as usage grows.

Keep at least one copy outside Render. A backup stored only on the same service
or disk is not disaster recovery.

## PostgreSQL backup

From Render Shell:

```bash
python tools/backup_database.py --output-dir /tmp/neuraltrend-backups
```

The command:

- reads `DATABASE_URL` without printing it;
- uses PostgreSQL custom format;
- excludes owner/ACL portability problems;
- verifies the catalog with `pg_restore --list` when available;
- prints file size and SHA-256.

Download/copy the resulting `.dump` off Render.

Verify a downloaded copy:

```bash
python tools/verify_database_backup.py /path/to/backup.dump
```

## Forward Record backup

From Render Shell:

```bash
python tools/backup_forward_record.py --output-dir /tmp/neuraltrend-backups
```

The archive contains only approved Forward Record files plus a manifest of
paths, sizes, and SHA-256 hashes. Symbolic links and unsafe paths are rejected.

Verify a downloaded copy:

```bash
python tools/verify_forward_record_backup.py /path/to/backup.tar.gz
```

## Restore rehearsal

Never make the first restore attempt against production.

PostgreSQL rehearsal:

1. Create a temporary empty PostgreSQL database.
2. Restore the `.dump` into that temporary database using `pg_restore`.
3. Point a temporary/staging NeuralTrend deployment to it.
4. Run migrations only if the restored backup predates required migrations.
5. Run `prelaunch_check.py`, automated tests where applicable, and smoke tests.
6. Confirm counts and key workflows before considering production recovery.

Forward Record rehearsal:

```bash
python tools/restore_forward_record.py /path/to/backup.tar.gz
```

The default is dry-run. To apply to the configured directory:

```bash
python tools/restore_forward_record.py /path/to/backup.tar.gz \
  --apply \
  --confirmation RESTORE-FORWARD-RECORD
```

Before replacing files, the script creates a safety backup of the current
Forward Record directory. Run `operational_check.py` immediately afterward.

## Production recovery order

1. Stop customer/admin write actions.
2. Identify the last consistent database and Forward Record backups.
3. Prefer restoring into new temporary resources first.
4. Restore PostgreSQL.
5. Restore Forward Record files from the corresponding time.
6. Verify environment variables and disk mount.
7. Run:

```bash
python tools/prelaunch_check.py
python tools/operational_check.py
python tools/recovery_check.py https://neuraltrend.org
```

8. Manually verify login, subscription state, watchlists, public performance,
   and admin publication controls.
9. Re-enable traffic/admin actions only after verification.
10. Record the incident, backup timestamps, restore commands, and final checks.

## Important limits

- Forward Record CSV backup does not replace the PostgreSQL backup because
  lifecycle metadata is stored in PostgreSQL.
- PostgreSQL backup does not include the approved CSV files on the persistent
  disk.
- Verification proves archive integrity/readability, not that every business
  workflow is semantically correct. A restore rehearsal is still required.
