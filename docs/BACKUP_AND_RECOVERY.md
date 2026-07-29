# NeuralTrend Backup and Recovery

## 1. Recovery scope

A complete NeuralTrend application-data recovery normally requires two separate
backup types created close together.

### PostgreSQL backup

Contains relational state such as:

- users and password hashes;
- subscription state;
- watchlists and simulations;
- signal-alert delivery ledger;
- Stripe webhook ledger;
- Forward Record lifecycle/integrity metadata.

### Forward Record backup

Contains the compact approved publication files stored under
`FORWARD_RECORD_STORAGE_DIR`.

```text
Complete recoverable application state
  = PostgreSQL backup
  + Forward Record backup
  + separately documented Render/external configuration
```

Backups do not recreate Render environment variables, Redis configuration,
custom domains, DNS, Stripe dashboard settings, email-provider configuration,
or GitHub/Render account settings.

## 2. Backup configuration

Use an actual persistent-disk path:

```text
NEURALTREND_BACKUP_DIR=/actual/render/disk/neuraltrend-backups
NEURALTREND_BACKUP_RETENTION=10
```

Requirements:

- persistent across deploys/restarts;
- outside `FORWARD_RECORD_STORAGE_DIR`;
- not inside the Git repository;
- not under `static/` or another public path;
- writable by the service;
- redeploy/restart after environment changes.

## 3. Preferred browser workflow

Open username menu → **Admin Operations → Backups**.

1. Create a PostgreSQL backup.
2. Create a Forward Record backup.
3. Confirm both verify.
4. Download both backup files.
5. Download both `.sha256` files.
6. Store all four in protected off-Render storage.
7. Open Recovery and confirm the latest pair is current.

## 4. Backup files

### PostgreSQL `.dump`

A PostgreSQL custom-format archive produced for `pg_restore`. It is not plain
text and should not be edited manually.

### Forward Record `.tar.gz`

A controlled archive of approved files plus a manifest containing expected
paths, sizes, and SHA-256 hashes. Verification rejects unsafe paths and symbolic
links.

### `.sha256`

Confirms that a backup file is byte-for-byte identical to the file originally
created. It does not prove that the selected backup represents the correct
business point in time.

## 5. Recommended schedule

Create both backup types:

- before every database migration;
- before high-risk releases;
- before bulk data correction;
- before Forward Record lifecycle/storage changes;
- after major enrollment, retirement, or removal changes;
- weekly during low usage;
- more frequently as customer and transaction activity grows.

Keep multiple generations and at least one verified off-Render copy.

## 6. Command-line backup alternatives

```bash
python tools/backup_database.py --output-dir /tmp/neuraltrend-backups
python tools/backup_forward_record.py --output-dir /tmp/neuraltrend-backups
```

Temporary shell output is not durable. Move/download verified files to protected
storage promptly.

## 7. Verification

Database:

```bash
python tools/verify_database_backup.py \
  /path/to/neuraltrend-postgres-....dump
```

Forward Record:

```bash
python tools/verify_forward_record_backup.py \
  /path/to/neuraltrend-forward-record-....tar.gz
```

Verify again after downloading, uploading, copying, or restoring from long-term
storage.

## 8. Off-Render storage

A copy stored only on the same Render disk is vulnerable to account, service,
region, or disk loss. Use encrypted, access-controlled storage outside Render.

Do not:

- commit backups to Git;
- upload them into public/static directories;
- share them through public links;
- leave them indefinitely in `/tmp`;
- store checksum files without the corresponding backup or vice versa.

## 9. Restore rehearsal policy

Never make the first restore attempt against production.

A restore rehearsal should confirm:

- the checksum passes;
- the archive/tooling can read the backup;
- PostgreSQL restores successfully;
- approved Forward Record files restore safely;
- database metadata and disk files are consistent;
- the application starts against restored state;
- critical account, billing, simulation, and performance workflows work.

## 10. PostgreSQL restore rehearsal

1. Create a temporary empty PostgreSQL database.
2. Upload the `.dump` and matching checksum.
3. Verify the backup.
4. Confirm the target database URL twice.
5. Restore:

```bash
pg_restore \
  --clean \
  --if-exists \
  --no-owner \
  --no-privileges \
  --dbname="$TARGET_DATABASE_URL" \
  /path/to/neuraltrend-postgres-....dump
```

6. Point a staging NeuralTrend deployment to the restored database.
7. Run migrations only when required by the restored backup/code combination.
8. Run automated checks.
9. Verify counts and representative user/subscription/watchlist/simulation
   records.

`--clean` is destructive to objects in the target database. Use only an
intentionally selected temporary target during rehearsal.

## 11. Forward Record restore rehearsal

Dry run:

```bash
python tools/restore_forward_record.py \
  /path/to/neuraltrend-forward-record-....tar.gz
```

Apply only after reviewing the dry-run output:

```bash
python tools/restore_forward_record.py \
  /path/to/neuraltrend-forward-record-....tar.gz \
  --apply \
  --confirmation RESTORE-FORWARD-RECORD
```

The script creates a safety copy of the current Forward Record directory before
replacement. Preserve that safety copy until recovery is fully validated.

## 12. Production recovery procedure

1. Restrict or pause customer/admin write actions where possible.
2. Preserve logs and identify the incident start time.
3. Determine whether the problem is code, PostgreSQL, Forward Record files,
   external configuration, or a combination.
4. Select the last consistent verified database and Forward Record backups.
5. Prefer restoring into new temporary resources first.
6. Restore PostgreSQL.
7. Restore the corresponding Forward Record files.
8. Verify environment variables, Redis, disk mounts, Stripe, email, DNS, and
   custom-domain configuration.
9. Run:

```bash
python tools/prelaunch_check.py
python tools/operational_check.py --strict
python tools/recovery_check.py https://neuraltrend.org
python tools/production_smoke_test.py https://neuraltrend.org --include-summary
```

10. Manually verify:
    - login/logout;
    - account and subscription state;
    - watchlists and simulations;
    - Signal Overview;
    - public Performance and Methodology;
    - Forward Record admin controls;
    - Stripe webhook behavior;
    - email/alert behavior;
    - backup/recovery pages.
11. Re-enable normal traffic/write actions only after validation.
12. Record backup timestamps, commands, results, and recovery decisions.

## 13. Selecting a consistent pair

The closest timestamps are not automatically the correct pair. Consider:

- whether a publication occurred between the two backups;
- whether a database migration occurred;
- whether subscriptions or customer activity changed materially;
- whether Forward Record metadata matches the archived files;
- whether either checksum or manifest fails.

When in doubt, rehearse more than one candidate pair in staging.

## 14. Code rollback versus data recovery

A code rollback is appropriate for a bad application release. A data restore is
appropriate for lost or corrupted state. They may be required together, but
one does not imply the other.

Restoring an older database can erase newer customer actions. Restoring an old
Forward Record can remove legitimate approved rows. Always identify the precise
failure boundary first.

## 15. Cleanup and retention

Inspect temporary locations before deletion:

```bash
find /tmp/neuraltrend-backups -maxdepth 1 -type f -printf '%f  %s bytes\n' 2>/dev/null
find /tmp/neuraltrend-admin-backups -maxdepth 1 -type f -printf '%f  %s bytes\n' 2>/dev/null
```

After preserving required files:

```bash
rm -rf /tmp/neuraltrend-backups
rm -rf /tmp/neuraltrend-admin-backups
```

Locate managed backups on persistent storage without deleting:

```bash
find /var/data -maxdepth 3 -type f \
  \( -name 'neuraltrend-postgres-*.dump' \
     -o -name 'neuraltrend-forward-record-*.tar.gz' \
     -o -name '*.sha256' \) -print 2>/dev/null
```

Review paths before any deletion. Never run broad recursive deletion against
`/var/data` or the configured Forward Record directory.

## 16. Important limitations

- Database backup does not include approved disk files.
- Forward Record backup does not replace database lifecycle metadata.
- Checksum verification proves integrity, not business correctness.
- Restore rehearsal is necessary even for verified files.
- Backup retention on Render is not off-site disaster recovery.
- A backup cannot recreate undocumented secrets or third-party configuration.
