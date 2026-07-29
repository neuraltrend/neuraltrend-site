# NeuralTrend Admin Backups — Setup and Use

## Purpose

The admin backup interface creates and manages the two backup types needed for
NeuralTrend data recovery:

1. PostgreSQL database backup (`.dump`)
2. Forward Record file backup (`.tar.gz`)

Each backup is paired with a `.sha256` checksum file.

## Render configuration

Configure a persistent directory that is outside the Forward Record directory.
Example only; use the actual Render disk mount:

```text
NEURALTREND_BACKUP_DIR=/var/data/neuraltrend-backups
NEURALTREND_BACKUP_RETENTION=10
```

Rules:

- `NEURALTREND_BACKUP_DIR` must be on the persistent disk.
- It must not equal or sit inside `FORWARD_RECORD_STORAGE_DIR`.
- It must not be inside the Git repository, `static/`, or another public path.
- Redeploy after changing environment variables.

## Open the admin interface

Sign in with an email listed in `ADMIN_EMAILS`.

From the username menu, select **Admin Operations**. The hub links to:

- Health & Operations
- Alerts & Forward Record
- Backups
- Recovery

Direct routes are:

```text
/admin/operations
/admin/signal-alerts
/admin/backups
/admin/recovery
```

## Create and download backups

On **Admin Operations → Backups**:

1. Create a PostgreSQL backup.
2. Create a Forward Record backup.
3. Confirm both show a verified checksum.
4. Download both backup files and both `.sha256` files.
5. Store the downloaded copies in protected storage outside Render.

The UI supports creation, verification, download, retention, and deletion.
Creation and deletion are CSRF-protected POST actions and require admin access.

## Retention and deletion

`NEURALTREND_BACKUP_RETENTION=10` keeps the latest ten backups of each managed
type and removes older managed copies automatically.

Do not delete the latest verified pair until an off-Render copy has been safely
downloaded and verified.

## Clean up earlier temporary shell backups

Inspect first:

```bash
find /tmp/neuraltrend-backups -maxdepth 1 -type f -printf '%f  %s bytes\n' 2>/dev/null
find /tmp/neuraltrend-admin-backups -maxdepth 1 -type f -printf '%f  %s bytes\n' 2>/dev/null
```

After preserving anything needed:

```bash
rm -rf /tmp/neuraltrend-backups
rm -rf /tmp/neuraltrend-admin-backups
```

To locate persistent-disk backups without deleting them:

```bash
find /var/data -maxdepth 3 -type f \
  \( -name 'neuraltrend-postgres-*.dump' \
     -o -name 'neuraltrend-forward-record-*.tar.gz' \
     -o -name '*.sha256' \) -print 2>/dev/null
```

Review every path before deleting anything. Never run a broad deletion command
against `/var/data`.

## Post-deployment verification

```bash
python tools/prelaunch_check.py
python tools/operational_check.py --strict
python tools/recovery_check.py https://neuraltrend.org
python tools/production_smoke_test.py https://neuraltrend.org --include-summary
```

Then open **Admin Operations** and confirm that Backups and Recovery both report
verified current files.

## Safety boundary

The browser UI intentionally does not provide one-click production restore.
Restore downloaded files into a temporary or staging environment first, verify
them, and only then perform a controlled production recovery.
