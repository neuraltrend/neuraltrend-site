# NeuralTrend Admin Backups — Render Setup

## 1. Deploy these files

Upload/commit the package, let GitHub Actions pass, and deploy to Render.

## 2. Configure persistent backup storage

In Render, use a directory inside the service's existing persistent disk, but **outside** the Forward Record directory.

Example only (adapt to your actual disk mount):

```text
NEURALTREND_BACKUP_DIR=/var/data/neuraltrend-backups
NEURALTREND_BACKUP_RETENTION=10
```

Do not set the backup directory equal to, or inside, `FORWARD_RECORD_STORAGE_DIR`.

After adding/changing these variables, redeploy.

## 3. Open the UI

Sign in with an email listed in `ADMIN_EMAILS`, then open:

```text
https://neuraltrend.org/admin/backups
```

You can create, verify, download, and delete database and Forward Record backups. Each backup receives a `.sha256` checksum file. Creation and deletion use CSRF-protected POST requests and all routes require admin access.

## 4. Important storage rule

The Render copy is convenient, but it is not sufficient disaster recovery by itself. Download backups and checksum files to protected storage outside Render. Never commit them to GitHub.

## 5. Clean up the earlier shell-created backups

First inspect only:

```bash
find /tmp/neuraltrend-backups -maxdepth 1 -type f -printf '%f  %s bytes\n' 2>/dev/null
find /tmp/neuraltrend-admin-backups -maxdepth 1 -type f -printf '%f  %s bytes\n' 2>/dev/null
```

After you have downloaded anything you need, remove temporary shell backups:

```bash
rm -rf /tmp/neuraltrend-backups
rm -rf /tmp/neuraltrend-admin-backups
```

If you previously saved backups on the persistent disk, locate them without deleting:

```bash
find /var/data -maxdepth 3 -type f \
  \( -name 'neuraltrend-postgres-*.dump' \
     -o -name 'neuraltrend-forward-record-*.tar.gz' \
     -o -name '*.sha256' \) -print 2>/dev/null
```

Review every path first. Delete only obsolete copies after downloading any backup you intend to retain. Do not run a broad deletion command against `/var/data`.

No Render setting is created by the old `/tmp` commands. The only new settings needed for this UI are `NEURALTREND_BACKUP_DIR` and optionally `NEURALTREND_BACKUP_RETENTION`.

## 6. Post-deploy checks

```bash
python tools/prelaunch_check.py
python tools/operational_check.py --strict
python tools/production_smoke_test.py https://neuraltrend.org --include-summary
```

Then create one backup of each type through the admin page, download both archives and both `.sha256` files, and delete the Render copies only if you do not want them retained there.
