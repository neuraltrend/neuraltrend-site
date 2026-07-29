# NeuralTrend Admin Operations Guide

## 1. Opening the administrator area

Sign in with an account whose email appears in `ADMIN_EMAILS`. Open the username
menu and select **Admin Operations**.

```text
/admin/operations
```

The hub links to:

```text
Health & Operations     /admin/operations
Alerts & Forward Record /admin/signal-alerts
Backups                 /admin/backups
Recovery                /admin/recovery
```

Non-administrator users must not be able to access these routes.

## 2. Health & Operations

### Purpose

Use this page for a compact read-only view of production health and operational
readiness. It complements—not replaces—Render logs, metrics, and shell checks.

### Review after

- every deployment;
- a restart or 502 incident;
- an external-service failure;
- a backup or Forward Record warning;
- an administrator report that a feature is not working.

### Interpretation

A healthy page means the checks represented there passed at that moment. It
does not prove that every customer flow, Stripe operation, email delivery, or
heavy market calculation is functioning.

For machine-readable inspection, the application also exposes:

```text
/admin/operations.json
```

This remains administrator-protected and should not be used as a public uptime
endpoint.

### Safe actions

The operations page is primarily observational. Use the specific Backups,
Recovery, or Alerts & Forward Record pages for controlled actions.

## 3. Alerts & Forward Record

### Purpose

This page groups operational controls for signal-change alerts and Forward
Record publication.

### Signal-alert workflow

1. Confirm source signal data is current.
2. Preview or review the pending alert state.
3. Confirm recipient and delivery counts are reasonable.
4. Trigger the controlled send action only once.
5. Review the result and Render logs.
6. Check the operational page for failed or stale delivery records.

Avoid repeated clicking while a request is still running. Repeated actions can
cause queueing and complicate diagnosis even when idempotency protections are
present.

### Forward Record workflow

```text
Large working CSVs
      ↓
Private sandbox preview
      ↓
Review pending new rows
      ↓
Manual approval/publication
      ↓
Compact approved CSV on persistent disk
```

The sandbox preview should evaluate only dates after the latest approved date.
Changes to already-approved dates in the working CSV must not rewrite or block
the approved performance history.

### Meaning of common outcomes

- **Pending asset updates**: newly dated candidate rows are available for
  review.
- **Unchanged asset**: there is no newly dated row to append.
- **Newly approved rows**: rows were accepted into the compact approved record.
- **Source-data error**: a new candidate row or required source field is invalid.
- **Approved-record integrity error**: the compact approved CSV, digest, or
  lifecycle metadata is inconsistent. Do not bypass this; investigate before
  publishing.

### Current source-of-truth behavior

For approved dates:

```text
Approved compact disk CSV = authoritative publication history
Working large CSV         = Signal Overview source
```

The working CSV can be corrected independently. A published-history correction
requires a separate explicit process and should preserve a record of the old
and new values, the reason, approver, and time.

## 4. Backups

### Purpose

Create, verify, download, retain, and delete managed backups of both production
data components.

### Required configuration

```text
NEURALTREND_BACKUP_DIR=/actual/persistent/disk/path
NEURALTREND_BACKUP_RETENTION=10
```

The backup directory must not be inside `FORWARD_RECORD_STORAGE_DIR`, the Git
repository, or a public/static directory.

### Create a complete backup pair

1. Select **Create PostgreSQL backup**.
2. Select **Create Forward Record backup**.
3. Confirm both display verified checksums.
4. Download both backup files.
5. Download both `.sha256` files.
6. Store them in protected off-Render storage.

Expected managed files:

```text
neuraltrend-postgres-<timestamp>.dump
neuraltrend-postgres-<timestamp>.dump.sha256
neuraltrend-forward-record-<timestamp>.tar.gz
neuraltrend-forward-record-<timestamp>.tar.gz.sha256
```

### Verify

Use the page verification action after creation and after any suspicious disk
or file event. A verified checksum proves file integrity, not that the backup
contains the intended business state. Periodic restore rehearsals remain
necessary.

### Download

Backups contain sensitive data. Store downloads in encrypted, access-controlled
storage. Do not email them, put them in a public cloud folder, or commit them to
Git.

### Delete

Delete only managed backup filenames shown by the application. Preserve at
least one verified off-Render copy before deleting the newest usable pair.
Deletion is a CSRF-protected administrator action.

### Retention

`NEURALTREND_BACKUP_RETENTION` keeps the configured number of recent managed
files for each backup type. Retention on the Render disk is convenience and
operational protection; it is not a substitute for off-Render recovery copies.

## 5. Recovery

### Purpose

The Recovery page summarizes whether a recent, verified database backup and
Forward Record backup are available and provides controlled recovery guidance.

### What to review

- latest database-backup timestamp and verification state;
- latest Forward Record-backup timestamp and verification state;
- age of both backups;
- availability of download actions;
- whether the pair is sufficiently close in time for the intended recovery.

### Important boundary

The browser intentionally has no one-click production restore. A mistaken
restore could erase new customer activity or replace correct published history.
Use the documented shell procedure, preferably against staging first.

## 6. Administrator safety rules

- Use a private browser window when verifying access-control behavior.
- Do not expose `/admin/*` pages to external uptime services.
- Do not refresh repeatedly during a slow action.
- Record the time and `X-Request-ID` for failures.
- Create both backup types before schema, storage, bulk-data, or publication
  changes.
- Never use a broad deletion command against the persistent-disk mount.
- Never treat a green `/healthz` response as proof that all business workflows
  are healthy.
- Never repair a published Forward Record by silently editing the working CSV.

## 7. Post-deployment admin checklist

1. Open **Admin Operations** from the username menu.
2. Review Health & Operations.
3. Open Alerts & Forward Record and confirm the page loads.
4. Open Backups and confirm managed files are visible.
5. Open Recovery and confirm verification and age are sensible.
6. Manually test the feature changed by the deployment.
7. Confirm a non-admin user cannot access administrator routes after any
   authorization-related change.
