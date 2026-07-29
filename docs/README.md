# NeuralTrend Operations and Architecture Manual

This directory is the starting point for understanding, operating, deploying,
and recovering NeuralTrend.

## Documentation map

| Document | Use it for |
|---|---|
| [System Architecture](SYSTEM_ARCHITECTURE.md) | Understanding how the application, data, subscriptions, alerts, Forward Record, storage, and operational controls fit together. |
| [Admin Operations](ADMIN_OPERATIONS.md) | Using the authenticated administrator pages safely. |
| [Operations Runbook](OPERATIONS_RUNBOOK.md) | Routine checks, deployments, incidents, and recurring maintenance. |
| [Deployment Guide](DEPLOYMENT_GUIDE.md) | Testing, releasing, configuring Render, and rolling back code. |
| [Backup and Recovery](BACKUP_AND_RECOVERY.md) | Creating, verifying, downloading, rehearsing, and restoring backups. |
| [Architecture Decisions](DECISIONS.md) | The reasons behind important design and safety choices. |

## Quick operating path

For normal administration:

```text
Username menu
    └── Admin Operations
          ├── Health & Operations
          ├── Alerts & Forward Record
          ├── Backups
          └── Recovery
```

For every production deployment, use this sequence in Render Shell:

```bash
python tools/prelaunch_check.py
python tools/operational_check.py --strict
python tools/recovery_check.py https://neuraltrend.org
python tools/production_smoke_test.py https://neuraltrend.org --include-summary
```

Then review the changed feature manually and inspect Render logs.

## System at a glance

```text
Users and administrators
          │
          ▼
      NeuralTrend
    Flask + Gunicorn
          │
  ┌───────┼───────────┬──────────────┐
  ▼       ▼           ▼              ▼
PostgreSQL Redis   Working CSVs   Persistent disk
  │       │           │              │
  │       │           └─ Signal      ├─ Approved Forward Record
  │       │              Overview    └─ Managed backups
  │       │
  │       └─ Runtime cache / rate-limit support
  │
  ├─ Accounts and subscriptions
  ├─ Watchlists and simulations
  ├─ Webhook and alert ledgers
  └─ Forward Record lifecycle metadata
```

External services include Stripe, an email provider, DNS/custom-domain
configuration, GitHub Actions, and Render hosting.

## Documentation principles

- The approved compact Forward Record files are authoritative for published
  historical performance.
- The large working CSVs power Signal Overview and may be recalculated or
  corrected without rewriting published performance.
- A code rollback is not a data restore.
- PostgreSQL and Forward Record files must be backed up separately.
- A backup kept only on the same Render disk is not complete disaster recovery.
- Production restore remains an explicit, controlled process rather than a
  one-click browser action.
