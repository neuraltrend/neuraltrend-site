# NeuralTrend System Architecture

## 1. Purpose and boundaries

NeuralTrend is a Flask web application that presents market signals, user
watchlists, live simulations, subscription-controlled features, public
performance history, and administrator operations.

The application combines four principal state layers:

1. **PostgreSQL** for relational application state.
2. **Redis** for runtime support such as caching or rate limiting where enabled.
3. **Large working market CSVs** for current Signal Overview calculations.
4. **Persistent disk files** for the approved Forward Record and managed
   backups.

External systems provide billing, email delivery, source control, continuous
integration, hosting, DNS, and monitoring.

## 2. High-level architecture

```text
                          Internet
                             │
                             ▼
                    Render web service
                  Gunicorn → Flask app
                             │
        ┌────────────────────┼─────────────────────┐
        │                    │                     │
        ▼                    ▼                     ▼
   PostgreSQL              Redis             Repository data/
        │                    │                     │
        │                    │                     └─ Large working CSVs
        │                    │                          │
        │                    │                          └─ Signal Overview
        │                    │
        │                    └─ Runtime support
        │
        ├─ users and credentials
        ├─ subscription state
        ├─ simulations and watchlists
        ├─ alert-delivery ledger
        ├─ Stripe webhook ledger
        └─ Forward Record lifecycle metadata

                    Render persistent disk
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
       Approved Forward Record      Managed backups
          compact CSV files       .dump / .tar.gz
```

## 3. Request lifecycle

A normal browser request follows this path:

```text
Browser
  ↓
Render / Gunicorn
  ↓
Flask route in app.py
  ↓
Authentication, authorization, CSRF, and rate-limit checks as applicable
  ↓
Business logic and database/storage access
  ↓
Jinja template or JSON response
  ↓
Operational logging with request ID
```

The application exposes `X-Request-ID` on responses. Use that value with the
request time, URL, method, and status when tracing failures in Render logs.

## 4. Major route groups

### Public and account routes

Examples include:

```text
/
/login
/signup
/logout
/dashboard
/subscription
/performance
/methodology
```

Account flows also include verification, password reset, account deletion, and
subscription-state endpoints.

### Market and simulation routes

```text
/data
/live-simulations
/live-simulations/<simulation_id>
/live-simulations/<simulation_id>/status
/watchlist
/watchlist/<ticker>
/watchlist/<ticker>/alerts
```

`/live-simulations` can be materially heavier than simple pages. Keep Render's
health check pointed at `/healthz`, not at a dashboard or simulation endpoint.

### Billing route group

```text
/create-checkout-session
/billing-portal
/stripe/webhook
/subscription-state
```

### Administrator route group

```text
/admin/operations
/admin/operations.json
/admin/signal-alerts
/admin/forward-record
/admin/backups
/admin/recovery
```

Administrator access is controlled by authenticated email membership in
`ADMIN_EMAILS`.

## 5. Signal Overview data flow

The large CSV files under the working data area power current and historical
Signal Overview calculations.

```text
Market/signal generation process
          ↓
Large working CSV for each asset
          ↓
Application reads current working values
          ↓
Signal Overview and related user-facing calculations
```

These files are operational inputs rather than immutable publication archives.
Historical rows may occasionally change because of a source correction,
regeneration, recalculation, timezone normalization, rounding change, or a
controlled signal-data correction.

## 6. Forward Record architecture

The Forward Record is deliberately separated from the large working CSVs.

```text
Large working CSV
      │
      ├─ already-approved dates → ignored by normal publication preview
      │
      └─ dates after latest approved date
                     ↓
              Private sandbox preview
                     ↓
               Manual approval
                     ↓
          Append to compact approved CSV
                     ↓
        Public methodology/performance history
```

### Source-of-truth rule

For a date that has already been approved, the compact approved CSV on
persistent disk is authoritative. Normal sandbox and public previews preserve
that approved row even when the corresponding row in the large working CSV:

- changes;
- is recalculated;
- is corrected for Signal Overview; or
- disappears.

The publisher considers only newly dated candidate rows after the latest
approved date. It does not reconstruct approved performance history from the
large working file.

### Integrity boundary

The system may still reject publication when the approved compact CSV itself no
longer matches its stored digest or lifecycle metadata. Changing published
performance requires a separate, explicit correction process with an audit
trail; editing a working CSV does not rewrite the public record.

### Why PostgreSQL is also involved

The disk files hold the approved rows, while PostgreSQL stores Forward Record
lifecycle and integrity metadata. A complete recovery therefore requires both:

```text
Approved compact CSV files + corresponding PostgreSQL metadata
```

## 7. Subscription and Stripe architecture

```text
User chooses a plan
      ↓
NeuralTrend creates Stripe Checkout session
      ↓
Stripe completes or changes billing state
      ↓
Stripe sends signed webhook
      ↓
NeuralTrend verifies STRIPE_WEBHOOK_SECRET
      ↓
Database subscription state is updated
      ↓
Feature access reads application subscription state
```

The Stripe webhook ledger in PostgreSQL helps identify failed, stale, or
repeated deliveries. The customer billing portal is exposed through the
application rather than by placing secret Stripe credentials in the browser.

Relevant configuration includes:

```text
STRIPE_SECRET_KEY
STRIPE_WEBHOOK_SECRET
STRIPE_PRO_MONTHLY_PRICE_ID
STRIPE_PRO_ANNUAL_PRICE_ID
```

`STRIPE_PRO_PRICE_ID` may remain for compatibility where used, but monthly and
annual identifiers should be treated as the explicit current plan settings.

## 8. Email and signal-alert architecture

```text
Watchlist / alert preference
          ↓
Database stores user configuration
          ↓
Admin or controlled alert tool evaluates signal changes
          ↓
Email provider sends notification
          ↓
Delivery result is recorded in alert ledger
```

The operational check reviews failed and stale alert deliveries. Unsubscribe
links use tokens rather than exposing internal identifiers.

Relevant configuration:

```text
EMAIL_USER
EMAIL_PASS
BASE_URL
```

## 9. Authentication, authorization, and safety controls

The application uses several independent controls:

- authenticated sessions for account state;
- password hashing in the database;
- email verification and tokenized reset/delete flows;
- administrator authorization through `ADMIN_EMAILS`;
- CSRF protection for state-changing browser actions;
- rate limiting where configured;
- path validation for backup filenames and archive extraction;
- Stripe webhook-signature verification;
- checksum and manifest verification for backups;
- request logging with sensitive-value redaction.

No backup or private application data should be served from `static/`, committed
to Git, or stored in a public path.

## 10. Persistent disk layout

Two configured directories serve different purposes:

```text
FORWARD_RECORD_STORAGE_DIR
    └─ approved compact Forward Record files

NEURALTREND_BACKUP_DIR
    ├─ neuraltrend-postgres-*.dump
    ├─ neuraltrend-postgres-*.dump.sha256
    ├─ neuraltrend-forward-record-*.tar.gz
    └─ neuraltrend-forward-record-*.tar.gz.sha256
```

The backup directory must be on persistent storage but must not equal, contain,
or sit inside the Forward Record directory. This separation prevents backup
management or retention from damaging the live approved record.

## 11. Backup architecture

```text
PostgreSQL
    ↓ pg_dump custom format
Database .dump + SHA-256

Forward Record directory
    ↓ safe archive with manifest
Forward Record .tar.gz + SHA-256
```

The two backup types are independent and should be created close together.
Downloading both backup files and both checksums off Render provides protection
against loss of the service or its disk.

## 12. Recovery and rollback architecture

Three procedures must remain distinct:

### Code rollback

Moves the application to a previous known-good deployment or commit. It does
not restore an older database or disk state.

### PostgreSQL restore

Restores relational application state. It can erase newer account,
subscription, webhook, alert, watchlist, or simulation activity if an older
backup is selected.

### Forward Record restore

Restores approved compact files on persistent disk. It must be consistent with
the accompanying database lifecycle metadata.

Preferred recovery path:

```text
Verified backup pair
      ↓
Restore into temporary/staging resources
      ↓
Run automated and manual validation
      ↓
Perform controlled production recovery
      ↓
Run post-recovery checks
```

## 13. Health and observability

### Fast platform health endpoint

```text
/healthz
```

Render should use this path. It must stay fast and avoid expensive Signal
Overview or simulation calculations.

### Operational visibility

- Render deployment events, logs, health state, CPU, and memory.
- GitHub Actions compilation and pytest results.
- Admin Operations status pages.
- `X-Request-ID` correlation.
- Optional structured logging.
- External uptime monitoring for `/` and `/healthz`.

Configuration:

```text
NEURALTREND_LOG_FORMAT=json
NEURALTREND_LOG_LEVEL=INFO
NEURALTREND_SLOW_REQUEST_SECONDS=2.0
```

## 14. Operational scripts

| Script | Purpose |
|---|---|
| `tools/prelaunch_check.py` | Validate required configuration, database/schema access, storage, and market-data prerequisites without sending email or calling Stripe. |
| `tools/operational_check.py` | Inspect operational state such as storage, stale/failed webhook and alert activity, market-data freshness, and disk space. |
| `tools/recovery_check.py` | Verify initialization and important routes after deploy, restart, rollback, or restore. |
| `tools/production_smoke_test.py` | Run read-only checks against the live site; optional summary mode exercises heavier calculations. |
| `tools/publish_forward_record.py` | Build sandbox/public Forward Record publication results while preserving approved disk history. |
| `tools/send_signal_change_alerts.py` | Execute controlled signal-change email delivery. |
| `tools/backup_database.py` | Create PostgreSQL custom-format backup and checksum. |
| `tools/backup_forward_record.py` | Create Forward Record archive, manifest, and checksum. |
| `tools/verify_database_backup.py` | Verify a database backup and checksum. |
| `tools/verify_forward_record_backup.py` | Verify archive safety, manifest, and checksum. |
| `tools/restore_forward_record.py` | Dry-run or explicitly apply a Forward Record restore. |
| `tools/audit_signal_history.py` | Inspect signal-history consistency without changing publication history. |

## 15. Configuration inventory

Core production variables currently referenced by the application and tools:

```text
ADMIN_EMAILS
BASE_URL
DATABASE_URL
REDIS_URL
SECRET_KEY
EMAIL_USER
EMAIL_PASS
STRIPE_SECRET_KEY
STRIPE_WEBHOOK_SECRET
STRIPE_PRO_MONTHLY_PRICE_ID
STRIPE_PRO_ANNUAL_PRICE_ID
FORWARD_RECORD_STORAGE_DIR
NEURALTREND_BACKUP_DIR
NEURALTREND_BACKUP_RETENTION
NEURALTREND_LOG_FORMAT
NEURALTREND_LOG_LEVEL
NEURALTREND_SLOW_REQUEST_SECONDS
```

Render-provided metadata such as `RENDER`, `RENDER_GIT_COMMIT`, and
`RENDER_SERVICE_NAME` may also be used for environment awareness and logging.
Never place production secret values in this documentation.
