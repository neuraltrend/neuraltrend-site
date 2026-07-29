# NeuralTrend Architecture Decisions

This log records important choices and the reasoning behind them. Update it when
a future change deliberately reverses or materially modifies a decision.

## ADR-001 — Approved Forward Record files are the publication source of truth

**Status:** Accepted

### Context

The large working CSVs power Signal Overview and may occasionally be corrected
or regenerated. Previously, publication preview compared already-approved rows
against the working CSV and blocked when `Date`, `Close`, or signal values
changed.

### Decision

After a date is approved, the compact approved CSV on persistent disk is
authoritative for public Methodology/Performance history. Normal publication
preview ignores later changes or deletions for those dates in the working CSV
and considers only dates after the latest approved date.

### Consequences

- Signal Overview history can be corrected without rewriting public
  performance.
- Published history remains stable and reproducible.
- Direct corruption/tampering of the approved compact CSV must still be blocked
  through digest/metadata checks.
- A genuine published-history correction requires a separate explicit audited
  workflow.

## ADR-002 — Forward Record files and PostgreSQL are backed up independently

**Status:** Accepted

### Context

PostgreSQL contains lifecycle metadata, while approved publication rows live on
persistent disk.

### Decision

Create one PostgreSQL custom-format backup and one Forward Record archive, each
with a SHA-256 checksum. Treat a close-in-time verified pair as the normal unit
of recovery.

### Consequences

- Either backup alone is incomplete for full recovery.
- Backup UI and documentation must always present both types.
- Recovery validation must check database/file consistency.

## ADR-003 — No one-click production restore in the browser

**Status:** Accepted

### Context

A restore can destroy newer customer activity or replace valid publication
history. Browser actions are easier to trigger accidentally than controlled
shell procedures.

### Decision

The admin UI may create, verify, download, retain, and delete backups, and may
show recovery readiness. Production restore remains an explicit shell-driven
procedure with staging rehearsal and confirmation.

### Consequences

- Recovery is slower but safer.
- Operators must retain command-line access and current documentation.
- Restore scripts should default to dry-run behavior and require explicit
  confirmation for application.

## ADR-004 — Admin tools are grouped under one operations hub

**Status:** Accepted

### Context

Health, alerts, publication, backups, and recovery controls became difficult to
find when exposed as unrelated links.

### Decision

The authenticated username menu contains one **Admin Operations** entry leading
to a hub for Health & Operations, Alerts & Forward Record, Backups, and Recovery.

### Consequences

- Administrator navigation is consistent.
- Access-control testing can focus on the `/admin/*` route group.
- New operational pages should be linked through the hub rather than scattered
  through the customer UI.

## ADR-005 — `/healthz` is intentionally narrow and fast

**Status:** Accepted

### Context

Render requires a fast, reliable health check. Dashboard and simulation routes
may perform expensive calculations and can be slow without the service being
fundamentally unhealthy.

### Decision

Use `/healthz` for Render health checks. Do not put expensive Signal Overview,
email, Stripe, or full business-flow validation into the endpoint.

### Consequences

- A green health check does not prove every feature works.
- Operational, recovery, smoke, and manual checks remain necessary.
- Heavy route performance must be monitored separately.

## ADR-006 — Separate operational checks by purpose

**Status:** Accepted

### Context

One monolithic check would be difficult to use safely across local, CI,
post-deployment, and incident contexts.

### Decision

Maintain separate tools:

- prelaunch for configuration/readiness;
- operational for live state and warnings;
- recovery for route/init validation;
- production smoke for read-only end-to-end checks.

### Consequences

- Operators can run the appropriate depth of check.
- Post-deployment procedure runs all four in a fixed order.
- Each script should remain focused and avoid destructive behavior.

## ADR-007 — Backups must also leave Render

**Status:** Accepted

### Context

A backup on the same provider/disk may be lost with the production environment.

### Decision

Keep managed on-disk backups for convenience and quick recovery, but regularly
download verified backup/checksum pairs to protected off-Render storage.

### Consequences

- Backup completion includes download and independent storage.
- Retention settings alone do not satisfy disaster recovery.
- Access controls and encryption are required for stored copies.

## ADR-008 — Code rollback and data restore are separate procedures

**Status:** Accepted

### Context

A failed deployment may not have damaged data, while data corruption may occur
without a bad code release.

### Decision

Document and execute code rollback, PostgreSQL restore, and Forward Record
restore as separate operations. Combine them only when incident evidence
requires it.

### Consequences

- Operators must identify the failure boundary before acting.
- Migration state must be checked before rollback.
- An older data restore is never an automatic consequence of a code rollback.

## ADR-009 — Administrator state changes require explicit protection

**Status:** Accepted

### Context

Backup creation/deletion, alert sending, and publication are sensitive actions.

### Decision

Require authenticated administrator authorization, CSRF protection for browser
POST actions, rate limits where appropriate, strict filename/path handling, and
clear confirmation boundaries.

### Consequences

- Non-admin users cannot use operational routes.
- GET requests remain non-destructive.
- Path traversal and accidental repeated actions must be tested.

## Updating this log

For a new decision, record:

1. identifier and title;
2. status;
3. context/problem;
4. decision;
5. consequences;
6. replacement decision when superseded.
