# Render Monitoring Setup

## Render health check

Keep this production setting enabled:

```text
Health Check Path: /healthz
```

Use `/healthz`, not the homepage, for Render's service health check. The endpoint
should remain fast and must not perform expensive dashboard calculations.

## Notifications and external monitoring

Enable Render notifications for:

- failed deployments;
- unhealthy services;
- unexpected restarts;
- available disk/resource alerts supported by the account.

Use an external uptime monitor for at least:

```text
https://neuraltrend.org/
https://neuraltrend.org/healthz
```

Do not monitor `/admin/operations`, `/admin/backups`, or `/admin/recovery`
externally because they require authentication and are intended for manual
admin review.

## Admin review

For manual operational review, sign in and choose **Admin Operations** from the
username menu. Use the hub to review:

- Health & Operations
- Alerts & Forward Record
- Backups
- Recovery

## Structured logs

Optional Render environment variables:

```text
NEURALTREND_LOG_FORMAT=json
NEURALTREND_LOG_LEVEL=INFO
NEURALTREND_SLOW_REQUEST_SECONDS=2.0
```

After changing environment variables, redeploy or restart and run:

```bash
python tools/prelaunch_check.py
python tools/operational_check.py --strict
python tools/recovery_check.py https://neuraltrend.org
```

## Deployment stability review

After deployment, inspect Render logs and metrics for:

- worker startup and readiness;
- `WORKER TIMEOUT`;
- worker exits or SIGKILL/OOM;
- repeated 500/502 responses;
- slow `/live-simulations` requests;
- CPU and memory pressure;
- health-check failures or restart loops.

Repeated browser refreshes can increase request queueing while a slow endpoint
is already running. Use logs and request IDs rather than repeated refreshes to
diagnose startup problems.
