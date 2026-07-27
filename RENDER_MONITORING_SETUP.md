# Render Monitoring Setup

Keep these production settings enabled:

```text
Health Check Path: /healthz
```

Enable Render notifications for failed deploys and unhealthy/restarted services
using the notification channels available to the account. Use an external
uptime monitor for at least the homepage and `/healthz` so outages are observed
from outside Render as well.

Suggested endpoints:

```text
https://neuraltrend.org/
https://neuraltrend.org/healthz
```

Do not monitor `/admin/operations` externally because it requires login and is
intended for manual admin review.

For structured Render logs, set:

```text
NEURALTREND_LOG_FORMAT=json
NEURALTREND_LOG_LEVEL=INFO
NEURALTREND_SLOW_REQUEST_SECONDS=2.0
```

After changing environment variables, redeploy/restart and run the recovery
check.
