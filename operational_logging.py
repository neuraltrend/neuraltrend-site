"""Production-safe request correlation and log redaction for NeuralTrend.

The module intentionally has no application-specific imports. It can be loaded
before the database and other Flask extensions are initialized.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import sys
import time
from datetime import datetime, timezone
from typing import Any

from flask import g, has_request_context, request


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_DATABASE_URL_RE = re.compile(
    r"(?P<scheme>postgres(?:ql)?://[^:\s/]+):[^@\s]+@",
    re.I,
)
_STRIPE_SECRET_RE = re.compile(
    r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9_]+\b|\bwhsec_[A-Za-z0-9_]+\b"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_KEY_VALUE_SECRET_RE = re.compile(
    r"(?i)\b(password|passwd|secret|token|api[_-]?key|authorization)"
    r"\s*([:=])\s*([^\s,;&]+)"
)

_SENSITIVE_PATH_PREFIXES = (
    "/verify/",
    "/reset-password/",
    "/confirm-delete/",
    "/unsubscribe-signal-alerts/",
)


def redact_text(value: Any) -> str:
    """Return a log-safe representation without common credentials or email."""
    text = str(value)
    text = _DATABASE_URL_RE.sub(r"\g<scheme>:[REDACTED]@", text)
    text = _STRIPE_SECRET_RE.sub("[REDACTED_STRIPE_SECRET]", text)
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _KEY_VALUE_SECRET_RE.sub(r"\1\2[REDACTED]", text)
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    return text


def safe_request_path() -> str | None:
    if not has_request_context():
        return None

    path = request.path or "/"
    for prefix in _SENSITIVE_PATH_PREFIXES:
        if path.startswith(prefix):
            return prefix + "[REDACTED]"
    return path


def current_request_id() -> str | None:
    if not has_request_context():
        return None
    return getattr(g, "nt_request_id", None)


def _current_user_id() -> int | None:
    if not has_request_context():
        return None
    try:
        from flask_login import current_user

        if current_user.is_authenticated:
            return int(current_user.get_id())
    except Exception:
        return None
    return None


class NeuralTrendFormatter(logging.Formatter):
    """Text or JSON formatter with request context and final-output redaction."""

    def __init__(self, *, json_mode: bool = False) -> None:
        super().__init__()
        self.json_mode = json_mode

    def format(self, record: logging.LogRecord) -> str:
        message = redact_text(record.getMessage())
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
            "request_id": current_request_id(),
            "method": request.method if has_request_context() else None,
            "path": safe_request_path(),
            "endpoint": request.endpoint if has_request_context() else None,
            "user_id": _current_user_id(),
        }

        if record.exc_info:
            payload["exception"] = redact_text(
                self.formatException(record.exc_info)
            )

        if self.json_mode:
            return json.dumps(payload, sort_keys=True, separators=(",", ":"))

        parts = [
            payload["timestamp"],
            f"level={payload['level']}",
            f"logger={payload['logger']}",
        ]
        for key in ("request_id", "method", "path", "endpoint", "user_id"):
            value = payload[key]
            if value is not None:
                parts.append(f"{key}={value}")
        parts.append(f"message={message}")
        if payload.get("exception"):
            parts.append(f"exception={payload['exception']}")
        return " ".join(parts)


def configure_operational_logging(app, *, testing_mode: bool = False) -> None:
    """Configure one stdout handler for application logs.

    Set ``NEURALTREND_LOG_FORMAT=json`` for one-line JSON logs. The default is
    human-readable key/value text. ``NEURALTREND_LOG_LEVEL`` accepts standard
    Python levels such as INFO or WARNING.
    """
    level_name = os.environ.get("NEURALTREND_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    json_mode = os.environ.get("NEURALTREND_LOG_FORMAT", "text").lower() == "json"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(NeuralTrendFormatter(json_mode=json_mode))
    handler.setLevel(level)

    app.logger.handlers.clear()
    app.logger.addHandler(handler)
    app.logger.setLevel(level)
    app.logger.propagate = False

    if testing_mode:
        app.logger.setLevel(logging.WARNING)


def register_request_observability(app) -> None:
    """Add request IDs and focused error/slow-request logging."""
    try:
        slow_seconds = max(
            0.1,
            float(os.environ.get("NEURALTREND_SLOW_REQUEST_SECONDS", "2.0")),
        )
    except ValueError:
        slow_seconds = 2.0

    @app.before_request
    def _start_request_observability():
        g.nt_request_id = secrets.token_hex(8)
        g.nt_request_started_at = time.perf_counter()

    @app.after_request
    def _finish_request_observability(response):
        request_id = getattr(g, "nt_request_id", None)
        if request_id:
            response.headers["X-Request-ID"] = request_id

        started = getattr(g, "nt_request_started_at", None)
        elapsed = time.perf_counter() - started if started is not None else 0.0

        # Successful Render probes are intentionally silent. Failed probes are
        # logged like all other server errors.
        if request.path == "/healthz" and response.status_code < 500:
            return response

        message = (
            "request_completed status=%s duration_ms=%.1f content_length=%s"
        )
        args = (
            response.status_code,
            elapsed * 1000.0,
            response.content_length,
        )

        if response.status_code >= 500:
            app.logger.error(message, *args)
        elif response.status_code >= 400:
            app.logger.warning(message, *args)
        elif elapsed >= slow_seconds:
            app.logger.warning("slow_" + message, *args)

        return response
