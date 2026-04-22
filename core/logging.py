"""Structured logging setup.

Call ``configure_logging()`` once at process start (API lifespan, collector main,
dashboard bootstrap). Subsequent calls are idempotent.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from core.config import get_settings

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Minimal JSON line formatter — one object per log record.

    Keeps structured fields (``extra=...``) alongside message/level/logger so
    downstream log collectors can parse without a custom grok pattern.
    """

    _RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in self._RESERVED or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = repr(value)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    """Configure root logger once; safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")
        )
    root.addHandler(handler)

    # Quieten noisy third-party loggers at INFO; they log at DEBUG if needed.
    for noisy in ("httpx", "httpcore", "urllib3", "sqlalchemy.engine.Engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring the root first if needed."""
    configure_logging()
    return logging.getLogger(name)
