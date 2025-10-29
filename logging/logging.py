# hydra-core/logging/logging.py
"""
Hydra Structured Logging
------------------------
Goals:
  • Single, consistent JSON logger across app, jobs, and tests.
  • Request-scoped context: tenant_id, req_id, span_id, route, ip.
  • PII safety: redact common fields (email, ssn, pan, phone).
  • Uvicorn/Starlette integration (optional): unify access/error logs.
  • Zero-config sane defaults; override via HYDRA_LOG_* envs.

Usage:
  from logging.logging import get_logger, bind_context, clear_context, LoggingMiddleware
  log = get_logger(__name__)
  log.info("dfi_score_computed", extra={"loan_id":"LN-12"})  # redacts PII automatically

  # FastAPI (in main.py):
  #   app.add_middleware(LoggingMiddleware)

Env:
  HYDRA_LOG_LEVEL=INFO|DEBUG|WARNING|ERROR (default INFO)
  HYDRA_ENV=sandbox|dev|prod (used in record)
  HYDRA_LOG_PRETTY=true|false (local dev pretty-print)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from contextvars import ContextVar

# ─────────────────────────────────────────────────────────────────────
# Context variables (propagate per request / task)
# ─────────────────────────────────────────────────────────────────────
_ctx_req_id: ContextVar[str] = ContextVar("req_id", default="")
_ctx_span_id: ContextVar[str] = ContextVar("span_id", default="")
_ctx_tenant: ContextVar[str] = ContextVar("tenant_id", default="")
_ctx_route: ContextVar[str] = ContextVar("route", default="")
_ctx_ip: ContextVar[str] = ContextVar("ip", default="")

def bind_context(
    *,
    req_id: Optional[str] = None,
    span_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    route: Optional[str] = None,
    ip: Optional[str] = None,
) -> None:
    if req_id is not None:
        _ctx_req_id.set(req_id)
    if span_id is not None:
        _ctx_span_id.set(span_id)
    if tenant_id is not None:
        _ctx_tenant.set(tenant_id)
    if route is not None:
        _ctx_route.set(route)
    if ip is not None:
        _ctx_ip.set(ip)

def clear_context() -> None:
    bind_context(req_id="", span_id="", tenant_id="", route="", ip="")

def new_ids() -> None:
    """Convenience: assign fresh req_id/span_id if missing."""
    if not _ctx_req_id.get():
        _ctx_req_id.set(str(uuid.uuid4()))
    _ctx_span_id.set(str(uuid.uuid4()))

# ─────────────────────────────────────────────────────────────────────
# PII redaction
# ─────────────────────────────────────────────────────────────────────
_PII_PATTERNS = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("pan", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),  # card PAN (loose)
    ("phone", re.compile(r"\+?\d[\d \-\(\)]{7,}\d")),
]

def _redact_val(v: Any) -> Any:
    if isinstance(v, str):
        redacted = v
        for _name, pat in _PII_PATTERNS:
            redacted = pat.sub("<REDACTED>", redacted)
        return redacted
    if isinstance(v, dict):
        return {k: _redact_val(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_redact_val(x) for x in v]
    return v

REDACT_KEYS = {
    "ssn", "social", "pan", "card", "card_number", "cvv", "cvc",
    "phone", "email", "dob", "address", "tax_id",
}

def _redact_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for k, v in d.items():
        if k.lower() in REDACT_KEYS:
            safe[k] = "<REDACTED>"
        else:
            safe[k] = _redact_val(v)
    return safe

# ─────────────────────────────────────────────────────────────────────
# JSON Formatter
# ─────────────────────────────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    def __init__(self, *, pretty: bool = False) -> None:
        super().__init__()
        self.pretty = pretty
        self.env = (os.getenv("HYDRA_ENV") or "sandbox").lower()
        self.service = "hydra-core"

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload: Dict[str, Any] = {
            "ts": ts,
            "lvl": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
            "env": self.env,
            "service": self.service,
            "pid": os.getpid(),
            "req_id": _ctx_req_id.get(),
            "span_id": _ctx_span_id.get(),
            "tenant_id": _ctx_tenant.get(),
            "route": _ctx_route.get(),
            "ip": _ctx_ip.get(),
            "file": f"{record.pathname}:{record.lineno}",
            "func": record.funcName,
        }

        # Merge extras (record.__dict__ includes args; filter standard attrs)
        # Only include dict-like "extra" provided by log calls via 'extra'
        extras: Dict[str, Any] = {}
        for k, v in record.__dict__.items():
            if k in (
                "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process",
            ):
                continue
            # Accept simple types and dicts/lists for JSON
            if isinstance(v, (str, int, float, bool, dict, list)) or v is None:
                extras[k] = v

        if extras:
            payload["extra"] = _redact_dict(extras)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        if self.pretty:
            return json.dumps(payload, indent=2, ensure_ascii=False)
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

# ─────────────────────────────────────────────────────────────────────
# Logger factory
# ─────────────────────────────────────────────────────────────────────
_LOGGERS: Dict[str, logging.Logger] = {}
_LEVEL = os.getenv("HYDRA_LOG_LEVEL", "INFO").upper()
_PRETTY = os.getenv("HYDRA_LOG_PRETTY", "false").lower() in ("1", "true", "yes")

def _base_handler() -> logging.Handler:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter(pretty=_PRETTY))
    return handler

def get_logger(name: str = "hydra") -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, _LEVEL, logging.INFO))
    # Avoid duplicate handlers in reloads
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(_base_handler())
    _LOGGERS[name] = logger
    return logger

# Root app logger shortcut
log = get_logger("hydra")

# ─────────────────────────────────────────────────────────────────────
# Uvicorn/Starlette integration (optional)
# ─────────────────────────────────────────────────────────────────────
def patch_uvicorn_loggers() -> None:
    """
    Optionally call in main.py to align uvicorn logs to JSON.
    """
    # Access log
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.setLevel(getattr(logging, _LEVEL, logging.INFO))
    access.propagate = False
    access.addHandler(_base_handler())

    # Error log
    uv = logging.getLogger("uvicorn")
    uv.handlers.clear()
    uv.setLevel(getattr(logging, _LEVEL, logging.INFO))
    uv.propagate = False
    uv.addHandler(_base_handler())

# ─────────────────────────────────────────────────────────────────────
# FastAPI middleware for context + access log
# ─────────────────────────────────────────────────────────────────────
try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response
except Exception:  # pragma: no cover - middleware is optional
    BaseHTTPMiddleware = object  # type: ignore
    Request = object  # type: ignore
    Response = object  # type: ignore

class LoggingMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]
    """
    Injects req_id/span_id/tenant/route/ip into contextvars.
    Emits a single access log JSON per request with timing and status.
    """
    def __init__(self, app, tenant_header: str = "x-tenant-id") -> None:
        super().__init__(app)
        self.tenant_header = tenant_header

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        started = time.perf_counter()
        try:
            # Bind context
            tenant_id = request.headers.get(self.tenant_header, "") or request.query_params.get("tenant_id", "")
            route = getattr(request.scope.get("route"), "path", request.url.path)
            ip = request.client.host if request.client else ""
            new_ids()
            bind_context(tenant_id=tenant_id, route=route, ip=ip)

            # Proceed
            response: Response = await call_next(request)

            # Access log
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            log.info(
                "http_access",
                extra={
                    "method": request.method,
                    "path": route,
                    "status": getattr(response, "status_code", 0),
                    "elapsed_ms": elapsed_ms,
                    "tenant_id": tenant_id or None,
                },
            )
            return response
        except Exception as e:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            log.error(
                "http_error",
                extra={
                    "method": getattr(request, "method", None),
                    "path": getattr(getattr(request, "url", None), "path", None),
                    "elapsed_ms": elapsed_ms,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise
        finally:
            # Always clear to avoid context bleed across tasks
            clear_context()
