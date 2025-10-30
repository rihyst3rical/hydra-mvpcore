# hydra-core/main.py
"""
HydraCore MVP — FastAPI entrypoint
Focus: DFI (Dynamic Fragility Index) + AFPS (Adaptive Funding Probability Signal)
Design goals:
  - Simple, surgical, explainable.
  - Enterprise-grade scaffolding (auth, audit, metrics, health, self-healing).
  - Sandbox-safe: no cross-tenant leaks; deterministic IDs for replays.
  - Ready for ICE EPC sandbox demos (OpenAPI snapshot, JSON logging, Prometheus).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError

# ──────────────────────────────────────────────────────────────────────────────
# External observability (Prometheus) — lightweight, no starlette-exporter req.
# ──────────────────────────────────────────────────────────────────────────────
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ──────────────────────────────────────────────────────────────────────────────
# Internal imports (we'll create these next, file-by-file)
# If you run main.py before creating them, import errors are expected.
# ──────────────────────────────────────────────────────────────────────────────
try:
    from config.settings import get_settings, Settings
    from core.auth import verify_tenant  # FastAPI dependency → returns tenant dict
    from core.db import init_db, db_healthcheck, get_async_session  # adapters
    from core.governance import audit_hash, record_decision  # compliance helpers
    from core.fi import compute_dfi  # core metric #1
    from core.afps import compute_afps  # core metric #2
    from core.hydra_voice import narrate_event  # plain-English narrative
    from telemetry.exporter import bind_tracing_context  # trace id, span stubs
    from logging.logging import get_logger  # structured JSON logger
    from chaos.chaos import chaos_inject  # safe chaos toggles (sandbox)
    from utils.utils import deterministic_uuid, gen_trace_id, retry_async
except Exception as _e:
    # Lightweight fallback for local file creation order.
    # We fail fast on execution, but allow `main.py` to be saved and committed.
    # Once the modules exist, this block never runs.
    print(f"[BOOT] Module hint: {_e}. This is expected until we add other files.", file=sys.stderr)

# ──────────────────────────────────────────────────────────────────────────────
# Prometheus registry + app-level metrics
# ──────────────────────────────────────────────────────────────────────────────
PROM_REGISTRY = CollectorRegistry(auto_describe=True)

REQ_COUNT = Counter(
    "hydra_http_requests_total",
    "Total HTTP requests",
    ["path", "method", "status"],
    registry=PROM_REGISTRY,
)

REQ_LATENCY = Histogram(
    "hydra_http_request_latency_seconds",
    "HTTP request latency",
    ["path", "method"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
    registry=PROM_REGISTRY,
)

DFI_COMPUTED = Counter(
    "hydra_dfi_computed_total",
    "Number of DFI computations",
    ["tenant"],
    registry=PROM_REGISTRY,
)

AFPS_COMPUTED = Counter(
    "hydra_afps_computed_total",
    "Number of AFPS computations",
    ["tenant"],
    registry=PROM_REGISTRY,
)

HEALTH_GAUGE = Gauge(
    "hydra_subsystem_health",
    "Subsystem health (1=ok,0=fail)",
    ["component"],
    registry=PROM_REGISTRY,
)

SELF_HEALING_ACTIONS = Counter(
    "hydra_self_heal_actions_total",
    "Count of self-healing actions invoked",
    ["kind"],
    registry=PROM_REGISTRY,
)

# ──────────────────────────────────────────────────────────────────────────────
# App factory with lifespan (startup/shutdown) for clean orchestration
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Settings + logger
    settings: Settings = get_settings()
    logger = get_logger(service="hydra-core", environment=settings.ENV)

    logger.info(
        "boot_start",
        extra={"env": settings.ENV, "version": settings.VERSION, "mode": settings.MODE},
    )

    # Initialize DB and mark health
    try:
        await init_db(settings)
        HEALTH_GAUGE.labels(component="db").set(1)
    except Exception as e:
        HEALTH_GAUGE.labels(component="db").set(0)
        logger.exception("db_init_failed", extra={"error": str(e)})
        # In sandbox we keep running to allow /healthz to reflect failure.
        # In prod you might raise here to fail fast.

    # Start background self-healing supervisor
    app.state.supervisor_task = asyncio.create_task(_supervisor_loop(app, settings, logger))

    logger.info("boot_complete")
    try:
        yield
    finally:
        logger.info("shutdown_start")
        # Cancel supervisor
        task = getattr(app.state, "supervisor_task", None)
        if task:
            task.cancel()
            with contextlib_suppress(asyncio.CancelledError):
                await task
        logger.info("shutdown_complete")


def create_app() -> FastAPI:
    settings: Settings = get_settings()

    app = FastAPI(
        title="HydraCore MVP",
        description="DFI + AFPS with narrative + audit. Sandbox-safe, enterprise-ready.",
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS (safe default; you can restrict on pilot)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach metrics middleware (micro)
    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next):
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
            status = getattr(response, "status_code", 500)
            return response
        finally:
            path = request.url.path
            method = request.method
            REQ_LATENCY.labels(path=path, method=method).observe(time.perf_counter() - start)
            REQ_COUNT.labels(path=path, method=method, status=str(status)).inc()

    # Exception handlers → consistent JSON + metrics do not break
    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": "validation_error", "details": json.loads(exc.json())},
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(_: Request, exc: Exception):
        # We keep this generic to avoid leaking internals
        return JSONResponse(status_code=500, content={"error": "internal_server_error"})

    # Routes
    _wire_routes(app)

    return app


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
def _wire_routes(app: FastAPI) -> None:
    settings: Settings = get_settings()
    logger = get_logger(service="hydra-core", environment=settings.ENV)

    @app.get("/healthz", tags=["ops"])
    async def healthz():
        # DB health only for now (extend later as needed)
        ok = await db_healthcheck()
        HEALTH_GAUGE.labels(component="db").set(1 if ok else 0)
        return {"status": "ok" if ok else "degraded", "components": {"db": ok}}

    @app.get("/readyz", tags=["ops"])
    async def readyz():
        # In MVP, ready when DB is reachable
        ok = await db_healthcheck()
        return {"ready": bool(ok)}

    @app.get("/metrics", tags=["ops"])
    async def metrics():
        return PlainTextResponse(generate_latest(PROM_REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.get("/spec/v1", tags=["ops"])
    async def spec_v1():
        """
        Version-pinned OpenAPI snapshot for auditors / SDKs.
        (In MVP this simply returns the live schema.)
        """
        return app.openapi()

    # ---- Core MVP: DFI ------------------------------------------------------
    @app.post("/v1/dfi/score", tags=["dfi"])
    async def dfi_score(
        payload: Dict[str, Any],
        request: Request,
        bg: BackgroundTasks,
        tenant=Depends(verify_tenant),
    ):
        """
        Input: normalized loan/process event payload (tenant-scoped).
        Output: DFI score + drivers + narrative + audit hash.
        """
        settings = get_settings()
        logger = get_logger(service="hydra-core", environment=settings.ENV)

        # Deterministic id for replays + fresh trace id
        run_id = deterministic_uuid(payload)
        trace_id = gen_trace_id()

        # Bind context for logs/traces
        bind_tracing_context(trace_id=trace_id, tenant_id=tenant["id"], run_id=run_id)

        # Optional safe chaos in sandbox
        await chaos_inject(kind="dfi_pre", enabled=settings.CHAOS_ENABLED, rate=0.02)

        # Compute DFI (core/fi.py) — quant-light, explainable
        try:
            result = await compute_dfi(payload, tenant=tenant, trace_id=trace_id)
        except Exception as e:
            logger.exception("dfi_compute_failed", extra={"tenant": tenant["id"], "run_id": run_id, "err": str(e)})
            raise HTTPException(500, "dfi_compute_failed")

        # Narrative (plain-English, negative visualization option)
        summary = narrate_event(
            metric="DFI",
            score=result.get("dfi_score"),
            drivers=result.get("drivers", []),
            what_if=result.get("what_if", {}),
            money_impact_hint=result.get("money_impact_hint"),
        )

        # Governance: immutable audit hash of the response envelope
        envelope = {
            "metric": "DFI",
            "tenant": tenant["id"],
            "trace_id": trace_id,
            "run_id": run_id,
            "result": result,
            "summary": summary,
            "ts": int(time.time() * 1000),
            "version": settings.VERSION,
        }
        envelope["audit_hash"] = audit_hash(envelope)

        # Async record (non-blocking)
        bg.add_task(record_decision, envelope)

        # Metrics
        DFI_COMPUTED.labels(tenant=tenant["id"]).inc()

        return envelope

    # ---- Core MVP: AFPS -----------------------------------------------------
    @app.post("/v1/afps/score", tags=["afps"])
    async def afps_score(
        payload: Dict[str, Any],
        request: Request,
        bg: BackgroundTasks,
        tenant=Depends(verify_tenant),
    ):
        """
        Input: same normalized payload (includes minimal tempo fields).
        Output: AFPS score (probability to fund in window) + narrative + audit hash.
        """
        settings = get_settings()
        logger = get_logger(service="hydra-core", environment=settings.ENV)

        run_id = deterministic_uuid(payload)
        trace_id = gen_trace_id()
        bind_tracing_context(trace_id=trace_id, tenant_id=tenant["id"], run_id=run_id)

        await chaos_inject(kind="afps_pre", enabled=settings.CHAOS_ENABLED, rate=0.02)

        try:
            result = await compute_afps(payload, tenant=tenant, trace_id=trace_id)
        except Exception as e:
            logger.exception("afps_compute_failed", extra={"tenant": tenant["id"], "run_id": run_id, "err": str(e)})
            raise HTTPException(500, "afps_compute_failed")

        summary = narrate_event(
            metric="AFPS",
            score=result.get("afps_score"),
            drivers=result.get("drivers", []),
            what_if=result.get("what_if", {}),
            money_impact_hint=result.get("money_impact_hint"),
        )

        envelope = {
            "metric": "AFPS",
            "tenant": tenant["id"],
            "trace_id": trace_id,
            "run_id": run_id,
            "result": result,
            "summary": summary,
            "ts": int(time.time() * 1000),
            "version": settings.VERSION,
        }
        envelope["audit_hash"] = audit_hash(envelope)

        bg.add_task(record_decision, envelope)
        AFPS_COMPUTED.labels(tenant=tenant["id"]).inc()

        return envelope


# ──────────────────────────────────────────────────────────────────────────────
# Background self-healing supervisor (kept tiny on purpose)
#   - pings DB every N seconds
#   - if degraded → attempts a soft reconnect
#   - increments self-heal metrics
#   - logs state transitions (green → yellow → green)
# ──────────────────────────────────────────────────────────────────────────────
async def _supervisor_loop(app: FastAPI, settings: "Settings", logger):
    probe_int = int(os.getenv("HYDRA_SUPERVISOR_INTERVAL_SEC", "15"))
    degraded = False

    while True:
        try:
            ok = await db_healthcheck()
            HEALTH_GAUGE.labels(component="db").set(1 if ok else 0)

            if not ok:
                if not degraded:
                    logger.warning("db_degraded_detected")
                degraded = True
                # Attempt soft heal (re-init connection pool)
                try:
                    await init_db(settings)
                    SELF_HEALING_ACTIONS.labels(kind="db_reinit").inc()
                    logger.info("db_soft_heal_success")
                    degraded = False
                except Exception as e:
                    logger.error("db_soft_heal_failed", extra={"err": str(e)})
            else:
                if degraded:
                    # Recovered
                    logger.info("db_recovered")
                    degraded = False

        except asyncio.CancelledError:
            # Shutdown path
            break
        except Exception as e:
            logger.exception("supervisor_loop_error", extra={"err": str(e)})

        await asyncio.sleep(probe_int)


# ──────────────────────────────────────────────────────────────────────────────
# Safe contextlib.suppress for Python 3.11+
# ──────────────────────────────────────────────────────────────────────────────
class contextlib_suppress:
    def __init__(self, *exceptions):
        self.exceptions = exceptions

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, self.exceptions)


# ──────────────────────────────────────────────────────────────────────────────
# Uvicorn entrypoint (optional for local run)
# In EPC you’ll deploy under the platform’s runtime, e.g., gunicorn/uvicorn worker
# ──────────────────────────────────────────────────────────────────────────────
app = create_app()

if __name__ == "__main__":
    # Local dev runner: python main.py
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)

from config.settings import get_settings
