"""
HydraCore MVP – Main Entry Point
--------------------------------
This is the front door of the Hydra monolith.
- Boots the app
- Wires up logging, telemetry, chaos, DB, and routes
- Runs health/self-checks at startup
- Sandbox + enterprise safe out the gate
"""

import asyncio
import uvicorn
from fastapi import FastAPI
from prometheus_client import make_asgi_app
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Hydra imports (internal modules)
from config.settings import settings
from logging.logging import init_logger, log_event
from telemetry.metrics import init_metrics, telemetry_span
from chaos.chaos import chaos_inject
from core.supervisor import supervisor_task
from api.routes import router as api_router

# Initialize FastAPI app
app = FastAPI(
    title="HydraCore MVP",
    description="Lean, surgical, enterprise-grade core proving FI impact",
    version="0.1.0",
    docs_url="/docs" if settings.env != "production" else None,
    redoc_url="/redoc" if settings.env != "production" else None,
)

# Attach Prometheus metrics endpoint at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Add CORS middleware (safe defaults for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware for chaos + telemetry
class HydraMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        trace_id = f"trace-{asyncio.get_event_loop().time()}"
        tenant = request.headers.get("X-Tenant", "sandbox")

        # Chaos injection
        try:
            await chaos_inject(subsystem="request")
        except Exception as e:
            log_event("chaos_warning", level="warning", tenant=tenant, trace_id=trace_id, error=str(e))

        # Telemetry span
        async with telemetry_span("request", tenant=tenant, trace_id=trace_id):
            response = await call_next(request)
            response.headers["X-Trace-ID"] = trace_id
            return response

app.add_middleware(HydraMiddleware)

# Health check route
@app.get("/healthz")
async def health_check():
    """Simple health check for k8s/docker/ICE sandbox."""
    return {"status": "ok", "env": settings.env}

# Register API routes
app.include_router(api_router, prefix="/api")

# Startup tasks
@app.on_event("startup")
async def on_startup():
    # Init logging
    init_logger()
    log_event("startup", level="info", message="HydraCore starting up")

    # Init telemetry
    init_metrics()

    # Kick off supervisor (self-healing orchestrator)
    asyncio.create_task(supervisor_task())

    log_event("startup_complete", level="info", message="HydraCore is ready")

# Shutdown tasks
@app.on_event("shutdown")
async def on_shutdown():
    log_event("shutdown", level="info", message="HydraCore shutting down gracefully")

# Entrypoint for local dev
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
