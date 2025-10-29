# hydra-core/core/db.py
"""
DB Adapter (async) — Postgres first, SQLite fallback
----------------------------------------------------
Goals:
  • Async SQLAlchemy engine + session factory.
  • Postgres (asyncpg) if DATABASE_URL present; else fallback to SQLite (aiosqlite).
  • FastAPI-friendly: dependency `get_session()` yields an AsyncSession.
  • Self-healing: lazy engine init, connect-retry with jitter/backoff on cold start.
  • Health checks: lightweight `db_health()` and `assert_db_ready()` helpers.
  • Safe defaults: small pool, short timeouts, statement_cache on.

Swap points (future):
  • Replace SQLite fallback with ephemeral DuckDB if desired for batch analytics.
  • Wire Alembic migrations once models stabilize (kept optional for MVP).

Env knobs (read via config.settings.Settings):
  • DATABASE_URL  (e.g., postgres+asyncpg://user:pass@host:5432/dbname)
  • DB_POOL_MIN   (default 1)
  • DB_POOL_MAX   (default 5)
  • DB_CONNECT_TIMEOUT_S (default 3)
"""

from __future__ import annotations

import asyncio
import os
import random
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Tuple

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

# Settings are defined in config/settings.py (we import defensively)
try:
    from config.settings import settings
except Exception:
    # Minimal emergency shim for local runs/tests if settings import fails
    class _Shim:
        DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
        DB_POOL_MIN: int = int(os.getenv("DB_POOL_MIN", "1"))
        DB_POOL_MAX: int = int(os.getenv("DB_POOL_MAX", "5"))
        DB_CONNECT_TIMEOUT_S: int = int(os.getenv("DB_CONNECT_TIMEOUT_S", "3"))
    settings = _Shim()  # type: ignore

# Optional: auto-create tables if models are present
try:
    from .models import Base  # Declarative base
except Exception:
    Base = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────
# Engine factory (lazy singleton)
# ──────────────────────────────────────────────────────────────────────

_ENGINE: Optional[AsyncEngine] = None
_SESSION_FACTORY: Optional[async_sessionmaker[AsyncSession]] = None


def _dsn_and_driver() -> Tuple[str, str]:
    """
    Decide DSN and driver. Prefer Postgres if DATABASE_URL set; else SQLite fallback.
    Returns (dsn, driver_label) for logging/metrics.
    """
    if settings.DATABASE_URL:
        return settings.DATABASE_URL, "postgres"
    # Fallback: file-based SQLite to persist across restarts in MVP
    sqlite_path = os.getenv("SQLITE_PATH", "./.hydra_mvp.sqlite3")
    dsn = f"sqlite+aiosqlite:///{sqlite_path}"
    return dsn, "sqlite"


def _build_engine(dsn: str) -> AsyncEngine:
    # Conservative, cloud-friendly pool sizing
    pool_min = max(0, getattr(settings, "DB_POOL_MIN", 1))
    pool_max = max(pool_min, getattr(settings, "DB_POOL_MAX", 5))
    connect_timeout = max(1, getattr(settings, "DB_CONNECT_TIMEOUT_S", 3))

    # SQLite ignores most pool args; harmless to pass.
    engine = create_async_engine(
        dsn,
        echo=False,
        pool_pre_ping=True,            # proactively test connections
        pool_size=pool_min,
        max_overflow=max(0, pool_max - pool_min),
        connect_args={"timeout": connect_timeout} if dsn.startswith("sqlite") else {},
        execution_options={"schema_translate_map": {}},
    )
    return engine


async def _init_engine_once() -> None:
    global _ENGINE, _SESSION_FACTORY
    if _ENGINE is not None and _SESSION_FACTORY is not None:
        return
    dsn, driver = _dsn_and_driver()
    _ENGINE = _build_engine(dsn)
    _SESSION_FACTORY = async_sessionmaker(
        bind=_ENGINE,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
        class_=AsyncSession,
    )
    # Best-effort: ensure DB reachable and (optionally) create tables
    await _connect_with_backoff()
    if Base is not None:
        await _safe_create_all()


# ──────────────────────────────────────────────────────────────────────
# Backoff connect + health
# ──────────────────────────────────────────────────────────────────────

async def _connect_with_backoff(max_attempts: int = 5) -> None:
    """
    Try to open a trivial connection with jittered exponential backoff.
    Fails fast on SQLite (should always succeed unless file locked).
    """
    assert _ENGINE is not None
    attempt = 0
    last_err: Optional[Exception] = None
    while attempt < max_attempts:
        try:
            async with _ENGINE.connect() as conn:
                await conn.execute(text("SELECT 1"))
                return
        except OperationalError as e:
            last_err = e
            # For SQLite, bubbling early is fine (file locks are developer env issues)
            if _ENGINE.url.get_backend_name().startswith("sqlite"):
                raise
            # Postgres: jittered backoff
            await asyncio.sleep((2 ** attempt) + random.random() * 0.25)
            attempt += 1
    # If we’re here, we failed all attempts
    if last_err:
        raise last_err


async def db_health() -> dict:
    """
    Lightweight health snapshot for /healthz and internal readiness checks.
    """
    await _init_engine_once()
    assert _ENGINE is not None
    try:
        async with _ENGINE.connect() as conn:
            # Simple latency probe
            await conn.execute(text("SELECT 1"))
        return {"db": "ok", "driver": _ENGINE.url.get_backend_name()}
    except Exception as e:
        return {"db": "fail", "driver": _ENGINE.url.get_backend_name(), "error": str(e)}


async def assert_db_ready() -> None:
    """
    Raise if DB is not reachable; used at startup/liveness.
    """
    status = await db_health()
    if status.get("db") != "ok":
        raise RuntimeError(f"DB not ready: {status}")


# ──────────────────────────────────────────────────────────────────────
# Schema bootstrap (optional, MVP convenience)
# ──────────────────────────────────────────────────────────────────────

async def _safe_create_all() -> None:
    """
    Create tables if missing (MVP convenience).
    In production, swap to Alembic migrations.
    """
    if Base is None:
        return
    assert _ENGINE is not None
    # SQLite needs run_sync; PostgreSQL is fine as well.
    async with _ENGINE.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ──────────────────────────────────────────────────────────────────────
# Session dependency + transaction helpers
# ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """
    Manual scope — yields a session and commits/rolls back automatically.
    Useful for scripts and background tasks.
    """
    await _init_engine_once()
    assert _SESSION_FACTORY is not None
    session: AsyncSession = _SESSION_FACTORY()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_session() -> AsyncIterator[AsyncSession]:
    """
    FastAPI dependency:
        from fastapi import Depends
        @app.get("/example")
        async def route(db: AsyncSession = Depends(get_session)):
            ...
    """
    await _init_engine_once()
    assert _SESSION_FACTORY is not None
    session: AsyncSession = _SESSION_FACTORY()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# ──────────────────────────────────────────────────────────────────────
# Shutdown hook
# ──────────────────────────────────────────────────────────────────────

async def shutdown_engine() -> None:
    global _ENGINE
    if _ENGINE is not None:
        await _ENGINE.dispose()
        _ENGINE = None


# ──────────────────────────────────────────────────────────────────────
# Developer helpers
# ──────────────────────────────────────────────────────────────────────

def backend_label() -> str:
    if _ENGINE is None:
        dsn, driver = _dsn_and_driver()
        return driver
    return _ENGINE.url.get_backend_name()


async def ensure_started() -> str:
    """
    Idempotent init + quick health return string ("postgres" | "sqlite").
    Good to call from main.py startup event.
    """
    await _init_engine_once()
    assert _ENGINE is not None
    return _ENGINE.url.get_backend_name()
