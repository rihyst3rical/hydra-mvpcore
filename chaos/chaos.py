# hydra-core/chaos/chaos.py
"""
Hydra Chaos & Sandbox Utilities
-------------------------------
Purpose:
  • Let us simulate flaky reality (latency, faults, dependency hiccups) ONLY in sandbox/dev.
  • Never harms prod: hard env + flag gate. Safe by default.
  • Deterministic when seeded; observable via hydra logger.
  • Decorators for easy, optional use on endpoints or internal sections.
  • Tiny circuit-breaker to model degraded dependencies and test fallbacks.

Environment Flags (all optional):
  HYDRA_ENV=sandbox|dev|prod           # default: sandbox
  CHAOS_ENABLED=true|false             # default: false
  CHAOS_LATENCY_MS_MIN=10              # min injected latency
  CHAOS_LATENCY_MS_MAX=250             # max injected latency
  CHAOS_ERROR_RATE=0.0..1.0            # probability to raise synthetic error
  CHAOS_SEED=integer                   # reproducible chaos
  CHAOS_FUZZ_RATE=0.0..1.0             # probability to fuzz input payloads
  CHAOS_BREAKER_THRESHOLD=5            # failures to trip breaker
  CHAOS_BREAKER_COOLDOWN_S=20          # time before half-open

Usage (FastAPI endpoint):

  from chaos.chaos import chaos_endpoint, CHAOS
  @router.post("/fi/score")
  @chaos_endpoint(name="fi_score")  # NOOP unless enabled + sandbox/dev
  async def fi_score(payload: dict): ...

Usage (internal function):

  from chaos.chaos import chaos_section
  @chaos_section("db_write")
  def write_row(...): ...

Manual injection:

  CHAOS.maybe_delay()
  CHAOS.maybe_fail("downstream_db")
  payload = CHAOS.maybe_fuzz(payload)

Observability:
  Uses Hydra structured logger (`logging/logging.py`) for JSON events.
"""

from __future__ import annotations

import os
import time
import random
import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timedelta

try:
    # Optional; only used when wrapping FastAPI endpoints
    from fastapi import HTTPException
except Exception:  # pragma: no cover
    HTTPException = None  # type: ignore

try:
    # Our app logger (safe import even if middleware isn’t installed)
    from logging.logging import get_logger
    log = get_logger("hydra.chaos")
except Exception:  # pragma: no cover
    # super-fallback
    class _Simple:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
    log = _Simple()  # type: ignore


# ─────────────────────────────────────────────────────────────────────
# Config & gating
# ─────────────────────────────────────────────────────────────────────

def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default


@dataclass
class ChaosConfig:
    env: str = field(default_factory=lambda: (os.getenv("HYDRA_ENV") or "sandbox").lower())
    enabled: bool = field(default_factory=lambda: _env_bool("CHAOS_ENABLED", False))
    latency_ms_min: int = field(default_factory=lambda: _env_int("CHAOS_LATENCY_MS_MIN", 10))
    latency_ms_max: int = field(default_factory=lambda: _env_int("CHAOS_LATENCY_MS_MAX", 250))
    error_rate: float = field(default_factory=lambda: _env_float("CHAOS_ERROR_RATE", 0.0))
    fuzz_rate: float = field(default_factory=lambda: _env_float("CHAOS_FUZZ_RATE", 0.0))
    seed: Optional[int] = field(default_factory=lambda: _env_int("CHAOS_SEED", 0) or None)
    breaker_threshold: int = field(default_factory=lambda: _env_int("CHAOS_BREAKER_THRESHOLD", 5))
    breaker_cooldown_s: int = field(default_factory=lambda: _env_int("CHAOS_BREAKER_COOLDOWN_S", 20))

    def active(self) -> bool:
        # Chaos is allowed only in sandbox/dev and when explicitly enabled
        return self.enabled and self.env in ("sandbox", "dev")


# ─────────────────────────────────────────────────────────────────────
# Tiny Circuit Breaker (per dependency name)
# ─────────────────────────────────────────────────────────────────────

class CircuitState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class Circuit:
    name: str
    threshold: int
    cooldown: timedelta
    failures: int = 0
    state: str = CircuitState.CLOSED
    opened_at: Optional[datetime] = None

    def record_success(self) -> None:
        if self.state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
            # success while half-open → close
            self.state = CircuitState.CLOSED
            self.failures = 0
            self.opened_at = None
        else:
            # closed: just ensure counter doesn’t drift up
            self.failures = 0

    def record_failure(self) -> None:
        self.failures += 1
        if self.state == CircuitState.CLOSED and self.failures >= self.threshold:
            self.state = CircuitState.OPEN
            self.opened_at = datetime.utcnow()
            log.warning("circuit_opened", extra={"dep": self.name, "failures": self.failures})

    def ready_to_try(self) -> bool:
        # If open and cooldown passed → move to half-open (allow a probe)
        if self.state == CircuitState.OPEN and self.opened_at:
            if datetime.utcnow() - self.opened_at >= self.cooldown:
                self.state = CircuitState.HALF_OPEN
                log.info("circuit_half_open", extra={"dep": self.name})
                return True
            return False
        # If half-open or closed, allow call
        return True


# ─────────────────────────────────────────────────────────────────────
# Chaos Controller
# ─────────────────────────────────────────────────────────────────────

class ChaosController:
    def __init__(self, cfg: Optional[ChaosConfig] = None) -> None:
        self.cfg = cfg or ChaosConfig()
        # Seeded randomness for reproducibility in tests/sandbox
        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)
        self._breakers: Dict[str, Circuit] = {}

        log.info(
            "chaos_boot",
            extra={
                "env": self.cfg.env,
                "enabled": self.cfg.enabled,
                "latency_ms_min": self.cfg.latency_ms_min,
                "latency_ms_max": self.cfg.latency_ms_max,
                "error_rate": self.cfg.error_rate,
                "fuzz_rate": self.cfg.fuzz_rate,
                "seed": self.cfg.seed,
            },
        )

    # ── Latency injection ─────────────────────────────────────────────
    def maybe_delay(self, name: str = "generic") -> None:
        if not self.cfg.active():
            return
        lo = max(0, self.cfg.latency_ms_min)
        hi = max(lo, self.cfg.latency_ms_max)
        if hi == 0:
            return
        delay_ms = random.randint(lo, hi)
        if delay_ms > 0:
            log.info("chaos_delay", extra={"name": name, "delay_ms": delay_ms})
            time.sleep(delay_ms / 1000.0)

    # ── Error injection ───────────────────────────────────────────────
    def maybe_fail(self, name: str = "generic", http_status: int = 503) -> None:
        if not self.cfg.active():
            return
        if self.cfg.error_rate <= 0.0:
            return
        if random.random() < self.cfg.error_rate:
            log.warning("chaos_error", extra={"name": name, "http_status": http_status})
            if HTTPException is not None:
                raise HTTPException(status_code=http_status, detail=f"chaos fault: {name}")
            raise RuntimeError(f"chaos fault: {name}")

    # ── Payload fuzzer (non-destructive) ──────────────────────────────
    def maybe_fuzz(self, payload: Any, *, name: str = "payload") -> Any:
        if not self.cfg.active():
            return payload
        if self.cfg.fuzz_rate <= 0.0 or random.random() >= self.cfg.fuzz_rate:
            return payload
        # Simple, safe fuzz: add an inert key or jitter numeric leaves ±5%
        mutated = self._jitter(payload)
        log.info("chaos_fuzz", extra={"name": name})
        return mutated

    def _jitter(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                out[k] = self._jitter(v)
            # add inert marker to test tolerant parsers
            out["_hydra_chaos"] = True
            return out
        if isinstance(obj, list):
            return [self._jitter(x) for x in obj]
        if isinstance(obj, (int, float)):
            # Jitter within ±5%
            return obj * (1 + (random.random() - 0.5) * 0.10)
        return obj

    # ── Circuit breaker per dependency ────────────────────────────────
    def breaker(self, dep: str) -> Circuit:
        if dep not in self._breakers:
            self._breakers[dep] = Circuit(
                name=dep,
                threshold=max(1, self.cfg.breaker_threshold),
                cooldown=timedelta(seconds=max(1, self.cfg.breaker_cooldown_s)),
            )
        return self._breakers[dep]

    def with_breaker(self, dep: str, func: Callable[[], Any]) -> Any:
        """
        Run a callable through circuit breaker. If open and not ready, raise immediately.
        """
        c = self.breaker(dep)
        if not c.ready_to_try():
            log.warning("circuit_blocked_call", extra={"dep": dep})
            if HTTPException is not None:
                raise HTTPException(status_code=503, detail=f"dependency unavailable: {dep}")
            raise RuntimeError(f"dependency unavailable: {dep}")

        try:
            res = func()
            c.record_success()
            return res
        except Exception as e:
            c.record_failure()
            raise e

    # ── Manual toggles ────────────────────────────────────────────────
    def enable(self) -> None:
        self.cfg.enabled = True
        log.info("chaos_enable")

    def disable(self) -> None:
        self.cfg.enabled = False
        log.info("chaos_disable")


# Singleton controller used by decorators
CHAOS = ChaosController()


# ─────────────────────────────────────────────────────────────────────
# Decorators
# ─────────────────────────────────────────────────────────────────────

def chaos_endpoint(name: str = "endpoint", *, use_fuzz: bool = False, http_status: int = 503):
    """
    Wrap FastAPI async/sync endpoints.
      • Injects latency, optional fuzz of JSON-like payloads, and error bursts.
      • NOOP unless CHAOS is active in sandbox/dev.

    Example:
      @router.post("/compute")
      @chaos_endpoint("compute", use_fuzz=True)
      async def compute(payload: dict): ...
    """
    def decorator(fn: Callable):
        is_coro = hasattr(fn, "__await__") or hasattr(fn, "__anext__")

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            CHAOS.maybe_delay(name=name)
            CHAOS.maybe_fail(name=name, http_status=http_status)
            if use_fuzz and kwargs:
                # best-effort: fuzz first JSON-like arg named 'payload' or 'body'
                for key in ("payload", "body"):
                    if key in kwargs:
                        kwargs[key] = CHAOS.maybe_fuzz(kwargs[key], name=f"{name}.{key}")
            return await fn(*args, **kwargs)

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            CHAOS.maybe_delay(name=name)
            CHAOS.maybe_fail(name=name, http_status=http_status)
            if use_fuzz and kwargs:
                for key in ("payload", "body"):
                    if key in kwargs:
                        kwargs[key] = CHAOS.maybe_fuzz(kwargs[key], name=f"{name}.{key}")
            return fn(*args, **kwargs)

        return async_wrapper if is_coro else sync_wrapper
    return decorator


def chaos_section(name: str = "section", *, error_status: int = 500):
    """
    Wrap any internal function (sync/async) with chaos knobs.
    Useful inside critical sections: DB write, cache read, external call.

    Example:
      @chaos_section("db_write")
      def write_row(...): ...
    """
    def decorator(fn: Callable):
        is_coro = hasattr(fn, "__await__") or hasattr(fn, "__anext__")

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            CHAOS.maybe_delay(name=name)
            CHAOS.maybe_fail(name=name, http_status=error_status)
            return await fn(*args, **kwargs)

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            CHAOS.maybe_delay(name=name)
            CHAOS.maybe_fail(name=name, http_status=error_status)
            return fn(*args, **kwargs)

        return async_wrapper if is_coro else sync_wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────
# Sanity probe (can be called from healthz)
# ─────────────────────────────────────────────────────────────────────

def chaos_status() -> Dict[str, Any]:
    """
    Introspect current chaos & breaker states. Good for /status/core.
    """
    return {
        "active": CHAOS.cfg.active(),
        "env": CHAOS.cfg.env,
        "enabled": CHAOS.cfg.enabled,
        "latency_ms": [CHAOS.cfg.latency_ms_min, CHAOS.cfg.latency_ms_max],
        "error_rate": CHAOS.cfg.error_rate,
        "fuzz_rate": CHAOS.cfg.fuzz_rate,
        "seed": CHAOS.cfg.seed,
        "breakers": [
            {
                "dep": name,
                "state": c.state,
                "failures": c.failures,
                "opened_at": c.opened_at.isoformat() if c.opened_at else None,
                "threshold": c.threshold,
                "cooldown_s": int(c.cooldown.total_seconds()),
            }
            for name, c in CHAOS._breakers.items()
        ],
    }


# ─────────────────────────────────────────────────────────────────────
# Example helpers for dependencies with breaker (optional usage)
# ─────────────────────────────────────────────────────────────────────

def call_with_breaker(dep_name: str, fn: Callable[[], Any]) -> Any:
    """
    One-liner wrapper when you want breaker semantics around a specific call.
    """
    return CHAOS.with_breaker(dep_name, fn)
