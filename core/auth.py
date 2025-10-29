# hydra-core/core/auth.py
"""
Auth — Tenant API Keys, Scopes, and KID Routing (MVP)
-----------------------------------------------------
Goals:
  • Simple, fast, explainable API-key auth for MVP (header-based).
  • Per-tenant config surface: scopes, rate limits, signing key (for governance).
  • Key rotation friendly: use KID (key id) so you can migrate secrets safely.
  • Minimal in-memory cache + DB hook (swap later) with constant-time compares.
  • No external network calls; integrates with FastAPI dependency in main.py.

Headers (default):
  • X-API-Key: <opaque secret>         (required)
  • X-API-KID: <key-id-string>         (optional; enables rotation/multi-key)
  • X-API-Tenant: <tenant-id>          (optional; can also be inferred)

Design:
  • Verifier checks: tenant exists → KID selected → constant-time secret match.
  • Scopes: coarse-grained ("score:dfi", "score:afps", "audit:read").
  • Rate limiting: lightweight token-bucket (in-memory) per tenant (burst + refill).
  • Governance bridge: expose GovernanceConfig with tenant’s signing key (b64).

Swap points for prod:
  • `load_tenant_from_store()` → plug real DB/Secrets Manager.
  • Rate limiter → replace with Redis or API Gateway when scaling.
"""

from __future__ import annotations

import base64
import hmac
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from fastapi import Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, ConfigDict

# Optional: we’ll reuse the governance config when signing audit packets
from .governance import GovernanceConfig


# ──────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────

class KeyRecord(BaseModel):
    kid: str = Field(..., description="Key ID")
    secret_b64: str = Field(..., description="Base64-encoded raw secret bytes")
    scopes: List[str] = Field(default_factory=lambda: ["score:dfi", "score:afps"])
    active: bool = True
    model_config = ConfigDict(extra="ignore")


class TenantRecord(BaseModel):
    tenant_id: str
    display_name: str = ""
    keys: List[KeyRecord] = Field(default_factory=list)
    # Used by governance for HMAC signing of audit exports
    signing_key_b64: Optional[str] = None
    # Rate limits (token bucket): burst capacity + tokens/sec
    rl_burst: int = 60
    rl_rate_per_sec: float = 2.0
    model_config = ConfigDict(extra="ignore")


class Principal(BaseModel):
    tenant_id: str
    kid: str
    scopes: List[str]
    governance: GovernanceConfig


# ──────────────────────────────────────────────────────────────────────
# Minimal In-Memory Store (MVP-friendly)
# (Replace via DB/Secrets in production)
# ──────────────────────────────────────────────────────────────────────

# Example: allow boot without DB by reading env (JSON not required; keep it simple)
# HYDRA_BOOT_TENANT_KEY_B64 may be provided for quickstart.
_BOOT_TENANT_ID = os.getenv("HYDRA_BOOT_TENANT_ID", "sandbox-tenant")
_BOOT_KID = os.getenv("HYDRA_BOOT_KID", "kid-0001")
_BOOT_KEY_B64 = os.getenv("HYDRA_BOOT_TENANT_KEY_B64")  # base64 raw secret
_BOOT_SIGNING_B64 = os.getenv("HYDRA_BOOT_SIGNING_B64")  # governance signing key

# In-memory registry populated on first access
_TENANT_CACHE: Dict[str, TenantRecord] = {}

def _bootstrap_tenant_if_needed() -> None:
    if _TENANT_CACHE:
        return
    if not _BOOT_KEY_B64:
        # Generate a dev-only 32-byte random key for local runs (not for prod!)
        os_key = os.urandom(32)
        key_b64 = base64.urlsafe_b64encode(os_key).decode("utf-8")
    else:
        key_b64 = _BOOT_KEY_B64

    tenant = TenantRecord(
        tenant_id=_BOOT_TENANT_ID,
        display_name="Hydra MVP Sandbox",
        signing_key_b64=_BOOT_SIGNING_B64,
        keys=[KeyRecord(kid=_BOOT_KID, secret_b64=key_b64, scopes=["score:dfi", "score:afps", "audit:read"])],
        rl_burst=120,
        rl_rate_per_sec=5.0,
    )
    _TENANT_CACHE[tenant.tenant_id] = tenant


def load_tenant_from_store(tenant_id: str) -> Optional[TenantRecord]:
    """
    MVP stub — returns from in-memory cache.
    Replace with a DB/Secrets Manager read in production.
    """
    _bootstrap_tenant_if_needed()
    return _TENANT_CACHE.get(tenant_id)


# ──────────────────────────────────────────────────────────────────────
# Constant-time comparisons
# ──────────────────────────────────────────────────────────────────────

def _b64_to_bytes(b64: str) -> Optional[bytes]:
    try:
        return base64.urlsafe_b64decode(b64.encode("utf-8"))
    except Exception:
        return None


def _ct_equal(a: bytes, b: bytes) -> bool:
    return hmac.compare_digest(a, b)


# ──────────────────────────────────────────────────────────────────────
# Token Bucket (in-memory)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Bucket:
    tokens: float
    last_ts: float
    burst: int
    rate: float  # tokens per second


_BUCKETS: Dict[str, Bucket] = {}

def rate_limit_check(tenant: TenantRecord) -> None:
    """
    Very small token-bucket limiter per tenant.
    Refill on each call; raise HTTPException 429 when empty.
    """
    now = time.time()
    b = _BUCKETS.get(tenant.tenant_id)
    if not b:
        b = Bucket(tokens=float(tenant.rl_burst), last_ts=now, burst=tenant.rl_burst, rate=tenant.rl_rate_per_sec)
        _BUCKETS[tenant.tenant_id] = b

    # Refill
    elapsed = max(0.0, now - b.last_ts)
    b.tokens = min(b.burst, b.tokens + elapsed * b.rate)
    b.last_ts = now

    if b.tokens < 1.0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded for tenant",
        )
    b.tokens -= 1.0


# ──────────────────────────────────────────────────────────────────────
# Core Verification
# ──────────────────────────────────────────────────────────────────────

def _select_key(tenant: TenantRecord, kid: Optional[str]) -> Optional[KeyRecord]:
    # If KID provided, honor it. Otherwise use first active key.
    if kid:
        for k in tenant.keys:
            if k.active and k.kid == kid:
                return k
        return None
    for k in tenant.keys:
        if k.active:
            return k
    return None


def verify_api_key(tenant: TenantRecord, provided_b64: str, kid: Optional[str]) -> Tuple[KeyRecord, bool]:
    key = _select_key(tenant, kid)
    if not key:
        return (None, False)  # type: ignore
    provided = _b64_to_bytes(provided_b64)
    stored = _b64_to_bytes(key.secret_b64)
    if not provided or not stored:
        return (None, False)  # type: ignore
    ok = _ct_equal(provided, stored)
    return (key, ok)


# ──────────────────────────────────────────────────────────────────────
# FastAPI Dependency
# ──────────────────────────────────────────────────────────────────────

def require_tenant(scopes_any: Optional[List[str]] = None):
    """
    Usage:
        @app.post("/score/dfi")
        async def route(payload: Payload, principal: Principal = Depends(require_tenant(["score:dfi"]))):
            ...
    """
    scopes_any = scopes_any or []

    async def _dep(request: Request) -> Principal:
        # Read headers (case-insensitive in ASGI)
        h = request.headers
        api_key = h.get("x-api-key")
        tenant_id = h.get("x-api-tenant", _BOOT_TENANT_ID)
        kid = h.get("x-api-kid")

        if not api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-API-Key")

        tenant = load_tenant_from_store(tenant_id)
        if not tenant:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown tenant")

        # Rate limit before heavy work
        rate_limit_check(tenant)

        keyrec, ok = verify_api_key(tenant, api_key, kid)
        if not ok or not keyrec or not keyrec.active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key or KID")

        # Scope check (any-match)
        if scopes_any:
            if not any(s in keyrec.scopes for s in scopes_any):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient scope")

        gov = GovernanceConfig(
            signing_key_b64=tenant.signing_key_b64,
            use_blake3=True,
            kid=keyrec.kid or "mvp-local",
        )

        return Principal(tenant_id=tenant.tenant_id, kid=keyrec.kid, scopes=keyrec.scopes, governance=gov)

    return _dep


# ──────────────────────────────────────────────────────────────────────
# Admin helpers (optional, used by tests or future admin endpoints)
# ──────────────────────────────────────────────────────────────────────

def add_tenant(tenant: TenantRecord) -> None:
    _TENANT_CACHE[tenant.tenant_id] = tenant


def rotate_key(tenant_id: str, new_key: KeyRecord, deactivate_old: bool = False) -> bool:
    tenant = _TENANT_CACHE.get(tenant_id)
    if not tenant:
        return False
    if deactivate_old:
        for k in tenant.keys:
            k.active = False
    tenant.keys.append(new_key)
    return True
