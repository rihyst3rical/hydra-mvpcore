# hydra-core/core/models.py
"""
SQLModel ORM + Drift Scaffolding (DFI & AFPS MVP)
-------------------------------------------------
Goals:
  • Async-friendly SQLModel entities for tenants, API keys, loan events, scores.
  • Minimal, auditable schema: timestamps, soft-deletes, indexes that match hot paths.
  • Model version registry for DFI/AFPS (so responses can cite version + params).
  • Feature drift telemetry (PSI/KS) stored per-feature and window to prove vigilance.
  • Zero heavy deps: pure-python PSI/KS fallback suitable for MVP checks.

Notes:
  • Works with Async SQLAlchemy engine from core/db.py.
  • `Base` alias exported so db bootstrap can call Base.metadata.create_all().
  • Keep JSON payloads compact; avoid PII in columns (PII lives redacted/encrypted upstream).

Tables:
  - Tenant             : Organizations (lenders/IMBs)
  - TenantKey          : Hashed API keys (prefix + hash), usage auditing
  - Event              : Loan/process events (readiness ticks, ops signals)
  - Score              : DFI/AFPS outputs per-loan snapshot (+reason codes)
  - ModelVersion       : Versioned params for DFI/AFPS
  - DriftStat          : PSI/KS per feature & time window

Utility:
  - hash_api_key()     : One-way key hashing (blake2b)
  - key_prefix()       : 6-char prefix for display/lookup
  - psi() / ks()       : Pure-Python drift tests (binned PSI, 2-sample KS)

MVP Contract:
  • Only DFI & AFPS are scored/store here; extend later without breaking schema.
  • No PII stored; use opaque IDs (tenant_id, loan_id).
"""

from __future__ import annotations

import math
import time
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlmodel import SQLModel, Field, Relationship, Column, JSON
from sqlalchemy import Index, UniqueConstraint, text as sa_text


# ──────────────────────────────────────────────────────────────────────
# Base alias so core/db.py can call Base.metadata.create_all(...)
# ──────────────────────────────────────────────────────────────────────
Base = SQLModel  # db._safe_create_all uses Base.metadata


# ──────────────────────────────────────────────────────────────────────
# Mixins
# ──────────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin(SQLModel):
    created_at: datetime = Field(default_factory=_utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=_utcnow, nullable=False)

    def touch(self) -> None:
        self.updated_at = _utcnow()


class SoftDeleteMixin(SQLModel):
    is_deleted: bool = Field(default=False, nullable=False, index=True)
    deleted_at: Optional[datetime] = Field(default=None, nullable=True)


# ──────────────────────────────────────────────────────────────────────
# Core Entities
# ──────────────────────────────────────────────────────────────────────

class Tenant(TimestampMixin, SoftDeleteMixin, table=True):
    """
    An org/tenant (lender, IMB, bank division).
    """
    __tablename__ = "tenant"

    id: Optional[int] = Field(default=None, primary_key=True)
    slug: str = Field(index=True, unique=True, regex=r"^[a-z0-9\-]{3,64}$")
    name: str = Field(index=True)
    is_active: bool = Field(default=True, index=True)

    keys: List["TenantKey"] = Relationship(back_populates="tenant")


class TenantKey(TimestampMixin, SoftDeleteMixin, table=True):
    """
    API key record (never store raw key). Format recommendation:
      <prefix>.<random> where prefix is stored separately for operator UX.
    """
    __tablename__ = "tenant_key"
    __table_args__ = (
        UniqueConstraint("tenant_id", "key_hash", name="uq_tenant_key_hash"),
        Index("ix_tenant_key_prefix", "prefix"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: int = Field(foreign_key="tenant.id", index=True)
    prefix: str = Field(index=True, max_length=12)
    key_hash: str = Field(index=True, max_length=128)  # blake2b hex
    scopes: List[str] = Field(sa_column=Column(JSON), default_factory=list)
    is_active: bool = Field(default=True, index=True)
    last_used_at: Optional[datetime] = Field(default=None, nullable=True)

    tenant: Optional[Tenant] = Relationship(back_populates="keys")


class Event(TimestampMixin, SoftDeleteMixin, table=True):
    """
    Process/loan events used by DFI/AFPS.
    Keep payload small & scrubbed (client should pre-redact PII).
    """
    __tablename__ = "event"
    __table_args__ = (
        Index("ix_event_tenant_loan_time", "tenant_id", "loan_id", "created_at"),
        Index("ix_event_type", "event_type"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: int = Field(foreign_key="tenant.id", index=True)
    loan_id: str = Field(index=True, max_length=64)
    event_type: str = Field(index=True, max_length=48)  # e.g., 'doc_upload', 'nudge_sent'
    readiness: Optional[float] = Field(default=None)    # 0–100, optional for DFI
    fragility_inputs: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
    )
    latency_ms: Optional[int] = Field(default=None)     # optional ops timing
    source: Optional[str] = Field(default=None, max_length=48)  # 'encompass', 'api', etc.


class Score(TimestampMixin, SoftDeleteMixin, table=True):
    """
    Snapshot scores per loan (DFI/AFPS) with reason codes & confidence.
    """
    __tablename__ = "score"
    __table_args__ = (
        Index("ix_score_tenant_loan_time", "tenant_id", "loan_id", "created_at"),
        Index("ix_score_models", "tenant_id", "model_version"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: int = Field(foreign_key="tenant.id", index=True)
    loan_id: str = Field(index=True, max_length=64)

    # Metrics (store what we compute in MVP)
    dfi: Optional[float] = Field(default=None, index=True)   # 0–100
    afps: Optional[float] = Field(default=None, index=True)  # 0–100 probability to fund expressed 0–100

    confidence: Optional[float] = Field(default=None)  # 0–1
    volatility_band: Optional[str] = Field(default=None, max_length=16)  # LOW/MED/HIGH
    reason_codes: List[Dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(JSON),
    )

    # Versioning
    model_name: str = Field(default="DFI_AFPS_MVP", index=True, max_length=48)
    model_version: str = Field(default="v1", index=True, max_length=24)


class ModelVersion(TimestampMixin, SoftDeleteMixin, table=True):
    """
    Registry of model configs/weights. Useful for audit + reproducibility.
    """
    __tablename__ = "model_version"
    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_model_name_version"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    model_name: str = Field(index=True, max_length=64)     # 'DFI', 'AFPS', 'DFI_AFPS_MVP'
    version: str = Field(index=True, max_length=24)        # 'v1', '2025-10-29'
    params: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    is_active: bool = Field(default=True, index=True)
    activated_at: Optional[datetime] = Field(default=None, nullable=True)


class DriftStat(TimestampMixin, SoftDeleteMixin, table=True):
    """
    Per-feature drift measurement over a time window.
    Supports PSI (population stability index) and KS (Kolmogorov–Smirnov).
    """
    __tablename__ = "drift_stat"
    __table_args__ = (
        Index("ix_drift_tenant_feature_window", "tenant_id", "feature_name", "window_start", "window_end"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: Optional[int] = Field(default=None, foreign_key="tenant.id", index=True)
    feature_name: str = Field(index=True, max_length=64)
    stat_type: str = Field(index=True, max_length=8)  # 'psi' | 'ks'
    stat_value: float = Field(index=True)
    window_start: datetime = Field(index=True)
    window_end: datetime = Field(index=True)
    sample_current: int = Field(default=0)
    sample_baseline: int = Field(default=0)
    alert_level: Optional[str] = Field(default=None, max_length=16)  # 'OK'|'WARN'|'ALERT'


# ──────────────────────────────────────────────────────────────────────
# Utility: API key hashing helpers
# ──────────────────────────────────────────────────────────────────────

def hash_api_key(raw: str) -> str:
    """
    One-way blake2b hash of API key (no salt required if keys are random & long).
    """
    h = hashlib.blake2b(digest_size=32)
    h.update(raw.encode("utf-8"))
    return h.hexdigest()


def key_prefix(raw: str, length: int = 6) -> str:
    """
    Short human-friendly prefix to show in logs/UX (first N non-delimiter chars).
    """
    base = raw.split(".")[0]
    return base[:length]


# ──────────────────────────────────────────────────────────────────────
# Utility: Drift metrics (PSI / KS) — pure Python, MVP-grade
# ──────────────────────────────────────────────────────────────────────

def psi(
    baseline: Sequence[float],
    current: Sequence[float],
    bins: int = 10,
    eps: float = 1e-9,
) -> float:
    """
    Population Stability Index (binned).
    Lower is better; common thresholds: <0.1 OK, 0.1–0.25 watch, >0.25 shift.
    """
    if not baseline or not current:
        return 0.0

    bmin, bmax = min(baseline), max(baseline)
    cmin, cmax = min(current), max(current)
    lo = min(bmin, cmin)
    hi = max(bmax, cmax)
    if math.isclose(lo, hi):
        return 0.0

    width = (hi - lo) / bins
    # Guard against degenerate width
    width = width if width > 0 else 1.0

    def hist(xs: Sequence[float]) -> List[int]:
        buckets = [0] * bins
        for x in xs:
            idx = int((x - lo) / width)
            if idx < 0:
                idx = 0
            elif idx >= bins:
                idx = bins - 1
            buckets[idx] += 1
        return buckets

    hb = hist(baseline)
    hc = hist(current)

    nb = max(1, sum(hb))
    nc = max(1, sum(hc))

    psi_val = 0.0
    for b, c in zip(hb, hc):
        pb = max(b / nb, eps)
        pc = max(c / nc, eps)
        psi_val += (pc - pb) * math.log(pc / pb)
    return float(abs(psi_val))


def ks(
    baseline: Sequence[float],
    current: Sequence[float],
) -> float:
    """
    Two-sample Kolmogorov–Smirnov statistic (D). 0 (identical) → 1 (disjoint).
    Pure Python implementation for MVP (O((n+m) log(n+m))).
    """
    if not baseline or not current:
        return 0.0
    a = sorted(baseline)
    b = sorted(current)
    na = len(a)
    nb = len(b)
    ia = ib = 0
    cdf_a = cdf_b = 0.0
    dmax = 0.0

    # Merge-walk
    while ia < na and ib < nb:
        if a[ia] <= b[ib]:
            x = a[ia]
            while ia < na and a[ia] == x:
                ia += 1
            cdf_a = ia / na
        else:
            x = b[ib]
            while ib < nb and b[ib] == x:
                ib += 1
            cdf_b = ib / nb
        dmax = max(dmax, abs(cdf_a - cdf_b))

    # Finish tails
    dmax = max(dmax, abs(1.0 - cdf_b))  # if a finished first
    dmax = max(dmax, abs(cdf_a - 1.0))  # if b finished first
    return float(dmax)


def drift_alert_level(stat_type: str, value: float) -> str:
    """
    Heuristic thresholds; keep conservative for MVP and tune with live data.
    """
    if stat_type == "psi":
        if value < 0.1:
            return "OK"
        if value < 0.25:
            return "WARN"
        return "ALERT"
    if stat_type == "ks":
        if value < 0.1:
            return "OK"
        if value < 0.2:
            return "WARN"
        return "ALERT"
    return "OK"


# ──────────────────────────────────────────────────────────────────────
# Lightweight SQL helpers (optional)
# ──────────────────────────────────────────────────────────────────────

DDL_INDEXES = [
    # A couple of defensive indexes for hot queries if DB lacks them
    ("CREATE INDEX IF NOT EXISTS ix_score_tenant_time ON score(tenant_id, created_at DESC)",),
    ("CREATE INDEX IF NOT EXISTS ix_event_tenant_time ON event(tenant_id, created_at DESC)",),
]


async def ensure_indexes(async_conn) -> None:
    """
    Best-effort creation of helpful indexes without a full migration system.
    Call once at startup in sandbox (safe on Postgres & SQLite).
    """
    for (sql,) in DDL_INDEXES:
        await async_conn.execute(sa_text(sql))


# ──────────────────────────────────────────────────────────────────────
# Mini self-check (for /healthz deep mode)
# ──────────────────────────────────────────────────────────────────────

def orm_smoke_signature() -> str:
    """
    Returns a short, deterministic signature proving model module loaded.
    Useful for debugging container mismatches.
    """
    payload = f"{Tenant.__tablename__}|{Score.__tablename__}|{int(time.time())//300}"
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()
