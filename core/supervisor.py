# hydra-core/core/supervisor.py
"""
Supervisor — Orchestrates DFI → AFPS → HydraVoice (MVP)
-------------------------------------------------------
Objectives:
  • Keep it stupid-simple but resilient: timeouts, bounded retries, tiny circuit breakers.
  • Deterministic, auditable outputs with minimal moving parts.
  • No external network calls. Pure in-process pipeline.
  • Enterprise-friendly: structured result, reason codes, timing metrics.

Flow:
  1) Compute DFI (fragility) with guardrails.
  2) Compute AFPS using DFI & payload features.
  3) Generate HydraVoice packet (negative visualization / actions).
  4) Return a single SupervisionResult the API can serve.

Design:
  • "Quant-light" where it matters (scoring), conventional everywhere else.
  • Self-heal if FI/AFPS raise (retry→fallback baseline).
  • Tiny circuit-breakers stop flapping when a stage keeps failing.
"""

from __future__ import annotations

import time
import math
from typing import Any, Dict, Optional

# Soft imports to avoid hard coupling during scaffold
try:
    from pydantic import BaseModel, Field, ConfigDict
except Exception:  # pragma: no cover
    class BaseModel:  # type: ignore
        pass
    def Field(*args, **kwargs):  # type: ignore
        return None
    ConfigDict = dict  # type: ignore

# Stage models
try:
    from core.fi import DFIConfig, DFIInput, DFIResult, compute_dfi
except Exception:  # pragma: no cover
    class DFIConfig(BaseModel):  # type: ignore
        pass
    class DFIInput(BaseModel):  # type: ignore
        timeline: list = []
        conditions_open: int = 0
        borrower_latency_days: float = 0.0
        branch_latency_days: float = 0.0
        rework_events: int = 0
    class DFIResult(BaseModel):  # type: ignore
        dfi_score: float = 75.0
        components: Dict[str, float] = {}
        band: str = "MEDIUM"
        reason_codes: list = []
        qc: Dict[str, Any] = {}
    def compute_dfi(*args, **kwargs) -> DFIResult:  # type: ignore
        return DFIResult()

try:
    from core.afps import AFPSConfig, AFPSInput, AFPSResult, score_afps
except Exception:  # pragma: no cover
    class AFPSConfig(BaseModel):  # type: ignore
        pass
    class AFPSInput(BaseModel):  # type: ignore
        dfi_score: float = 75.0
        timeline: list = []
        conditions_open: int = 0
        borrower_latency_days: float = 0.0
        branch_latency_days: float = 0.0
        verifications_pending: int = 0
        appraisal_received: bool = False
        compliance_flags_open: int = 0
    class AFPSResult(BaseModel):  # type: ignore
        afps_score: float = 72.0
        p_fund: float = 0.78
        band: str = "MEDIUM"
        drivers: list = []
        actions: list = []
        confidence: float = 0.7
        qc: Dict[str, Any] = {}
    def score_afps(*args, **kwargs) -> AFPSResult:  # type: ignore
        return AFPSResult()

try:
    from core.hydra_voice import VoiceConfig, VoicePacket, hydra_voice
except Exception:  # pragma: no cover
    class VoiceConfig(BaseModel):  # type: ignore
        pass
    class VoicePacket(BaseModel):  # type: ignore
        headline: str = "ok"
        risk: str = "LOW"
        why: list = []
        do_next: list = []
        impact: Dict[str, Any] = {}
        metrics: Dict[str, Any] = {}
        qc: Dict[str, Any] = {}
    def hydra_voice(*args, **kwargs) -> VoicePacket:  # type: ignore
        return VoicePacket()

# Optional governance hook (hashing/audit) — soft import
try:
    from core.governance import audit_hash
except Exception:  # pragma: no cover
    def audit_hash(payload: Dict[str, Any]) -> str:  # type: ignore
        # Deterministic lightweight fallback
        import hashlib, json
        j = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(j).hexdigest()[:16]


# -----------------------------
# Config & Result Models
# -----------------------------

class CircuitState(BaseModel):
    failed_count: int = 0
    open_until_epoch: float = 0.0
    model_config = ConfigDict(extra="ignore")


class SupervisorConfig(BaseModel):
    # Timeouts (seconds)
    fi_timeout_s: float = Field(0.75, ge=0.05)
    afps_timeout_s: float = Field(0.75, ge=0.05)
    # Retries
    fi_retries: int = Field(1, ge=0)
    afps_retries: int = Field(1, ge=0)
    # Circuit breaker
    cb_fail_threshold: int = Field(3, ge=1)
    cb_cooldown_s: float = Field(20.0, ge=1.0)
    # Voice economics
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    # Defaults for fallbacks
    default_dfi: float = Field(72.0, ge=0.0, le=100.0)
    default_afps: float = Field(70.0, ge=0.0, le=100.0)
    model_config = ConfigDict(extra="ignore")


class SupervisionInput(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    fi_payload: DFIInput
    afps_payload: Optional[AFPSInput] = None  # if None, we’ll derive from fi+payload
    fi_cfg: Optional[DFIConfig] = None
    afps_cfg: Optional[AFPSConfig] = None
    model_config = ConfigDict(extra="ignore")


class StageTiming(BaseModel):
    fi_ms: int = 0
    afps_ms: int = 0
    voice_ms: int = 0


class SupervisionResult(BaseModel):
    """
    Single, API-ready packet containing all stage outputs + meta.
    """
    tenant_id: str
    dfi: DFIResult
    afps: AFPSResult
    voice: VoicePacket
    hash16: str
    timings: StageTiming
    degraded: bool  # true if any stage used fallback or circuit-open
    reason: str     # short reason for degradation if any
    model_config = ConfigDict(extra="ignore")


# -----------------------------
# Tiny Circuit Breaker
# -----------------------------

class _CB:
    """
    In-memory, per-process circuit-breaker store.
    MVP: Per-stage global keys (fi, afps). Good enough for a single pod.
    """
    _map: Dict[str, CircuitState] = {}

    @classmethod
    def get(cls, key: str) -> CircuitState:
        st = cls._map.get(key)
        if not st:
            st = CircuitState()
            cls._map[key] = st
        return st

    @classmethod
    def maybe_open(cls, key: str, cfg: SupervisorConfig) -> None:
        st = cls.get(key)
        st.failed_count += 1
        if st.failed_count >= cfg.cb_fail_threshold:
            st.open_until_epoch = time.time() + cfg.cb_cooldown_s

    @classmethod
    def reset(cls, key: str) -> None:
        st = cls.get(key)
        st.failed_count = 0
        st.open_until_epoch = 0.0

    @classmethod
    def is_open(cls, key: str) -> bool:
        st = cls.get(key)
        return time.time() < st.open_until_epoch


# -----------------------------
# Public Orchestration
# -----------------------------

def supervise_case(inp: SupervisionInput, cfg: Optional[SupervisorConfig] = None) -> SupervisionResult:
    """
    Run DFI → AFPS → HydraVoice with guardrails, return unified result.
    """
    scfg = cfg or SupervisorConfig()
    timings = StageTiming()
    degraded = False
    reason = ""

    # -------- Stage 1: DFI --------
    t0 = time.perf_counter()
    dfi_res: DFIResult
    if _CB.is_open("fi"):
        dfi_res = _fallback_dfi(scfg)
        degraded, reason = True, "dfi_circuit_open"
    else:
        try:
            dfi_res = _run_with_guard_fi(inp.fi_payload, inp.fi_cfg, scfg)
            _CB.reset("fi")
        except Exception:
            _CB.maybe_open("fi", scfg)
            dfi_res = _fallback_dfi(scfg)
            degraded, reason = True, "dfi_fallback"
    timings.fi_ms = int((time.perf_counter() - t0) * 1000)

    # -------- Stage 2: AFPS --------
    t1 = time.perf_counter()
    # Build AFPS payload if not provided
    afps_payload = inp.afps_payload or _derive_afps_payload(inp.fi_payload, dfi_res)

    if _CB.is_open("afps"):
        afps_res = _fallback_afps(scfg)
        degraded, reason = True, reason or "afps_circuit_open"
    else:
        try:
            afps_res = _run_with_guard_afps(afps_payload, inp.afps_cfg, scfg)
            _CB.reset("afps")
        except Exception:
            _CB.maybe_open("afps", scfg)
            afps_res = _fallback_afps(scfg)
            degraded, reason = True, reason or "afps_fallback"
    timings.afps_ms = int((time.perf_counter() - t1) * 1000)

    # -------- Stage 3: HydraVoice --------
    t2 = time.perf_counter()
    voice_packet = hydra_voice(dfi_res, afps_res, scfg.voice)
    timings.voice_ms = int((time.perf_counter() - t2) * 1000)

    # -------- Governance hash (deterministic) --------
    # We hash only minimal, stable fields to avoid churn.
    hash_payload = {
        "tenant_id": inp.tenant_id,
        "dfi": round(dfi_res.dfi_score, 2),
        "afps": round(afps_res.afps_score, 2),
        "p": round(afps_res.p_fund, 4),
        "risk": afps_res.band,
        "headline": voice_packet.headline,
    }
    h16 = audit_hash(hash_payload)

    return SupervisionResult(
        tenant_id=inp.tenant_id,
        dfi=dfi_res,
        afps=afps_res,
        voice=voice_packet,
        hash16=h16,
        timings=timings,
        degraded=degraded,
        reason=reason,
    )


# -----------------------------
# Guarded runners + fallbacks
# -----------------------------

def _run_with_guard_fi(fi_payload: DFIInput, fi_cfg: Optional[DFIConfig], scfg: SupervisorConfig) -> DFIResult:
    # Bounded retries with exponential backoff and wall-time timeout.
    attempts = scfg.fi_retries + 1
    last_exc: Optional[Exception] = None
    for i in range(attempts):
        start = time.perf_counter()
        try:
            res = compute_dfi(fi_payload, fi_cfg or DFIConfig())
            # guard: cheap time check (simulated timeout gate)
            if (time.perf_counter() - start) > scfg.fi_timeout_s:
                raise TimeoutError("dfi_timeout")
            return res
        except Exception as e:  # pragma: no cover
            last_exc = e
            # Small backoff (non-sleepy for API thread): spin loop with monotonic check
            _spin_backoff(ms=30 * (i + 1))
    # If we’re here, fail hard to outer try/except
    if last_exc:
        raise last_exc
    raise RuntimeError("dfi_unknown_failure")


def _run_with_guard_afps(afps_payload: AFPSInput, afps_cfg: Optional[AFPSConfig], scfg: SupervisorConfig) -> AFPSResult:
    attempts = scfg.afps_retries + 1
    last_exc: Optional[Exception] = None
    for i in range(attempts):
        start = time.perf_counter()
        try:
            res = score_afps(afps_payload, afps_cfg or AFPSConfig())
            if (time.perf_counter() - start) > scfg.afps_timeout_s:
                raise TimeoutError("afps_timeout")
            return res
        except Exception as e:  # pragma: no cover
            last_exc = e
            _spin_backoff(ms=30 * (i + 1))
    if last_exc:
        raise last_exc
    raise RuntimeError("afps_unknown_failure")


def _fallback_dfi(cfg: SupervisorConfig) -> DFIResult:
    """
    Safe median-like default that won't trigger chaos, with explicit reason codes.
    """
    return DFIResult(
        dfi_score=cfg.default_dfi,
        components={"fallback": 1.0},
        band="MEDIUM",
        reason_codes=["dfi_fallback_default"],
        qc={"fallback": True},
    )


def _fallback_afps(cfg: SupervisorConfig) -> AFPSResult:
    """
    Safe, conservative AFPS default (keeps risk in MEDIUM to force attention).
    """
    score = cfg.default_afps
    p = 0.75 if score >= 70 else 0.6
    band = "MEDIUM"
    return AFPSResult(
        afps_score=score,
        p_fund=p,
        band=band,
        drivers=[],
        actions=["Stabilize file: reduce open conditions; enforce 24h SLA; schedule appraisal if missing."],
        confidence=0.5,
        qc={"fallback": True},
    )


def _derive_afps_payload(fi_payload: DFIInput, dfi_res: DFIResult) -> AFPSInput:
    """
    Minimal deterministic derivation when caller doesn’t provide AFPSInput.
    """
    # Pull common signals from DFIInput; pass through DFI score for coupling.
    verifications_pending = 0  # unknown → 0 in MVP
    appraisal_received = True  # assume received unless flagged elsewhere (keeps noise low)
    compliance_flags_open = 0  # MVP assumes none unless provided

    return AFPSInput(
        dfi_score=dfi_res.dfi_score,
        timeline=fi_payload.timeline,
        conditions_open=fi_payload.conditions_open,
        borrower_latency_days=fi_payload.borrower_latency_days,
        branch_latency_days=fi_payload.branch_latency_days,
        verifications_pending=verifications_pending,
        appraisal_received=appraisal_received,
        compliance_flags_open=compliance_flags_open,
    )


def _spin_backoff(ms: int) -> None:
    """
    Ultra-light backoff without sleeping the thread (keeps API responsive in MVP).
    Spins for ~ms using monotonic time. Replace with asyncio/await later if needed.
    """
    end = time.perf_counter() + (ms / 1000.0)
    while time.perf_counter() < end:
        pass
