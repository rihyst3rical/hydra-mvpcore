# hydra-core/core/fi.py
"""
Dynamic Fragility Index (DFI) — HydraCore MVP
----------------------------------------------
Purpose: quantify process fragility for a single loan/application using only
behavioral/process telemetry that lenders already have (readiness timeline +
basic event counters). Returns:
  • dfi_score (0–100, higher = more stable)
  • band (LOW | MEDIUM | HIGH volatility)
  • confidence (0–1)
  • reason_codes (top detractors with contributions)
  • counterfactuals (simple "do X → +Y" guidance)
  • qc (non-breaking quality flags for auditors)

Design goals:
  • Quant-light, fully explainable, deterministic.
  • Self-healing: tolerates messy/unsorted timestamps & sparse inputs.
  • Tenant-safe: uses plain dict inputs; Pydantic validators ensure safety.
  • Compliance-first: stable math, no hidden side effects, no outbound calls.

Usage (inside an endpoint or service layer):
    from core.fi import compute_dfi, FIPayload

    payload = FIPayload(
        loan_id="LN-123",
        borrower_id="B-456",
        timeline=[{"t": "2025-01-02", "readiness": 42},
                  {"t": "2025-01-12", "readiness": 60},
                  {"t": "2025-01-19", "readiness": 64},
                  {"t": "2025-02-02", "readiness": 83}],
        # Optional event counters (can be 0 if unknown)
        escalations=1,
        resubmissions=2,
        avg_response_latency_hours=12.0,
        docs_missing_ratio=0.15
    )

    result = compute_dfi(payload)
    # result.dict() → JSON-safe response

Notes:
- Weights are conservative and documented below.
- Counterfactuals are linear and transparent (no black box).
- Telemetry is optional; will no-op if exporter is absent or disabled.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Pydantic v2 preferred; fall back to v1 aliases if needed.
try:
    from pydantic import BaseModel, Field, field_validator, ConfigDict
except Exception:  # pragma: no cover
    from pydantic import BaseModel, Field  # type: ignore
    from pydantic import validator as field_validator  # type: ignore
    ConfigDict = dict  # type: ignore

# Local settings (no heavy imports)
try:
    from config.settings import get_settings
except Exception:  # pragma: no cover
    def get_settings():
        class _S:  # minimal stub for local runs
            PROMETHEUS_ENABLED = False
        return _S()

# Optional: light-touch telemetry (won't break if missing)
try:
    from telemetry.exporter import metrics_inc, metrics_observe
except Exception:  # pragma: no cover
    def metrics_inc(*args, **kwargs):  # type: ignore
        return None
    def metrics_observe(*args, **kwargs):  # type: ignore
        return None


# -----------------------------
# Input / Output Models
# -----------------------------

class TimelinePoint(BaseModel):
    t: datetime = Field(..., description="Timestamp (ISO8601)")
    readiness: float = Field(..., ge=0, le=100, description="Readiness 0–100")

    model_config = ConfigDict(extra="ignore")


class FIPayload(BaseModel):
    loan_id: Optional[str] = Field(default=None)
    borrower_id: Optional[str] = Field(default=None)

    # Time series of readiness snapshots (minimally 2 recommended)
    timeline: List[TimelinePoint] = Field(default_factory=list)

    # Optional observed process frictions (can be zeros if unknown)
    escalations: int = Field(default=0, ge=0)
    resubmissions: int = Field(default=0, ge=0)
    avg_response_latency_hours: float = Field(default=0.0, ge=0.0)
    docs_missing_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of required docs missing at last check (0–1)."
    )

    # Optional: per-tenant knobs (leave None to use defaults)
    # Keep MVP extremely simple: just an optional volatility window
    ewma_alpha: Optional[float] = Field(
        default=None, description="EWMA smoothing factor for momentum (0–1)."
    )

    model_config = ConfigDict(extra="ignore")

    @field_validator("timeline")
    @classmethod
    def _ensure_sorted(cls, v: List[TimelinePoint]) -> List[TimelinePoint]:
        # Self-heal unsorted/messy input (stable sort by timestamp)
        return sorted(v, key=lambda p: p.t)


class ReasonCode(BaseModel):
    code: str
    contribution: float  # points deducted
    details: Dict[str, Any] = Field(default_factory=dict)


class CounterfactualSuggestion(BaseModel):
    action: str
    expected_delta: float  # +points if action taken
    rationale: str


class DFIResult(BaseModel):
    loan_id: Optional[str]
    borrower_id: Optional[str]

    dfi_score: float = Field(..., ge=0, le=100)
    band: str  # "LOW" | "MEDIUM" | "HIGH" volatility (LOW volatility == high DFI)
    confidence: float = Field(..., ge=0, le=1)
    reason_codes: List[ReasonCode] = Field(default_factory=list)
    counterfactuals: List[CounterfactualSuggestion] = Field(default_factory=list)

    # QA/QC and transparency
    qc: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


# -----------------------------
# Core Math (transparent)
# -----------------------------

# Weight model (points deducted; tuned conservatively for MVP)
# Rationale:
#   • series_volatility: penalize sawtooth behavior
#   • idle_gaps: penalize long no-progress periods
#   • resubmissions/escalations/docs/latency: operational frictions
DEFAULT_WEIGHTS = {
    "series_volatility": 22.0,   # up to 22 pts
    "idle_gap": 18.0,            # up to 18 pts
    "resubmissions": 16.0,       # up to 16 pts
    "escalations": 14.0,         # up to 14 pts
    "docs_missing": 18.0,        # up to 18 pts
    "response_latency": 12.0,    # up to 12 pts
    "momentum_bonus": 10.0,      # adds back up to 10 pts (not a deduction)
}

# Feature caps (normalize raw → [0,1] risk factors)
CAPS = {
    "volatility_std_cap": 18.0,        # std of first differences (readiness pts)
    "idle_gap_days_cap": 21.0,         # max penalized idle gap
    "resubmissions_cap": 6,
    "escalations_cap": 4,
    "docs_missing_cap": 0.60,          # 60% missing docs saturates penalty
    "response_latency_hours_cap": 72.0 # >3 days saturates penalty
}


def _std(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return math.sqrt(var)


def _first_diffs(xs: List[float]) -> List[float]:
    return [xs[i] - xs[i - 1] for i in range(1, len(xs))]


def _max_idle_gap_days(points: List[TimelinePoint]) -> float:
    if len(points) < 2:
        return 0.0
    gaps_days = []
    for i in range(1, len(points)):
        dt = points[i].t - points[i - 1].t
        gaps_days.append(abs(dt.total_seconds()) / 86400.0)
    return max(gaps_days) if gaps_days else 0.0


def _ewma_momentum(points: List[TimelinePoint], alpha: float = 0.35) -> float:
    """
    Simple EWMA slope proxy scaled 0–1:
      - Compute EWMA of readiness; derive last 3 deltas.
      - Positive and consistent deltas → momentum near 1.
      - Flat/negative → near 0.
    """
    if len(points) < 2:
        return 0.0
    # EWMA sequence
    m = points[0].readiness
    e = [m]
    for p in points[1:]:
        m = alpha * p.readiness + (1 - alpha) * m
        e.append(m)
    # last 3 diffs
    diffs = _first_diffs(e[-4:]) if len(e) >= 4 else _first_diffs(e)
    if not diffs:
        return 0.0
    pos = sum(d for d in diffs if d > 0)
    neg = -sum(d for d in diffs if d < 0)
    total = pos + neg
    if total <= 0:
        return 0.0
    # Momentum = share of positive directional progress (tempered)
    ratio = pos / total  # 0..1
    # Magnitude tempering: larger total change → slightly higher momentum
    mag = min(1.0, (pos - neg) / 25.0 + 0.5)  # keep gentle
    return max(0.0, min(1.0, 0.6 * ratio + 0.4 * mag))


def _normalize(x: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return max(0.0, min(1.0, x / cap))


def _band(score: float) -> str:
    # Interpret as volatility band (lower volatility → higher DFI)
    if score >= 80:
        return "LOW"     # low volatility (good)
    if score >= 60:
        return "MEDIUM"
    return "HIGH"        # high volatility (risky)


def _confidence(points: List[TimelinePoint]) -> float:
    """
    Heuristic confidence:
      - +0.2 if >= 4 points
      - +0.2 if span >= 21 days
      - +0.2 if last readiness != first readiness (has movement)
      - +0.2 if timestamps strictly increasing (we enforce)
      - +0.2 if variance not degenerate
    """
    if not points:
        return 0.0
    score = 0.2 if len(points) >= 4 else 0.0
    if len(points) >= 2:
        span_days = (points[-1].t - points[0].t).total_seconds() / 86400.0
        if span_days >= 21:
            score += 0.2
        if abs(points[-1].readiness - points[0].readiness) >= 5:
            score += 0.2
        diffs = _first_diffs([p.readiness for p in points])
        if _std(diffs) > 0:
            score += 0.2
    # Timestamps are sorted by validator → award the final 0.2
    score += 0.2
    return max(0.0, min(1.0, score))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# -----------------------------
# Public API
# -----------------------------

def compute_dfi(payload: FIPayload, tenant_weights: Optional[Dict[str, float]] = None) -> DFIResult:
    """
    Main entry for DFI computation.
    - Accepts validated FIPayload.
    - Allows optional per-tenant weight overrides (subset OK).
    - Returns fully formed DFIResult.
    """
    s = get_settings()
    metrics_inc("hydra_dfi_requests_total", labels={"env": getattr(s, "ENV", "SANDBOX")})

    weights = dict(DEFAULT_WEIGHTS)
    if tenant_weights:
        for k, v in tenant_weights.items():
            if k in weights and isinstance(v, (int, float)):
                weights[k] = float(v)

    # Extract timeline safely
    points = payload.timeline
    rvals = [p.readiness for p in points]

    # Compute volatility on first differences (stability proxy)
    diffs = _first_diffs(rvals) if len(rvals) >= 2 else []
    vol_std = _std(diffs)

    # Compute idle gap (max days between events)
    idle_gap_days = _max_idle_gap_days(points)

    # Normalize features → [0,1]
    f_vol = _normalize(vol_std, CAPS["volatility_std_cap"])
    f_idle = _normalize(idle_gap_days, CAPS["idle_gap_days_cap"])
    f_resub = _normalize(float(payload.resubmissions), float(CAPS["resubmissions_cap"]))
    f_escal = _normalize(float(payload.escalations), float(CAPS["escalations_cap"]))
    f_docs = _normalize(payload.docs_missing_ratio, CAPS["docs_missing_cap"])
    f_latency = _normalize(payload.avg_response_latency_hours, CAPS["response_latency_hours_cap"])

    # Momentum bonus (scaled 0–1)
    alpha = payload.ewma_alpha if payload.ewma_alpha is not None else 0.35
    momentum = _ewma_momentum(points, alpha=alpha)

    # Deduction components (each up to their weight cap)
    deductions = {
        "series_volatility": weights["series_volatility"] * f_vol,
        "idle_gap":          weights["idle_gap"] * f_idle,
        "resubmissions":     weights["resubmissions"] * f_resub,
        "escalations":       weights["escalations"] * f_escal,
        "docs_missing":      weights["docs_missing"] * f_docs,
        "response_latency":  weights["response_latency"] * f_latency,
    }

    # Score assembly
    base = 100.0 - sum(deductions.values())
    # add momentum bonus after deductions (bounded)
    score = max(0.0, min(100.0, base + weights["momentum_bonus"] * momentum))

    confidence = _confidence(points)
    band = _band(score)

    # Reason codes (top 3 detractors)
    rc_sorted = sorted(deductions.items(), key=lambda kv: kv[1], reverse=True)
    reason_codes = [
        ReasonCode(code=k, contribution=round(v, 2), details=_rc_details(k, payload, vol_std, idle_gap_days))
        for k, v in rc_sorted[:3]
        if v > 0.5  # hide negligible noise
    ]

    # Counterfactuals — simple linear what-ifs (transparent)
    counterfactuals = _counterfactuals(weights, deductions, target_gain=6.0)

    # QC flags for auditors (non-breaking; visible in JSON)
    qc = {
        "points_count": len(points),
        "span_days": ((points[-1].t - points[0].t).total_seconds() / 86400.0) if len(points) >= 2 else 0.0,
        "vol_std": round(vol_std, 3),
        "idle_gap_days": round(idle_gap_days, 2),
        "features": {
            "f_vol": round(f_vol, 3),
            "f_idle": round(f_idle, 3),
            "f_resub": round(f_resub, 3),
            "f_escal": round(f_escal, 3),
            "f_docs": round(f_docs, 3),
            "f_latency": round(f_latency, 3),
            "momentum": round(momentum, 3),
        },
        "weights": {k: round(v, 3) for k, v in weights.items()},
        "safe_mode": bool(len(points) < 2),  # if true, volatility may be understated
    }

    # Telemetry (soft)
    metrics_observe("hydra_dfi_score", value=score, labels={"band": band})
    metrics_inc("hydra_dfi_low_band_total", labels={}) if band == "HIGH" else None

    return DFIResult(
        loan_id=payload.loan_id,
        borrower_id=payload.borrower_id,
        dfi_score=round(score, 2),
        band=band,
        confidence=round(confidence, 2),
        reason_codes=reason_codes,
        counterfactuals=counterfactuals,
        qc=qc,
    )


def safe_compute_dfi(data: Dict[str, Any]) -> DFIResult:
    """
    Convenience wrapper:
    - Accepts raw dict, validates to FIPayload, and computes DFI.
    - Use in endpoints to keep handlers tiny.
    """
    try:
        payload = FIPayload(**data)
    except Exception as e:
        # Minimal safe fallback: no timeline, rely on counters if any
        payload = FIPayload(
            loan_id=data.get("loan_id"),
            borrower_id=data.get("borrower_id"),
            timeline=[],
            escalations=int(_safe_float(data.get("escalations", 0))),
            resubmissions=int(_safe_float(data.get("resubmissions", 0))),
            avg_response_latency_hours=_safe_float(data.get("avg_response_latency_hours", 0.0)),
            docs_missing_ratio=max(0.0, min(1.0, _safe_float(data.get("docs_missing_ratio", 0.0)))),
        )
    return compute_dfi(payload)


# -----------------------------
# Explainability helpers
# -----------------------------

def _rc_details(code: str, payload: FIPayload, vol_std: float, idle_gap_days: float) -> Dict[str, Any]:
    if code == "series_volatility":
        return {"std_first_diff": round(vol_std, 3), "cap": CAPS["volatility_std_cap"]}
    if code == "idle_gap":
        return {"max_idle_gap_days": round(idle_gap_days, 2), "cap": CAPS["idle_gap_days_cap"]}
    if code == "resubmissions":
        return {"resubmissions": payload.resubmissions, "cap": CAPS["resubmissions_cap"]}
    if code == "escalations":
        return {"escalations": payload.escalations, "cap": CAPS["escalations_cap"]}
    if code == "docs_missing":
        return {"docs_missing_ratio": payload.docs_missing_ratio, "cap": CAPS["docs_missing_cap"]}
    if code == "response_latency":
        return {"avg_response_latency_hours": payload.avg_response_latency_hours, "cap": CAPS["response_latency_hours_cap"]}
    return {}


def _counterfactuals(weights: Dict[str, float], deductions: Dict[str, float], target_gain: float = 5.0) -> List[CounterfactualSuggestion]:
    """
    Linear counterfactuals: if you reduce one driver by X%, expected +delta ≈ weight * (reduction_fraction).
    We propose the top 2 levers that can collectively add ~target_gain points.
    """
    # Sort by biggest deduction
    levers = sorted(deductions.items(), key=lambda kv: kv[1], reverse=True)

    suggestions: List[CounterfactualSuggestion] = []
    total_target = target_gain

    for code, contrib in levers:
        if contrib <= 1.0 or total_target <= 0.5:
            continue
        # propose a 25–40% reduction depending on the lever size, capped
        frac = 0.25 if contrib < 6 else 0.35 if contrib < 12 else 0.4
        expected_delta = round(contrib * frac, 2)
        action = _action_text(code, frac)
        rationale = f"Reducing {code} by ~{int(frac*100)}% should recover ≈ {expected_delta} DFI points."
        suggestions.append(CounterfactualSuggestion(action=action, expected_delta=expected_delta, rationale=rationale))
        total_target -= expected_delta
        if len(suggestions) >= 2:
            break

    # Momentum nudge (if nothing else)
    if total_target > 0.5 and "series_volatility" not in [c for c, _ in levers[:2]]:
        suggestions.append(CounterfactualSuggestion(
            action="Sustain steady weekly progress (avoid multi-week lulls).",
            expected_delta=min(3.0, total_target),
            rationale="Consistent small gains increase momentum bonus and reduce volatility."
        ))

    return suggestions


def _action_text(code: str, frac: float) -> str:
    if code == "series_volatility":
        return "Reduce bursty work: schedule 2–3 smaller updates/week instead of sporadic spikes."
    if code == "idle_gap":
        return "Shorten idle gaps: ensure at least one readiness action every 5–7 days."
    if code == "resubmissions":
        return "Cut resubmissions: add checklist validation before uploads to avoid rework."
    if code == "escalations":
        return "Lower escalations: route files with early warnings to senior review first."
    if code == "docs_missing":
        return "Close doc gaps: auto-request missing docs and gate next steps until resolved."
    if code == "response_latency":
        return "Respond faster: enforce <24h borrower/LO response SLAs on open questions."
    return "Improve process hygiene."
