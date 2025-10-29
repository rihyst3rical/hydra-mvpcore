# hydra-core/core/afps.py
"""
AFPS — Adaptive Funding Probability Score (MVP)
-----------------------------------------------
Goal: turn process health (DFI) + a few high-signal operational frictions
into a clean, auditable probability of funding and a 0–100 score lenders
can act on today.

Outputs:
  • afps_score (0–100) — higher means more likely to fund on time
  • p_fund (0–1)       — calibrated probability proxy
  • band               — LOW | MEDIUM | HIGH risk (HIGH = watchlist)
  • drivers            — top subtractors with point impacts
  • actions            — “do X → +Y points” (deterministic, linear)
  • confidence         — data sufficiency heuristic
  • qc                 — feature reveals for auditors

Design:
  • Quant-light: logistic-style mapping, no black-box ML.
  • Explainable: fully decomposed contributions (no hidden weights).
  • Self-healing: sane defaults; accepts either a DFI score OR raw timeline.
  • Finance-grade: deterministic, side-effect free, no outbound calls.
  • Future-proof: plug-in weights; can later swap mapper for calibrated Platt/Isotonic.

Modes:
  1) Provide an existing DFI result (recommended for pipeline):
        from core.fi import FIPayload, compute_dfi
        dfi = compute_dfi(FIPayload(...))
        afps = compute_afps(AFPSPayload.from_dfi(dfi))
  2) Provide raw process frictions + (optionally) a minimal timeline:
        afps = compute_afps(AFPSPayload(dfi_score=73, conditions_open=4, ...))

"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# Local models (we import types but do not require runtime call if user passes dfi_score)
try:
    from pydantic import BaseModel, Field, ConfigDict, field_validator
except Exception:  # pragma: no cover
    from pydantic import BaseModel, Field  # type: ignore
    from pydantic import validator as field_validator  # type: ignore
    ConfigDict = dict  # type: ignore

try:
    from core.fi import FIPayload, compute_dfi, DFIResult, TimelinePoint
except Exception:  # pragma: no cover
    # Minimal stubs for type hints if fi.py isn’t imported yet
    class TimelinePoint(BaseModel):  # type: ignore
        t: Any
        readiness: float
    class FIPayload(BaseModel):  # type: ignore
        timeline: List[TimelinePoint] = []
    class DFIResult(BaseModel):  # type: ignore
        dfi_score: float = 0.0

# Optional telemetry (no-op if absent)
try:
    from telemetry.exporter import metrics_inc, metrics_observe
except Exception:  # pragma: no cover
    def metrics_inc(*args, **kwargs):  # type: ignore
        return None
    def metrics_observe(*args, **kwargs):  # type: ignore
        return None


# -----------------------------
# Payload / Result
# -----------------------------

class AFPSPayload(BaseModel):
    # Preferred: pass an existing DFI score (or we’ll compute if timeline provided)
    dfi_score: Optional[float] = Field(default=None, ge=0, le=100)

    # High-signal, low-noise frictions (keep MVP tiny & universal)
    conditions_open: int = Field(0, ge=0, description="Outstanding underwriting/closing conditions.")
    docs_missing_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Share of required docs missing (0–1).")
    verifications_pending: int = Field(0, ge=0, description="VOE/VOI/VOD pending count.")
    appraisal_received: bool = Field(True, description="Appraisal in? If False, penalize timing risk.")
    compliance_flags: int = Field(0, ge=0, description="Open compliance findings (curable).")
    # Timing proxy (optional but helpful)
    avg_response_latency_hours: float = Field(0.0, ge=0.0)

    # If dfi_score is not provided, user may pass timeline to compute DFI inline (light path)
    timeline: Optional[List[TimelinePoint]] = Field(default=None)

    # Optional tenor knob for mapping slope around decision boundary
    # (purely deterministic; can tune per-tenant later)
    slope_bias: float = Field(1.0, ge=0.2, le=3.0)

    model_config = ConfigDict(extra="ignore")

    @classmethod
    def from_dfi(cls, dfi: DFIResult, **kwargs) -> "AFPSPayload":
        return cls(dfi_score=float(dfi.dfi_score), **kwargs)

    @field_validator("timeline")
    @classmethod
    def _sort_timeline(cls, v):
        if not v:
            return v
        return sorted(v, key=lambda p: p.t)


class AFPSDriver(BaseModel):
    code: str
    points: float
    details: Dict[str, Any] = Field(default_factory=dict)


class AFPSResult(BaseModel):
    afps_score: float = Field(..., ge=0, le=100)
    p_fund: float = Field(..., ge=0, le=1)
    band: str
    drivers: List[AFPSDriver] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)
    qc: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


# -----------------------------
# Weights / Caps (MVP)
# -----------------------------

WEIGHTS = {
    # start from DFI; strong positive anchor
    "dfi_weight": 0.62,          # contributes up to +62 points when DFI=100
    # subtractors (capped contributions)
    "conditions": 14.0,
    "docs_missing": 12.0,
    "verifications": 6.0,
    "no_appraisal": 4.0,
    "latency": 6.0,
    "compliance": 6.0,
    # action boost (deterministic nudge for “do this now”)
    "action_bonus": 8.0,
}

CAPS = {
    "conditions_cap": 8,
    "docs_missing_cap": 0.6,
    "verifications_cap": 4,
    "latency_hours_cap": 72.0,
    "compliance_cap": 3,
}


# -----------------------------
# Public API
# -----------------------------

def compute_afps(payload: AFPSPayload) -> AFPSResult:
    """
    Deterministic additive score with logistic-style mapping to p_fund.
    The score starts from a DFI-derived base, subtracts friction points,
    then applies a small action bonus if obvious fixes exist.
    """
    metrics_inc("hydra_afps_requests_total", labels={})

    # 1) Determine DFI anchor
    dfi = payload.dfi_score
    qc_dfi_source = "provided"
    if dfi is None:
        # If a minimal timeline is present, compute a DFI on the fly (transparent path)
        if payload.timeline:
            dfi_res = compute_dfi(FIPayload(timeline=payload.timeline))
            dfi = float(dfi_res.dfi_score)
            qc_dfi_source = "computed_from_timeline"
        else:
            dfi = 50.0  # conservative neutral
            qc_dfi_source = "defaulted"

    # 2) Start score from DFI (0–100 → 0–62 pts)
    base = (dfi / 100.0) * WEIGHTS["dfi_weight"] * 100.0  # equals dfi * 0.62

    # 3) Subtract friction contributions (normalized to caps)
    drivers: List[AFPSDriver] = []

    def _sub(code: str, raw: float, cap: float, weight: float, detail_key: str):
        frac = 0.0 if cap <= 0 else max(0.0, min(1.0, raw / cap))
        pts = weight * frac
        return pts, AFPSDriver(code=code, points=round(pts, 2), details={detail_key: raw, "cap": cap})

    cond_pts, cond_drv = _sub("conditions_open", payload.conditions_open, CAPS["conditions_cap"], WEIGHTS["conditions"], "conditions_open")
    docs_pts, docs_drv = _sub("docs_missing", payload.docs_missing_ratio, CAPS["docs_missing_cap"], WEIGHTS["docs_missing"], "docs_missing_ratio")
    verf_pts, verf_drv = _sub("verifications_pending", payload.verifications_pending, CAPS["verifications_cap"], WEIGHTS["verifications"], "verifications_pending")
    lat_pts,  lat_drv  = _sub("latency", payload.avg_response_latency_hours, CAPS["latency_hours_cap"], WEIGHTS["latency"], "avg_response_latency_hours")
    comp_pts, comp_drv = _sub("compliance_flags", payload.compliance_flags, CAPS["compliance_cap"], WEIGHTS["compliance"], "compliance_flags")
    appr_pts = WEIGHTS["no_appraisal"] if not payload.appraisal_received else 0.0
    appr_drv = AFPSDriver(code="no_appraisal", points=round(appr_pts, 2), details={"appraisal_received": payload.appraisal_received})

    subtractors = [cond_pts, docs_pts, verf_pts, lat_pts, comp_pts, appr_pts]
    base_after = max(0.0, base - sum(subtractors))

    # 4) Action bonus: if there are obvious, curable blockers, add a modest nudge (bounded)
    actions: List[str] = []
    action_pool = WEIGHTS["action_bonus"]

    if payload.docs_missing_ratio > 0.0:
        delta = min(action_pool, 3.0)
        actions.append(f"Close missing docs to <5% → +{delta:.1f} pts")
        action_pool -= delta

    if payload.conditions_open > 0:
        delta = min(action_pool, 3.0)
        actions.append(f"Resolve top 2 conditions (critical path) → +{delta:.1f} pts")
        action_pool -= delta

    if not payload.appraisal_received and action_pool > 0.0:
        delta = min(action_pool, 2.0)
        actions.append(f"Prioritize appraisal scheduling/receipt → +{delta:.1f} pts")
        action_pool -= delta

    score = max(0.0, min(100.0, base_after + (WEIGHTS["action_bonus"] - action_pool)))

    # 5) Map score → p_fund with a shallow logistic so it’s intuitive near decision bands
    #    Center the curve near 60–70 (typical decision gray zone). slope_bias tunes steepness.
    center = 65.0
    slope = 0.12 * float(payload.slope_bias)  # gentle; avoids overconfidence
    p = 1.0 / (1.0 + math.exp(-slope * (score - center)))
    p_fund = max(0.0, min(1.0, p))

    band = _band(score)
    confidence = _confidence(dfi_source=qc_dfi_source, payload=payload)

    # Collect drivers (top 3 by absolute subtraction)
    all_drivers = [cond_drv, docs_drv, verf_drv, lat_drv, comp_drv, appr_drv]
    drivers_sorted = sorted(all_drivers, key=lambda d: d.points, reverse=True)
    top_drivers = [d for d in drivers_sorted if d.points > 0.5][:3]

    metrics_observe("hydra_afps_score", value=score, labels={"band": band})
    metrics_observe("hydra_afps_pfund", value=p_fund, labels={"band": band})

    qc = {
        "dfi_score": round(dfi, 2),
        "dfi_source": qc_dfi_source,
        "base_points_from_dfi": round(base, 2),
        "subtractors_total": round(sum(subtractors), 2),
        "slope_bias": payload.slope_bias,
        "caps": CAPS,
        "weights": WEIGHTS,
    }

    return AFPSResult(
        afps_score=round(score, 2),
        p_fund=round(p_fund, 3),
        band=band,
        drivers=top_drivers,
        actions=actions,
        confidence=round(confidence, 2),
        qc=qc,
    )


def safe_compute_afps(data: Dict[str, Any]) -> AFPSResult:
    """
    Accepts a dict, validates, and computes AFPS. If DFI is missing and no timeline
    is supplied, defaults DFI to 50 (neutral).
    """
    try:
        payload = AFPSPayload(**data)
    except Exception:
        # Ultra-safe defaults
        payload = AFPSPayload(
            dfi_score=float(data.get("dfi_score", 50.0)),
            conditions_open=int(data.get("conditions_open", 0)),
            docs_missing_ratio=float(data.get("docs_missing_ratio", 0.0)),
            verifications_pending=int(data.get("verifications_pending", 0)),
            appraisal_received=bool(data.get("appraisal_received", True)),
            compliance_flags=int(data.get("compliance_flags", 0)),
            avg_response_latency_hours=float(data.get("avg_response_latency_hours", 0.0)),
        )
    return compute_afps(payload)


# -----------------------------
# Helpers
# -----------------------------

def _band(score: float) -> str:
    if score >= 80:
        return "LOW"      # LOW risk, high funding likelihood
    if score >= 60:
        return "MEDIUM"
    return "HIGH"         # HIGH risk, watchlist


def _confidence(dfi_source: str, payload: AFPSPayload) -> float:
    """
    Simple sufficiency heuristic:
      - +0.4 if DFI provided or computed
      - +0.2 if docs_missing_ratio present
      - +0.2 if conditions_open present
      - +0.2 if appraisal_received explicitly set / verifications present
    """
    c = 0.0
    if dfi_source in ("provided", "computed_from_timeline"):
        c += 0.4
    if payload.docs_missing_ratio is not None:
        c += 0.2
    c += 0.2 if payload.conditions_open is not None else 0.0
    c += 0.1 if payload.verifications_pending is not None else 0.0
    c += 0.1 if payload.appraisal_received is not None else 0.0
    return max(0.0, min(1.0, c))
