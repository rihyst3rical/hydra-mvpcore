# hydra-core/core/hydra_voice.py
"""
HydraVoice — Narrative + Negative Visualization (MVP)
-----------------------------------------------------
Purpose: Convert DFI + AFPS outputs into plain-English guidance that
forces action. No black boxes. Everything is deterministic and auditable.

Inputs:
  • DFIResult (from core.fi)      — fragility/process stability (0–100)
  • AFPSResult (from core.afps)   — funding likelihood + drivers/actions
  • VoiceConfig (below)           — lightweight economics to estimate $ impact

Outputs (VoicePacket):
  • headline     — one-liner summary
  • risk         — LOW/MEDIUM/HIGH (mirrors AFPS band)
  • why          — top 2–3 factors with human language
  • do_next      — prioritized actions (from AFPS + minimal DFI cues)
  • impact       — $/loan and $/pipeline negative visualization (7/30 day)
  • metrics      — compact JSON for dashboards (Grafana tiles)
  • qc           — what assumptions we used (dollars_per_day, penalties, etc.)

Design constraints:
  • Zero external calls. Pure functions. Deterministic.
  • Pairs directly with existing FI/AFPS schemas.
  • “Hydra voice” = concise, surgical, slightly urgent, never fluffy.
  • All math is transparent in qc/assumptions for audits.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Soft imports for types; we avoid hard coupling at import time.
try:
    from pydantic import BaseModel, Field, ConfigDict
except Exception:  # pragma: no cover
    class BaseModel:  # type: ignore
        pass
    def Field(*args, **kwargs):  # type: ignore
        return None
    ConfigDict = dict  # type: ignore

try:
    from core.fi import DFIResult
except Exception:  # pragma: no cover
    class DFIResult(BaseModel):  # type: ignore
        dfi_score: float = 0.0

try:
    from core.afps import AFPSResult, AFPSDriver
except Exception:  # pragma: no cover
    class AFPSDriver(BaseModel):  # type: ignore
        code: str
        points: float
        details: Dict[str, Any] = {}
    class AFPSResult(BaseModel):  # type: ignore
        afps_score: float = 0.0
        p_fund: float = 0.0
        band: str = "HIGH"
        drivers: List[AFPSDriver] = []
        actions: List[str] = []
        confidence: float = 0.5
        qc: Dict[str, Any] = {}


# -----------------------------
# Config & Outputs
# -----------------------------

class VoiceConfig(BaseModel):
    """
    Minimal economics to make the dollar math real but simple.
    Tweak per-tenant later without touching logic.
    """
    dollars_per_ops_day: float = Field(300.0, ge=0.0, description="Avg daily ops burn per loan stuck in pipeline.")
    fallout_penalty_per_loan: float = Field(450.0, ge=0.0, description="Typical cost of a rate-lock fallout / repull friction.")
    labor_cost_per_touch: float = Field(28.0, ge=0.0, description="Blended cost per manual touch/rework.")
    touches_per_open_condition: float = Field(1.5, ge=0.0, description="Avg touches to resolve one condition.")
    pipeline_loan_count: int = Field(100, ge=1, description="Loans currently under management for a given branch/view.")
    days_window_short: int = Field(7, ge=1)
    days_window_long: int = Field(30, ge=7)

    # AFPS thresholds to translate score → risk of fallout / delay deltas.
    afps_good: float = Field(80.0, ge=0.0, le=100.0)
    afps_ok: float = Field(60.0, ge=0.0, le=100.0)

    model_config = ConfigDict(extra="ignore")


class ImpactBlock(BaseModel):
    per_loan_usd_7d: float
    per_loan_usd_30d: float
    pipeline_usd_7d: float
    pipeline_usd_30d: float
    est_lost_fundings_30d: float


class VoicePacket(BaseModel):
    headline: str
    risk: str
    why: List[str]
    do_next: List[str]
    impact: ImpactBlock
    metrics: Dict[str, Any]
    qc: Dict[str, Any]

    model_config = ConfigDict(extra="ignore")


# -----------------------------
# Public API
# -----------------------------

def hydra_voice(dfi: DFIResult, afps: AFPSResult, config: Optional[VoiceConfig] = None) -> VoicePacket:
    """
    Compose the narrative + negative visualization from DFI/AFPS.
    """
    cfg = config or VoiceConfig()
    # 1) Risk headline
    headline = _build_headline(dfi_score=dfi.dfi_score, afps_score=afps.afps_score, band=afps.band, p=afps.p_fund)

    # 2) Why (top drivers in human language)
    why_lines = _drivers_to_why(afps.drivers)

    # 3) Actions (prioritized)
    do_next = _prioritize_actions(afps.actions, afps.drivers, dfi_score=dfi.dfi_score)

    # 4) Impact math (negative visualization)
    impact = _estimate_impact(
        dfi_score=dfi.dfi_score,
        afps_score=afps.afps_score,
        p_fund=afps.p_fund,
        band=afps.band,
        drivers=afps.drivers,
        cfg=cfg
    )

    # 5) Metrics tile for dashboards
    metrics = {
        "dfi": round(dfi.dfi_score, 1),
        "afps": round(afps.afps_score, 1),
        "p_fund": round(afps.p_fund, 3),
        "risk_band": afps.band,
        "confidence": round(afps.confidence, 2),
        "impact_per_loan_7d": round(impact.per_loan_usd_7d, 2),
        "impact_pipeline_30d": round(impact.pipeline_usd_30d, 2),
        "est_lost_fundings_30d": round(impact.est_lost_fundings_30d, 2),
    }

    qc = {
        "afps_thresholds": {"good": cfg.afps_good, "ok": cfg.afps_ok},
        "economics": {
            "dollars_per_ops_day": cfg.dollars_per_ops_day,
            "fallout_penalty_per_loan": cfg.fallout_penalty_per_loan,
            "labor_cost_per_touch": cfg.labor_cost_per_touch,
            "touches_per_open_condition": cfg.touches_per_open_condition,
        },
        "windows": {"short_days": cfg.days_window_short, "long_days": cfg.days_window_long},
        "pipeline_loan_count": cfg.pipeline_loan_count,
    }

    return VoicePacket(
        headline=headline,
        risk=afps.band,
        why=why_lines,
        do_next=do_next,
        impact=impact,
        metrics=metrics,
        qc=qc,
    )


# -----------------------------
# Internals
# -----------------------------

def _build_headline(dfi_score: float, afps_score: float, band: str, p: float) -> str:
    # Keep it blunt, one line, specific.
    if band == "HIGH":
        return f"Watchlist: AFPS {afps_score:.0f} (p={p:.2f}) with DFI {dfi_score:.0f} — risk of fallout and delay is elevated."
    if band == "MEDIUM":
        return f"Manage closely: AFPS {afps_score:.0f} (p={p:.2f}); DFI {dfi_score:.0f} — stabilize to push into green."
    return f"Green lane: AFPS {afps_score:.0f} (p={p:.2f}); DFI {dfi_score:.0f} — protect velocity."


def _drivers_to_why(drivers: List[AFPSDriver]) -> List[str]:
    friendly = {
        "conditions_open": "Too many open conditions",
        "docs_missing": "Missing documents slowing underwriting",
        "verifications_pending": "Outstanding verifications (VOE/VOI/VOD)",
        "latency": "Slow borrower/branch response times",
        "compliance_flags": "Open compliance findings",
        "no_appraisal": "Appraisal not received",
    }
    lines: List[str] = []
    for d in drivers[:3]:
        label = friendly.get(d.code, d.code.replace("_", " ").title())
        lines.append(f"{label} (−{d.points:.1f} pts)")
    return lines or ["No major blockers detected; maintain cadence."]


def _prioritize_actions(actions: List[str], drivers: List[AFPSDriver], dfi_score: float) -> List[str]:
    # Keep vendor-agnostic, deterministic ordering: biggest subtractors first, then AFPS actions.
    ordered = []
    if drivers:
        for d in sorted(drivers, key=lambda x: x.points, reverse=True):
            if d.code == "docs_missing":
                ordered.append("Close document gaps to <5% immediately (owner: borrower + processor).")
            elif d.code == "conditions_open":
                ordered.append("Resolve top 2 critical conditions first; defer non-blockers.")
            elif d.code == "verifications_pending":
                ordered.append("Prioritize VOE/VOI/VOD calls; schedule follow-up windows.")
            elif d.code == "no_appraisal":
                ordered.append("Schedule and chase appraisal receipt now; block downstream work until received.")
            elif d.code == "latency":
                ordered.append("Cut response latency: set 24h SLA, automate nudges, escalate after 48h.")
            elif d.code == "compliance_flags":
                ordered.append("Clear open compliance findings; attach proof to file.")
    # Merge with AFPS’ computed nudges (they include +points estimates).
    for a in actions:
        if a not in ordered:
            ordered.append(a)
    # If the file is already stable, add a protect-velocity reminder.
    if dfi_score >= 80:
        ordered.append("Protect velocity: avoid rework; freeze requirements; only critical touches.")
    return ordered[:5]  # keep it scannable


def _estimate_impact(
    dfi_score: float,
    afps_score: float,
    p_fund: float,
    band: str,
    drivers: List[AFPSDriver],
    cfg: VoiceConfig
) -> ImpactBlock:
    """
    Turn signals into $ math. We use small, transparent heuristics:

    • Delay cost per day rises as AFPS falls below “ok” (60) and DFI < 70.
    • Fallout probability penalty scales with (1 - p_fund).
    • Each open condition implies extra touches (labor).
    • Pipeline impact = per-loan × pipeline_loan_count.

    All tunable via VoiceConfig; auditors get exact assumptions in QC.
    """
    # 1) Delay multiplier by risk band and DFI
    delay_mult = 1.0
    if band == "MEDIUM":
        delay_mult = 1.25
    elif band == "HIGH":
        delay_mult = 1.6
    if dfi_score < 70:
        delay_mult *= 1.2

    # 2) Daily ops burn per loan
    daily_ops = cfg.dollars_per_ops_day * delay_mult

    # 3) Fallout expected cost share
    fallout_ev = (1.0 - p_fund) * cfg.fallout_penalty_per_loan

    # 4) Labor from open conditions (if present in drivers)
    cond = _get_detail(drivers, "conditions_open", "conditions_open", default=0)
    labor_ev = cond * cfg.touches_per_open_condition * cfg.labor_cost_per_touch

    # 5) Sum for 7d and 30d per-loan
    per_loan_7d = daily_ops * cfg.days_window_short + fallout_ev * 0.25 + labor_ev  # smaller share of fallout in 7d
    per_loan_30d = daily_ops * cfg.days_window_long + fallout_ev + labor_ev

    # 6) Translate AFPS to expected lost fundings over 30d (rough heuristic)
    # Assume “good” band would achieve ~0.9 p_fund. Shortfall vs that benchmark → lost closings.
    p_benchmark = 0.90
    shortfall = max(0.0, p_benchmark - p_fund)
    est_lost_fundings_30d = shortfall * cfg.pipeline_loan_count

    # 7) Pipeline totals
    pipeline_7d = per_loan_7d * cfg.pipeline_loan_count
    pipeline_30d = per_loan_30d * cfg.pipeline_loan_count

    return ImpactBlock(
        per_loan_usd_7d=round(per_loan_7d, 2),
        per_loan_usd_30d=round(per_loan_30d, 2),
        pipeline_usd_7d=round(pipeline_7d, 2),
        pipeline_usd_30d=round(pipeline_30d, 2),
        est_lost_fundings_30d=round(est_lost_fundings_30d, 2),
    )


def _get_detail(drivers: List[AFPSDriver], code: str, key: str, default: float) -> float:
    for d in drivers:
        if d.code == code:
            try:
                return float(d.details.get(key, default))
            except Exception:
                return default
    return default
