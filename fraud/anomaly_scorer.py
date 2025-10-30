# hydra-core/fraud/anomaly_scorer.py
"""
Process/Fraud Anomaly Scorer — AFPS v0
--------------------------------------
Goal:
  • Turn a compact feature vector (see fraud/feature_builder.py) into an interpretable
    anomaly score that flags both PROCESS risk (pipeline drag) and FRAUD-leaning patterns.
  • Deterministic, quant-light, auditable. No ML needed for MVP.

Outputs (example):
{
  "schema_version": "afps-v0",
  "score": 72.4,                 # 0..100 = anomaly risk (higher = riskier)
  "band": "HIGH",                # LOW / MED / HIGH / CRITICAL
  "reasons": [                   # sorted, top drivers first
     {"key":"leverage_pressure","impact":18.3,"note":"High DTI×LTV"},
     {"key":"rf_income_gap","impact":12.0,"note":"Risk flag present"},
     ...
  ],
  "triage": {
     "action":"Escalate UW review",
     "urgency_minutes": 120,
     "negative_visualization": "Expected fallout cost ~$1,050 if unaddressed in 48h."
  },
  "governance": {
     "fv_hash":"...",            # passthrough from features
     "score_hash":"...",         # blake2b of canonical json
     "policy_id":"afps-ruleset-2025-10-a"
  }
}

Scoring philosophy:
  • Linear-combo of interpretable terms with weights.
  • Each term produces a bounded contribution.
  • Reasons carry their contribution for HydraVoice to narrate later.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
from collections import OrderedDict

from utils.utils import (
    clamp,
    blake2b_hex,
    canonical_json,
)

AFPS_SCHEMA_VERSION = "afps-v0"
POLICY_ID = "afps-ruleset-2025-10-a"

# ─────────────────────────────────────────────────────────────────────
# Rule weights (calibrated by common sense for MVP; tune later)
# Each function below computes a 0..impact_max contribution.
# Keep the set small and surgical.
# ─────────────────────────────────────────────────────────────────────
IMPACT = {
    "leverage_pressure": 25.0,    # DTI × LTV — pipeline + fraud-adjacent
    "credit_cushion":    10.0,    # (850 - FICO) normalized
    "rate_delta":         8.0,    # pricing misalignment
    "rf_any":            15.0,    # any risk flags
    "rf_income_gap":     12.0,
    "rf_id_mismatch":    14.0,
    "rf_address_mismatch": 8.0,
    "rf_thin_file":       8.0,
    "rf_recent_inquiry_burst": 8.0,
    "rf_manual_calc":     6.0,
    "rf_doc_missing":     9.0,
    "stage_penalty":      8.0,    # late-stage anomalies cost more
    "night_ops":          5.0,    # odd-hour activity (weak signal, bound low)
}

BANDS = [
    (0.0,   35.0, "LOW"),
    (35.0,  60.0, "MED"),
    (60.0,  80.0, "HIGH"),
    (80.0, 101.0, "CRITICAL"),
]

LATE_STAGE_IDX = {  # matches feature_builder cat encoding (see CAT_BUCKETS["stage"])
    "lead": 0, "app": 1, "uw": 2, "ctc": 3, "clear": 4, "funded": 5, "post": 6, "_other": 7
}

# Simple dollarization heuristic for negative visualization.
# We keep it explicit so finance teams can challenge/adjust it.
EXPECTED_FALLOUT_COST_BY_BAND = {
    "LOW":       150.0,
    "MED":       450.0,
    "HIGH":     1050.0,
    "CRITICAL": 2500.0,
}

# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def score_features(fv: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute AFPS anomaly score from feature vector `fv` (see feature_builder.build_features()).
    Returns a dict with score, band, reasons, triage, governance.
    """
    x = fv.get("x", {})
    reasons: List[Dict[str, Any]] = []

    # 1) Leverage pressure: DTI×LTV on 0..20000 scale; normalize to 0..1 at 10000.
    lev = float(x.get("leverage_pressure", 0.0))
    lev_norm = clamp(lev / 10000.0, 0.0, 1.0)
    _add_reason(reasons, "leverage_pressure", lev_norm * IMPACT["leverage_pressure"], "High DTI×LTV")

    # 2) Credit cushion: 0..1 where 1 = weak cushion → up to max impact.
    cushion = clamp(float(x.get("credit_cushion", 0.0)), 0.0, 1.0)
    _add_reason(reasons, "credit_cushion", cushion * IMPACT["credit_cushion"], "Thin credit cushion")

    # 3) Rate delta: >0 = worse than baseline; cap at +2.0% for MVP scaling.
    rd = float(x.get("rate_delta", 0.0))
    rd_norm = clamp(rd / 2.0, 0.0, 1.0)
    _add_reason(reasons, "rate_delta", rd_norm * IMPACT["rate_delta"], "Pricing misaligned vs baseline")

    # 4) Risk flags — any + specific
    rf_any = int(x.get("rf_any", 0))
    if rf_any:
        _add_reason(reasons, "rf_any", IMPACT["rf_any"], "One or more risk flags present")

    for rf_key in ("rf_income_gap","rf_id_mismatch","rf_address_mismatch","rf_thin_file",
                   "rf_recent_inquiry_burst","rf_manual_calc","rf_doc_missing"):
        if int(x.get(rf_key, 0)) == 1:
            _add_reason(reasons, rf_key, IMPACT[rf_key], _rf_note(rf_key))

    # 5) Stage penalty — heavier if anomaly appears at/after UW (idx ≥2)
    stage_ix = int(x.get("stage_ix", 7))
    late_stage = 1.0 if stage_ix >= 2 else 0.0
    _add_reason(reasons, "stage_penalty", late_stage * IMPACT["stage_penalty"], "Late-stage anomaly")

    # 6) Night ops — event_hour in [0..5] or [22..23]
    hour = int(x.get("event_hour", 0))
    night = 1.0 if (hour <= 5 or hour >= 22) else 0.0
    _add_reason(reasons, "night_ops", night * IMPACT["night_ops"], "Odd-hour activity")

    # 7) Aggregate score (bounded 0..100)
    raw = sum(r["impact"] for r in reasons)
    score = clamp(raw, 0.0, 100.0)
    band = _band_for(score)

    # Sort reasons by impact desc, prune tiny noise (< 0.5)
    reasons = sorted([r for r in reasons if r["impact"] >= 0.5],
                     key=lambda r: r["impact"], reverse=True)

    # 8) Triage suggestion
    triage = _triage_for(band, stage_ix)

    # 9) Governance fingerprints
    fv_hash = (fv.get("fingerprints") or {}).get("feature_hash", "")
    out = OrderedDict()
    out["schema_version"] = AFPS_SCHEMA_VERSION
    out["score"] = round(score, 2)
    out["band"] = band
    out["reasons"] = [
        {"key": r["key"], "impact": round(r["impact"], 2), "note": r["note"]}
        for r in reasons
    ]
    out["triage"] = triage
    out["governance"] = {
        "fv_hash": fv_hash,
        "score_hash": blake2b_hex(canonical_json({
            "schema_version": AFPS_SCHEMA_VERSION,
            "fv_hash": fv_hash,
            "score": round(score, 4),
            "reasons": [(r["key"], round(r["impact"], 4)) for r in reasons],
            "policy_id": POLICY_ID
        })),
        "policy_id": POLICY_ID,
    }
    return out


# ─────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────

def _add_reason(reasons: List[Dict[str, Any]], key: str, impact: float, note: str):
    reasons.append({"key": key, "impact": float(impact), "note": note})

def _band_for(score: float) -> str:
    for lo, hi, name in BANDS:
        if lo <= score < hi:
            return name
    return "CRITICAL"

def _rf_note(k: str) -> str:
    return {
        "rf_income_gap": "Income inconsistency",
        "rf_id_mismatch": "Identity mismatch",
        "rf_address_mismatch": "Address mismatch",
        "rf_thin_file": "Thin credit file",
        "rf_recent_inquiry_burst": "Recent inquiry burst",
        "rf_manual_calc": "Manual calc used (bypass risk)",
        "rf_doc_missing": "Missing documentation",
    }.get(k, "Risk flag")

def _triage_for(band: str, stage_ix: int) -> Dict[str, Any]:
    # Urgency: later stages compress the window.
    base_urgency = {
        "LOW": 1440,       # 24h
        "MED": 480,        # 8h
        "HIGH": 120,       # 2h
        "CRITICAL": 30,    # 30m
    }[band]

    # Late stage multiplier
    mult = 0.5 if stage_ix >= 2 else 1.0
    urgency = int(max(base_urgency * mult, 15))

    action = {
        "LOW": "Queue standard review",
        "MED": "Prioritize processor review",
        "HIGH": "Escalate UW review",
        "CRITICAL": "Freeze & verify identity/income",
    }[band]

    neg = f"Expected fallout cost ~${EXPECTED_FALLOUT_COST_BY_BAND[band]:,.0f} if unaddressed in 48h."
    return {"action": action, "urgency_minutes": urgency, "negative_visualization": neg}


# ─────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Minimal inline demo using the builder to show end-to-end shape.
    from fraud.ingest_normalizer import normalize_event
    from fraud.feature_builder import build_features

    raw = {
        "loanId": "LN-001",
        "lenderId": "L-ABC",
        "Channel": "Retail",
        "LoanPurpose": "Purchase",
        "Stage": "UW",
        "FICO": "642",
        "DTI": "48%",
        "LTV": "0.92",
        "LoanAmount": "585000",
        "Rate": "7.625",
        "Occupancy": "Investment",
        "LockStatus": "Locked",
        "riskFlags": ["income_gap", "doc_missing", "recent_inquiry_burst"],
        "eventTime": "2025-10-28T23:41:12Z",
    }
    ev = normalize_event(raw)
    fv = build_features(ev)
    result = score_features(fv)
    print(result)
