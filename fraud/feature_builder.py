# hydra-core/fraud/feature_builder.py
"""
Feature Builder — AFPS/DFI v0
-----------------------------
Purpose:
  • Convert a NormalizedLoanEvent → compact, deterministic feature vector.
  • Keep it tiny: numeric core, a few high-signal interactions, bounded categorical encodings,
    light time features, and governance fingerprints.
  • Zero external dependencies; stable ordering; sandbox-safe.

Design choices:
  • "Quant-light": only transforms that have clear, monotonic value for AFPS & DFI.
  • Deterministic key order to ensure consistent hashing & caching.
  • Categorical encoding uses bounded "enum bucket" approach (not exploding one-hots).
  • All units are explicit; percent fields remain in [0..100] domain to match ingest/DFI.

Inputs:
  • NormalizedLoanEvent (from fraud/ingest_normalizer.py)

Outputs:
  • FeatureVector dict:
      {
        "schema_version": "fv-v0",
        "x": { ... ordered feature map ... },
        "meta": {
           "loan_id": "...",
           "lender_id": "...",
           "stage": "...",
           "event_ts": "...",
           "quality_score": 93.0,
           "issues": ["..."]
        },
        "fingerprints": {
           "feature_hash": "<blake2b hex>",
           "source_norm_hash": "<norm hash passthrough>"
        }
      }

Security:
  • No PII fields serialized. Meta is limited to routing context.

Extensibility:
  • Add new features behind version bump "fv-v1" later without breaking callers.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import math
from collections import OrderedDict

from fraud.ingest_normalizer import NormalizedLoanEvent
from utils.utils import (
    blake2b_hex,
    canonical_json,
    clamp,
)

FV_SCHEMA_VERSION = "fv-v0"

# ─────────────────────────────────────────────────────────────────────
# Bounded categorical spaces (keep small, deterministic)
# Anything outside falls into "_other"
# ─────────────────────────────────────────────────────────────────────
CAT_BUCKETS = {
    "channel": ("retail", "wholesale", "correspondent", "consumer", "_other"),
    "purpose": ("purchase", "refi", "cashout", "heloc", "other", "_other"),
    "stage":   ("lead", "app", "uw", "ctc", "clear", "funded", "post", "_other"),
    "occupancy": ("primary", "second", "investment", "_other"),
    "lock_status": ("unlocked", "locked", "expired", "relock", "_other"),
}

# Risk flags we care about early (bounded; others collapse into "rf_other")
RISK_FLAGS_CANON = (
    "thin_file",
    "income_gap",
    "id_mismatch",
    "address_mismatch",
    "credit_spike",
    "recent_inquiry_burst",
    "manual_calc",
    "doc_missing",
)

# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def build_features(ev: NormalizedLoanEvent) -> Dict[str, Any]:
    """
    Build a compact, ordered feature vector from a normalized event.
    Deterministic output for stable hashing and cache keys.
    """

    # 1) Numeric core (raw domain retained for transparency)
    fico = _nz_int(ev.fico)
    dti = _clip_pct(ev.dti_pct, 0.0, 100.0)
    ltv = _clip_pct(ev.ltv_pct, 0.0, 200.0)
    rate = _clip(ev.rate_pct, 0.0, 30.0)
    amt = _clip(ev.loan_amount, 0.0, 20_000_000.0)

    # 2) Interactions (clear, interpretable)
    #    - leverage pressure: DTI × LTV (scaled to 0..20000)
    #    - rate-to-amount pressure proxy: rate × log1p(amt)
    #    - credit cushion: (850 - fico) normalized
    leverage_pressure = (dti * ltv)  # domain: 0..20,000
    rate_amount_pressure = rate * _log1p(amt)
    credit_cushion = clamp((850 - fico) / 550.0, 0.0, 1.0)  # 0 (great) .. 1 (weak)

    # 3) Rate delta vs a dumb baseline (placeholder until we wire pricing feed)
    bench_rate = _baseline_rate_guess(ev)  # very simple heuristic
    rate_delta = rate - bench_rate  # positive means worse than baseline

    # 4) Categorical buckets (bounded)
    cat_enc = _encode_cats(
        channel=ev.channel,
        purpose=ev.purpose,
        stage=ev.stage,
        occupancy=ev.occupancy,
        lock_status=ev.lock_status,
    )

    # 5) Risk flags (bounded)
    rf_map = _encode_risk_flags(ev.risk_flags)

    # 6) Time slivers (low-risk, interpretable)
    tod_hour, dow = _time_bins(ev.event_ts)

    # 7) Assemble ordered feature map (fixed key order)
    x = OrderedDict()

    # Numeric core
    x["fico"] = fico
    x["dti_pct"] = dti
    x["ltv_pct"] = ltv
    x["rate_pct"] = rate
    x["loan_amount"] = amt

    # Interactions
    x["leverage_pressure"] = leverage_pressure
    x["rate_amount_pressure"] = rate_amount_pressure
    x["credit_cushion"] = credit_cushion
    x["rate_benchmark"] = bench_rate
    x["rate_delta"] = rate_delta

    # Time features
    x["event_hour"] = tod_hour          # 0..23
    x["event_dow"] = dow                # 0..6 (Mon=0)

    # Categoricals (bounded)
    for k, v in cat_enc.items():
        x[k] = v

    # Risk flags (bounded counts/indicators)
    for k, v in rf_map.items():
        x[k] = v

    # 8) Governance fingerprints
    feature_hash = blake2b_hex(canonical_json({"schema": FV_SCHEMA_VERSION, "x": x}))

    # 9) Pack final vector
    fv = {
        "schema_version": FV_SCHEMA_VERSION,
        "x": x,
        "meta": {
            "loan_id": ev.loan_id,
            "lender_id": ev.lender_id,
            "stage": ev.stage,
            "event_ts": ev.event_ts,
            "quality_score": ev.quality_score,
            "issues": list(ev.issues),
        },
        "fingerprints": {
            "feature_hash": feature_hash,
            "source_norm_hash": ev.fingerprints.get("norm_hash", ""),
        },
    }
    return fv


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _nz_int(v: Optional[int]) -> int:
    try:
        return int(v or 0)
    except Exception:
        return 0

def _clip(v: float, lo: float, hi: float) -> float:
    if v < lo: return lo
    if v > hi: return hi
    return v

def _clip_pct(v: float, lo: float, hi: float) -> float:
    # v already in percent domain (0..100 or up to 200 for LTV)
    return _clip(float(v or 0.0), lo, hi)

def _log1p(v: float) -> float:
    try:
        return math.log1p(max(v, 0.0))
    except Exception:
        return 0.0

def _encode_cats(**cats: str) -> Dict[str, int]:
    """
    Map each categorical into a small integer code.
    e.g., channel: retail=0, wholesale=1, correspondent=2, consumer=3, _other=4
    """
    out: Dict[str, int] = {}
    for field, raw in cats.items():
        buckets = CAT_BUCKETS[field]
        val = (raw or "").strip().lower()
        if val not in buckets:
            idx = buckets.index("_other")
        else:
            idx = buckets.index(val)
        out[f"{field}_ix"] = idx
    return out

def _encode_risk_flags(flags: List[str]) -> Dict[str, int]:
    flags = [str(f or "").strip().lower() for f in (flags or []) if str(f or "").strip()]
    known = 0
    out: Dict[str, int] = {}
    for rf in RISK_FLAGS_CANON:
        present = 1 if rf in flags else 0
        out[f"rf_{rf}"] = present
        known += present
    # bucket the rest to avoid dimensional blow-up
    other = max(len(flags) - known, 0)
    out["rf_other_count"] = other
    out["rf_any"] = 1 if (known + other) > 0 else 0
    return out

def _time_bins(iso_ts: str) -> Tuple[int, int]:
    """
    Extract hour (0..23) and weekday (0..6, Mon=0) from an ISO string if possible.
    We avoid importing datetime parsing heavy libs; use simple slice heuristics.
    """
    try:
        # Expect formats like '2025-10-28T13:22:10Z' or with timezone offset.
        hh = int(iso_ts[11:13])
        # Very rough weekday estimation requires a real parser; keep 0 (Mon) placeholder
        # to avoid pulling in datetime with timezone math in this minimal layer.
        dow = 0
        return (min(max(hh, 0), 23), dow)
    except Exception:
        return (0, 0)

def _baseline_rate_guess(ev: NormalizedLoanEvent) -> float:
    """
    Dumb, transparent baseline rate model:
      • Start at 6.75
      • +0.50 if fico < 660
      • +0.25 if dti > 45
      • +0.25 if ltv > 85
      • +0.125 if investment occupancy
    This is a placeholder until we wire a proper pricing feed or rate card.
    """
    base = 6.75
    if ev.fico and ev.fico < 660:
        base += 0.50
    if (ev.dti_pct or 0) > 45.0:
        base += 0.25
    if (ev.ltv_pct or 0) > 85.0:
        base += 0.25
    if (ev.occupancy or "").lower() == "investment":
        base += 0.125
    return round(base, 3)


# ─────────────────────────────────────────────────────────────────────
# Smoke test (manual)
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from fraud.ingest_normalizer import normalize_event

    raw = {
        "loanId": "AB12CD34",
        "lenderId": "LEND-001",
        "Channel": "Retail",
        "LoanPurpose": "Purchase",
        "Stage": "UW",
        "FICO": "742",
        "DTI": "36.5%",
        "LTV": "0.78",
        "LoanAmount": "425000",
        "State": "FL",
        "Occupancy": "Owner",
        "LockStatus": "Locked",
        "Rate": "6.625",
        "riskFlags": ["thin_file", "income_gap", "random_custom_flag"],
        "eventType": "status_change",
        "eventTime": "2025-10-28T13:22:10Z",
        "source": "vendor_api",
    }
    ev = normalize_event(raw)
    fv = build_features(ev)
    # Pretty print without importing json (stay minimal)
    print("schema:", fv["schema_version"])
    print("meta:", fv["meta"])
    print("fingerprints:", fv["fingerprints"])
    print("features:")
    for k, v in fv["x"].items():
        print(f"  {k}: {v}")
