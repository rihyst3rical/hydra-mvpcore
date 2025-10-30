# hydra-core/fraud/ingest_normalizer.py
"""
Ingest Normalizer — AFPS/DFI entry point
----------------------------------------
Goal:
  • Accept messy upstream “loan event” payloads (EPC webhooks, CSV->JSON rows, vendor APIs).
  • Normalize to a compact, deterministic schema used by AFPS & DFI.
  • Do *light* quality checks (types, ranges), stamp governance digests (BLAKE2b),
    and produce a tiny provenance record (source, received_at, raw_hash).
  • Zero external deps; fast-path; sandbox-safe.

Design:
  • Standard library only; wired to hydra.utils and core.governance later.
  • Deterministic canonical JSON → BLAKE2b digest for dedupe/audit.
  • “Just enough” coercion: dates, currency, percents, enums, IDs.
  • Non-fatal errors collected into `issues[]` for AFPS triage; severity gate optional.

Outputs:
  • `NormalizedLoanEvent` (dict-like) ready for feature_builder and DFI.
  • `quality_score` (0..100) and `issues` list (strings).
  • `fingerprints`: {raw_hash, norm_hash, stable_key}

Security:
  • Never logs PII by default. Use `redact=True` in summarize() to mask.

Notes:
  • Keep this file tiny; domain specifics grow in feature_builder.py not here.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from utils.utils import (
    canonical_json,
    blake2b_hex,
    clamp,
    percent_str,
    looks_like_loan_id,
    utcnow,
    iso_now,
    scrub_pii,
)

# ─────────────────────────────────────────────────────────────────────
# Canonical AFPS/DFI minimal schema (v0)
# ─────────────────────────────────────────────────────────────────────
# We intentionally keep this tight. Feature explosion belongs in feature_builder.
# Keys are lowercase snake_case; nested dicts are shallow and bounded.

CANON_FIELDS = {
    "loan_id": str,                # stable ID (EPC, LOS, or vendored)
    "lender_id": str,              # tenant/partner ID (auth-mapped)
    "channel": str,                # retail | wholesale | correspondent | consumer
    "purpose": str,                # purchase | refi | cashout | heloc | other
    "stage": str,                  # lead | app | uw | ctc | clear | funded | post
    "fico": int,                   # 300..850
    "dti_pct": float,              # 0..100
    "ltv_pct": float,              # 0..200 (HELOC/second lien can exceed 100)
    "loan_amount": float,          # USD
    "property_state": str,         # CA, TX, FL, ...
    "occupancy": str,              # primary | second | investment
    "lock_status": str,            # unlocked | locked | expired | relock
    "rate_pct": float,             # e.g., 6.75
    "risk_flags": list,            # normalized string flags (["thin_file","income_gap"])
    "event_type": str,             # snapshot | status_change | doc_update | pricing | alert
    "event_ts": str,               # ISO-8601 (producer) if present; else we set it
    # Provenance (we set these)
    "source": str,                 # epc_webhook | csv_bulk | vendor_api | manual
    "received_ts": str,            # ISO-8601 (ingest time)
    "ingest_version": str,         # "ingest-v0"
}

# Whitelist enums (lenient; we map variants)
ENUM_MAP = {
    "channel": {
        "retail": "retail",
        "wholesale": "wholesale",
        "broker": "wholesale",
        "correspondent": "correspondent",
        "consumer": "consumer",
        "direct": "consumer",
    },
    "purpose": {
        "purchase": "purchase",
        "refi": "refi",
        "refinance": "refi",
        "cashout": "cashout",
        "cash-out": "cashout",
        "heloc": "heloc",
        "other": "other",
    },
    "stage": {
        "lead": "lead",
        "application": "app",
        "app": "app",
        "uw": "uw",
        "underwriting": "uw",
        "ctc": "ctc",
        "clear_to_close": "ctc",
        "clear": "clear",
        "funded": "funded",
        "post": "post",
    },
    "occupancy": {
        "primary": "primary",
        "owner": "primary",
        "owner_occupied": "primary",
        "second": "second",
        "secondary": "second",
        "investment": "investment",
        "non-owner": "investment",
    },
    "lock_status": {
        "unlocked": "unlocked",
        "locked": "locked",
        "expired": "expired",
        "relock": "relock",
        "re-locked": "relock",
    },
    "event_type": {
        "snapshot": "snapshot",
        "status_change": "status_change",
        "doc_update": "doc_update",
        "pricing": "pricing",
        "alert": "alert",
    },
}

# Lightweight field aliases from common feeds → canonical keys
ALIASES = {
    "loanId": "loan_id",
    "LoanID": "loan_id",
    "guid": "loan_id",
    "lenderId": "lender_id",
    "tenant_id": "lender_id",
    "Channel": "channel",
    "LoanPurpose": "purpose",
    "Stage": "stage",
    "FICO": "fico",
    "ficoScore": "fico",
    "DTI": "dti_pct",
    "DTI_pct": "dti_pct",
    "LTV": "ltv_pct",
    "LTV_pct": "ltv_pct",
    "LoanAmount": "loan_amount",
    "amount": "loan_amount",
    "State": "property_state",
    "Occupancy": "occupancy",
    "LockStatus": "lock_status",
    "Rate": "rate_pct",
    "Rate_pct": "rate_pct",
    "riskFlags": "risk_flags",
    "eventType": "event_type",
    "event_ts": "event_ts",
    "eventTime": "event_ts",
    "source": "source",
}

# ─────────────────────────────────────────────────────────────────────
# dataclass container
# ─────────────────────────────────────────────────────────────────────

@dataclass
class NormalizedLoanEvent:
    # Canonical fields (see CANON_FIELDS)
    loan_id: str
    lender_id: str
    channel: str
    purpose: str
    stage: str
    fico: int
    dti_pct: float
    ltv_pct: float
    loan_amount: float
    property_state: str
    occupancy: str
    lock_status: str
    rate_pct: float
    risk_flags: List[str]
    event_type: str
    event_ts: str

    # Provenance
    source: str
    received_ts: str
    ingest_version: str = "ingest-v0"

    # Additions for downstream ops (not part of CANON_FIELDS)
    quality_score: float = 0.0               # 0..100
    issues: List[str] = field(default_factory=list)
    fingerprints: Dict[str, str] = field(default_factory=dict)  # raw_hash, norm_hash, stable_key

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summarize(self, redact: bool = True) -> str:
        """Short human summary for logs or HydraVoice; PII scrubbed when requested."""
        msg = (
            f"loan_id={self.loan_id} lender={self.lender_id} stage={self.stage} "
            f"purpose={self.purpose} channel={self.channel} fico={self.fico} "
            f"dti={percent_str(self.dti_pct)} ltv={percent_str(self.ltv_pct)} "
            f"amt=${self.loan_amount:,.0f} rate={self.rate_pct:.3f}% "
            f"lock={self.lock_status} occ={self.occupancy} "
            f"quality={self.quality_score:.1f} issues={len(self.issues)} src={self.source}"
        )
        return scrub_pii(msg) if redact else msg


# ─────────────────────────────────────────────────────────────────────
# Core normalization pipeline
# ─────────────────────────────────────────────────────────────────────

def normalize_event(raw: Dict[str, Any], *, default_source: str = "epc_webhook") -> NormalizedLoanEvent:
    """
    Convert a raw dict → NormalizedLoanEvent with:
      • field aliasing
      • type coercion
      • enum mapping
      • gentle bounds checks
      • governance fingerprints
    """
    received_ts = iso_now()
    source = (raw.get("source") or default_source).strip().lower()

    # 1) Alias map (shallow)
    mapped: Dict[str, Any] = {}
    for k, v in raw.items():
        key = ALIASES.get(k, k)
        mapped[key] = v

    issues: List[str] = []

    # 2) Pull + coerce
    loan_id = _coerce_loan_id(mapped.get("loan_id"))
    if not loan_id:
        issues.append("missing_or_bad.loan_id")

    lender_id = _as_str(mapped.get("lender_id"))
    if not lender_id:
        issues.append("missing.lender_id")

    channel = _enum("channel", mapped.get("channel"), issues)
    purpose = _enum("purpose", mapped.get("purpose"), issues)
    stage = _enum("stage", mapped.get("stage"), issues)

    fico = _as_int(mapped.get("fico"), lo=250, hi=900, soft=(300, 850), issues=issues, label="fico")
    dti_pct = _as_pct(mapped.get("dti_pct"), issues, "dti_pct", lo=0, hi=100)
    ltv_pct = _as_pct(mapped.get("ltv_pct"), issues, "ltv_pct", lo=0, hi=200)
    loan_amount = _as_float(mapped.get("loan_amount"), lo=0, hi=20_000_000, issues=issues, label="loan_amount")
    property_state = _as_state(mapped.get("property_state"), issues)
    occupancy = _enum("occupancy", mapped.get("occupancy"), issues)
    lock_status = _enum("lock_status", mapped.get("lock_status"), issues)
    rate_pct = _as_float(mapped.get("rate_pct"), lo=0.0, hi=30.0, issues=issues, label="rate_pct")
    risk_flags = _as_flags(mapped.get("risk_flags"))

    event_type = _enum("event_type", mapped.get("event_type"), issues)
    event_ts = _as_iso(mapped.get("event_ts")) or received_ts

    # 3) Compute fingerprints
    raw_hash = blake2b_hex(canonical_json(raw))
    norm_payload = {
        "loan_id": loan_id,
        "lender_id": lender_id,
        "channel": channel,
        "purpose": purpose,
        "stage": stage,
        "fico": fico,
        "dti_pct": dti_pct,
        "ltv_pct": ltv_pct,
        "loan_amount": loan_amount,
        "property_state": property_state,
        "occupancy": occupancy,
        "lock_status": lock_status,
        "rate_pct": rate_pct,
        "risk_flags": risk_flags,
        "event_type": event_type,
        "event_ts": event_ts,
        "source": source,
        "received_ts": received_ts,
        "ingest_version": "ingest-v0",
    }
    norm_hash = blake2b_hex(canonical_json(norm_payload))
    stable_key = blake2b_hex(f"{loan_id}|{lender_id}|{event_type}|{event_ts}")

    # 4) Quality score (simple, transparent)
    quality = _quality_score(
        loan_id=bool(loan_id and looks_like_loan_id(loan_id)),
        lender_id=bool(lender_id),
        enums_ok=all([channel, purpose, stage, occupancy, lock_status, event_type]),
        ranges_ok=_ranges_ok(fico, dti_pct, ltv_pct, rate_pct, loan_amount),
        issues=len(issues),
    )

    return NormalizedLoanEvent(
        loan_id=loan_id or "UNKNOWN",
        lender_id=lender_id or "UNKNOWN",
        channel=channel or "retail",
        purpose=purpose or "other",
        stage=stage or "lead",
        fico=fico or 0,
        dti_pct=dti_pct,
        ltv_pct=ltv_pct,
        loan_amount=loan_amount,
        property_state=property_state or "NA",
        occupancy=occupancy or "primary",
        lock_status=lock_status or "unlocked",
        rate_pct=rate_pct,
        risk_flags=risk_flags,
        event_type=event_type or "snapshot",
        event_ts=event_ts,
        source=source,
        received_ts=received_ts,
        quality_score=quality,
        issues=issues,
        fingerprints={
            "raw_hash": raw_hash,
            "norm_hash": norm_hash,
            "stable_key": stable_key,
        },
    )


def normalize_stream(rows: Iterable[Dict[str, Any]], *, default_source: str = "csv_bulk") -> Iterator[NormalizedLoanEvent]:
    """
    Normalize a stream of dict rows. Avoids raising; yields events with issues[] populated.
    """
    for row in rows:
        try:
            yield normalize_event(row, default_source=default_source)
        except Exception as e:
            # Last-resort guard: produce a stub with the error captured.
            received_ts = iso_now()
            payload = {"row": str(row)[:500], "error": str(e)}
            yield NormalizedLoanEvent(
                loan_id="UNKNOWN",
                lender_id="UNKNOWN",
                channel="retail",
                purpose="other",
                stage="lead",
                fico=0,
                dti_pct=0.0,
                ltv_pct=0.0,
                loan_amount=0.0,
                property_state="NA",
                occupancy="primary",
                lock_status="unlocked",
                rate_pct=0.0,
                risk_flags=["ingest_error"],
                event_type="snapshot",
                event_ts=received_ts,
                source=default_source,
                received_ts=received_ts,
                quality_score=0.0,
                issues=[f"exception.{type(e).__name__}"],
                fingerprints={"raw_hash": blake2b_hex(canonical_json(payload)), "norm_hash": "", "stable_key": ""},
            )


# ─────────────────────────────────────────────────────────────────────
# Coercers & validators (tight, predictable)
# ─────────────────────────────────────────────────────────────────────

def _as_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s or None

def _coerce_loan_id(x: Any) -> Optional[str]:
    s = _as_str(x)
    if not s:
        return None
    s = s.upper()
    return s if looks_like_loan_id(s) else s  # let feature layer decide strictness

def _as_int(x: Any, *, lo: int, hi: int, soft: Tuple[int, int], issues: List[str], label: str) -> Optional[int]:
    if x is None or x == "":
        issues.append(f"missing.{label}")
        return None
    try:
        v = int(float(str(x).replace(",", "")))
    except Exception:
        issues.append(f"type.{label}")
        return None
    if v < lo or v > hi:
        issues.append(f"range.{label}.hard")
    if v < soft[0] or v > soft[1]:
        issues.append(f"range.{label}.soft")
    return v

def _as_float(x: Any, *, lo: float, hi: float, issues: List[str], label: str) -> float:
    if x is None or x == "":
        issues.append(f"missing.{label}")
        return 0.0
    try:
        v = float(str(x).replace(",", "").replace("%", ""))
    except Exception:
        issues.append(f"type.{label}")
        return 0.0
    if math.isnan(v) or math.isinf(v):
        issues.append(f"type.{label}.naninf")
        return 0.0
    if v < lo or v > hi:
        issues.append(f"range.{label}.hard")
    return v

def _as_pct(x: Any, issues: List[str], label: str, *, lo: float, hi: float) -> float:
    """Accepts 0..1 or 0..100 or '45%' and normalizes to 0..100."""
    if x is None or x == "":
        issues.append(f"missing.{label}")
        return 0.0
    s = str(x).strip()
    try:
        if s.endswith("%"):
            v = float(s[:-1])
        else:
            v = float(s.replace(",", ""))
            if v <= 1.0:  # treat small values as ratio
                v *= 100.0
    except Exception:
        issues.append(f"type.{label}")
        return 0.0
    if v < lo or v > hi:
        issues.append(f"range.{label}.hard")
    return v

def _as_state(x: Any, issues: List[str]) -> Optional[str]:
    s = _as_str(x)
    if not s:
        issues.append("missing.property_state")
        return None
    s = s.upper()
    if not re.fullmatch(r"[A-Z]{2}", s):
        issues.append("format.property_state")
    return s

def _enum(field: str, x: Any, issues: List[str]) -> Optional[str]:
    s = _as_str(x)
    if not s:
        issues.append(f"missing.{field}")
        return None
    s = s.strip().lower().replace(" ", "_")
    mapped = ENUM_MAP.get(field, {}).get(s)
    if not mapped:
        issues.append(f"enum.{field}.unknown:{s}")
    return mapped or s

def _as_flags(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        # Allow comma or semicolon separated
        parts = re.split(r"[;,]", x)
        return [p.strip().lower().replace(" ", "_") for p in parts if p.strip()]
    if isinstance(x, (list, tuple)):
        return [str(p).strip().lower().replace(" ", "_") for p in x if str(p).strip()]
    return [str(x).strip().lower()]

def _ranges_ok(fico: Optional[int], dti: float, ltv: float, rate: float, amount: float) -> bool:
    ok = True
    if fico is None or fico < 300 or fico > 850:
        ok = False
    if dti < 0 or dti > 100:
        ok = False
    if ltv < 0 or ltv > 200:
        ok = False
    if rate < 0 or rate > 30:
        ok = False
    if amount <= 0:
        ok = False
    return ok

def _quality_score(*, loan_id: bool, lender_id: bool, enums_ok: bool, ranges_ok: bool, issues: int) -> float:
    base = 60.0
    if loan_id: base += 10
    if lender_id: base += 10
    if enums_ok: base += 10
    if ranges_ok: base += 10
    penalty = min(issues * 2.0, 25.0)
    return clamp(base - penalty, 0.0, 100.0)


# ─────────────────────────────────────────────────────────────────────
# Tiny smoke test (manual)
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":  # quick local sanity check
    raw = {
        "loanId": "ab12cd34",
        "lenderId": "LEND-001",
        "Channel": "Broker",
        "LoanPurpose": "Refinance",
        "Stage": "UW",
        "FICO": "742",
        "DTI": "36.5%",
        "LTV": "0.78",
        "LoanAmount": "425,000",
        "State": "fl",
        "Occupancy": "Owner",
        "LockStatus": "Locked",
        "Rate": "6.625",
        "riskFlags": ["thin_file", "income_gap"],
        "eventType": "status_change",
        "eventTime": "2025-10-28T13:22:10Z",
        "source": "vendor_api",
    }
    ev = normalize_event(raw)
    print(ev.summarize())
    print(ev.to_dict())
