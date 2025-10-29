# hydra-core/core/governance.py
"""
Governance — Audit, Hashing, and Signed Export (MVP)
----------------------------------------------------
Goals:
  • Deterministic hashing for audit trails (short and long forms).
  • Canonical JSON serialization so hashes are stable across runs.
  • Lightweight signing/verification for result packets (no heavy deps).
  • Zero external calls; easy to swap in real KMS/HSM later.

Design Notes:
  • Prefers BLAKE3 if available (fast, modern). Falls back to SHA256.
  • Canonical JSON: sorted keys, no whitespace variance, UTF-8 bytes.
  • "Seal" uses HMAC-SHA256 over canonical bytes (keyed with a tenant/cluster secret).
    This is NOT encryption; it’s tamper-evident signing. Good enough for MVP.
  • KMS hooks are interface stubs so prod can wire AWS KMS without touching call sites.

Public Surface (used by Supervisor/API):
  • canonical_json(obj) -> bytes
  • audit_hash(obj) -> str (16-hex short)
  • long_hash(obj) -> str (64-hex SHA256/BLAKE3)
  • sign_packet(packet_bytes, key) -> str (base64 token)
  • verify_packet(packet_bytes, token, key) -> bool
  • make_audit_record(tenant_id, dfi, afps, voice, timings, degraded, reason) -> dict
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional

try:
    from pydantic import BaseModel, Field, ConfigDict
except Exception:  # pragma: no cover
    class BaseModel:  # type: ignore
        pass
    def Field(*args, **kwargs):  # type: ignore
        return None
    ConfigDict = dict  # type: ignore

# Optional blake3 acceleration
try:
    import blake3  # type: ignore
    _HAS_BLAKE3 = True
except Exception:  # pragma: no cover
    blake3 = None
    _HAS_BLAKE3 = False


# ---------------------------------------------------------------------
# Config Models
# ---------------------------------------------------------------------

class GovernanceConfig(BaseModel):
    """
    Governance knobs. Keys should be provided by settings or secret store.
    """
    signing_key_b64: Optional[str] = Field(default=None, description="Base64 key for HMAC signing")
    use_blake3: bool = Field(default=True, description="Prefer blake3 when available")
    kid: str = Field(default="mvp-local", description="Key ID tag for signatures")
    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------
# Canonicalization & Hashing
# ---------------------------------------------------------------------

def canonical_json(obj: Any) -> bytes:
    """
    Stable JSON encoding: sorted keys, no whitespace, UTF-8.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _blake3_hex(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()  # type: ignore


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def long_hash(obj: Any, prefer_blake3: bool = True) -> str:
    """
    64-hex digest for storage/chain-linking.
    """
    data = canonical_json(obj)
    if prefer_blake3 and _HAS_BLAKE3:
        return _blake3_hex(data)
    return _sha256_hex(data)


def audit_hash(obj: Any, prefer_blake3: bool = True) -> str:
    """
    Short 16-hex for dashboards/ids (not collision-proof like the long form).
    """
    h = long_hash(obj, prefer_blake3=prefer_blake3)
    return h[:16]


# ---------------------------------------------------------------------
# Signing / Verification (tamper-evident seal)
# ---------------------------------------------------------------------

def _key_bytes(cfg: GovernanceConfig) -> Optional[bytes]:
    if not cfg.signing_key_b64:
        return None
    try:
        return base64.urlsafe_b64decode(cfg.signing_key_b64.encode("utf-8"))
    except Exception:
        return None


def sign_packet(packet_bytes: bytes, cfg: GovernanceConfig) -> Optional[str]:
    """
    HMAC-SHA256 signature over canonical bytes.
    Returns compact token: base64url(hmac || kid)
    """
    key = _key_bytes(cfg)
    if not key:
        return None
    mac = hmac.new(key, packet_bytes, hashlib.sha256).digest()
    token = base64.urlsafe_b64encode(mac + b"." + cfg.kid.encode("utf-8")).decode("utf-8").rstrip("=")
    return token


def verify_packet(packet_bytes: bytes, token: str, cfg: GovernanceConfig) -> bool:
    key = _key_bytes(cfg)
    if not key or not token:
        return False
    try:
        raw = base64.urlsafe_b64decode(_pad(token))
        mac, dot, kid = raw.partition(b".")
        if dot != b".":
            return False
        calc = hmac.new(key, packet_bytes, hashlib.sha256).digest()
        return hmac.compare_digest(mac, calc)
    except Exception:
        return False


def _pad(t: str) -> bytes:
    # restore padding for urlsafe base64
    missing = (4 - len(t) % 4) % 4
    return (t + "=" * missing).encode("utf-8")


# ---------------------------------------------------------------------
# Audit Record Assembly
# ---------------------------------------------------------------------

class AuditStamp(BaseModel):
    at_epoch: float = Field(default_factory=lambda: round(time.time(), 3))
    hash16: str = ""
    hash_algo: str = Field(default="blake3" if _HAS_BLAKE3 else "sha256")
    kid: str = "mvp-local"
    model_config = ConfigDict(extra="ignore")


def make_audit_record(
    tenant_id: str,
    dfi: Any,
    afps: Any,
    voice: Any,
    timings: Any,
    degraded: bool,
    reason: str,
    cfg: Optional[GovernanceConfig] = None,
) -> Dict[str, Any]:
    """
    Compose a compact, signed audit record for storage/streaming.
    Safe to log in JSON (contains no PII by design if upstream payloads are clean).
    """
    gc = cfg or GovernanceConfig()
    body = {
        "tenant_id": tenant_id,
        "dfi_score": round(float(getattr(dfi, "dfi_score", 0.0)), 3),
        "afps_score": round(float(getattr(afps, "afps_score", 0.0)), 3),
        "p_fund": round(float(getattr(afps, "p_fund", 0.0)), 6),
        "band": getattr(afps, "band", "UNKNOWN"),
        "headline": getattr(voice, "headline", ""),
        "timings_ms": {
            "fi": int(getattr(timings, "fi_ms", 0)),
            "afps": int(getattr(timings, "afps_ms", 0)),
            "voice": int(getattr(timings, "voice_ms", 0)),
        },
        "degraded": bool(degraded),
        "reason": reason or "",
    }

    # hashes
    h16 = audit_hash(
        {
            "tenant_id": tenant_id,
            "dfi": body["dfi_score"],
            "afps": body["afps_score"],
            "p": body["p_fund"],
            "band": body["band"],
            "headline": body["headline"],
        },
        prefer_blake3=gc.use_blake3,
    )
    stamp = AuditStamp(hash16=h16, kid=gc.kid, hash_algo=("blake3" if (gc.use_blake3 and _HAS_BLAKE3) else "sha256"))

    # signature (optional)
    pkt_bytes = canonical_json({"stamp": stamp.__dict__, "body": body})
    sig = sign_packet(pkt_bytes, gc)

    record = {
        "stamp": stamp.__dict__,
        "body": body,
        "sig": sig,  # None if no signing key configured
    }
    return record


# ---------------------------------------------------------------------
# KMS/HSM Interfaces (stubs for later swap-in)
# ---------------------------------------------------------------------

class KMSClient(BaseModel):
    """
    Interface placeholder for a real KMS/HSM.
    Swap this out with AWS KMS or GCP KMS client in production.
    """
    model_config = ConfigDict(extra="ignore")

    def sign(self, payload: bytes, key_id: str) -> bytes:  # pragma: no cover
        raise NotImplementedError("Wire a real KMS before enabling")

    def verify(self, payload: bytes, signature: bytes, key_id: str) -> bool:  # pragma: no cover
        raise NotImplementedError("Wire a real KMS before enabling")


# ---------------------------------------------------------------------
# Redaction helpers (PII safety in logs/exports)
# ---------------------------------------------------------------------

_REDACT_KEYS = {"ssn", "dob", "email", "phone", "name", "address", "ein", "itin"}

def redact(obj: Any) -> Any:
    """
    Best-effort redaction: masks common PII-like keys in nested dict/list structures.
    Safe on non-collection types (returns as-is).
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).lower() in _REDACT_KEYS:
                out[k] = "***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    return obj
