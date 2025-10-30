# hydra-core/fraud/alert_writer.py
"""
AFPS Alert Writer (MVP)
-----------------------
Purpose:
  • Persist AFPS alerts as append-only JSONL with deterministic governance hashes.
  • Keep it boring: one file per day, rotate on size, no external deps.
  • Hand back a concise, HydraVoice-ready line to display in UI / Grafana.

Design notes:
  • Schema-first: small, stable envelope + free-form 'facts' and 'hints'.
  • Deterministic hash covers only stable fields to avoid timestamp churn.
  • Ready for later swap to Kafka/S3/DB — callers won't change.

File layout:
  data/alerts/YYYY/MM/DD/alerts-YYYYMMDD.jsonl

Core APIs:
  • writer = AlertWriter()
  • rec = writer.write_alert(...)

"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import datetime

from utils.utils import (
    ensure_dir,
    canonical_json,
    blake2b_hex,
    utc_iso,
    clamp,
)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

ALERT_SCHEMA_VERSION = "afps-alert-v0"
POLICY_ID = "afps-policy-2025-10-a"

DEFAULT_BASE_DIR = os.environ.get("HYDRA_ALERT_DIR", "data/alerts")
MAX_FILE_SIZE_MB_DEFAULT = int(os.environ.get("HYDRA_ALERT_MAX_MB", "64"))

# ─────────────────────────────────────────────────────────────────────
# Dataclasses (lightweight type hints)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ScoreBand:
    band: str           # "LOW" | "MED" | "HIGH" | "CRITICAL"
    dfi: float          # 0..100
    afps: float         # 0..100

# ─────────────────────────────────────────────────────────────────────
# Alert Writer
# ─────────────────────────────────────────────────────────────────────

class AlertWriter:
    def __init__(
        self,
        base_dir: str = DEFAULT_BASE_DIR,
        max_file_size_mb: int = MAX_FILE_SIZE_MB_DEFAULT,
    ):
        self.base_dir = base_dir
        self.max_bytes = max_file_size_mb * 1024 * 1024

    # Public -----------------------------------------------------------------

    def write_alert(
        self,
        *,
        tenant_id: str,
        loan_id: str,
        borrower_id: Optional[str],
        score_band: ScoreBand,
        reasons: Iterable[str],
        hints: Iterable[Dict[str, Any]],               # from fraud.graph_engine.hints_for_loan(...)
        remediation: Iterable[str],                     # concrete next actions
        facts: Optional[Dict[str, Any]] = None,         # structured, e.g. feature deltas
        meta: Optional[Dict[str, Any]] = None,          # free-form tracing: pipeline_id, model_id, etc.
    ) -> Dict[str, Any]:
        """
        Build record, compute deterministic governance hash, append to JSONL,
        and return a compact summary for immediate use in UI/voice.
        """
        now = datetime.utcnow()
        # stable envelope
        record = {
            "schema_version": ALERT_SCHEMA_VERSION,
            "policy_id": POLICY_ID,
            "tenant_id": tenant_id,
            "loan_id": loan_id,
            "borrower_id": borrower_id,
            "scores": {
                "band": score_band.band.upper(),
                "dfi": round(clamp(score_band.dfi, 0.0, 100.0), 2),
                "afps": round(clamp(score_band.afps, 0.0, 100.0), 2),
            },
            "reasons": list(dict.fromkeys([str(r).strip() for r in reasons if str(r).strip()])),
            "hints": self._normalize_hints(hints),
            "remediation": list(dict.fromkeys([str(r).strip() for r in remediation if str(r).strip()])),
            "facts": facts or {},
            "meta": meta or {},
            "timestamps": {
                "ts_iso": utc_iso(now),
                "ts_epoch": int(now.timestamp()),
            },
        }

        # governance hash excludes volatile timestamps/meta to keep diff noise low
        hash_payload = {
            "schema_version": record["schema_version"],
            "policy_id": record["policy_id"],
            "tenant_id": record["tenant_id"],
            "loan_id": record["loan_id"],
            "scores": record["scores"],
            "reasons": record["reasons"],
            "hints": [(h.get("link_to"), h.get("via"), h.get("band")) for h in record["hints"][:8]],
            "remediation": record["remediation"],
            "facts": record["facts"],
        }
        record["governance"] = {
            "alert_hash": blake2b_hex(canonical_json(hash_payload)),
        }

        # append to file
        path = self._current_file_path(now)
        ensure_dir(os.path.dirname(path))
        self._rotate_if_needed(path)
        offset = self._append_jsonl(path, record)

        # build UI/Voice line
        voice_line = self.compose_voice_line(record)

        return {
            "alert_path": path,
            "offset": offset,
            "hash": record["governance"]["alert_hash"],
            "band": record["scores"]["band"],
            "dfi": record["scores"]["dfi"],
            "afps": record["scores"]["afps"],
            "loan_id": loan_id,
            "voice_line": voice_line,
        }

    # Utilities ---------------------------------------------------------------

    def compose_voice_line(self, record: Dict[str, Any]) -> str:
        """
        Short, punchy narrative for Grafana/HydraVoice panels.
        Uses negative visualization (cost & throughput drag) if present.
        """
        loan = record["loan_id"]
        band = record["scores"]["band"]
        dfi = record["scores"]["dfi"]
        afps = record["scores"]["afps"]

        cost_impact = None
        tput_impact = None

        facts = record.get("facts") or {}
        if isinstance(facts, dict):
            ci = facts.get("estimated_cost_impact_usd")
            tp = facts.get("estimated_throughput_delta_loans")
            cost_impact = f"${int(ci):,}" if isinstance(ci, (int, float)) else None
            tput_impact = f"{int(tp)} loans" if isinstance(tp, (int, float)) else None

        # pick strongest hint
        hint_txt = ""
        hints = record.get("hints") or []
        if hints:
            h = hints[0]
            via = h.get("via", "signal")
            link_to = h.get("link_to", "")
            band_hint = h.get("band", "")
            hint_txt = f" | link: {via}→{link_to} [{band_hint}]"

        neg_viz = ""
        if cost_impact and tput_impact:
            neg_viz = f" | risk of ~{cost_impact} loss and {tput_impact}/mo drag if ignored"
        elif cost_impact:
            neg_viz = f" | risk of ~{cost_impact} loss if ignored"
        elif tput_impact:
            neg_viz = f" | risk of ~{tput_impact}/mo drag if ignored"

        return f"Loan {loan}: band={band}, DFI={dfi}, AFPS={afps}{hint_txt}{neg_viz}"

    # Internals ---------------------------------------------------------------

    def _normalize_hints(self, hints: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for h in hints or []:
            if not isinstance(h, dict):
                continue
            out.append({
                "link_to": str(h.get("link_to", "")),
                "via": str(h.get("via", "")),
                "strength": int(h.get("strength", 0)) if str(h.get("strength", "")).isdigit() else h.get("strength", 0),
                "band": str(h.get("band", "LOW")).upper(),
                "notes": str(h.get("notes", ""))[:240],
            })
        return out[:8]

    def _current_file_path(self, now: datetime) -> str:
        y = now.strftime("%Y")
        m = now.strftime("%m")
        d = now.strftime("%d")
        fname = f"alerts-{y}{m}{d}.jsonl"
        return os.path.join(self.base_dir, y, m, d, fname)

    def _rotate_if_needed(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            size = os.path.getsize(path)
        except OSError:
            return
        if size < self.max_bytes:
            return
        # rotate by suffix .N (monotonic)
        base, ext = os.path.splitext(path)
        n = 1
        while os.path.exists(f"{base}.{n}{ext}"):
            n += 1
        os.rename(path, f"{base}.{n}{ext}")

    def _append_jsonl(self, path: str, record: Dict[str, Any]) -> int:
        # Return byte offset (rough locator for audits)
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        encoded = line.encode("utf-8")
        with open(path, "ab") as f:
            offset = f.tell()
            f.write(encoded)
            return offset


# ─────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from fraud.graph_engine import hints_for_loan, ingest_event_links
    # Minimal demo data
    ingest_event_links(
        loan_id="LN-1001",
        borrower_id="B-77",
        email_fp="e:alpha",
        addr_fp="a:oaks",
        device_fp="d:xyz",
        risk_flags=["income_gap"],
    )
    ingest_event_links(
        loan_id="LN-1002",
        borrower_id="B-19",
        email_fp="e:alpha",
        addr_fp="a:oaks",
        device_fp="d:xyz",
        risk_flags=["doc_missing"],
    )
    hints = hints_for_loan("LN-1002", "B-19", "e:alpha", "a:oaks", "d:xyz", ["doc_missing"])
    writer = AlertWriter()
    rec = writer.write_alert(
        tenant_id="tenant-acme",
        loan_id="LN-1002",
        borrower_id="B-19",
        score_band=ScoreBand(band="HIGH", dfi=72.4, afps=81.3),
        reasons=["shared_device_cluster", "email_cooccurs_with_high_risk_loans"],
        hints=hints,
        remediation=[
            "Verify employer income docs within 24h",
            "Trigger KBA on borrower and freeze e-sign until pass",
        ],
        facts={
            "estimated_cost_impact_usd": 1800,
            "estimated_throughput_delta_loans": 12,
            "window_days": 30,
        },
        meta={"pipeline_id": "afps-mvp", "env": "local"},
    )
    print("Wrote alert:", rec["alert_path"], "hash:", rec["hash"])
    print("Voice:", rec["voice_line"])
