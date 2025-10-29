# hydra-core/telemetry/suite.py
"""
Hydra Telemetry Suite
---------------------
Purpose:
  • Produce a compact JSON snapshot for dashboards & audits:
      - health: env, version, uptime, last_score_at
      - sanity: config flags, clock skew, input entropy
      - drift: PSI/KS by feature (if provided)
  • Keep it boring, testable, and explainable.
  • No PII in outputs. Stable keys for Grafana JSON panels.

Design:
  - Stateless calculators + a tiny TTL cache for the last snapshot.
  - PSI / KS are simple, bounded, and documented.
  - Optional FastAPI binder to serve GET /telemetry (application/json).
  - Integrates with telemetry.exporter.metrics for drift gauges.

Inputs:
  • ref_dist / cur_dist per feature are simple histograms:
      {"bins": [0,10,20,30,40,50,60,70,80,90,100], "counts": [..]}
    or already-normalized densities "probs": [..]
  • last_events: dict with minimal timing context:
      {"last_score_ts": epoch_seconds, "last_request_ts": epoch_seconds}

Usage:
  from telemetry.suite import TelemetrySuite
  suite = TelemetrySuite(env="sandbox", model="DFI_AFPS_MVP", version="v1")
  snapshot = suite.snapshot(tenant_id="t1", drift_inputs={...}, last_events={...})
  suite.attach_fastapi(app)  # exposes /telemetry

Notes:
  • PSI rule of thumb: <0.1 little/no drift, 0.1–0.25 moderate, >0.25 significant.
  • KS rule of thumb: <0.1 small, 0.1–0.2 medium, >0.2 large distribution shift.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Dict, List, Optional, Tuple, Any

from .exporter import metrics  # Prometheus helper (no-op labels if not set)

_EPS = 1e-12


def _now() -> float:
    return time.time()


def _safe_ratio(p: float, q: float) -> float:
    """Avoid division by zero for PSI."""
    p = max(p, _EPS)
    q = max(q, _EPS)
    return p / q


def _normalize_counts(counts: List[float]) -> List[float]:
    total = float(sum(counts))
    if total <= 0:
        # Uniform fallback to avoid NaNs
        n = max(1, len(counts))
        return [1.0 / n] * n
    return [c / total for c in counts]


def psi_from_histograms(ref: List[float], cur: List[float]) -> float:
    """
    Population Stability Index:
      PSI = sum( (p_i - q_i) * ln(p_i / q_i) ), p=cur, q=ref
    Inputs are raw counts or probabilities; we normalize internally.
    """
    p = _normalize_counts(cur)
    q = _normalize_counts(ref)
    psi = 0.0
    for pi, qi in zip(p, q):
        psi += (pi - qi) * math.log(_safe_ratio(pi, qi))
    # PSI is non-negative in theory; clamp small negatives from FP error
    return float(max(0.0, psi))


def cdf_from_counts(counts: List[float]) -> List[float]:
    probs = _normalize_counts(counts)
    cdf = []
    acc = 0.0
    for p in probs:
        acc += p
        cdf.append(min(1.0, acc))
    return cdf


def ks_from_histograms(ref: List[float], cur: List[float]) -> float:
    """
    Kolmogorov–Smirnov statistic between two discrete distributions
    given as hist counts/probs (same binning).
    """
    cdf_ref = cdf_from_counts(ref)
    cdf_cur = cdf_from_counts(cur)
    diffs = [abs(a - b) for a, b in zip(cdf_ref, cdf_cur)]
    return float(max(diffs) if diffs else 0.0)


def _entropy(probs: List[float]) -> float:
    probs = [max(_EPS, p) for p in _normalize_counts(probs)]
    return -sum(p * math.log(p) for p in probs)


def _bounded_entropy(counts: List[float]) -> Tuple[float, float]:
    """Return (entropy, entropy_max) for normalization to [0,1]."""
    n = max(1, len(counts))
    h = _entropy(counts)
    h_max = math.log(n)
    return h, h_max


def _fmt_seconds(s: float) -> float:
    return round(float(s), 6)


class TelemetrySuite:
    def __init__(
        self,
        *,
        env: Optional[str] = None,
        model: str = "DFI_AFPS_MVP",
        version: str = "v1",
        ttl_seconds: int = 10,
        build_id: Optional[str] = None,
        boot_ts: Optional[float] = None,
    ) -> None:
        self.env = (env or os.getenv("HYDRA_ENV") or "sandbox").lower()
        self.model = model
        self.version = version
        self.build_id = build_id or os.getenv("HYDRA_BUILD_ID", "")
        self.boot_ts = float(boot_ts or _now())
        self.ttl = int(ttl_seconds)

        self._cache_lock = threading.Lock()
        self._cache_expiry = 0.0
        self._cache: Dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────
    def snapshot(
        self,
        *,
        tenant_id: Optional[str] = None,
        drift_inputs: Optional[Dict[str, Dict[str, List[float]]]] = None,
        last_events: Optional[Dict[str, float]] = None,
        config_flags: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build or return a cached JSON-friendly snapshot.
        drift_inputs example:
          {
            "feature_name": {
               "ref":   [c1, c2, ...],
               "cur":   [c1, c2, ...],
               "bins":  [optional bin edges],
            },
            ...
          }
        last_events example:
          {"last_score_ts": 1730000000.12, "last_request_ts": 1730000101.03}
        """
        now = _now()
        with self._cache_lock:
            if now <= self._cache_expiry and self._cache:
                return self._cache

        # Health
        health = self._health(now, last_events or {})

        # Sanity
        sanity = self._sanity(now, last_events or {}, config_flags or {})

        # Drift (and export to Prometheus gauges)
        drift, drift_summary = self._drift_block(tenant_id, drift_inputs or {})

        snap = {
            "env": self.env,
            "model": self.model,
            "version": self.version,
            "build_id": self.build_id,
            "timestamp": now,
            "uptime_seconds": _fmt_seconds(now - self.boot_ts),
            "health": health,
            "sanity": sanity,
            "drift": drift,
            "drift_summary": drift_summary,
        }

        with self._cache_lock:
            self._cache = snap
            self._cache_expiry = now + self.ttl
        return snap

    def snapshot_json(self, **kwargs) -> str:
        return json.dumps(self.snapshot(**kwargs), separators=(",", ":"), ensure_ascii=False)

    def attach_fastapi(self, app, route: str = "/telemetry") -> None:
        """
        Adds a read-only JSON endpoint to a FastAPI app.
        """
        @app.get(route)
        async def _telemetry():
            snap = self.snapshot()
            return snap  # FastAPI will serialize to JSON

    # ──────────────────────────────────────────────────────────────────
    # Sections
    # ──────────────────────────────────────────────────────────────────
    def _health(self, now: float, last: Dict[str, float]) -> Dict[str, Any]:
        last_score = float(last.get("last_score_ts", 0.0))
        last_req = float(last.get("last_request_ts", 0.0))
        return {
            "status": "up",
            "clock": now,
            "last_score_ts": last_score,
            "last_request_ts": last_req,
            "age_since_last_score_s": _fmt_seconds(now - last_score) if last_score > 0 else None,
            "age_since_last_request_s": _fmt_seconds(now - last_req) if last_req > 0 else None,
        }

    def _sanity(
        self,
        now: float,
        last: Dict[str, float],
        flags: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Lightweight sanity signals:
          - clock skew (relative to env var or given ts)
          - input entropy (if provided in flags["entropy_counts"])
          - config posture (booleans only, no secrets)
        """
        skew_ref = float(flags.get("skew_ref_ts", now))
        clock_skew_s = _fmt_seconds(now - skew_ref)

        entropy_counts = flags.get("entropy_counts", None)  # Optional list[int]
        entropy_norm = None
        if isinstance(entropy_counts, list) and entropy_counts:
            h, h_max = _bounded_entropy([float(x) for x in entropy_counts])
            entropy_norm = round(h / max(_EPS, h_max), 6)

        # Only allow safe, boolean-ish config echoes
        safe_cfg = {}
        for k, v in (flags or {}).items():
            if k in ("skew_ref_ts", "entropy_counts"):
                continue
            if isinstance(v, (bool, int, float, str)):
                safe_cfg[k] = v

        return {
            "clock_skew_s": clock_skew_s,
            "input_entropy_norm": entropy_norm,
            "config_echo": safe_cfg,
        }

    def _drift_block(
        self,
        tenant_id: Optional[str],
        drift_inputs: Dict[str, Dict[str, List[float]]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compute PSI / KS for each feature; update Prometheus gauges.
        Returns (per_feature_map, summary_stats).
        """
        per_feature: Dict[str, Dict[str, Any]] = {}
        psi_vals: List[float] = []
        ks_vals: List[float] = []

        for feature, payload in (drift_inputs or {}).items():
            ref = payload.get("ref", [])
            cur = payload.get("cur", [])
            if not ref or not cur or len(ref) != len(cur):
                continue

            psi = psi_from_histograms(ref, cur)
            ks = ks_from_histograms(ref, cur)

            # Export to Prometheus (tenant-safe, no PII)
            try:
                metrics.set_drift(tenant_id or "unknown", feature, psi=psi, ks=ks)
            except Exception:
                pass  # metrics may not be initialized yet during early boot/tests

            per_feature[feature] = {
                "psi": round(float(psi), 6),
                "ks": round(float(ks), 6),
                "bins": payload.get("bins", None),
            }
            psi_vals.append(float(psi))
            ks_vals.append(float(ks))

        summary = {
            "features_evaluated": len(per_feature),
            "psi_max": round(max(psi_vals), 6) if psi_vals else None,
            "psi_median": round(_median(psi_vals), 6) if psi_vals else None,
            "ks_max": round(max(ks_vals), 6) if ks_vals else None,
            "ks_median": round(_median(ks_vals), 6) if ks_vals else None,
            "psi_flag": _psi_flag(max(psi_vals) if psi_vals else 0.0),
            "ks_flag": _ks_flag(max(ks_vals) if ks_vals else 0.0),
        }
        return per_feature, summary


# ─────────────────────────────────────────────────────────────────────
# Small utilities (kept here to avoid extra imports)
# ─────────────────────────────────────────────────────────────────────
def _median(xs: List[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    xs_sorted = sorted(xs)
    mid = n // 2
    if n % 2:
        return xs_sorted[mid]
    return 0.5 * (xs_sorted[mid - 1] + xs_sorted[mid])


def _psi_flag(psi: float) -> str:
    if psi < 0.1:
        return "low"
    if psi < 0.25:
        return "moderate"
    return "high"


def _ks_flag(ks: float) -> str:
    if ks < 0.1:
        return "low"
    if ks < 0.2:
        return "moderate"
    return "high"
