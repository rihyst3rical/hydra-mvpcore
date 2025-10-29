# hydra-core/telemetry/exporter.py
"""
Hydra Telemetry Exporter (Prometheus + optional OTEL)
-----------------------------------------------------
Design goals:
  • Zero-drama metrics for MVP: Prometheus first, OTEL optional.
  • One tiny API for the rest of the app: inc/set/observe/timeit.
  • Stable, minimal label set (no PII): tenant_id, model, version, env.
  • Safe in single-process and gunicorn/uvicorn multi-worker modes.
  • A lightweight /metrics FastAPI route binder.

What you get:
  - Counters:
      hydra_requests_total
      hydra_scores_total
  - Gauges:
      hydra_up
      hydra_drift_psi{feature=...}
      hydra_drift_ks{feature=...}
  - Histograms:
      hydra_request_latency_seconds
      hydra_score_latency_seconds
      hydra_dfi_score
      hydra_afps_score

Usage:
  from telemetry.exporter import metrics
  metrics.init(env="sandbox")
  metrics.attach_fastapi(app)  # exposes /metrics
  metrics.inc("hydra_requests_total", tenant_id="t1")
  with metrics.timeit("hydra_score_latency_seconds", tenant_id="t1"):
      ... do work ...
  metrics.observe("hydra_dfi_score", 73.2, tenant_id="t1")

Notes:
  • Do not pass PII as labels. Ever.
  • If OTEL libs/env are absent, OTEL is silently disabled (no errors).
"""

from __future__ import annotations

import os
import time
import logging
from contextlib import contextmanager
from typing import Dict, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    CONTENT_TYPE_LATEST,
    generate_latest,
)

# Optional OpenTelemetry (no hard dependency)
try:
    from opentelemetry import metrics as otel_metrics  # type: ignore
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter  # type: ignore
    from opentelemetry.sdk.metrics import MeterProvider  # type: ignore
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader  # type: ignore
    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover
    _OTEL_AVAILABLE = False


log = logging.getLogger("hydra.telemetry")


class _Metrics:
    # Prometheus registry + instruments
    _registry: Optional[CollectorRegistry] = None
    _gauges: Dict[str, Gauge] = {}
    _counters: Dict[str, Counter] = {}
    _histos: Dict[str, Histogram] = {}
    _env: str = "sandbox"
    _model: str = "DFI_AFPS_MVP"
    _version: str = "v1"

    # OpenTelemetry (optional)
    _otel_enabled: bool = False
    _otel_meter = None

    # ──────────────────────────────────────────────────────────────────
    # Init & wiring
    # ──────────────────────────────────────────────────────────────────
    def init(
        self,
        *,
        env: str = None,
        model: str = None,
        version: str = None,
        enable_otel: Optional[bool] = None,
        otlp_endpoint: Optional[str] = None,
        otlp_interval_sec: int = 30,
    ) -> None:
        """
        Initialize the metrics registry and create standard instruments.
        Safe to call multiple times; subsequent calls are ignored.
        """
        if self._registry is not None:
            return

        self._env = (env or os.getenv("HYDRA_ENV") or "sandbox").lower()
        self._model = model or os.getenv("HYDRA_MODEL", "DFI_AFPS_MVP")
        self._version = version or os.getenv("HYDRA_MODEL_VERSION", "v1")

        self._registry = CollectorRegistry(auto_describe=True)

        # Counters
        self._counters["hydra_requests_total"] = Counter(
            "hydra_requests_total",
            "Total API requests received",
            ["tenant_id", "env"],
            registry=self._registry,
        )
        self._counters["hydra_scores_total"] = Counter(
            "hydra_scores_total",
            "Total score computations (DFI/AFPS)",
            ["tenant_id", "model", "version", "env"],
            registry=self._registry,
        )

        # Gauges
        self._gauges["hydra_up"] = Gauge(
            "hydra_up",
            "Hydra exporter is up (1) / down (0)",
            ["env"],
            registry=self._registry,
        )
        self._gauges["hydra_drift_psi"] = Gauge(
            "hydra_drift_psi",
            "Population Stability Index per feature (lower is better)",
            ["tenant_id", "feature", "env"],
            registry=self._registry,
        )
        self._gauges["hydra_drift_ks"] = Gauge(
            "hydra_drift_ks",
            "KS Statistic per feature (lower is better)",
            ["tenant_id", "feature", "env"],
            registry=self._registry,
        )

        # Histograms
        # Buckets tuned for API latencies and quick scoring paths.
        self._histos["hydra_request_latency_seconds"] = Histogram(
            "hydra_request_latency_seconds",
            "End-to-end request latency",
            ["tenant_id", "route", "env"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5, 10),
            registry=self._registry,
        )
        self._histos["hydra_score_latency_seconds"] = Histogram(
            "hydra_score_latency_seconds",
            "Latency of DFI/AFPS scoring",
            ["tenant_id", "model", "version", "env"],
            buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self._registry,
        )
        # Score distributions (0–100); bucket every 5 points.
        self._histos["hydra_dfi_score"] = Histogram(
            "hydra_dfi_score",
            "Distribution of DFI (0–100)",
            ["tenant_id", "env"],
            buckets=tuple(i for i in range(0, 105, 5)),
            registry=self._registry,
        )
        self._histos["hydra_afps_score"] = Histogram(
            "hydra_afps_score",
            "Distribution of AFPS (0–100)",
            ["tenant_id", "env"],
            buckets=tuple(i for i in range(0, 105, 5)),
            registry=self._registry,
        )

        # Flip 'up' to 1 at init
        try:
            self._gauges["hydra_up"].labels(env=self._env).set(1)
        except Exception as e:  # pragma: no cover
            log.warning("Failed to set hydra_up: %s", e)

        # Optional OpenTelemetry
        if enable_otel is None:
            enable_otel = bool(os.getenv("HYDRA_OTEL_ENABLE", "").strip())

        if enable_otel and _OTEL_AVAILABLE:
            try:
                endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
                if endpoint:
                    reader = PeriodicExportingMetricReader(
                        OTLPMetricExporter(endpoint=endpoint),
                        export_interval_millis=max(1000, int(otlp_interval_sec * 1000)),
                    )
                    provider = MeterProvider(metric_readers=[reader])
                    otel_metrics.set_meter_provider(provider)
                    self._otel_meter = otel_metrics.get_meter("hydra.core")
                    self._otel_enabled = True
                    log.info("OpenTelemetry metrics enabled → %s", endpoint)
                else:
                    log.info("OTEL requested but no endpoint set; skipping.")
            except Exception as e:  # pragma: no cover
                log.warning("OTEL init failed; continuing without OTEL: %s", e)
                self._otel_enabled = False

    # ──────────────────────────────────────────────────────────────────
    # Label helpers (keep PII out)
    # ──────────────────────────────────────────────────────────────────
    def _safe_tenant(self, tenant_id: Optional[str]) -> str:
        return (tenant_id or "unknown")[:32]

    def _base_env(self) -> str:
        return self._env

    def _model_labels(self, tenant_id: Optional[str] = None) -> Dict[str, str]:
        return {
            "tenant_id": self._safe_tenant(tenant_id),
            "model": self._model,
            "version": self._version,
            "env": self._base_env(),
        }

    # ──────────────────────────────────────────────────────────────────
    # Public helpers
    # ──────────────────────────────────────────────────────────────────
    def inc(self, name: str, amount: int = 1, *, tenant_id: Optional[str] = None, **extra_labels) -> None:
        """
        Increment a counter with sane default labels.
        """
        c = self._counters.get(name)
        if not c:
            raise KeyError(f"Unknown counter: {name}")
        labels = {"env": self._base_env()}
        if "hydra_scores_total" == name:
            labels.update(self._model_labels(tenant_id))
        else:
            labels.update({"tenant_id": self._safe_tenant(tenant_id)})
        labels.update(extra_labels)
        c.labels(**labels).inc(amount)

    def set_gauge(self, name: str, value: float, *, tenant_id: Optional[str] = None, **extra_labels) -> None:
        """
        Set a gauge value (e.g., drift stats).
        """
        g = self._gauges.get(name)
        if not g:
            raise KeyError(f"Unknown gauge: {name}")
        labels = {"env": self._base_env()}
        if name in ("hydra_drift_psi", "hydra_drift_ks"):
            labels.update({"tenant_id": self._safe_tenant(tenant_id)})
        labels.update(extra_labels)
        g.labels(**labels).set(value)

    def observe(self, name: str, value: float, *, tenant_id: Optional[str] = None, **extra_labels) -> None:
        """
        Observe a histogram value (latency, score distributions).
        """
        h = self._histos.get(name)
        if not h:
            raise KeyError(f"Unknown histogram: {name}")
        labels = {"env": self._base_env()}
        if name in ("hydra_score_latency_seconds",):
            labels.update(self._model_labels(tenant_id))
        elif name in ("hydra_dfi_score", "hydra_afps_score"):
            labels.update({"tenant_id": self._safe_tenant(tenant_id)})
        else:
            labels.update({"tenant_id": self._safe_tenant(tenant_id)})
        labels.update(extra_labels)
        h.labels(**labels).observe(value)

    @contextmanager
    def timeit(self, name: str, *, tenant_id: Optional[str] = None, **extra_labels):
        """
        Context manager to time a block and record to a histogram.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.observe(name, duration, tenant_id=tenant_id, **extra_labels)

    # ──────────────────────────────────────────────────────────────────
    # FastAPI integration
    # ──────────────────────────────────────────────────────────────────
    def attach_fastapi(self, app, route: str = "/metrics") -> None:
        """
        Adds a /metrics endpoint to a FastAPI app.
        """
        if self._registry is None:
            raise RuntimeError("metrics.init(...) must be called before attach_fastapi()")

        @app.get(route)
        async def _metrics():
            data = generate_latest(self._registry)
            return app.responses.Response(content=data, media_type=CONTENT_TYPE_LATEST)

    # ──────────────────────────────────────────────────────────────────
    # Convenience shims for common events
    # ──────────────────────────────────────────────────────────────────
    def mark_request(self, tenant_id: Optional[str], route: str, duration_s: float) -> None:
        self.inc("hydra_requests_total", tenant_id=tenant_id)
        self.observe("hydra_request_latency_seconds", duration_s, tenant_id=tenant_id, route=route)

    def mark_score(self, tenant_id: Optional[str], dfi: Optional[float], afps: Optional[float], duration_s: float) -> None:
        self.inc("hydra_scores_total", tenant_id=tenant_id)
        self.observe("hydra_score_latency_seconds", duration_s, tenant_id=tenant_id)
        if dfi is not None:
            self.observe("hydra_dfi_score", max(0.0, min(100.0, float(dfi))), tenant_id=tenant_id)
        if afps is not None:
            self.observe("hydra_afps_score", max(0.0, min(100.0, float(afps))), tenant_id=tenant_id)

    def set_drift(self, tenant_id: str, feature: str, *, psi: Optional[float] = None, ks: Optional[float] = None) -> None:
        if psi is not None:
            self.set_gauge("hydra_drift_psi", float(psi), tenant_id=tenant_id, feature=feature)
        if ks is not None:
            self.set_gauge("hydra_drift_ks", float(ks), tenant_id=tenant_id, feature=feature)


# Singleton-ish instance
metrics = _Metrics()
