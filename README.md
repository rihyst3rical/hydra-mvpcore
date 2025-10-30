HYDRA-CORE (MVP) — DFI & AFPS

Purpose: a lean, enterprise-grade API that reduces fallout and cycle time using two explainable metrics:
	•	DFI — Dynamic Fragility Index (process volatility & friction)
	•	AFPS — Adaptive Funding Probability Score (funding probability & expected time to fund)

Designed for ICE EPC sandbox → pilot → production. Simple to operate, quant-light where it matters, and hands-off (self-healing retries, guardrails, audit hashes, telemetry).

⸻

TL;DR
	•	Stack: FastAPI + Uvicorn, Pydantic, NumPy, Prometheus, OpenTelemetry, PyJWT, BLAKE3.
	•	MVP scope: DFI & AFPS + narratives + orchestration + governance + telemetry. Fraud/graph code scaffolded only.
	•	Security: JWT, tenant isolation hooks, request hashing, no PII required for scoring.
	•	Compliance: audit trails & deterministic versioning; blake3 content hashes; KMS-ready secret wrapping (AWS KMS/GCP KMS envelopes).
	•	Ops: /metrics (Prometheus), OTLP traces, structured logs.
	•	Outcome: lenders act earlier on volatile files → fewer dead locks, faster funding → $800–$2,300+/loan savings potential (pilot-dependent).

⸻

Repo Layout

hydra-core/
├─ main.py                      # Entry point (boot, orchestrate, serve)
├─ requirements.txt
├─ README.md
│
├─ config/
│  └─ settings.py               # Env & defaults (sandbox/prod)
│
├─ core/
│  ├─ fi.py                     # DFI scoring (explainable, quant-light)
│  ├─ afps.py                   # AFPS scoring (probability + time-to-fund)
│  ├─ hydra_voice.py            # Narrative: risk → action, neg. visualization
│  ├─ supervisor.py             # Orchestration, fallbacks, retries
│  ├─ governance.py             # Compliance: hashing, model versions, audit packs
│  ├─ db.py                     # (Optional later) DB adapter (SQLModel scaffold)
│  ├─ auth.py                   # JWT + API key hooks + tenant guardrails
│  └─ models.py                 # (Optional later) ORM + drift monitors
│
├─ telemetry/
│  ├─ exporter.py               # Prometheus + OTLP wiring
│  └─ suite.py                  # Sanity probes, JSON diagnostics
│
├─ logging/
│  └─ logging.py                # JSON logs, correlation IDs
│
├─ chaos/
│  └─ chaos.py                  # Fault injection (sandbox only)
│
├─ utils/
│  └─ utils.py                  # Stat helpers, percentiles, normalization
│
├─ fraud/                       # (Scaffold only for later)
│  ├─ ingest_normalizer.py
│  ├─ feature_builder.py
│  ├─ anomaly_scorer.py
│  ├─ graph_engine.py
│  └─ alert_writer.py
│
└─ tests/
   └─ test_fi.py                # Unit tests for DFI math & edge cases


⸻

Prerequisites
	•	Python 3.10–3.12
	•	pip or uv (optional)
	•	(Optional later) Docker and docker compose
	•	(Optional later) Postgres if you enable persistence beyond in-memory.

⸻

Install & Run (Local)

# 1) Clone
git clone https://github.com/<you>/hydra-core.git
cd hydra-core

# 2) Create venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Install deps
pip install -r requirements.txt

# 4) Configure env (create .env)
cp .env.example .env

# 5) Start API
uvicorn main:app --reload --port 8080

.env.example

# Runtime
HYDRA_ENV=sandbox
HYDRA_SERVICE_NAME=hydra-core
HYDRA_VERSION=v0.1.0

# Auth
HYDRA_JWT_ISSUER=hydra-core
HYDRA_JWT_AUDIENCE=hydra-tenants
HYDRA_JWT_SECRET=change_me_in_prod      # In prod: rotate & wrap with KMS

# Telemetry
PROM_ENABLED=true
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=hydra-core

# Governance
AUDIT_HASH_ALGO=blake3
MODEL_FI_VERSION=fi-0.1.0
MODEL_AFPS_VERSION=afps-0.1.0

# Chaos (sandbox only)
CHAOS_ENABLED=false
CHAOS_RATE=0.01

Prod secret management: wrap HYDRA_JWT_SECRET with AWS KMS or GCP KMS (envelope encryption). The code already expects a cleartext secret at runtime; use an init container or bootstrap step to decrypt.

⸻

API Overview

Base path defaults to /api/v1.

1) Health & Diagnostics
	•	GET /healthz → liveness probe
	•	GET /readyz → readiness probe
	•	GET /spec → OpenAPI JSON
	•	GET /metrics → Prometheus metrics
	•	GET /diag/suite → JSON suite (sanity stats, build info)

2) DFI — Dynamic Fragility Index

POST /dfi/score

Request (example):

{
  "tenant_id": "lndr_001",
  "loan_id": "LN-90342",
  "timeline": [
    {"t": "2025-01-02T10:00:00Z", "readiness": 42, "events": {"docs_missing": 1}},
    {"t": "2025-01-12T10:00:00Z", "readiness": 60, "events": {"escalations": 1}},
    {"t": "2025-01-19T10:00:00Z", "readiness": 64, "events": {}},
    {"t": "2025-02-02T10:00:00Z", "readiness": 83, "events": {"reopen_steps": 2}},
    {"t": "2025-02-06T10:00:00Z", "readiness": 84, "events": {}}
  ]
}

Response (example):

{
  "loan_id": "LN-90342",
  "dfi_score": 72.4,
  "volatility_band": "MED",
  "reason_codes": [
    {"feature": "variance", "impact": 12.1},
    {"feature": "idle_gap", "impact": 8.3},
    {"feature": "reopen_steps", "impact": 6.0}
  ],
  "narrative": "DFI=72 (stable enough). If idle gaps reduce by ~25%, DFI +5–7 points.",
  "audit": {
    "model_version": "fi-0.1.0",
    "hash": "b3:2f8a84…",
    "generated_at": "2025-10-30T15:32:11Z"
  }
}

3) AFPS — Adaptive Funding Probability Score

POST /afps/score

Request (example):

{
  "tenant_id": "lndr_001",
  "loan_id": "LN-90342",
  "dfi_score": 72.4,
  "tempo": {
    "avg_inactivity_days": 3.1,
    "milestone_pace": 0.78
  },
  "ops": {
    "lo_response_time": 0.9,
    "branch_delay_index": 0.2
  },
  "history": {
    "pull_through_peer": 0.87
  }
}

Response (example):

{
  "loan_id": "LN-90342",
  "afps_score": 83.9,
  "probability_to_fund_30d": 0.86,
  "expected_time_to_fund_days": 27.4,
  "narrative": "High probability to fund inside 30 days. Watch LO response time; +10% speed → +2 pts AFPS.",
  "audit": {
    "model_version": "afps-0.1.0",
    "hash": "b3:0d9f11…",
    "generated_at": "2025-10-30T15:33:12Z"
  }
}

Both endpoints include Hydra Voice one-liners that convert signals to actions (negative visualization + “do this now” cues).

⸻

Auth
	•	Bearer JWT in Authorization header.
	•	Claims validated: iss, aud, exp.
	•	Tenant guardrails: hook for tenant_id scoping & rate limits.

Authorization: Bearer <JWT>

In pilots, you can use static API keys → JWT for production. Rotation policy included.

⸻

Telemetry & Observability
	•	Prometheus at /metrics:
	•	hydra_requests_total{route=...,status=...}
	•	hydra_request_latency_seconds_bucket
	•	hydra_scores_computed_total{type="dfi|afps"}
	•	hydra_early_warnings_total
	•	OpenTelemetry (OTLP) traces (FastAPI auto-instrumented):
	•	context propagation, spans per request, error tagging
	•	Structured logging (JSON): correlation IDs, tenant tags, model versions.
	•	Diagnostics suite: /diag/suite returns lightweight system health snapshot (env, build, feature flags, last N scoring stats).

Quick Grafana (local)

Point Grafana/Prometheus at http://localhost:8080/metrics.
(OTLP → Tempo/Jaeger optional.)

⸻

Compliance & Security (MVP+)
	•	Audit hashing: each response carries a BLAKE3 hash of normalized inputs + outputs + model version.
	•	Determinism: same inputs + model version → same outputs (unit tests included).
	•	Model version pins: MODEL_FI_VERSION, MODEL_AFPS_VERSION in env + response payloads.
	•	Secrets: KMS-ready; rotate JWT secrets; no hardcoded secrets.
	•	PII: not required for scoring; design supports tokenized identifiers.
	•	Logs: no PII in logs; redact headers; cap payload size.
	•	Rate limits: hooks available (enable slowapi later if needed).
	•	Chaos: opt-in (CHAOS_ENABLED=true) only in sandbox; never in prod builds.

Intent: when ICE audits, ≤5% work remains; ~3% is paperwork (policy PDFs, runbooks, diagrams). The code already enforces hashes, versioning, redaction, and determinism.

⸻

Configuration Matrix

Setting	Sandbox Default	Production Hint
HYDRA_ENV	sandbox	prod
HYDRA_JWT_SECRET	local string	KMS-wrapped + rotated
PROM_ENABLED	true	true
OTEL_ENABLED	true	true
CHAOS_ENABLED	false	false
MODEL_FI_VERSION	fi-0.1.0	increment via CI on change
MODEL_AFPS_VERSION	afps-0.1.0	increment via CI on change


⸻

Running Tests

pytest -q

	•	tests/test_fi.py covers:
	•	increasing idle gaps ↓ DFI
	•	high variance ↓ DFI
	•	stable monotone readiness ↑ DFI
	•	determinism (same input → same output)

Add tests/test_afps.py similarly as you extend AFPS.

⸻

Deployment (Simple)

Bare Uvicorn

uvicorn main:app --host 0.0.0.0 --port 8080

Docker (Optional)

# Dockerfile (minimal)
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HYDRA_ENV=prod
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

docker build -t hydra-core:0.1.0 .
docker run -p 8080:8080 --env-file .env hydra-core:0.1.0


⸻

Operational Playbook (MVP)
	•	SLOs: p95 latency < 150ms; uptime > 99.5% (single region).
	•	Error budget: < 0.5% 5xx; alert at 0.3%.
	•	Autoscaling: horizontal (container replicas) if CPU > 70% / 5m.
	•	Backoff & retries: tenacity for outbound calls; circuit breaker hooks available.
	•	Versioning: /api/v1 path + model_version fields in responses.

⸻

Roadmap (Post-MVP)
	1.	FPS Portfolio dashboard (CVI-lite tiles through Grafana)
	2.	Tenant overrides (weights/tolerances at branch level)
	3.	Sandbox → pilot outcomes loop (calibrate AFPS with real pull-through)
	4.	KMS envelopes in init path; secrets rotation job
	5.	EPC certification artifacts (runbooks, model cards, DPA, BCP/DR)

⸻

FAQ

Q: Do we need a database now?
A: No. MVP computes in-memory and returns scores with audit hashes. Add DB once you need historical analytics or lender-facing history.

Q: Is this explainable enough for ICE/lenders?
A: Yes. We return reason codes, version pins, deterministic outputs, and plain-English Hydra Voice guidance.

Q: Can this hit four-figure savings per loan?
A: Yes—when lenders act on DFI early warnings & AFPS prioritization (pilot results dependent). That’s the whole point of simple, actionable narratives.

⸻

Support Scripts (Nice to have)
	•	scripts/bootstrap.sh → create venv, install deps, generate sample JWT.
	•	scripts/smoke.sh → curl /healthz, /dfi/score, /afps/score with sample JSON.

⸻

License / Use

Internal MVP for pilots. Add license before external distribution.

⸻
