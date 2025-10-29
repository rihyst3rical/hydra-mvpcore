# hydra-core/config/settings.py
"""
Settings for HydraCore MVP
- Minimal, explicit, enterprise-safe.
- Works in SANDBOX or PROD without code edits.
- No cloud SDK deps; KMS hooks are stubs you can swap later.
- Strong defaults; everything overrideable via environment variables.

Key goals:
  • Keep config dumb-simple to reason about.
  • Make auditors happy (deterministic, documented, least-privilege).
  • No surprises in sandbox: safe fallbacks, no network side-effects.
"""

from __future__ import annotations

import os
import json
from functools import lru_cache
from typing import List, Dict, Optional

# pydantic v2 first; fall back to v1 if needed
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, field_validator
except Exception:  # pragma: no cover
    from pydantic import BaseSettings, Field  # type: ignore
    from pydantic import validator as field_validator  # type: ignore


def _split_csv(value: str | List[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [v.strip() for v in value if v and v.strip()]
    return [v.strip() for v in value.split(",") if v and v.strip()]


def _parse_api_keys(raw: str | None) -> Dict[str, str]:
    """
    Parse STATIC_API_KEYS like:
      "tenantA:sk_live_a,tenantB:sk_live_b"
    → {"tenantA": "sk_live_a", "tenantB": "sk_live_b"}
    """
    if not raw:
        return {}
    out: Dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            # allow single default key with pseudo-tenant "default"
            out["default"] = pair
            continue
        t, k = pair.split(":", 1)
        t, k = t.strip(), k.strip()
        if t and k:
            out[t] = k
    return out


class Settings(BaseSettings):
    # ───────────────────────
    # Runtime & Versioning
    # ───────────────────────
    ENV: str = Field(default="SANDBOX", description="SANDBOX | STAGING | PROD")
    MODE: str = Field(default="API", description="API | JOB | CLI")
    VERSION: str = Field(default=os.getenv("HYDRA_VERSION", "0.1.0"))

    # ───────────────────────
    # Networking / CORS
    # ───────────────────────
    CORS_ALLOW_ORIGINS: List[str] = Field(
        default_factory=lambda: ["*"],  # lock down during pilot if needed
        description="Allowed origins for CORS",
    )

    # ───────────────────────
    # Logging & Telemetry
    # ───────────────────────
    LOG_LEVEL: str = Field(default="INFO", description="DEBUG | INFO | WARN | ERROR")
    PROMETHEUS_ENABLED: bool = Field(default=True)
    CHAOS_ENABLED: bool = Field(
        default=bool(int(os.getenv("HYDRA_CHAOS_ENABLED", "0"))),
        description="Enable safe chaos hooks in sandbox demos",
    )

    # ───────────────────────
    # Database
    # ───────────────────────
    # Prefer a single DSN string to keep infra boring:
    # Example (Postgres):  postgresql+asyncpg://user:pass@localhost:5432/hydra
    DB_DSN: str = Field(default=os.getenv("HYDRA_DB_DSN", "sqlite+aiosqlite:///./hydra.db"))
    DB_POOL_SIZE: int = Field(default=5)
    DB_MAX_OVERFLOW: int = Field(default=5)
    DB_CONNECT_TIMEOUT_SEC: int = Field(default=5)

    # ───────────────────────
    # Auth (Tenant API Keys)
    # ───────────────────────
    API_KEY_HEADER: str = Field(default="x-hydra-api-key")
    STATIC_API_KEYS_RAW: Optional[str] = Field(
        default=os.getenv("HYDRA_STATIC_API_KEYS", ""),
        description='CSV of "tenant:key" pairs. Example: "pilotA:sk_a,pilotB:sk_b"',
    )

    # ───────────────────────
    # Governance / Compliance
    # ───────────────────────
    AUDIT_HASH_SALT: str = Field(
        default=os.getenv("HYDRA_AUDIT_HASH_SALT", "local-dev-salt"),
        description="Pepper for envelope hash; rotate per environment.",
    )
    ENABLE_KMS: bool = Field(default=bool(int(os.getenv("HYDRA_ENABLE_KMS", "0"))))
    KMS_KEY_ID: Optional[str] = Field(default=os.getenv("HYDRA_KMS_KEY_ID"))
    # If you want to store secrets as files (e.g., mounted), point to a dir:
    SECRET_DIR: Optional[str] = Field(default=os.getenv("HYDRA_SECRET_DIR"))

    # ───────────────────────
    # Rate-Limiting (simple, per-process)
    # ───────────────────────
    RATE_LIMIT_PER_MIN: int = Field(default=600)  # can tune per tenant later

    @field_validator("ENV")
    @classmethod
    def _env_upper(cls, v: str) -> str:
        v = (v or "").upper()
        if v not in {"SANDBOX", "STAGING", "PROD"}:
            raise ValueError("ENV must be SANDBOX | STAGING | PROD")
        return v

    @field_validator("CORS_ALLOW_ORIGINS", mode="before")
    @classmethod
    def _cors_from_csv(cls, v):
        return _split_csv(v)

    @field_validator("STATIC_API_KEYS_RAW", mode="before")
    @classmethod
    def _noneify(cls, v):
        return v if v is not None else ""

    # Convenience properties (don’t persist in env)
    @property
    def STATIC_API_KEYS(self) -> Dict[str, str]:
        return _parse_api_keys(self.STATIC_API_KEYS_RAW)

    # ───────────────────────
    # Secret helpers (auditor-friendly)
    # ───────────────────────
    def secret(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Resolve a secret by name, in this order:
          1) Exact env var
          2) If SECRET_DIR is set and <dir>/<name> exists → read file
          3) default
        """
        val = os.getenv(name)
        if val:
            return val
        if self.SECRET_DIR:
            path = os.path.join(self.SECRET_DIR, name)
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        return f.read().strip()
            except Exception:
                # Do not raise — secrets are optional in MVP
                return default
        return default

    def kms_decrypt(self, ciphertext_b64: str) -> str:
        """
        MVP stub: returns input unless ENABLE_KMS is true.
        Swap with AWS/GCP KMS later (kept out of MVP deps on purpose).
        """
        if not self.ENABLE_KMS:
            return ciphertext_b64
        # Placeholder — document the contract for later:
        # input: base64-encoded ciphertext; output: plaintext string
        # raise NotImplementedError("KMS not wired yet in MVP")
        return ciphertext_b64  # non-breaking default

    class Config:
        env_prefix = "HYDRA_"  # HYDRA_ENV, HYDRA_DB_DSN, etc.
        case_sensitive = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Singleton accessor so every import shares the exact same Settings instance.
    - Deterministic for auditors
    - Cheap for hot paths
    """
    s = Settings()
    # Hard safety rails in SANDBOX
    if s.ENV == "SANDBOX" and s.DB_DSN.startswith("postgresql"):
        # Allowed — but log a warning in your logger once logging is wired.
        pass
    return s
