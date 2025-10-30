# hydra-core/utils/utils.py
"""
Hydra Utils — Lean, Standard-Library Helpers
--------------------------------------------
Purpose:
  • Keep core math, hashing, time, and retry logic tight and dependency-free.
  • Power DFI/AFPS without dragging in heavy libs (numpy/pandas/etc).
  • Compliant by design: canonical JSON + BLAKE2b for auditability and dedup.

Key helpers:
  • safe_div, clamp, pct, zscore, scale_0_100  → score math without footguns.
  • ema, rolling_mean, rolling_std             → small-window stats.
  • canonical_json, blake2b_hex                → governance-grade digests.
  • redact_keys, scrub_pii                     → safe logs/exports.
  • chunked, uniq_stable                       → iteration helpers.
  • utcnow, iso_now, parse_iso                 → time, ISO-8601.
  • new_id                                     → collision-safe sortable IDs.
  • retry, aretry                              → exponential backoff w/ jitter.
  • memoize_ttl                                → tiny in-proc cache for hot paths.

No external deps. Works in Python 3.10+.
"""

from __future__ import annotations

import os
import re
import math
import json
import uuid
import time
import hmac
import hashlib
import random
import functools
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union

T = TypeVar("T")


# ─────────────────────────────────────────────────────────────────────
# Logging (optional structured logger, safe fallback)
# ─────────────────────────────────────────────────────────────────────

try:
    from logging.logging import get_logger
    log = get_logger("hydra.utils")
except Exception:  # pragma: no cover
    class _Log:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
        def debug(self, *a, **k): pass
    log = _Log()  # type: ignore


# ─────────────────────────────────────────────────────────────────────
# Math helpers — safe and boring (the good kind)
# ─────────────────────────────────────────────────────────────────────

def safe_div(n: float, d: float, default: float = 0.0) -> float:
    """Division that never throws. Returns default on zero/NaN."""
    try:
        if d == 0 or math.isnan(n) or math.isnan(d):
            return default
        v = n / d
        return v if not math.isnan(v) and math.isfinite(v) else default
    except Exception:
        return default


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp x into [lo, hi]."""
    if lo > hi:
        lo, hi = hi, lo
    return lo if x < lo else hi if x > hi else x


def pct(part: float, whole: float, default: float = 0.0) -> float:
    """Percent helper (0..100)."""
    return safe_div(part, whole, default) * 100.0


def zscore(x: float, mean: float, std: float, default: float = 0.0) -> float:
    """Compute z = (x - mean) / std safely."""
    if std <= 0 or math.isnan(std):
        return default
    z = (x - mean) / std
    return z if math.isfinite(z) else default


def scale_0_100(x: float, lo: float, hi: float) -> float:
    """
    Linear scale x from [lo, hi] → [0, 100], clamped. Handles swapped bounds.
    """
    if lo == hi:
        return 50.0
    if lo > hi:
        lo, hi = hi, lo
    return clamp((x - lo) / (hi - lo), 0.0, 1.0) * 100.0


def ema(values: Sequence[float], alpha: float = 0.2, seed: Optional[float] = None) -> float:
    """
    Exponential moving average for small sequences.
    alpha in (0,1], larger = more reactive.
    """
    if not values:
        return seed if seed is not None else 0.0
    if seed is None:
        seed = values[0]
    s = seed
    a = clamp(alpha, 1e-6, 1.0)
    for v in values:
        if not math.isfinite(v):
            continue
        s = a * v + (1 - a) * s
    return s


def rolling_mean(values: Sequence[float]) -> float:
    """Simple mean with guards."""
    if not values:
        return 0.0
    s = 0.0
    n = 0
    for v in values:
        if math.isfinite(v):
            s += v
            n += 1
    return safe_div(s, n, 0.0)


def rolling_std(values: Sequence[float], ddof: int = 1) -> float:
    """Sample std dev (ddof=1) with guards; returns 0 for tiny n."""
    clean = [v for v in values if math.isfinite(v)]
    n = len(clean)
    if n <= ddof:
        return 0.0
    m = rolling_mean(clean)
    s2 = sum((v - m) ** 2 for v in clean) / (n - ddof)
    return math.sqrt(s2)


# ─────────────────────────────────────────────────────────────────────
# Deterministic JSON + Hashing — for governance, dedup, audit trails
# ─────────────────────────────────────────────────────────────────────

def canonical_json(obj: Any) -> str:
    """
    Deterministic JSON with:
      • sorted keys
      • no spaces
      • ensure_ascii=False (keeps UTF-8 readable)
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def blake2b_hex(data: Union[str, bytes], *, key: Optional[bytes] = None, digest_size: int = 32) -> str:
    """
    BLAKE2b hex digest. If key is provided → MAC (blake2 keyed).
    digest_size 16..64 bytes (we default to 32).
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    h = hashlib.blake2b(digest_size=digest_size, key=key or b"")
    h.update(data)
    return h.hexdigest()


def hash_json(obj: Any, *, key: Optional[bytes] = None, digest_size: int = 32) -> str:
    """BLAKE2b of canonical JSON. Stable across runs and hosts."""
    return blake2b_hex(canonical_json(obj), key=key, digest_size=digest_size)


def hmac_sha256_hex(message: Union[str, bytes], secret: Union[str, bytes]) -> str:
    """HMAC-SHA256 for webhook signing or lightweight authenticity tags."""
    if isinstance(message, str):
        message = message.encode("utf-8")
    if isinstance(secret, str):
        secret = secret.encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


# ─────────────────────────────────────────────────────────────────────
# Redaction / PII scrub (log-safe transforms)
# ─────────────────────────────────────────────────────────────────────

_DEFAULT_REDACT_KEYS = re.compile(r"(ssn|sin|dob|password|pass|secret|token|key|pan|card|cvv|email|phone|address)", re.I)

def redact_keys(d: Dict[str, Any], *, pattern: re.Pattern = _DEFAULT_REDACT_KEYS, mask: str = "██REDACTED██") -> Dict[str, Any]:
    """
    Returns a shallow-copied dict with sensitive-looking keys masked.
    Only redacts top-level keys by default (log-friendly, cheap).
    """
    out = {}
    for k, v in d.items():
        if isinstance(k, str) and pattern.search(k):
            out[k] = mask
        else:
            out[k] = v
    return out


# Heuristic scrubs (cheap, not perfect)
_EMAIL_RX = re.compile(r"([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
_PHONE_RX = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b")

def scrub_pii(text: str, mask: str = "████") -> str:
    """Mask emails and phone-like strings in free text."""
    text = _EMAIL_RX.sub(mask, text)
    text = _PHONE_RX.sub(mask, text)
    return text


# ─────────────────────────────────────────────────────────────────────
# Time & IDs
# ─────────────────────────────────────────────────────────────────────

def utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)

def iso_now() -> str:
    """ISO-8601 string with 'Z' suffix."""
    return utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def parse_iso(s: str) -> datetime:
    """Lenient ISO-8601 parser for our formats."""
    # Accept both with 'Z' and with timezone offset; fallback to fromisoformat
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        # very lenient last resort
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")

def new_id(prefix: str = "hyd") -> str:
    """
    Sortable, unique-ish ID: <prefix>_<epoch_ms>_<uuid4-8>
    Good enough for logs/joins without external coordination.
    """
    ts = int(time.time() * 1000)
    u8 = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{u8}"


# ─────────────────────────────────────────────────────────────────────
# Iteration helpers
# ─────────────────────────────────────────────────────────────────────

def chunked(seq: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield fixed-size slices from seq (last chunk may be smaller)."""
    if size <= 0:
        size = 1
    for i in range(0, len(seq), size):
        yield seq[i : i + size]

def uniq_stable(it: Iterable[T]) -> List[T]:
    """Stable uniqueness preserving first occurrence order."""
    seen: set = set()
    out: List[T] = []
    for x in it:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ─────────────────────────────────────────────────────────────────────
# Retry (sync & async) with exp backoff + jitter — minimal, predictable
# ─────────────────────────────────────────────────────────────────────

def _sleep_s(seconds: float) -> None:
    time.sleep(max(0.0, seconds))

def _compute_backoff(attempt: int, base: float, cap: float, jitter: float) -> float:
    """
    attempt: 1..n
    backoff = min(cap, base * 2^(attempt-1)) * (1 ± jitter)
    """
    back = min(cap, base * (2 ** (attempt - 1)))
    if jitter > 0:
        j = 1.0 + random.uniform(-jitter, jitter)
        back *= j
    return max(0.0, back)

def retry(
    *,
    tries: int = 3,
    base: float = 0.1,
    cap: float = 1.5,
    jitter: float = 0.2,
    retry_on: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator: retry sync call with exponential backoff + jitter.
    """
    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            attempt = 1
            while True:
                try:
                    return fn(*args, **kwargs)
                except retry_on as e:  # type: ignore
                    if attempt >= tries:
                        raise
                    if on_retry:
                        on_retry(attempt, e)
                    delay = _compute_backoff(attempt, base, cap, jitter)
                    log.warning("retry", extra={"fn": fn.__name__, "attempt": attempt, "delay": round(delay, 3)})
                    _sleep_s(delay)
                    attempt += 1
        return wrapper
    return deco

# Async variant
def aretry(
    *,
    tries: int = 3,
    base: float = 0.1,
    cap: float = 1.5,
    jitter: float = 0.2,
    retry_on: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator: retry async coroutine with exponential backoff + jitter.
    """
    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs) -> T:
            import asyncio
            attempt = 1
            while True:
                try:
                    return await fn(*args, **kwargs)
                except retry_on as e:  # type: ignore
                    if attempt >= tries:
                        raise
                    if on_retry:
                        on_retry(attempt, e)
                    delay = _compute_backoff(attempt, base, cap, jitter)
                    log.warning("aretry", extra={"fn": fn.__name__, "attempt": attempt, "delay": round(delay, 3)})
                    await asyncio.sleep(delay)
                    attempt += 1
        return wrapper
    return deco


# ─────────────────────────────────────────────────────────────────────
# Tiny TTL memoization (for hot, pure-ish functions)
# ─────────────────────────────────────────────────────────────────────

def memoize_ttl(ttl_s: float = 5.0, maxsize: int = 256):
    """
    Cache results for ttl_s seconds. Evicts by time or size.
    Not thread-safe across processes (simple in-proc cache).
    """
    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        cache: Dict[Tuple[Any, ...], Tuple[float, T]] = {}
        order: List[Tuple[Any, ...]] = []

        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            # Key must be hashable; we limit to args only for simplicity.
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            # Purge expired
            if order and (now - (cache.get(order[0], (0, None))[0])) > ttl_s:
                # cheap lazy cleanup
                stale = []
                for k in order:
                    ts, _ = cache.get(k, (0, None))
                    if now - ts > ttl_s:
                        stale.append(k)
                for k in stale:
                    cache.pop(k, None)
                    try:
                        order.remove(k)
                    except ValueError:
                        pass

            if key in cache:
                ts, val = cache[key]
                if now - ts <= ttl_s:
                    return val
                else:
                    cache.pop(key, None)
                    try:
                        order.remove(key)
                    except ValueError:
                        pass

            val = fn(*args, **kwargs)
            cache[key] = (now, val)
            order.append(key)
            if len(order) > maxsize:
                oldest = order.pop(0)
                cache.pop(oldest, None)
            return val

        return wrapper
    return deco


# ─────────────────────────────────────────────────────────────────────
# Domain-ish convenience
# ─────────────────────────────────────────────────────────────────────

_LOAN_ID_RX = re.compile(r"^[A-Z0-9]{6,32}$", re.I)

def looks_like_loan_id(s: str) -> bool:
    """Cheap validation to avoid junk IDs flowing through DFI/AFPS."""
    return bool(_LOAN_ID_RX.match(s))


def percent_str(x: float, places: int = 1) -> str:
    """Format like '97.3%' with clamped bounds."""
    x = clamp(x, 0.0, 100.0)
    return f"{x:.{places}f}%"
