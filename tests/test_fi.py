# hydra-core/tests/test_fi.py
"""
Unit tests for core.fi (DFI – Dynamic Fragility Index)

Goals (MVP):
  • Sanity: returns required keys, values in valid ranges.
  • Behavior: smooth timelines score higher than jagged/idle-heavy.
  • Banding: high scores map to LOW band; low scores map to HIGH/CRITICAL.
  • Robustness: missing optional fields do not crash; clamping works.
  • Determinism: identical inputs → identical outputs.

Assumptions about core.fi API (per MVP spec):
  compute_dfi(payload: dict) -> dict with at least:
    {
      "dfi_score": float,           # 0..100
      "band": str,                  # "LOW" | "MED" | "HIGH" | "CRITICAL"
      "reason_codes": list[str],    # non-empty for non-perfect scores
      "deltas": dict,               # optional: variance, idle_penalty, corrections
    }

If your implementation uses different names, update these tests accordingly.
"""

import math
import time
from datetime import datetime, timedelta

import pytest

# Import target
from core.fi import compute_dfi


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def make_timeline(
    start: datetime,
    steps: list[tuple[int, float]],
):
    """
    Build a timeline array for DFI:
      steps: list of (day_offset, readiness_value 0..100)
    """
    return [{"t": iso(start + timedelta(days=dd)), "readiness": float(r)} for dd, r in steps]


@pytest.fixture
def base_payload():
    """Minimal viable payload fields expected by compute_dfi."""
    return {
        "borrower_id": "B-001",
        "loan_id": "LN-001",
        # timeline filled per-test
    }


def test_returns_required_keys(base_payload):
    now = datetime.utcnow() - timedelta(days=10)
    payload = base_payload | {
        "timeline": make_timeline(now, [(0, 40), (3, 55), (6, 70), (9, 84)]),
        "corrections": 0,
    }
    out = compute_dfi(payload)

    assert isinstance(out, dict)
    for k in ("dfi_score", "band", "reason_codes"):
        assert k in out, f"missing key: {k}"

    assert isinstance(out["dfi_score"], (int, float))
    assert 0.0 <= out["dfi_score"] <= 100.0, "score should be clamped 0..100"
    assert out["band"] in {"LOW", "MED", "HIGH", "CRITICAL"}, "unexpected band"
    assert isinstance(out["reason_codes"], list)


def test_smooth_progress_scores_high_and_is_LOW_band(base_payload):
    now = datetime.utcnow() - timedelta(days=20)
    # Smooth, steady improvements; short idle gaps; no reopens
    tl = make_timeline(
        now,
        [(0, 35), (3, 50), (7, 65), (10, 78), (13, 86), (17, 92)],
    )
    out = compute_dfi(base_payload | {"timeline": tl, "corrections": 0})

    assert out["dfi_score"] >= 80.0
    assert out["band"] == "LOW"


def test_jagged_with_long_idle_scores_low_and_is_HIGH_or_CRITICAL(base_payload):
    now = datetime.utcnow() - timedelta(days=60)
    # Sawtooth + long idle gap
    tl = make_timeline(
        now,
        [
            (0, 40),
            (5, 70),
            (20, 42),   # regression after idle
            (40, 68),
            (59, 55),   # late stall
        ],
    )
    out = compute_dfi(base_payload | {"timeline": tl, "corrections": 3})

    assert out["dfi_score"] <= 60.0
    assert out["band"] in {"HIGH", "CRITICAL"}


@pytest.mark.parametrize(
    "idle_days,expected_max",
    [
        (0, 100.0),
        (14, 95.0),
        (30, 85.0),
        (45, 75.0),
    ],
)
def test_idle_penalty_caps_score(idle_days, expected_max, base_payload):
    now = datetime.utcnow() - timedelta(days=idle_days + 5)
    # Two points with an idle gap in between; readiness still improves
    tl = make_timeline(now, [(0, 50), (idle_days, 80)])
    out = compute_dfi(base_payload | {"timeline": tl, "corrections": 0})

    assert out["dfi_score"] <= expected_max + 1e-6


def test_missing_optional_fields_does_not_crash(base_payload):
    now = datetime.utcnow() - timedelta(days=5)
    tl = make_timeline(now, [(0, 20), (5, 60)])
    # No 'corrections' key; also include unknown keys
    payload = base_payload | {"timeline": tl, "unknown": 123}
    out = compute_dfi(payload)

    assert "dfi_score" in out
    assert 0.0 <= out["dfi_score"] <= 100.0


def test_extreme_values_are_clamped(base_payload):
    now = datetime.utcnow() - timedelta(days=3)
    tl = [
        {"t": iso(now), "readiness": -50.0},
        {"t": iso(now + timedelta(days=1)), "readiness": 150.0},
    ]
    out = compute_dfi(base_payload | {"timeline": tl, "corrections": -10})

    assert 0.0 <= out["dfi_score"] <= 100.0
    assert out["band"] in {"LOW", "MED", "HIGH", "CRITICAL"}


def test_determinism_same_input_same_output(base_payload):
    now = datetime.utcnow() - timedelta(days=12)
    tl = make_timeline(now, [(0, 45), (4, 59), (8, 76), (11, 82)])
    payload = base_payload | {"timeline": tl, "corrections": 1}

    out1 = compute_dfi(payload)
    time.sleep(0.01)  # ensure wall-clock differs; should not affect result
    out2 = compute_dfi(payload)

    assert math.isclose(out1["dfi_score"], out2["dfi_score"], rel_tol=0, abs_tol=1e-9)
    assert out1["band"] == out2["band"]
    assert out1.get("reason_codes", []) == out2.get("reason_codes", [])


def test_reason_codes_present_when_score_isnt_perfect(base_payload):
    now = datetime.utcnow() - timedelta(days=10)
    tl = make_timeline(now, [(0, 40), (5, 60)])
    out = compute_dfi(base_payload | {"timeline": tl, "corrections": 2})

    if out["dfi_score"] < 100.0:
        assert len(out["reason_codes"]) >= 1
