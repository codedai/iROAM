"""Unit tests for the ``forecast.run_forecast`` orchestrator with a stubbed predictor.

The actual LightGBM model is covered by ``test_bunching_predictor_smoke``. Here
we pin down aggregate rollup semantics and per-bus payload shape with a
predictable fake predictor so failures are precise.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from apps.analytics.anomalies import BusTrajectory, TrajectoryPoint
from apps.api.services.forecast import run_forecast
from apps.api.services.forecast_features import SEQ_LEN, STEP_SECONDS


def _mk_bus(idx: int, t0: datetime, *, start_dist: float, speed: float = 8.0) -> BusTrajectory:
    pts: list[TrajectoryPoint] = []
    for i in range(80):
        t = t0 + timedelta(seconds=i * STEP_SECONDS)
        d = start_dist + speed * (i * STEP_SECONDS)
        pts.append(
            TrajectoryPoint(
                datetime=t,
                travel_distance_m=d,
                moving_speed_m_s=speed,
                occupancy_status=None,
                stop_index=d / 300.0,
            )
        )
    return BusTrajectory(
        bus_index=idx,
        trip_id=f"T{idx}",
        start_date="20260422",
        vehicle_id=f"V{idx}",
        points=pts,
    )


def _mod(dt_utc: datetime) -> float:
    from zoneinfo import ZoneInfo
    local = dt_utc.astimezone(ZoneInfo("America/Toronto"))
    return local.hour * 60 + local.minute + local.second / 60.0


class _StubPredictor:
    """Predictable stand-in for BunchingPredictor.

    ``predict_proba`` returns a ramp ``[h/pred_len for h in ...]`` scaled by a
    per-bus factor that grows with ``bus.max_prob``. ``alert`` delegates.
    """

    pred_len = 30

    def __init__(self, scale: float = 1.0) -> None:
        self._scale = scale
        self.thresholds = {
            h: {"threshold": 0.5, "f2": 1.0, "precision": 1.0, "recall": 1.0}
            for h in range(self.pred_len)
        }

    def predict_proba(self, batch: np.ndarray, *, is_scaled: bool) -> np.ndarray:
        assert is_scaled is False  # the service must pass raw
        assert batch.ndim == 3
        n = batch.shape[0]
        # Ramp 0→0.9 per horizon; same for every bus (keeps the maths simple).
        ramp = np.linspace(0.0, 0.9, self.pred_len, dtype=np.float32)
        return np.broadcast_to(ramp, (n, self.pred_len)).astype(np.float32)

    def alert(self, batch: np.ndarray, *, is_scaled: bool) -> list[dict]:
        probs = self.predict_proba(batch, is_scaled=is_scaled)
        thrs = np.array([self.thresholds[h]["threshold"] for h in range(self.pred_len)])
        out = []
        for i in range(probs.shape[0]):
            exceed = probs[i] >= thrs
            any_hit = bool(exceed.any())
            out.append(
                {
                    "any_alert": any_hit,
                    "first_alert_step": int(np.argmax(exceed)) if any_hit else None,
                    "max_prob": float(probs[i].max()),
                    "max_prob_step": int(np.argmax(probs[i])),
                    "per_horizon": probs[i].tolist(),
                }
            )
        return out


def test_forecast_tags_eligible_and_ineligible_buses():
    t0 = datetime(2026, 4, 22, 16, 0, 0, tzinfo=timezone.utc)
    # Three in-band buses in a line: the first two each have an upstream behind
    # them; the trailing one has none and is ineligible for that reason.
    leader = _mk_bus(0, t0, start_dist=2000.0)
    middle = _mk_bus(1, t0, start_dist=1600.0)
    tail = _mk_bus(2, t0, start_dist=1200.0)
    # Near terminus → edge-exclude
    edge = _mk_bus(3, t0, start_dist=11600.0)

    t_ref = _mod(leader.points[75].datetime)

    result = run_forecast(
        [leader, middle, tail, edge],
        num_stops=40,
        t_ref_min=t_ref,
        predictor=_StubPredictor(),
    )

    assert result.num_buses_total == 4
    eligible_ids = {r["bus_id"] for r in result.per_bus if r["eligible"]}
    ineligible = {r["bus_id"]: r for r in result.per_bus if not r["eligible"]}
    assert eligible_ids == {0, 1}
    assert "no upstream" in (ineligible[2]["ineligible_reason"] or "")
    assert "edge-exclude" in (ineligible[3]["ineligible_reason"] or "")


def test_forecast_aggregate_horizon_rollup():
    t0 = datetime(2026, 4, 22, 16, 0, 0, tzinfo=timezone.utc)
    target = _mk_bus(0, t0, start_dist=2000.0)
    upstream = _mk_bus(1, t0, start_dist=1600.0)

    t_ref = _mod(target.points[75].datetime)
    result = run_forecast(
        [target, upstream],
        num_stops=40,
        t_ref_min=t_ref,
        predictor=_StubPredictor(),
    )

    # The stub returns a 0→0.9 ramp; threshold=0.5 → any_alert_rate = 1.0 at
    # horizons where ramp ≥ 0.5, else 0.0.
    ramp = np.linspace(0.0, 0.9, 30, dtype=np.float32)
    expected_rate = (ramp >= 0.5).astype(float)
    got = np.array(result.horizon_summary["any_alert_rate"])
    assert got.shape == (30,)
    assert np.allclose(got, expected_rate)

    expected_mean = ramp.astype(float)
    assert np.allclose(np.array(result.horizon_summary["mean_prob"]), expected_mean)


def test_forecast_empty_eligible_still_returns_shape():
    # All buses fail eligibility (stale).
    t0 = datetime(2026, 4, 22, 8, 0, 0, tzinfo=timezone.utc)
    buses = [_mk_bus(0, t0, start_dist=2000.0), _mk_bus(1, t0, start_dist=1600.0)]
    # t_ref 2 hours later — far beyond freshness.
    t_ref = _mod(t0 + timedelta(hours=2))

    result = run_forecast(
        buses,
        num_stops=40,
        t_ref_min=t_ref,
        predictor=_StubPredictor(),
    )
    assert result.num_eligible == 0
    assert len(result.horizon_summary["any_alert_rate"]) == 30
    assert np.allclose(result.horizon_summary["mean_prob"], 0.0)
    assert all(not r["eligible"] for r in result.per_bus)
