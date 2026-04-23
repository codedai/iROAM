"""Build raw-unit (60, 9) feature windows for the bunching predictor.

The vendor model expects a rolling 60-step × 9-channel window per bus in the
order ``(target_speed, target_gap, target_aux,
         u1_speed, u1_gap, u1_aux,
         u2_speed, u2_gap, u2_aux)``
in *raw* units (m/s for speed, metres for gap). The ``aux`` channel is a
passthrough route-position / categorical feature in the training pipeline; the
vendor README recommends filling it with the training-time mean (0.0 in scaled
space) when the live feed does not include it. We use 0.0 and document this as
an assumption — see the top-level plan for rationale.

This module is pure: no DB access, no I/O. It takes the same
``list[BusTrajectory]`` objects that ``/iroam/buses`` assembles and a
``t_ref_minute`` (minute-of-day in America/Toronto, matching the minute-of-day
axis used throughout the dashboard), and emits per-bus windows + eligibility
verdicts.

"Running" bus definition (what makes a bus predictable at ``t_ref``):

* Has a sample within ``freshness_s`` of ``t_ref`` (default 90 s).
* Rounded stop index at ``t_ref`` satisfies ``edge_exclude ≤ si < N - edge_exclude``
  (default excludes first-2 and last-2 stops).
* Has ≥ ``seq_len`` consecutive samples ending at ``t_ref`` with per-sample
  gaps ≤ ``2 * step_seconds``.
* Window values are finite.
* At least one upstream bus exists on at least one of the 60 ticks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np

from apps.analytics.anomalies import BusTrajectory, TrajectoryPoint, _to_minute_of_day

# Channel/geometry constants — match the shipped model's metadata.json.
SEQ_LEN = 60
N_CHANNELS = 9
STEP_SECONDS = 10

# Per-tick matching tolerance: a sample at most ±MATCH_TOL seconds off the tick
# is accepted. Any gap > MAX_GAP_SECONDS disqualifies the whole window.
MATCH_TOL_S = 5.0
MAX_GAP_S = 2 * STEP_SECONDS

# Fallback "no leader" gap in metres. Picked to be larger than any realistic TTC
# route length so the model treats "nobody ahead of me" as "very low bunching
# pressure from ahead". The model was trained on bounded gaps; setting an
# arbitrarily huge value would leak outside the training distribution.
NO_LEADER_GAP_M = 20_000.0

# Fallback "no upstream" sentinels — used only when SOME ticks had an upstream
# and some did not. All-missing is rejected outright.
NO_UPSTREAM_SPEED = 0.0
NO_UPSTREAM_GAP_M = NO_LEADER_GAP_M


@dataclass(frozen=True)
class BusWindowResult:
    bus_index: int
    window: np.ndarray | None  # (SEQ_LEN, N_CHANNELS) float32, or None if ineligible
    reason: str | None         # human-readable rejection reason, or None if eligible
    stop_idx_at_ref: float | None


def _stop_idx_at(bus: BusTrajectory, t_ref_min: float) -> tuple[float | None, TrajectoryPoint | None]:
    """Return (stop_idx, sample) nearest to ``t_ref_min`` within MATCH_TOL_S, or (None, None)."""
    best: tuple[float, TrajectoryPoint] | None = None
    for p in bus.points:
        dt = abs(_to_minute_of_day(p.datetime) - t_ref_min) * 60.0
        if dt <= MATCH_TOL_S:
            if best is None or dt < best[0]:
                best = (dt, p)
    if best is None:
        return None, None
    return best[1].stop_index, best[1]


def _points_sorted_by_mod(bus: BusTrajectory) -> list[tuple[float, TrajectoryPoint]]:
    """Return ``[(minute_of_day, point), ...]`` sorted by time. Cheap to recompute."""
    out = [(_to_minute_of_day(p.datetime), p) for p in bus.points]
    out.sort(key=lambda r: r[0])
    return out


def _nearest_within(
    sorted_pts: Sequence[tuple[float, TrajectoryPoint]],
    target_mod: float,
    tol_s: float,
) -> TrajectoryPoint | None:
    """Linear-scan nearest sample with ``|mod - target| <= tol_s``."""
    tol_min = tol_s / 60.0
    lo, hi = target_mod - tol_min, target_mod + tol_min
    best: tuple[float, TrajectoryPoint] | None = None
    for mod, p in sorted_pts:
        if mod < lo:
            continue
        if mod > hi:
            break
        dt = abs(mod - target_mod)
        if best is None or dt < best[0]:
            best = (dt, p)
    return best[1] if best else None


def _freshness_min(bus: BusTrajectory, t_ref_min: float) -> float:
    """Minutes between ``t_ref`` and the most-recent sample at or before it."""
    most_recent: float | None = None
    for p in bus.points:
        mod = _to_minute_of_day(p.datetime)
        if mod <= t_ref_min and (most_recent is None or mod > most_recent):
            most_recent = mod
    if most_recent is None:
        return math.inf
    return t_ref_min - most_recent


def _forward_gap(
    target: TrajectoryPoint, others: list[TrajectoryPoint]
) -> float:
    """Metres ahead to the nearest leader on the same shape; cap at NO_LEADER_GAP_M."""
    best = NO_LEADER_GAP_M
    td = target.travel_distance_m
    for o in others:
        if o.travel_distance_m > td:
            gap = o.travel_distance_m - td
            if gap < best:
                best = gap
    return float(best)


def _upstream_samples(
    target: TrajectoryPoint, others: list[TrajectoryPoint]
) -> list[TrajectoryPoint]:
    """Up to 2 buses just behind target, sorted by increasing distance-behind (closest first)."""
    behind: list[tuple[float, TrajectoryPoint]] = []
    td = target.travel_distance_m
    for o in others:
        if o.travel_distance_m < td:
            behind.append((td - o.travel_distance_m, o))
    behind.sort(key=lambda r: r[0])
    return [p for _, p in behind[:2]]


def build_bus_window(
    target: BusTrajectory,
    peers: Sequence[BusTrajectory],
    *,
    t_ref_min: float,
    num_stops: int,
    freshness_s: float = 90.0,
    edge_exclude: int = 2,
) -> BusWindowResult:
    """Build a single (60, 9) raw-unit window for ``target`` ending at ``t_ref_min``.

    ``peers`` must be the full list of buses on the same slice; ``target`` itself
    may be included and is filtered out internally.
    """
    stop_at_ref, _ = _stop_idx_at(target, t_ref_min)

    # Freshness — bus has sampled recently enough.
    fresh_min = _freshness_min(target, t_ref_min)
    if fresh_min > freshness_s / 60.0:
        return BusWindowResult(
            bus_index=target.bus_index,
            window=None,
            stop_idx_at_ref=stop_at_ref,
            reason=(
                f"stale: last sample {fresh_min * 60:.0f}s before t_ref "
                f"(threshold {freshness_s:.0f}s)"
            ),
        )

    if stop_at_ref is None:
        return BusWindowResult(
            bus_index=target.bus_index,
            window=None,
            stop_idx_at_ref=None,
            reason="no sample within ±5s of t_ref",
        )

    si = int(round(stop_at_ref))
    if si < edge_exclude or si >= num_stops - edge_exclude:
        return BusWindowResult(
            bus_index=target.bus_index,
            window=None,
            stop_idx_at_ref=stop_at_ref,
            reason=(
                f"stop_idx={si} inside edge-exclude zone [{edge_exclude}, "
                f"{num_stops - edge_exclude})"
            ),
        )

    # Grid of 60 ticks ending at t_ref.
    step_min = STEP_SECONDS / 60.0
    grid = [t_ref_min - (SEQ_LEN - 1 - k) * step_min for k in range(SEQ_LEN)]

    target_sorted = _points_sorted_by_mod(target)
    peers_sorted = {
        p.bus_index: _points_sorted_by_mod(p) for p in peers if p.bus_index != target.bus_index
    }

    window = np.zeros((SEQ_LEN, N_CHANNELS), dtype=np.float32)
    any_upstream_tick = False

    for k, tick_mod in enumerate(grid):
        t_sample = _nearest_within(target_sorted, tick_mod, MAX_GAP_S)
        if t_sample is None:
            return BusWindowResult(
                bus_index=target.bus_index,
                window=None,
                stop_idx_at_ref=stop_at_ref,
                reason=f"missing target sample at tick {k} (±{MAX_GAP_S:.0f}s)",
            )

        peers_at_tick: list[TrajectoryPoint] = []
        for _, pts in peers_sorted.items():
            q = _nearest_within(pts, tick_mod, MAX_GAP_S)
            if q is not None:
                peers_at_tick.append(q)

        # target channels
        t_speed = t_sample.moving_speed_m_s if t_sample.moving_speed_m_s is not None else 0.0
        t_gap = _forward_gap(t_sample, peers_at_tick)
        window[k, 0] = t_speed
        window[k, 1] = t_gap
        window[k, 2] = 0.0  # aux — passthrough, vendor-recommended fallback

        # upstream channels
        ups = _upstream_samples(t_sample, peers_at_tick)
        if ups:
            any_upstream_tick = True
        for j in range(2):
            col = 3 + 3 * j
            if j < len(ups):
                up = ups[j]
                up_speed = up.moving_speed_m_s if up.moving_speed_m_s is not None else 0.0
                up_gap = (t_sample.travel_distance_m - up.travel_distance_m)
                window[k, col + 0] = up_speed
                window[k, col + 1] = up_gap
                window[k, col + 2] = 0.0
            else:
                window[k, col + 0] = NO_UPSTREAM_SPEED
                window[k, col + 1] = NO_UPSTREAM_GAP_M
                window[k, col + 2] = 0.0

    if not any_upstream_tick:
        return BusWindowResult(
            bus_index=target.bus_index,
            window=None,
            stop_idx_at_ref=stop_at_ref,
            reason="no upstream bus on any tick of the 10-min window",
        )

    if not np.all(np.isfinite(window)):
        return BusWindowResult(
            bus_index=target.bus_index,
            window=None,
            stop_idx_at_ref=stop_at_ref,
            reason="non-finite value in feature window",
        )

    return BusWindowResult(
        bus_index=target.bus_index,
        window=window,
        stop_idx_at_ref=stop_at_ref,
        reason=None,
    )
