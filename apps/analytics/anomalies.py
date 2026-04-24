"""Anomaly detection over upsampled trip trajectories.

Produces the three event types the iROAM dashboard renders:

  * ``idle``  — bus stationary for ≥ ``idle_min_threshold`` minutes
  * ``bunch`` — two consecutive trip instances pass the same point within
                ≤ ``bunch_seconds_threshold`` of each other (``method="time"``),
                **or** along-route separation < ``bunch_distance_threshold_m``
                for a contiguous stretch (``method="distance"``).
  * ``crowd`` — a point's GTFS-RT OccupancyStatus maps to ≥ ``crowd_pct_threshold``

Input is a list of ``BusTrajectory`` rows (one per trip instance) with their
time/distance samples. Output is a flat list of ``AnomalyEvent`` rows keyed by
``bus_index`` so the caller can merge them back per-bus.

All functions here are pure — no DB calls, no I/O — so they're trivial to
unit-test and cheap to re-run on every threshold change.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

IDLE_SPEED_THRESHOLD_M_S = 0.5

# Map GTFS-RT OccupancyStatus enum to a coarse "percent full" value.
# Values are advisory only — the feed doesn't give true load factors.
OCCUPANCY_PCT = {
    "EMPTY": 0,
    "MANY_SEATS_AVAILABLE": 25,
    "FEW_SEATS_AVAILABLE": 55,
    "STANDING_ROOM_ONLY": 75,
    "CRUSHED_STANDING_ROOM_ONLY": 95,
    "FULL": 100,
    "NOT_ACCEPTING_PASSENGERS": 100,
}

AnomalyType = Literal["bunch", "idle", "crowd"]
BunchMethod = Literal["time", "distance"]
BunchMethodSelector = Literal["time", "distance", "both"]


@dataclass(frozen=True)
class TrajectoryPoint:
    datetime: datetime
    travel_distance_m: float
    moving_speed_m_s: float | None
    occupancy_status: str | None
    stop_index: float  # fractional — precomputed by stop_projection


@dataclass(frozen=True)
class BusTrajectory:
    bus_index: int         # per-request zero-based index used by the dashboard
    trip_id: str
    start_date: str
    vehicle_id: str | None
    points: list[TrajectoryPoint]


@dataclass(frozen=True)
class AnomalyEvent:
    bus_index: int
    minute_of_day: float    # time in minutes since midnight (UTC), float for sub-minute precision
    stop_index: float
    type: AnomalyType
    # Only meaningful when ``type == "bunch"`` — distinguishes the time-gap
    # detector from the along-route-distance detector. None for idle/crowd.
    method: BunchMethod | None = None


def _to_minute_of_day(dt: datetime) -> float:
    """Convert a UTC datetime to a float minute-of-day in America/Toronto.

    The design plots minute-of-day from 360 (6:00) to 1320 (22:00) on a local
    wall-clock axis; we convert from UTC once here so callers can just hand
    back the value.
    """
    # Lazy import — zoneinfo is stdlib in 3.9+ but importing once per call is fine.
    try:
        from zoneinfo import ZoneInfo
        local = dt.astimezone(ZoneInfo("America/Toronto"))
    except Exception:  # pragma: no cover
        local = dt
    return local.hour * 60 + local.minute + local.second / 60.0


def detect_idle_events(
    bus: BusTrajectory, *, idle_min_threshold: float
) -> list[AnomalyEvent]:
    """Emit one event per contiguous stationary run ≥ ``idle_min_threshold`` minutes."""
    events: list[AnomalyEvent] = []
    threshold_s = idle_min_threshold * 60.0
    run_start: TrajectoryPoint | None = None
    last_in_run: TrajectoryPoint | None = None

    def flush(start: TrajectoryPoint, end: TrajectoryPoint) -> None:
        dur = (end.datetime - start.datetime).total_seconds()
        if dur < threshold_s:
            return
        mid_epoch = (start.datetime.timestamp() + end.datetime.timestamp()) / 2.0
        mid_dt = datetime.fromtimestamp(mid_epoch, tz=start.datetime.tzinfo)
        mid_dist_m = (start.travel_distance_m + end.travel_distance_m) / 2.0
        # Weighted stop_index at midpoint (start/end have same stop_index when
        # idle, but use the average to be safe).
        mid_stop_idx = (start.stop_index + end.stop_index) / 2.0
        events.append(
            AnomalyEvent(
                bus_index=bus.bus_index,
                minute_of_day=_to_minute_of_day(mid_dt),
                stop_index=mid_stop_idx,
                type="idle",
            )
        )

    for p in bus.points:
        speed = p.moving_speed_m_s or 0.0
        is_idle = speed < IDLE_SPEED_THRESHOLD_M_S
        if is_idle:
            if run_start is None:
                run_start = p
            last_in_run = p
        else:
            if run_start is not None and last_in_run is not None:
                flush(run_start, last_in_run)
            run_start = None
            last_in_run = None

    if run_start is not None and last_in_run is not None:
        flush(run_start, last_in_run)

    return events


def detect_crowd_events(
    bus: BusTrajectory, *, crowd_pct_threshold: float
) -> list[AnomalyEvent]:
    """Emit one event per contiguous run where occupancy%≥threshold."""
    events: list[AnomalyEvent] = []
    run_start: TrajectoryPoint | None = None
    last_in_run: TrajectoryPoint | None = None

    def flush(start: TrajectoryPoint, end: TrajectoryPoint) -> None:
        mid_epoch = (start.datetime.timestamp() + end.datetime.timestamp()) / 2.0
        mid_dt = datetime.fromtimestamp(mid_epoch, tz=start.datetime.tzinfo)
        mid_stop = (start.stop_index + end.stop_index) / 2.0
        events.append(
            AnomalyEvent(
                bus_index=bus.bus_index,
                minute_of_day=_to_minute_of_day(mid_dt),
                stop_index=mid_stop,
                type="crowd",
            )
        )

    for p in bus.points:
        pct = OCCUPANCY_PCT.get((p.occupancy_status or "").upper(), -1)
        if pct >= crowd_pct_threshold and pct >= 0:
            if run_start is None:
                run_start = p
            last_in_run = p
        else:
            if run_start is not None and last_in_run is not None:
                flush(run_start, last_in_run)
            run_start = None
            last_in_run = None

    if run_start is not None and last_in_run is not None:
        flush(run_start, last_in_run)

    return events


def detect_bunch_events(
    buses: list[BusTrajectory], *, bunch_seconds_threshold: float
) -> list[AnomalyEvent]:
    """Flag trip pairs whose passage through the same ~stop is closer than threshold.

    Strategy: for every integer stop index, compute the time each bus passes it
    (by linear interpolation of ``(datetime, stop_index)`` samples). Sort by
    time; any pair with Δt < threshold → emit an event on the trailing bus
    at the crossing time/stop.
    """
    events: list[AnomalyEvent] = []
    if not buses:
        return events

    max_stop = max((p.stop_index for b in buses for p in b.points), default=0.0)
    max_stop_int = int(max_stop)

    for si in range(max_stop_int + 1):
        passes: list[tuple[datetime, int, float]] = []  # (time, bus_index, stop_idx)
        for bus in buses:
            crossing = _interpolate_crossing_time(bus.points, si)
            if crossing is not None:
                passes.append((crossing, bus.bus_index, float(si)))
        passes.sort(key=lambda r: r[0])
        for a, b in zip(passes, passes[1:]):
            gap_s = (b[0] - a[0]).total_seconds()
            if 0 < gap_s < bunch_seconds_threshold:
                events.append(
                    AnomalyEvent(
                        bus_index=b[1],
                        minute_of_day=_to_minute_of_day(b[0]),
                        stop_index=b[2],
                        type="bunch",
                        method="time",
                    )
                )
    return events


def detect_bunch_events_distance(
    buses: list[BusTrajectory],
    *,
    bunch_distance_threshold_m: float,
    grid_seconds: float = 30.0,
) -> list[AnomalyEvent]:
    """Flag pairs of buses whose along-route separation stays < threshold.

    Strategy: sweep a uniform time grid (default 30 s, matches GTFS-RT cadence).
    At each tick, for every bus active at that tick, linearly interpolate
    ``travel_distance_m`` and ``stop_index``. Sort active buses by distance and
    check each consecutive (follower, leader) pair — their along-route gap is
    ``leader.dist - follower.dist``. When the same ordered pair stays inside
    the threshold across consecutive ticks we treat it as one contiguous run;
    on run-end we emit a single event tagged on the follower at the run's
    time/stop midpoint.

    Tagging the *follower* (trailing bus, lower ``travel_distance_m``) matches
    the time-based detector, which tags the later-arriving bus at the stop.
    """
    events: list[AnomalyEvent] = []
    if not buses:
        return events

    # Per-bus sorted arrays for O(log n) linear interpolation.
    tracks: list[tuple[int, list[float], list[float], list[float]]] = []
    for bus in buses:
        if len(bus.points) < 2:
            continue
        ts = [p.datetime.timestamp() for p in bus.points]
        ds = [p.travel_distance_m for p in bus.points]
        si = [p.stop_index for p in bus.points]
        tracks.append((bus.bus_index, ts, ds, si))

    if not tracks:
        return events

    t_min = min(ts[0] for _, ts, _, _ in tracks)
    t_max = max(ts[-1] for _, ts, _, _ in tracks)
    if t_max <= t_min or grid_seconds <= 0:
        return events

    tz = buses[0].points[0].datetime.tzinfo

    # Open runs keyed by (follower_idx, leader_idx) → (start_t, start_si, last_t, last_si).
    open_runs: dict[tuple[int, int], tuple[float, float, float, float]] = {}

    def flush(key: tuple[int, int], run: tuple[float, float, float, float]) -> None:
        start_t, start_si, last_t, last_si = run
        mid_t = (start_t + last_t) / 2.0
        mid_si = (start_si + last_si) / 2.0
        events.append(
            AnomalyEvent(
                bus_index=key[0],  # follower
                minute_of_day=_to_minute_of_day(datetime.fromtimestamp(mid_t, tz=tz)),
                stop_index=mid_si,
                type="bunch",
                method="distance",
            )
        )

    t = t_min
    # +epsilon on the bound so the last tick isn't dropped by float drift.
    while t <= t_max + 1e-9:
        active: list[tuple[int, float, float]] = []  # (bus_index, dist, stop_idx)
        for bus_idx, ts, ds, si in tracks:
            if t < ts[0] or t > ts[-1]:
                continue
            active.append(
                (bus_idx, _interp_sorted(t, ts, ds), _interp_sorted(t, ts, si))
            )
        active.sort(key=lambda r: r[1])

        pairs_now: set[tuple[int, int]] = set()
        for a, b in zip(active, active[1:]):
            gap = b[1] - a[1]
            if 0 < gap < bunch_distance_threshold_m:
                key = (a[0], b[0])  # (follower, leader)
                pairs_now.add(key)
                prev = open_runs.get(key)
                if prev is None:
                    open_runs[key] = (t, a[2], t, a[2])
                else:
                    open_runs[key] = (prev[0], prev[1], t, a[2])

        # Flush runs whose pair didn't re-appear this tick.
        for key in list(open_runs.keys()):
            if key not in pairs_now:
                flush(key, open_runs.pop(key))

        t += grid_seconds

    # Flush any still-open runs at the end of the window.
    for key, run in open_runs.items():
        flush(key, run)

    return events


def _interp_sorted(t: float, ts: list[float], ys: list[float]) -> float:
    """Linear interpolation on a sorted ``ts`` array; clamps at the endpoints."""
    i = bisect.bisect_left(ts, t)
    if i <= 0:
        return ys[0]
    if i >= len(ts):
        return ys[-1]
    t0, t1 = ts[i - 1], ts[i]
    if t1 <= t0:
        return ys[i - 1]
    frac = (t - t0) / (t1 - t0)
    return ys[i - 1] + frac * (ys[i] - ys[i - 1])


def _interpolate_crossing_time(
    points: list[TrajectoryPoint], target_stop_index: int
) -> datetime | None:
    """Linear interpolate the datetime at which ``stop_index`` first reaches target."""
    for a, b in zip(points, points[1:]):
        lo, hi = (a, b) if a.stop_index <= b.stop_index else (b, a)
        if lo.stop_index <= target_stop_index <= hi.stop_index:
            span = hi.stop_index - lo.stop_index
            if span <= 0:
                return lo.datetime
            frac = (target_stop_index - lo.stop_index) / span
            epoch = lo.datetime.timestamp() + frac * (
                hi.datetime.timestamp() - lo.datetime.timestamp()
            )
            return datetime.fromtimestamp(epoch, tz=lo.datetime.tzinfo)
    return None


def detect_all(
    buses: list[BusTrajectory],
    *,
    bunch_seconds_threshold: float,
    idle_min_threshold: float,
    crowd_pct_threshold: float,
    bunch_distance_threshold_m: float = 150.0,
    bunch_method: BunchMethodSelector = "time",
) -> list[AnomalyEvent]:
    """Run all detectors and return the flat event list.

    ``bunch_method`` controls which bunching detector(s) fire:
      * ``"time"``     — only the stop-crossing time-gap detector (default; preserves
                         pre-distance-detector behaviour).
      * ``"distance"`` — only the along-route separation detector.
      * ``"both"``     — both detectors run; events are tagged with ``method`` so
                         callers can tell them apart.
    """
    out: list[AnomalyEvent] = []
    for bus in buses:
        out.extend(detect_idle_events(bus, idle_min_threshold=idle_min_threshold))
        out.extend(detect_crowd_events(bus, crowd_pct_threshold=crowd_pct_threshold))
    if bunch_method in ("time", "both"):
        out.extend(
            detect_bunch_events(buses, bunch_seconds_threshold=bunch_seconds_threshold)
        )
    if bunch_method in ("distance", "both"):
        out.extend(
            detect_bunch_events_distance(
                buses, bunch_distance_threshold_m=bunch_distance_threshold_m
            )
        )
    return out
