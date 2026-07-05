"""Week-1 Task 3 — trajectory data-quality profiler.

For one ``--route`` / ``--date`` slice, reports the drop rate at each filter
stage (dedup / off-route / teleport / ghost-segment), the orthogonal-distance
distribution, GPS cadence, clock skew, and feed-field availability. Read-only;
writes a markdown report under ``out/qa/``.

    python -m scripts.analysis.profile_quality --route 29 --date 2026-06-20
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from apps.analytics.gtfs_static import load_all, resolve_shape_id
from apps.analytics.pipeline import _EFFECTIVE_START_DATE
from apps.analytics.project_to_shape import project_trajectory
from apps.analytics.shapes import build_linestrings
from apps.analytics.stop_projection import compute_route_stops
from apps.analytics.trajectory_extract import build_trip_trajectory
from apps.api.services.bus_grouping import group_into_buses
from core.logging import configure_logging
from db.models.trip_trajectory import TripTrajectory
from db.models.vehicle_position import VehiclePosition
from db.session import SessionLocal

configure_logging()

# Production thresholds (mirror core/config defaults) used to isolate each filter.
_ORTHO_M = 200.0
_TELEPORT_M_S = 35.0


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Profile trajectory data quality")
    p.add_argument("--date", required=True, type=date.fromisoformat, help="service date (YYYY-MM-DD)")
    p.add_argument("--route", type=str, default=None, help="filter to a single route_id")
    return p.parse_args(argv)


def _run(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    route_id = args.route
    service_date = args.date
    yyyymmdd = service_date.strftime("%Y%m%d")

    with SessionLocal() as session:
        stmt = (
            select(VehiclePosition)
            .where(VehiclePosition.route_id == route_id)
            .where(_EFFECTIVE_START_DATE == yyyymmdd)
        )
        vehicle_positions = session.execute(stmt).scalars().all()
    vp_df = pd.DataFrame(
        [{c.name: getattr(vp, c.name) for c in VehiclePosition.__table__.columns} for vp in vehicle_positions]
    )
    print(f"Pulled {len(vp_df)} vehicle_positions rows for route {route_id} on {service_date}")

    with SessionLocal() as session:
        stmt = (
            select(TripTrajectory)
            .where(TripTrajectory.route_id == route_id)
            .where(TripTrajectory.service_date == service_date)
        )
        trip_trajectories = session.execute(stmt).scalars().all()
    tt_df = pd.DataFrame(
        [{c.name: getattr(tt, c.name) for c in TripTrajectory.__table__.columns} for tt in trip_trajectories]
    )
    print(f"Pulled {len(tt_df)} trip_trajectories rows for route {route_id} on {service_date}")

    static = load_all()
    shape_lines = build_linestrings(static.shapes)
    unique_trip_ids = vp_df["trip_id"].dropna().unique() if not vp_df.empty else []
    print(f"\nFound {len(unique_trip_ids)} unique trip_ids in vehicle_positions")

    # ── Drop rates per filter stage + orthogonal-distance distribution ──
    all_ortho = []  # orthogonal distances of ALL points before any distance filter
    dropped_ortho = dropped_teleport = raw_points = dropped_dedup = 0

    for trip_id in unique_trip_ids:
        rows = [vp for vp in vehicle_positions if vp.trip_id == trip_id]
        if not rows:
            continue
        df = build_trip_trajectory(rows, static.trips)  # dedup + null lat/lon drop
        dropped_dedup += len(rows) - len(df)
        if df.empty:
            continue
        shape_id = resolve_shape_id(static, trip_id)
        if shape_id is None or shape_id not in shape_lines:
            continue

        # All filters off (huge thresholds) → baseline with orthogonal_distance_m.
        df_raw = project_trajectory(
            df, shape_lines[shape_id], max_orthogonal_distance_m=1e12, max_implied_speed_m_s=None
        )
        all_ortho.extend(df_raw["orthogonal_distance_m"].tolist())  # <- populated (was dead code)
        df_ortho = project_trajectory(
            df, shape_lines[shape_id], max_orthogonal_distance_m=_ORTHO_M, max_implied_speed_m_s=None
        )
        df_both = project_trajectory(
            df, shape_lines[shape_id], max_orthogonal_distance_m=_ORTHO_M, max_implied_speed_m_s=_TELEPORT_M_S
        )
        dropped_ortho += len(df_raw) - len(df_ortho)
        dropped_teleport += len(df_ortho) - len(df_both)
        raw_points += len(df_raw)

    print("\n── Filter drop rates ──")
    print(f"exact-timestamp dedup + null-drop: {dropped_dedup} rows removed at extraction")
    if raw_points:
        print(f"off-route (>200 m):   {dropped_ortho} ({dropped_ortho / raw_points * 100:.1f}%)")
        print(f"teleport (>35 m/s):   {dropped_teleport} ({dropped_teleport / raw_points * 100:.1f}%)")
        print(f"total projected-in:   {raw_points} points")

    if all_ortho:
        a = np.array(all_ortho)
        print("\n── Orthogonal distance distribution (m, before 200 m filter) ──")
        for q in (50, 90, 95, 99):
            print(f"  p{q}: {np.percentile(a, q):.1f}")
        print(f"  max: {a.max():.1f}")
        print(f"  % beyond 200 m cutoff: {(a > _ORTHO_M).mean() * 100:.1f}%")
    else:
        print("\nNo orthogonal distance data collected (no projectable trips)")

    # ── Ghost-segment / stale-tail drop (from stored trip_trajectories) ──
    if not tt_df.empty:
        for direction_id in tt_df["direction_id"].dropna().unique():
            route_stops = compute_route_stops(route_id, int(direction_id))
            if route_stops is None:
                print(f"\ndir {direction_id}: no canonical route stops — skipping ghost-filter stat")
                continue
            slice_df = tt_df[tt_df["direction_id"] == direction_id].sort_values(
                by=["trip_id", "start_date", "vehicle_id", "datetime"]
            )
            buses = group_into_buses(list(slice_df.itertuples(index=False)), route_stops)
            before = len(slice_df)
            after = sum(len(b.points) for b in buses)
            pct = ((before - after) / before * 100) if before else 0.0
            print(f"\ndir {direction_id}: ghost/stale filter kept {after}/{before} "
                  f"points ({pct:.1f}% dropped) across {len(buses)} bus segments")

    # ── GPS cadence ──
    trips_with_gap = 0
    for trip_id in unique_trip_ids:
        tv = vp_df[vp_df["trip_id"] == trip_id].sort_values("vehicle_timestamp")
        if len(tv) < 2:
            continue
        dt = tv["vehicle_timestamp"].diff().dt.total_seconds()
        if (dt > 120).any():
            trips_with_gap += 1
    if len(unique_trip_ids):
        print(f"\n── GPS cadence ── trips with an internal gap > 2 min: "
              f"{trips_with_gap}/{len(unique_trip_ids)} ({trips_with_gap / len(unique_trip_ids) * 100:.1f}%)")

    # ── Clock skew ──
    if not vp_df.empty:
        vp_df["skew_s"] = (vp_df["fetched_at"] - vp_df["vehicle_timestamp"]).dt.total_seconds()
        print("\n── Clock skew (fetched_at − vehicle_timestamp, s) ──")
        print(vp_df["skew_s"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())

    # ── Field availability ──
    fields = ["occupancy_status", "occupancy_percentage", "current_stop_sequence", "stop_id",
              "current_status", "bearing", "speed_mps", "odometer", "direction_id", "vehicle_id"]
    print("\n── Field availability (% non-null) ──")
    for f in fields:
        if f in vp_df.columns and not vp_df.empty:
            print(f"  {f}: {vp_df[f].notnull().mean() * 100:.1f}%")

    return route_id, service_date


def main(argv=None):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        route_id, service_date = _run(argv)
    output = buffer.getvalue()
    sys.stdout.write(output)

    out_dir = Path("out/qa")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"route{route_id}_{service_date}_quality_profile.md"
    out_path.write_text("\n".join([
        f"# Quality Profile — Route {route_id} — {service_date}", "", "```", output.rstrip(), "```",
    ]))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
