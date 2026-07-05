"""Week-1 Task 1 — shape-variant coordinate-mismatch report.

Measures how often trips run a *non-canonical* shape (the canonical shape being
the one ``stop_projection._pick_canonical_trip`` picks per route+direction), and
demonstrates on one example that the same ``travel_distance_m`` lands at a
different place on the trip shape vs the canonical shape. Read-only analysis.

    python -m scripts.analysis.shape_variant_report
    python -m scripts.analysis.shape_variant_report --route 29 --direction 0
"""

from __future__ import annotations

import argparse
import sys

from shapely.geometry import Point

from apps.analytics.gtfs_static import load_all
from apps.analytics.shapes import build_linestrings, transform_lonlat_to_meters
from apps.analytics.stop_projection import _pick_canonical_trip


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shape-variant coordinate-mismatch report")
    p.add_argument("--route", type=str, default=None, help="limit to one route_id")
    p.add_argument("--direction", type=int, default=None, help="limit to one direction_id")
    return p.parse_args(argv if argv is not None else sys.argv[1:])


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    static = load_all()
    trips = static.trips

    # 1. Distinct shape_ids per (route, direction) and how many trips use each.
    if args.route is not None:
        trips = trips[trips["route_id"].astype(str) == str(args.route)]
    if args.direction is not None:
        trips = trips[trips["direction_id"] == args.direction]
    if trips.empty:
        print("no trips match the given --route/--direction filter")
        return 1

    shape_variant = (
        trips.groupby(["route_id", "direction_id", "shape_id"], dropna=False).size()
        .reset_index(name="trip_count")
    )
    print("shape usage per (route, direction):")
    print(shape_variant.to_string(index=False))

    # 2. Canonical shape per (route, direction); % of trips on a non-canonical shape.
    route_direction_combos = set(zip(trips["route_id"], trips["direction_id"], strict=False))
    canonical_shapes: dict[tuple[str, int], str] = {}
    for route_id, direction_id in route_direction_combos:
        canonical_trip_id = _pick_canonical_trip(static, route_id, direction_id)
        if canonical_trip_id is not None:
            row = static.trips.loc[static.trips["trip_id"] == canonical_trip_id].iloc[0]
            canonical_shapes[(route_id, direction_id)] = str(row["shape_id"])

    non_canonical_count = 0
    total_trips = 0
    example = None  # (trip_id, route_id, direction_id, actual_shape_id, canonical_shape_id)
    for row in trips.itertuples(index=False):
        total_trips += 1
        key = (row.route_id, row.direction_id)
        canonical = canonical_shapes.get(key)
        if str(row.shape_id) != str(canonical):
            non_canonical_count += 1
            if example is None:
                example = (row.trip_id, row.route_id, row.direction_id, str(row.shape_id), canonical)

    pct = (non_canonical_count / total_trips * 100) if total_trips else 0.0
    print(f"\ntrips on a non-canonical shape: {non_canonical_count}/{total_trips} ({pct:.2f}%)")

    # 3. For one non-canonical trip that shares a stop with its canonical trip,
    #    show the same stop projects to a different distance-along on each shape.
    if example is None:
        print("\nno non-canonical trip found — canonical == trip shape for every trip")
        return 0

    stop_info = None
    for row in trips.itertuples(index=False):
        key = (row.route_id, row.direction_id)
        canonical = canonical_shapes.get(key)
        if str(row.shape_id) == str(canonical):
            continue
        canonical_trip_id = _pick_canonical_trip(static, row.route_id, row.direction_id)
        can_st = static.stop_times.loc[static.stop_times["trip_id"] == canonical_trip_id]
        non_st = static.stop_times.loc[static.stop_times["trip_id"] == row.trip_id]
        shared = set(can_st["stop_id"]).intersection(set(non_st["stop_id"]))
        if not shared:
            continue
        shared_stop_id = sorted(shared)[0]  # deterministic pick
        s = static.stops.loc[static.stops["stop_id"] == shared_stop_id]
        if s.empty:
            continue
        stop_info = s.iloc[0]
        actual_shape_id = str(row.shape_id)
        canonical_shape_id = str(canonical)
        print(f"\nNon-canonical trip {row.trip_id} route {row.route_id} dir {row.direction_id} "
              f"shape {actual_shape_id} (canonical {canonical_shape_id})")
        print(f"Shared stop {shared_stop_id} @ ({stop_info['stop_lat']}, {stop_info['stop_lon']})")
        break

    if stop_info is None:
        print("\nnon-canonical trips exist but none shares a stop with its canonical trip")
        return 0

    can_lines = build_linestrings(static.shapes.loc[static.shapes["shape_id"] == canonical_shape_id])
    non_lines = build_linestrings(static.shapes.loc[static.shapes["shape_id"] == actual_shape_id])
    if canonical_shape_id not in can_lines or actual_shape_id not in non_lines:
        print("one of the shapes has no geometry in shapes.txt — cannot compare projections")
        return 0

    x, y = transform_lonlat_to_meters(float(stop_info["stop_lon"]), float(stop_info["stop_lat"]))
    pt = Point(x, y)
    can_d = can_lines[canonical_shape_id].project(pt)
    non_d = non_lines[actual_shape_id].project(pt)
    print(f"\nstop projected onto canonical shape: {can_d:.1f} m")
    print(f"stop projected onto non-canonical shape: {non_d:.1f} m")
    print(f"difference (same stop, different travel_distance_m): {abs(can_d - non_d):.1f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
