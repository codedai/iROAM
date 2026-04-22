"""Lazy cached loader for the GTFS static bundle under ``Complete GTFS/``.

The four tables the pipeline actually uses are ``trips``, ``stops``,
``stop_times``, and ``shapes``. ``routes`` is loaded for completeness.

The load is cached by directory mtime so dev iterations on the GTFS bundle are
picked up without a process restart, but production reuses a single copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from core.config import get_settings


@dataclass(frozen=True)
class GtfsStatic:
    trips: pd.DataFrame          # trip_id, route_id, service_id, direction_id, shape_id, ...
    stops: pd.DataFrame          # stop_id, stop_lat, stop_lon, ...
    stop_times: pd.DataFrame     # trip_id, stop_id, stop_sequence, arrival_time, ...
    shapes: pd.DataFrame         # shape_id, shape_pt_lat, shape_pt_lon, shape_pt_sequence, ...
    routes: pd.DataFrame         # route_id, route_short_name, ...


def _dir_mtime_key(path: Path) -> float:
    """Max mtime across every .txt in the bundle, for cache invalidation."""
    if not path.is_dir():
        return 0.0
    mtimes = [p.stat().st_mtime for p in path.glob("*.txt")]
    return max(mtimes) if mtimes else 0.0


@lru_cache(maxsize=4)
def _load(path_str: str, mtime_key: float) -> GtfsStatic:
    path = Path(path_str)
    trips = pd.read_csv(path / "trips.txt", dtype={"trip_id": str, "route_id": str, "shape_id": str})
    stops = pd.read_csv(path / "stops.txt", dtype={"stop_id": str})
    stop_times = pd.read_csv(
        path / "stop_times.txt",
        dtype={"trip_id": str, "stop_id": str},
    )
    shapes = pd.read_csv(path / "shapes.txt", dtype={"shape_id": str})
    routes = pd.read_csv(path / "routes.txt", dtype={"route_id": str})
    return GtfsStatic(trips=trips, stops=stops, stop_times=stop_times, shapes=shapes, routes=routes)


def load_all(gtfs_dir: Path | None = None) -> GtfsStatic:
    """Return the cached ``GtfsStatic`` bundle."""
    path = gtfs_dir if gtfs_dir is not None else get_settings().gtfs_static_dir
    return _load(str(path), _dir_mtime_key(path))


def resolve_shape_id(static: GtfsStatic, trip_id: str) -> str | None:
    """Return the ``shape_id`` for a given ``trip_id`` via ``trips.txt``, or None."""
    hit = static.trips.loc[static.trips["trip_id"] == trip_id, "shape_id"]
    if hit.empty:
        return None
    value = hit.iloc[0]
    if pd.isna(value):
        return None
    return str(value)


def resolve_direction_id(static: GtfsStatic, trip_id: str) -> int | None:
    """Return ``direction_id`` (0 or 1) for a trip, or None."""
    hit = static.trips.loc[static.trips["trip_id"] == trip_id, "direction_id"]
    if hit.empty:
        return None
    value = hit.iloc[0]
    if pd.isna(value):
        return None
    return int(value)
