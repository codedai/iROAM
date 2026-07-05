"""End-to-end integration test for ``apps.analytics.runner.run_for_date``.

Seeds a small synthetic trip instance (≈20 GPS rows along a straight line) plus
a matching GTFS static bundle in a temp directory, runs the full pipeline, and
asserts the ``analytics_runs`` row finalized and ``trip_trajectories`` rows
were written. Requires a reachable Postgres; auto-skips via ``db_session``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from pyproj import Transformer
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.analytics import runner
from apps.analytics.shapes import METRIC_CRS
from db.models.feed_fetch_log import FeedFetchLog
from db.models.raw_snapshot import RawGtfsrtSnapshot
from db.models.trip_trajectory import AnalyticsRun, TripTrajectory
from db.models.vehicle_position import VehiclePosition

_TRIP_ID = "T_SMOKE"
_SHAPE_ID = "S_SMOKE"
_ROUTE_ID = "R_SMOKE"
_START_DATE = "20260420"
_START_TIME = "08:00:00"


def _write_synthetic_gtfs(tmp: Path) -> Path:
    """Minimal GTFS bundle with one trip on a straight 2km line in Toronto."""
    gtfs = tmp / "gtfs"
    gtfs.mkdir(parents=True, exist_ok=True)

    # A straight north-south LineString of ~2km in Toronto.
    # Anchor in the metric CRS near Toronto (UTM 17N easting/northing).
    transformer = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)
    x0, y0 = 630_000.0, 4_833_000.0
    pts = [(x0, y0), (x0, y0 + 1000.0), (x0, y0 + 2000.0)]
    shape_rows = []
    for seq, (x, y) in enumerate(pts, start=1):
        lon, lat = transformer.transform(x, y)
        shape_rows.append(
            {
                "shape_id": _SHAPE_ID,
                "shape_pt_lat": lat,
                "shape_pt_lon": lon,
                "shape_pt_sequence": seq,
                "shape_dist_traveled": (seq - 1) * 1000.0,
            }
        )
    pd.DataFrame(shape_rows).to_csv(gtfs / "shapes.txt", index=False)

    pd.DataFrame(
        [
            {
                "trip_id": _TRIP_ID,
                "route_id": _ROUTE_ID,
                "service_id": "svc1",
                "direction_id": 0,
                "shape_id": _SHAPE_ID,
            }
        ]
    ).to_csv(gtfs / "trips.txt", index=False)

    pd.DataFrame(columns=["stop_id", "stop_lat", "stop_lon"]).to_csv(
        gtfs / "stops.txt", index=False
    )
    pd.DataFrame(
        columns=["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"]
    ).to_csv(gtfs / "stop_times.txt", index=False)
    pd.DataFrame([{"route_id": _ROUTE_ID, "route_short_name": _ROUTE_ID}]).to_csv(
        gtfs / "routes.txt", index=False
    )
    return gtfs


def _seed_vehicle_positions(session: Session, n: int = 20) -> None:
    """Insert ``n`` evenly-spaced GPS rows progressing along the synthetic shape."""
    # Anchor in the metric CRS near Toronto (UTM 17N easting/northing).
    transformer = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)
    x0, y0 = 630_000.0, 4_833_000.0
    base_dt = datetime(2026, 4, 20, 13, 0, 0, tzinfo=timezone.utc)

    log = FeedFetchLog(
        feed_name="vehicle-positions",
        feed_url="http://example/feed",
        fetched_at=base_dt,
        http_status=200,
        success=True,
        duration_ms=10,
        response_bytes=100,
        feed_header_timestamp=base_dt,
        entity_count=n,
    )
    session.add(log)
    session.flush()
    snap = RawGtfsrtSnapshot(
        fetch_log_id=log.id,
        feed_name="vehicle-positions",
        fetched_at=base_dt,
        feed_header_timestamp=base_dt,
        content_sha256="0" * 64,
    )
    session.add(snap)
    session.flush()

    for i in range(n):
        x = x0
        y = y0 + (2000.0 * i / (n - 1))
        lon, lat = transformer.transform(x, y)
        dt = base_dt + timedelta(seconds=15 * i)
        session.add(
            VehiclePosition(
                snapshot_id=snap.id,
                fetched_at=dt,
                feed_header_timestamp=dt,
                entity_id=f"e{i}",
                vehicle_timestamp=dt,
                vehicle_id="V_SMOKE",
                trip_id=_TRIP_ID,
                route_id=_ROUTE_ID,
                direction_id=0,
                start_date=_START_DATE,
                start_time=_START_TIME,
                latitude=lat,
                longitude=lon,
                occupancy_status="MANY_SEATS_AVAILABLE",
                raw_entity={"id": f"e{i}"},
            )
        )
    session.commit()


def test_run_for_date_roundtrip(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Point settings + gtfs_static cache at the synthetic bundle.
    gtfs_dir = _write_synthetic_gtfs(tmp_path)

    # Patching get_settings at the gtfs_static module redirects load_all,
    # bundle_token, and load_shape_linestrings together — they must agree on
    # the bundle or the runner drops every trip as "shape not found".
    from types import SimpleNamespace

    import apps.analytics.gtfs_static as gs
    monkeypatch.setattr(gs, "get_settings", lambda: SimpleNamespace(gtfs_static_dir=gtfs_dir))

    _seed_vehicle_positions(db_session, n=20)

    outcome = runner.run_for_date(
        db_session,
        date_from_str(_START_DATE),
        route_id=_ROUTE_ID,
        upsample_resolution_s=10,
        max_orthogonal_distance_m=200.0,
    )

    assert outcome.status == "ok"
    assert outcome.trip_instances_processed == 1
    assert outcome.rows_written > 0

    run_row = db_session.execute(
        select(AnalyticsRun).where(AnalyticsRun.id == outcome.run_id)
    ).scalars().one()
    assert run_row.status == "ok"
    assert run_row.finished_at is not None
    assert run_row.rows_written == outcome.rows_written

    traj_rows = db_session.execute(
        select(TripTrajectory).where(TripTrajectory.run_id == outcome.run_id)
    ).scalars().all()
    assert len(traj_rows) == outcome.rows_written
    assert {r.trip_id for r in traj_rows} == {_TRIP_ID}
    assert {r.route_id for r in traj_rows} == {_ROUTE_ID}
    assert {r.shape_id for r in traj_rows} == {_SHAPE_ID}
    # Travel distances should fall within the 0..2000m synthetic shape.
    for r in traj_rows:
        assert 0.0 <= r.travel_distance_m <= 2100.0


def _seed_vehicle_positions_with_gap(
    session: Session, *, before: int = 8, after: int = 8, gap_seconds: int = 300
) -> tuple[datetime, datetime]:
    """Seed two clusters of GPS rows separated by a ``gap_seconds`` outage.

    Returns the (start, end) UTC bounds of the gap so a test can assert nothing
    was synthesized inside it. Rows within each cluster are 15 s apart.
    """
    transformer = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)
    x0, y0 = 630_000.0, 4_833_000.0
    base_dt = datetime(2026, 4, 20, 13, 0, 0, tzinfo=timezone.utc)

    log = FeedFetchLog(
        feed_name="vehicle-positions", feed_url="http://example/feed", fetched_at=base_dt,
        http_status=200, success=True, duration_ms=10, response_bytes=100,
        feed_header_timestamp=base_dt, entity_count=before + after,
    )
    session.add(log)
    session.flush()
    snap = RawGtfsrtSnapshot(
        fetch_log_id=log.id, feed_name="vehicle-positions", fetched_at=base_dt,
        feed_header_timestamp=base_dt, content_sha256="1" * 64,
    )
    session.add(snap)
    session.flush()

    total = before + after
    offsets = [15 * i for i in range(before)]
    gap_lo = base_dt + timedelta(seconds=offsets[-1])
    resume = offsets[-1] + gap_seconds
    offsets += [resume + 15 * i for i in range(after)]
    gap_hi = base_dt + timedelta(seconds=offsets[before])

    for i, off in enumerate(offsets):
        y = y0 + (2000.0 * i / (total - 1))
        lon, lat = transformer.transform(x0, y)
        dt = base_dt + timedelta(seconds=off)
        session.add(
            VehiclePosition(
                snapshot_id=snap.id, fetched_at=dt, feed_header_timestamp=dt,
                entity_id=f"g{i}", vehicle_timestamp=dt, vehicle_id="V_GAP",
                trip_id=_TRIP_ID, route_id=_ROUTE_ID, direction_id=0,
                start_date=_START_DATE, start_time=_START_TIME,
                latitude=lat, longitude=lon, occupancy_status="MANY_SEATS_AVAILABLE",
                raw_entity={"id": f"g{i}"},
            )
        )
    session.commit()
    return gap_lo, gap_hi


def test_max_gap_seconds_threads_through_run_for_date(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The online path (run_for_date → process_trip_instance → upsample_df)
    honors ``max_gap_seconds``: a long outage is not bridged, and unset keeps
    the legacy (bridged) behavior."""
    gtfs_dir = _write_synthetic_gtfs(tmp_path)
    from types import SimpleNamespace

    import apps.analytics.gtfs_static as gs
    monkeypatch.setattr(gs, "get_settings", lambda: SimpleNamespace(gtfs_static_dir=gtfs_dir))

    gap_lo, gap_hi = _seed_vehicle_positions_with_gap(db_session, gap_seconds=300)
    service_date = date_from_str(_START_DATE)

    uncapped = runner.run_for_date(
        db_session, service_date, route_id=_ROUTE_ID,
        upsample_resolution_s=10, max_orthogonal_distance_m=200.0,
    )
    capped = runner.run_for_date(
        db_session, service_date, route_id=_ROUTE_ID,
        upsample_resolution_s=10, max_orthogonal_distance_m=200.0,
        max_gap_seconds=60.0,
    )

    assert uncapped.status == capped.status == "ok"
    # Bridging the 300 s outage adds ~30 synthetic points; capping drops them.
    assert capped.rows_written < uncapped.rows_written

    # The capped config is recorded on the analytics_runs row.
    run_row = db_session.execute(
        select(AnalyticsRun).where(AnalyticsRun.id == capped.run_id)
    ).scalars().one()
    assert run_row.config_json["max_gap_seconds"] == 60.0

    # No stored (capped) trajectory row lands strictly inside the outage.
    capped_rows = db_session.execute(
        select(TripTrajectory).where(TripTrajectory.run_id == capped.run_id)
    ).scalars().all()
    assert not any(gap_lo < r.datetime < gap_hi for r in capped_rows)


def test_analytics_max_gap_seconds_defaults_off() -> None:
    """The new config knob is off by default (stored data unchanged unless set)."""
    from core.config import Settings

    # _env_file=None isolates the code default from any local .env override.
    assert Settings(_env_file=None).analytics_max_gap_seconds is None


def date_from_str(yyyymmdd: str):
    from datetime import date
    return date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
