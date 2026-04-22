"""Analytics runner — transaction owner and ``analytics_runs`` lifecycle.

One run = one service date. Commits per trip instance so a mid-day crash
preserves all earlier work. The analytics_runs row is created up front with
``status='running'`` and finalized at the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from apps.analytics import csv_export, pipeline
from apps.analytics.gtfs_static import load_all
from apps.analytics.shapes import build_linestrings
from core.logging import get_logger
from db.models.trip_trajectory import AnalyticsRun, TripTrajectory

_logger = get_logger(__name__)


@dataclass
class RunOutcome:
    run_id: int
    service_date: date
    trip_instances_processed: int
    rows_written: int
    status: str
    error_message: str | None = None


def _df_to_orm(df: pd.DataFrame, run_id: int) -> list[TripTrajectory]:
    """Convert the canonical trajectory DataFrame into ORM rows."""
    rows: list[TripTrajectory] = []
    for rec in df.to_dict(orient="records"):
        rows.append(
            TripTrajectory(
                run_id=run_id,
                trip_id=rec["trip_id"],
                start_date=rec["start_date"],
                service_date=rec["service_date"],
                route_id=rec.get("route_id"),
                direction_id=rec.get("direction_id") if pd.notna(rec.get("direction_id")) else None,
                shape_id=rec.get("shape_id"),
                vehicle_id=rec.get("vehicle_id"),
                datetime=rec["datetime"].to_pydatetime() if hasattr(rec["datetime"], "to_pydatetime") else rec["datetime"],
                time_offset_seconds=(
                    int(rec["time_offset_seconds"]) if pd.notna(rec.get("time_offset_seconds")) else None
                ),
                travel_distance_m=float(rec["travel_distance_m"]),
                moving_speed_m_s=(
                    float(rec["moving_speed_m_s"]) if pd.notna(rec.get("moving_speed_m_s")) else None
                ),
                observed=bool(rec["observed"]),
                occupancy_status=rec.get("occupancy_status"),
                source_vehicle_position_id=(
                    int(rec["source_vehicle_position_id"])
                    if pd.notna(rec.get("source_vehicle_position_id"))
                    else None
                ),
            )
        )
    return rows


def run_for_date(
    session: Session,
    service_date: date,
    *,
    route_id: str | None = None,
    upsample_resolution_s: int = 10,
    max_orthogonal_distance_m: float = 200.0,
    export_csv_dir: Path | None = None,
) -> RunOutcome:
    """Process every trip instance whose ``start_date`` matches ``service_date``."""
    config = {
        "upsample_resolution_s": upsample_resolution_s,
        "max_orthogonal_distance_m": max_orthogonal_distance_m,
        "route_id": route_id,
    }
    run = AnalyticsRun(
        service_date=service_date,
        route_id=route_id,
        config_json=config,
        status="running",
    )
    session.add(run)
    session.commit()
    run_id = run.id

    total_rows = 0
    instances_processed = 0
    csv_buckets: dict[tuple[str, str, int], list[pd.DataFrame]] = {}

    try:
        static = load_all()
        shape_lines = build_linestrings(static.shapes)

        instances = pipeline.list_trip_instances(session, service_date, route_id=route_id)
        _logger.info(
            "analytics_start",
            extra={
                "run_id": run_id,
                "service_date": service_date.isoformat(),
                "route_id": route_id,
                "trip_instances": len(instances),
            },
        )

        for trip_id, start_date in instances:
            df = pipeline.process_trip_instance(
                session,
                static,
                shape_lines,
                trip_id,
                start_date,
                upsample_resolution_s=upsample_resolution_s,
                max_orthogonal_distance_m=max_orthogonal_distance_m,
            )
            if df.empty:
                continue
            orm_rows = _df_to_orm(df, run_id)
            session.add_all(orm_rows)
            session.commit()
            total_rows += len(orm_rows)
            instances_processed += 1

            if export_csv_dir is not None:
                route_val = df["route_id"].iloc[0] if "route_id" in df.columns else "NA"
                dir_val = df["direction_id"].iloc[0] if "direction_id" in df.columns else None
                dir_int = int(dir_val) if pd.notna(dir_val) else -1
                key = (str(route_val), service_date.isoformat(), dir_int)
                csv_buckets.setdefault(key, []).append(df)

        # Finalize the run row.
        run.status = "ok"
        run.finished_at = datetime.now(tz=timezone.utc)
        run.rows_written = total_rows
        session.commit()

        if export_csv_dir is not None:
            written = csv_export.write_day_csvs(export_csv_dir, csv_buckets)
            _logger.info(
                "analytics_csv_export",
                extra={"run_id": run_id, "files": len(written), "dir": str(export_csv_dir)},
            )

        _logger.info(
            "analytics_ok",
            extra={
                "run_id": run_id,
                "trip_instances_processed": instances_processed,
                "rows_written": total_rows,
            },
        )
        return RunOutcome(
            run_id=run_id,
            service_date=service_date,
            trip_instances_processed=instances_processed,
            rows_written=total_rows,
            status="ok",
        )

    except Exception as exc:  # noqa: BLE001 — finalize and re-raise
        session.rollback()
        run.status = "failed"
        run.finished_at = datetime.now(tz=timezone.utc)
        run.rows_written = total_rows
        run.error_message = str(exc)[:8000]
        session.commit()
        _logger.exception("analytics_failed", extra={"run_id": run_id})
        raise
