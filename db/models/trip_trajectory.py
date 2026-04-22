"""Analytics output tables.

``AnalyticsRun`` is one row per ``apps/analytics`` invocation (mirrors
``FeedFetchLog`` for the collector). ``TripTrajectory`` is one row per upsampled
trajectory point; children cascade-delete with their run.

Both are append-only: re-running for the same ``service_date`` inserts a new
run and new rows. Latest-per-point is a ``DISTINCT ON`` query against the
indexes, not a mutation.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import Boolean, Date, Float, ForeignKey, Index, Integer, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class AnalyticsRun(Base):
    __tablename__ = "analytics_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    service_date: Mapped[date] = mapped_column(Date, nullable=False)
    route_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    rows_written: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_analytics_runs_service_date_started", "service_date", "started_at"),
        Index("ix_analytics_runs_status", "status"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<AnalyticsRun id={self.id} service_date={self.service_date} "
            f"status={self.status} rows={self.rows_written}>"
        )


class TripTrajectory(Base):
    __tablename__ = "trip_trajectories"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("analytics_runs.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Trip identity (key on (trip_id, start_date) — a trip_id repeats across days).
    trip_id: Mapped[str] = mapped_column(String(64), nullable=False)
    start_date: Mapped[str] = mapped_column(String(8), nullable=False)
    service_date: Mapped[date] = mapped_column(Date, nullable=False)
    route_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    direction_id: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    shape_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    vehicle_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    datetime: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    time_offset_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    travel_distance_m: Mapped[float] = mapped_column(DOUBLE_PRECISION, nullable=False)
    moving_speed_m_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    observed: Mapped[bool] = mapped_column(Boolean, nullable=False)

    occupancy_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source_vehicle_position_id: Mapped[int | None] = mapped_column(
        ForeignKey("vehicle_positions.id", ondelete="SET NULL"),
        nullable=True,
    )

    __table_args__ = (
        # Unique at the trip-instance grain. Enforces the runner's
        # delete-then-insert contract at the schema level — any future code
        # path that tries to simple-append will get a DB-level error rather
        # than silently doubling the dataset. Doubles as the hot-path lookup
        # index for per-trip-instance queries.
        Index(
            "ux_trip_trajectories_instance_dt",
            "trip_id",
            "start_date",
            "datetime",
            unique=True,
        ),
        Index("ix_tt_route_service_dt", "route_id", "service_date", "datetime"),
        Index("ix_tt_run", "run_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<TripTrajectory trip={self.trip_id} start_date={self.start_date} "
            f"dt={self.datetime.isoformat() if self.datetime else None} "
            f"dist={self.travel_distance_m}>"
        )
