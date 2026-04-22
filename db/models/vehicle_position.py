"""VehiclePosition — one row per FeedEntity.vehicle per snapshot.

Append-only. The hottest read path is "latest row per vehicle_id" and
"latest row per route_id," both served by composite `(vehicle_id|route_id,
fetched_at DESC)` indexes.

The ``geom`` column is a PostgreSQL GENERATED column derived from
``(longitude, latitude)`` by the migration; it is not mapped as an
attribute on this ORM class because Python code never writes to it.
Queries that need spatial filtering access ``geom`` through raw SQL
expressions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Float, ForeignKey, Index, Integer, SmallInteger, String
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class VehiclePosition(Base):
    __tablename__ = "vehicle_positions"

    id: Mapped[int] = mapped_column(primary_key=True)
    snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("raw_gtfsrt_snapshots.id", ondelete="CASCADE"),
        nullable=False,
    )
    fetched_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    feed_header_timestamp: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    entity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    vehicle_timestamp: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    # VehicleDescriptor
    vehicle_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vehicle_label: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # TripDescriptor
    trip_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    route_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    direction_id: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    start_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    start_time: Mapped[str | None] = mapped_column(String(8), nullable=True)
    schedule_relationship: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # Position
    latitude: Mapped[float | None] = mapped_column(Float(precision=53), nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float(precision=53), nullable=True)
    bearing: Mapped[float | None] = mapped_column(Float, nullable=True)
    odometer: Mapped[float | None] = mapped_column(Float(precision=53), nullable=True)
    speed_mps: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Progress
    current_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    current_stop_sequence: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stop_id: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Load
    occupancy_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    occupancy_percentage: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    congestion_level: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Full per-entity JSON for row-level debugging without re-decoding the snapshot.
    raw_entity: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    __table_args__ = (
        Index("ix_vp_vehicle_fetched", "vehicle_id", "fetched_at"),
        Index("ix_vp_route_fetched", "route_id", "fetched_at"),
        Index("ix_vp_trip_fetched", "trip_id", "fetched_at"),
        Index("ix_vp_fetched_at", "fetched_at"),
        Index("ix_vp_snapshot", "snapshot_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return (
            f"<VehiclePosition id={self.id} vehicle={self.vehicle_id} "
            f"route={self.route_id} at={self.fetched_at.isoformat() if self.fetched_at else None}>"
        )
