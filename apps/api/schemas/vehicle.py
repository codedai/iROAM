"""VehiclePosition response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class VehiclePositionResponse(BaseModel):
    """Flat view of a ``vehicle_positions`` row.

    ``raw_entity`` is excluded by default to keep map payloads small —
    callers that need it can opt in via ``?include=raw``.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    snapshot_id: int
    fetched_at: datetime
    feed_header_timestamp: datetime | None
    entity_id: str
    vehicle_timestamp: datetime | None
    vehicle_id: str | None
    vehicle_label: str | None
    trip_id: str | None
    route_id: str | None
    direction_id: int | None
    start_date: str | None
    start_time: str | None
    schedule_relationship: str | None
    latitude: float | None
    longitude: float | None
    bearing: float | None
    odometer: float | None
    speed_mps: float | None
    current_status: str | None
    current_stop_sequence: int | None
    stop_id: str | None
    occupancy_status: str | None
    occupancy_percentage: int | None
    congestion_level: str | None


class VehiclePositionWithRaw(VehiclePositionResponse):
    """Full row including ``raw_entity`` JSON (per-entity protobuf dict)."""

    raw_entity: dict[str, Any]
