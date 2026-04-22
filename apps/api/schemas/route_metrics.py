"""RouteMetrics response schema."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class BoundingBox(BaseModel):
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


class RouteMetricsResponse(BaseModel):
    route_id: str
    window_minutes: int
    active_vehicle_count: int
    latest_fetched_at: datetime | None
    avg_speed_mps: float | None
    max_speed_mps: float | None
    status_breakdown: dict[str, int]
    occupancy_breakdown: dict[str, int]
    bbox: BoundingBox | None
