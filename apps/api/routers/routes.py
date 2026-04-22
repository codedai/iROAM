"""Route-centric endpoints.

  GET /routes                            — list routes observed recently
  GET /routes/{route_id}/vehicles/latest — latest vehicle per vehicle_id on route
  GET /routes/{route_id}/metrics         — fleet-health aggregates
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from apps.api.schemas import (
    BoundingBox,
    RouteMetricsResponse,
    VehiclePositionResponse,
    VehiclePositionWithRaw,
)
from core.config import get_settings
from db.queries.vehicles import (
    active_route_ids,
    latest_vehicle_positions,
    route_metrics,
)

router = APIRouter(tags=["routes"])


@router.get("/routes", response_model=list[str])
def routes_index(
    minutes: int = Query(default=15, ge=1, le=180),
    db: Session = Depends(get_db),
) -> list[str]:
    """Routes with at least one observation in the last ``minutes`` minutes."""
    return active_route_ids(db, window=timedelta(minutes=minutes))


@router.get("/routes/{route_id}/vehicles/latest")
def route_vehicles_latest(
    route_id: str,
    limit: int | None = Query(default=None, ge=1),
    minutes: int = Query(default=5, ge=1, le=180),
    include: Literal["raw"] | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Latest observation per vehicle currently on ``route_id``."""
    settings = get_settings()
    effective_limit = min(limit or settings.max_page_size, settings.max_page_size)
    since = datetime.now(tz=timezone.utc) - timedelta(minutes=minutes)
    rows = latest_vehicle_positions(
        db, route_id=route_id, since=since, limit=effective_limit
    )
    if include == "raw":
        return [VehiclePositionWithRaw.model_validate(r) for r in rows]
    return [VehiclePositionResponse.model_validate(r) for r in rows]


@router.get("/routes/{route_id}/metrics", response_model=RouteMetricsResponse)
def route_metrics_endpoint(
    route_id: str,
    window_minutes: int = Query(default=15, ge=1, le=180),
    db: Session = Depends(get_db),
) -> RouteMetricsResponse:
    """Fleet-level metrics over the latest observation per vehicle on a route."""
    m = route_metrics(db, route_id, window_minutes=window_minutes)
    return RouteMetricsResponse(
        route_id=m.route_id,
        window_minutes=m.window_minutes,
        active_vehicle_count=m.active_vehicle_count,
        latest_fetched_at=m.latest_fetched_at,
        avg_speed_mps=m.avg_speed_mps,
        max_speed_mps=m.max_speed_mps,
        status_breakdown=m.status_breakdown,
        occupancy_breakdown=m.occupancy_breakdown,
        bbox=BoundingBox(**m.bbox) if m.bbox else None,
    )
