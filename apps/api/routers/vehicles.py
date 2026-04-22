"""Vehicle-centric endpoints.

  GET /vehicles/latest[?route_id=&limit=]     — latest row per vehicle
  GET /vehicles/{vehicle_id}/latest           — single latest row (404 if none)
  GET /vehicles/{vehicle_id}/history          — append-ordered history window
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from apps.api.schemas import VehiclePositionResponse, VehiclePositionWithRaw
from core.config import get_settings
from db.queries.vehicles import (
    latest_vehicle_position,
    latest_vehicle_positions,
    vehicle_history,
)

router = APIRouter(prefix="/vehicles", tags=["vehicles"])


def _render(row, include: str | None):
    if include == "raw":
        return VehiclePositionWithRaw.model_validate(row)
    return VehiclePositionResponse.model_validate(row)


@router.get("/latest")
def vehicles_latest(
    route_id: str | None = Query(default=None),
    limit: int | None = Query(default=None, ge=1),
    minutes: int = Query(default=5, ge=1, le=180, description="Only vehicles seen within this window"),
    include: Literal["raw"] | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Latest observation per ``vehicle_id``, optionally filtered to a route.

    ``minutes`` bounds staleness — default 5 min keeps the map clean; set
    higher for sparse debugging or to see recently-parked vehicles.
    """
    settings = get_settings()
    effective_limit = min(limit or settings.max_page_size, settings.max_page_size)
    since = datetime.now(tz=timezone.utc) - timedelta(minutes=minutes)
    rows = latest_vehicle_positions(
        db, route_id=route_id, since=since, limit=effective_limit
    )
    return [_render(r, include) for r in rows]


@router.get("/{vehicle_id}/latest")
def vehicle_latest(
    vehicle_id: str,
    include: Literal["raw"] | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Single most-recent observation for a vehicle (404 if never seen)."""
    row = latest_vehicle_position(db, vehicle_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no observations for vehicle_id={vehicle_id}")
    return _render(row, include)


@router.get("/{vehicle_id}/history")
def vehicle_history_endpoint(
    vehicle_id: str,
    start: datetime = Query(...),
    end: datetime = Query(...),
    limit: int | None = Query(default=None, ge=1),
    include: Literal["raw"] | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Append-ordered observations in [start, end) for one vehicle."""
    if end <= start:
        raise HTTPException(status_code=400, detail="end must be after start")
    settings = get_settings()
    effective_limit = min(limit or settings.max_page_size, settings.max_page_size)
    rows = vehicle_history(db, vehicle_id, start=start, end=end, limit=effective_limit)
    return [_render(r, include) for r in rows]
