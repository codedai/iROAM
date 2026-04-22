"""Append-ordered replay over a time window."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from apps.api.schemas import VehiclePositionResponse, VehiclePositionWithRaw
from core.config import get_settings
from db.queries.vehicles import replay_vehicles

router = APIRouter(prefix="/replay", tags=["replay"])


@router.get("/vehicles")
def replay_vehicles_endpoint(
    start: datetime = Query(..., description="Inclusive UTC start."),
    end: datetime = Query(..., description="Exclusive UTC end."),
    route_id: str | None = Query(default=None),
    limit: int | None = Query(default=None, ge=1),
    include: Literal["raw"] | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Append-ordered vehicle positions for map replay."""
    if end <= start:
        raise HTTPException(status_code=400, detail="end must be after start")
    settings = get_settings()
    effective_limit = min(limit or settings.max_page_size, settings.max_page_size)
    rows = replay_vehicles(
        db,
        start=start,
        end=end,
        route_id=route_id,
        limit=effective_limit,
    )
    if include == "raw":
        return [VehiclePositionWithRaw.model_validate(r) for r in rows]
    return [VehiclePositionResponse.model_validate(r) for r in rows]
