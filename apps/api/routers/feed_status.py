"""Feed health endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from apps.api.schemas import FeedStatusResponse, FetchLogEntry
from core.constants import FEED_VEHICLE_POSITIONS
from db.queries.feed_stats import feed_status, recent_fetches

router = APIRouter(prefix="/feed-status", tags=["feed"])


@router.get("/vehicle-positions", response_model=FeedStatusResponse)
def vehicle_positions_status(db: Session = Depends(get_db)) -> FeedStatusResponse:
    """Summary + recent fetches for the VehiclePositions feed."""
    status = feed_status(db, FEED_VEHICLE_POSITIONS)
    logs = recent_fetches(db, FEED_VEHICLE_POSITIONS, limit=50)
    return FeedStatusResponse(
        feed_name=status.feed_name,
        last_fetched_at=status.last_fetched_at,
        last_success_at=status.last_success_at,
        last_http_status=status.last_http_status,
        last_error_type=status.last_error_type,
        last_error_message=status.last_error_message,
        last_entity_count=status.last_entity_count,
        last_feed_header_timestamp=status.last_feed_header_timestamp,
        fetches_last_hour=status.fetches_last_hour,
        successes_last_hour=status.successes_last_hour,
        failures_last_hour=status.failures_last_hour,
        success_rate_last_hour=status.success_rate_last_hour,
        lag_seconds=status.lag_seconds,
        recent=[FetchLogEntry.model_validate(row) for row in logs],
    )
