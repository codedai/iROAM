"""Health and feed-status schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    status: str
    db_ok: bool
    feed_name: str
    now: datetime


class FetchLogEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    feed_name: str
    fetched_at: datetime
    http_status: int | None
    success: bool
    duration_ms: int | None
    response_bytes: int | None
    feed_header_timestamp: datetime | None
    entity_count: int | None
    error_type: str | None
    error_message: str | None


class FeedStatusResponse(BaseModel):
    feed_name: str
    last_fetched_at: datetime | None
    last_success_at: datetime | None
    last_http_status: int | None
    last_error_type: str | None
    last_error_message: str | None
    last_entity_count: int | None
    last_feed_header_timestamp: datetime | None
    fetches_last_hour: int
    successes_last_hour: int
    failures_last_hour: int
    success_rate_last_hour: float | None
    lag_seconds: float | None
    recent: list[FetchLogEntry]
