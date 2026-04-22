"""Health check."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from apps.api.schemas import HealthResponse
from core.constants import CANONICAL_FEED
from core.time import utc_now

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - we want db_ok=false for any error
        db_ok = False
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        db_ok=db_ok,
        feed_name=CANONICAL_FEED,
        now=utc_now(),
    )
