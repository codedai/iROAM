"""Replay-oriented queries.

Thin wrappers that exist so future additions (replay of raw payloads,
reconstruction from snapshots) have a natural home.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models.raw_snapshot import RawGtfsrtSnapshot


def snapshots_in_window(
    session: Session,
    *,
    feed_name: str,
    start: datetime,
    end: datetime,
    limit: int = 500,
) -> Sequence[RawGtfsrtSnapshot]:
    """All snapshots for a feed in a time window, oldest-first."""
    stmt = (
        select(RawGtfsrtSnapshot)
        .where(RawGtfsrtSnapshot.feed_name == feed_name)
        .where(RawGtfsrtSnapshot.fetched_at >= start)
        .where(RawGtfsrtSnapshot.fetched_at <= end)
        .order_by(RawGtfsrtSnapshot.fetched_at.asc())
        .limit(limit)
    )
    return session.execute(stmt).scalars().all()
