"""Feed health aggregates derived from ``feed_fetch_logs``."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import Integer, case, func, select
from sqlalchemy.orm import Session

from db.models.feed_fetch_log import FeedFetchLog


@dataclass
class FeedStatus:
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
    lag_seconds: float | None  # now() - last_feed_header_timestamp


def feed_status(session: Session, feed_name: str) -> FeedStatus:
    """Return a summary of the last fetch and the last-hour success rate."""
    now = datetime.now(tz=timezone.utc)
    one_hour_ago = now - timedelta(hours=1)

    last = session.execute(
        select(FeedFetchLog)
        .where(FeedFetchLog.feed_name == feed_name)
        .order_by(FeedFetchLog.fetched_at.desc())
        .limit(1)
    ).scalars().first()

    last_success = session.execute(
        select(FeedFetchLog)
        .where(FeedFetchLog.feed_name == feed_name)
        .where(FeedFetchLog.success.is_(True))
        .order_by(FeedFetchLog.fetched_at.desc())
        .limit(1)
    ).scalars().first()

    success_as_int = case((FeedFetchLog.success.is_(True), 1), else_=0).cast(Integer)
    row = session.execute(
        select(
            func.count().label("total"),
            func.coalesce(func.sum(success_as_int), 0).label("succ"),
        )
        .where(FeedFetchLog.feed_name == feed_name)
        .where(FeedFetchLog.fetched_at >= one_hour_ago)
    ).one()
    total = int(row.total or 0)
    succ = int(row.succ or 0)
    fail = total - succ
    rate: float | None = (succ / total) if total else None

    header_ts = last.feed_header_timestamp if last else None
    lag = (now - header_ts).total_seconds() if header_ts else None

    return FeedStatus(
        feed_name=feed_name,
        last_fetched_at=last.fetched_at if last else None,
        last_success_at=last_success.fetched_at if last_success else None,
        last_http_status=last.http_status if last else None,
        last_error_type=last.error_type if last else None,
        last_error_message=last.error_message if last else None,
        last_entity_count=last.entity_count if last else None,
        last_feed_header_timestamp=header_ts,
        fetches_last_hour=total,
        successes_last_hour=succ,
        failures_last_hour=fail,
        success_rate_last_hour=rate,
        lag_seconds=lag,
    )


def recent_fetches(
    session: Session,
    feed_name: str,
    *,
    limit: int = 50,
) -> list[FeedFetchLog]:
    """The last N fetch-log rows, newest-first."""
    return list(
        session.execute(
            select(FeedFetchLog)
            .where(FeedFetchLog.feed_name == feed_name)
            .order_by(FeedFetchLog.fetched_at.desc())
            .limit(limit)
        ).scalars().all()
    )
