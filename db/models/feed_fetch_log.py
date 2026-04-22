"""Log of every fetch attempt against a GTFS-RT feed.

Every poll — success or failure — creates exactly one row here. On success,
exactly one child ``RawGtfsrtSnapshot`` row is also written.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Index, Integer, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base

if TYPE_CHECKING:
    from db.models.raw_snapshot import RawGtfsrtSnapshot


class FeedFetchLog(Base):
    __tablename__ = "feed_fetch_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    feed_name: Mapped[str] = mapped_column(String(32), nullable=False)
    feed_url: Mapped[str] = mapped_column(String(512), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    http_status: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feed_header_timestamp: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    entity_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    snapshot: Mapped["RawGtfsrtSnapshot | None"] = relationship(
        back_populates="fetch_log",
        uselist=False,
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_fetch_logs_feed_fetched_desc", "feed_name", "fetched_at"),
        Index("ix_fetch_logs_success_fetched_desc", "success", "fetched_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return (
            f"<FeedFetchLog id={self.id} feed={self.feed_name} "
            f"success={self.success} at={self.fetched_at.isoformat() if self.fetched_at else None}>"
        )
