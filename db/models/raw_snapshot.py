"""Raw GTFS-RT protobuf snapshot, one row per successful fetch.

Keeping the bytes lets us re-normalize if the parser changes and lets the
future /replay endpoint serve byte-identical historical payloads.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, LargeBinary, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base

if TYPE_CHECKING:
    from db.models.feed_fetch_log import FeedFetchLog


class RawGtfsrtSnapshot(Base):
    __tablename__ = "raw_gtfsrt_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    fetch_log_id: Mapped[int] = mapped_column(
        ForeignKey("feed_fetch_logs.id", ondelete="CASCADE"),
        nullable=False,
    )
    feed_name: Mapped[str] = mapped_column(String(32), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    feed_header_timestamp: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    gtfs_realtime_version: Mapped[str | None] = mapped_column(String(16), nullable=True)
    incrementality: Mapped[str | None] = mapped_column(String(16), nullable=True)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    fetch_log: Mapped["FeedFetchLog"] = relationship(back_populates="snapshot")

    __table_args__ = (
        UniqueConstraint("fetch_log_id", name="uq_snapshots_fetch_log_id"),
        Index("ix_snapshots_feed_fetched_desc", "feed_name", "fetched_at"),
        Index("ix_snapshots_content_sha256", "content_sha256"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return (
            f"<RawGtfsrtSnapshot id={self.id} feed={self.feed_name} "
            f"bytes={len(self.payload) if self.payload else 0}>"
        )
