"""initial schema: fetch logs, raw snapshots, trip_updates, stop_times

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-21
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import TIMESTAMP

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "feed_fetch_logs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("feed_name", sa.String(length=32), nullable=False),
        sa.Column("feed_url", sa.String(length=512), nullable=False),
        sa.Column(
            "fetched_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("http_status", sa.SmallInteger(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("response_bytes", sa.Integer(), nullable=True),
        sa.Column("feed_header_timestamp", TIMESTAMP(timezone=True), nullable=True),
        sa.Column("entity_count", sa.Integer(), nullable=True),
        sa.Column("error_type", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_fetch_logs_feed_fetched_desc",
        "feed_fetch_logs",
        ["feed_name", sa.text("fetched_at DESC")],
    )
    op.create_index(
        "ix_fetch_logs_success_fetched_desc",
        "feed_fetch_logs",
        ["success", sa.text("fetched_at DESC")],
    )

    op.create_table(
        "raw_gtfsrt_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("fetch_log_id", sa.BigInteger(), nullable=False),
        sa.Column("feed_name", sa.String(length=32), nullable=False),
        sa.Column("fetched_at", TIMESTAMP(timezone=True), nullable=False),
        sa.Column("feed_header_timestamp", TIMESTAMP(timezone=True), nullable=True),
        sa.Column("gtfs_realtime_version", sa.String(length=16), nullable=True),
        sa.Column("incrementality", sa.String(length=16), nullable=True),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.LargeBinary(), nullable=False),
        sa.ForeignKeyConstraint(
            ["fetch_log_id"],
            ["feed_fetch_logs.id"],
            ondelete="CASCADE",
            name="fk_snapshots_fetch_log_id",
        ),
        sa.UniqueConstraint("fetch_log_id", name="uq_snapshots_fetch_log_id"),
    )
    op.create_index(
        "ix_snapshots_feed_fetched_desc",
        "raw_gtfsrt_snapshots",
        ["feed_name", sa.text("fetched_at DESC")],
    )
    op.create_index(
        "ix_snapshots_content_sha256",
        "raw_gtfsrt_snapshots",
        ["content_sha256"],
    )

    op.create_table(
        "trip_updates",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("snapshot_id", sa.BigInteger(), nullable=False),
        sa.Column("entity_id", sa.String(length=128), nullable=False),
        sa.Column("trip_id", sa.String(length=64), nullable=True),
        sa.Column("route_id", sa.String(length=32), nullable=True),
        sa.Column("direction_id", sa.SmallInteger(), nullable=True),
        sa.Column("start_date", sa.String(length=8), nullable=True),
        sa.Column("start_time", sa.String(length=8), nullable=True),
        sa.Column("schedule_relationship", sa.String(length=16), nullable=True),
        sa.Column("vehicle_id", sa.String(length=64), nullable=True),
        sa.Column("vehicle_label", sa.String(length=64), nullable=True),
        sa.Column("vehicle_license_plate", sa.String(length=32), nullable=True),
        sa.Column("trip_update_timestamp", TIMESTAMP(timezone=True), nullable=True),
        sa.Column("delay_seconds", sa.Integer(), nullable=True),
        sa.Column("fetched_at", TIMESTAMP(timezone=True), nullable=False),
        sa.Column("feed_header_timestamp", TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["raw_gtfsrt_snapshots.id"],
            ondelete="CASCADE",
            name="fk_trip_updates_snapshot_id",
        ),
    )
    op.create_index(
        "ix_trip_updates_trip_fetched_desc",
        "trip_updates",
        ["trip_id", sa.text("fetched_at DESC")],
    )
    op.create_index(
        "ix_trip_updates_route_fetched_desc",
        "trip_updates",
        ["route_id", sa.text("fetched_at DESC")],
    )
    op.create_index(
        "ix_trip_updates_vehicle_fetched_desc",
        "trip_updates",
        ["vehicle_id", sa.text("fetched_at DESC")],
    )
    op.create_index(
        "ix_trip_updates_fetched_desc",
        "trip_updates",
        [sa.text("fetched_at DESC")],
    )
    op.create_index("ix_trip_updates_snapshot", "trip_updates", ["snapshot_id"])

    op.create_table(
        "trip_update_stop_times",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("trip_update_id", sa.BigInteger(), nullable=False),
        sa.Column("stop_sequence", sa.Integer(), nullable=True),
        sa.Column("stop_id", sa.String(length=32), nullable=True),
        sa.Column("arrival_time", TIMESTAMP(timezone=True), nullable=True),
        sa.Column("arrival_delay", sa.Integer(), nullable=True),
        sa.Column("arrival_uncertainty", sa.Integer(), nullable=True),
        sa.Column("departure_time", TIMESTAMP(timezone=True), nullable=True),
        sa.Column("departure_delay", sa.Integer(), nullable=True),
        sa.Column("departure_uncertainty", sa.Integer(), nullable=True),
        sa.Column("schedule_relationship", sa.String(length=16), nullable=True),
        sa.ForeignKeyConstraint(
            ["trip_update_id"],
            ["trip_updates.id"],
            ondelete="CASCADE",
            name="fk_tu_stop_times_trip_update_id",
        ),
    )
    op.create_index(
        "ix_tu_stop_times_trip_update",
        "trip_update_stop_times",
        ["trip_update_id"],
    )
    op.create_index(
        "ix_tu_stop_times_stop",
        "trip_update_stop_times",
        ["stop_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_tu_stop_times_stop", table_name="trip_update_stop_times")
    op.drop_index("ix_tu_stop_times_trip_update", table_name="trip_update_stop_times")
    op.drop_table("trip_update_stop_times")

    op.drop_index("ix_trip_updates_snapshot", table_name="trip_updates")
    op.drop_index("ix_trip_updates_fetched_desc", table_name="trip_updates")
    op.drop_index("ix_trip_updates_vehicle_fetched_desc", table_name="trip_updates")
    op.drop_index("ix_trip_updates_route_fetched_desc", table_name="trip_updates")
    op.drop_index("ix_trip_updates_trip_fetched_desc", table_name="trip_updates")
    op.drop_table("trip_updates")

    op.drop_index("ix_snapshots_content_sha256", table_name="raw_gtfsrt_snapshots")
    op.drop_index("ix_snapshots_feed_fetched_desc", table_name="raw_gtfsrt_snapshots")
    op.drop_table("raw_gtfsrt_snapshots")

    op.drop_index("ix_fetch_logs_success_fetched_desc", table_name="feed_fetch_logs")
    op.drop_index("ix_fetch_logs_feed_fetched_desc", table_name="feed_fetch_logs")
    op.drop_table("feed_fetch_logs")
