"""pivot to vehicle positions

Drops the trip-updates schema, enables PostGIS, creates ``vehicle_positions``
with a generated ``geom`` column and a GiST index.

Revision ID: 0002_pivot_vp
Revises: 0001_initial
Create Date: 2026-04-21
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002_pivot_vp"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. PostGIS extension — required for the generated geom column.
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    # 2. Drop the previous TripUpdates schema (children first). Indexes drop
    # implicitly with their tables — using raw SQL with IF EXISTS keeps the
    # migration idempotent regardless of which 0001 indexes landed.
    op.execute("DROP TABLE IF EXISTS trip_update_stop_times CASCADE")
    op.execute("DROP TABLE IF EXISTS trip_updates CASCADE")

    # 3. Create vehicle_positions (columns only; geom added separately as GENERATED).
    op.create_table(
        "vehicle_positions",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "snapshot_id",
            sa.BigInteger,
            sa.ForeignKey("raw_gtfsrt_snapshots.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("fetched_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("feed_header_timestamp", postgresql.TIMESTAMP(timezone=True)),
        sa.Column("entity_id", sa.String(128), nullable=False),
        sa.Column("vehicle_timestamp", postgresql.TIMESTAMP(timezone=True)),
        sa.Column("vehicle_id", sa.String(64)),
        sa.Column("vehicle_label", sa.String(64)),
        sa.Column("trip_id", sa.String(64)),
        sa.Column("route_id", sa.String(32)),
        sa.Column("direction_id", sa.SmallInteger),
        sa.Column("start_date", sa.String(8)),
        sa.Column("start_time", sa.String(8)),
        sa.Column("schedule_relationship", sa.String(16)),
        sa.Column("latitude", sa.Float(precision=53)),
        sa.Column("longitude", sa.Float(precision=53)),
        sa.Column("bearing", sa.Float),
        sa.Column("odometer", sa.Float(precision=53)),
        sa.Column("speed_mps", sa.Float),
        sa.Column("current_status", sa.String(32)),
        sa.Column("current_stop_sequence", sa.Integer),
        sa.Column("stop_id", sa.String(32)),
        sa.Column("occupancy_status", sa.String(32)),
        sa.Column("occupancy_percentage", sa.SmallInteger),
        sa.Column("congestion_level", sa.String(32)),
        sa.Column("raw_entity", postgresql.JSONB, nullable=False),
    )

    # 4. Generated geom column — can't be expressed via SA's Column() cleanly.
    op.execute(
        """
        ALTER TABLE vehicle_positions
        ADD COLUMN geom geometry(Point, 4326)
        GENERATED ALWAYS AS (
            CASE
                WHEN longitude IS NOT NULL AND latitude IS NOT NULL
                THEN ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
                ELSE NULL
            END
        ) STORED
        """
    )

    # 5. Indexes
    op.create_index(
        "ix_vp_vehicle_fetched",
        "vehicle_positions",
        ["vehicle_id", sa.text("fetched_at DESC")],
    )
    op.create_index(
        "ix_vp_route_fetched",
        "vehicle_positions",
        ["route_id", sa.text("fetched_at DESC")],
    )
    op.create_index(
        "ix_vp_trip_fetched",
        "vehicle_positions",
        ["trip_id", sa.text("fetched_at DESC")],
    )
    op.create_index(
        "ix_vp_fetched_at",
        "vehicle_positions",
        [sa.text("fetched_at DESC")],
    )
    op.create_index("ix_vp_snapshot", "vehicle_positions", ["snapshot_id"])
    op.execute("CREATE INDEX ix_vp_geom ON vehicle_positions USING GIST (geom)")


def downgrade() -> None:
    # Drop VP artifacts (indexes cascade with the table).
    op.execute("DROP TABLE IF EXISTS vehicle_positions CASCADE")

    # Recreate the TripUpdates tables to restore the 0001 state. Data not restored.
    op.create_table(
        "trip_updates",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "snapshot_id",
            sa.BigInteger,
            sa.ForeignKey("raw_gtfsrt_snapshots.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("entity_id", sa.String(128), nullable=False),
        sa.Column("trip_id", sa.String(64)),
        sa.Column("route_id", sa.String(32)),
        sa.Column("direction_id", sa.SmallInteger),
        sa.Column("start_date", sa.String(8)),
        sa.Column("start_time", sa.String(8)),
        sa.Column("schedule_relationship", sa.String(16)),
        sa.Column("vehicle_id", sa.String(64)),
        sa.Column("vehicle_label", sa.String(64)),
        sa.Column("vehicle_license_plate", sa.String(32)),
        sa.Column("trip_update_timestamp", postgresql.TIMESTAMP(timezone=True)),
        sa.Column("delay_seconds", sa.Integer),
        sa.Column("fetched_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("feed_header_timestamp", postgresql.TIMESTAMP(timezone=True)),
    )
    op.create_table(
        "trip_update_stop_times",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "trip_update_id",
            sa.BigInteger,
            sa.ForeignKey("trip_updates.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stop_sequence", sa.Integer),
        sa.Column("stop_id", sa.String(32)),
        sa.Column("arrival_time", postgresql.TIMESTAMP(timezone=True)),
        sa.Column("arrival_delay", sa.Integer),
        sa.Column("arrival_uncertainty", sa.Integer),
        sa.Column("departure_time", postgresql.TIMESTAMP(timezone=True)),
        sa.Column("departure_delay", sa.Integer),
        sa.Column("departure_uncertainty", sa.Integer),
        sa.Column("schedule_relationship", sa.String(16)),
    )
    # PostGIS extension left in place — harmless if nothing uses it.
